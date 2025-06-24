from types import SimpleNamespace
from typing import OrderedDict
import torch
# import torchvision
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from visual_net import resnet18

from ipdb import set_trace
import timm
from video_swin_wrapper import Swin3DWrapper
from einops import rearrange, repeat
import os

import loralib as lora
from transformers.activations import get_activation
import copy
from torch.nn import MultiheadAttention
import random
import torch.utils.checkpoint as checkpoint

from htsat import HTSAT_Swin_Transformer
import esc_config as esc_config
from utils import do_mixup, get_mix_lambda, do_mixup_label
from prompted_models import PromptedSwinV2

class VisualAdapter(nn.Module):
	"""Conventional Adapter layer, in which the weights of up and down sampler modules
	are parameters and are optimized."""

	def __init__(self, input_dim, output_dim, adapter_kind, dim_list=None, layer_idx=0, reduction_factor=16, opt=None ,use_bn=True, use_gate=True, conv_dim_in=0, conv_dim_out=0, linear_in=0, linear_out=0):
		super().__init__()
		self.adapter_kind = adapter_kind
		self.use_bn = use_bn
		self.is_multimodal = opt.is_multimodal
		self.opt = opt
		self.conv_adapter = nn.Conv2d(conv_dim_in, conv_dim_out, kernel_size=1)  
		self.fc = nn.Linear(linear_in, linear_out)
        
		d_model = linear_out // 2
		self.fc_affine_audio_1 = nn.Linear(linear_out, linear_out)
		self.fc_affine_video_1 = nn.Linear(linear_out, linear_out)
		self.fc_affine_bottleneck = nn.Linear(linear_out, d_model)
		self.fc_affine_video_2 = nn.Linear(linear_out, d_model)
		self.fc_affine_audio_2 = nn.Linear(linear_out, d_model)     
		self.fc_affine_v_s_att = nn.Linear(d_model, 1) 
		self.fc_tanh = nn.Tanh()
		self.fc_softmax = nn.Softmax(dim=-1)
		self.fc_affine_v_c_att = nn.Linear(d_model, linear_out)
        
		if use_gate:
			self.gate = nn.Parameter(torch.zeros(1))
		else:
			self.gate = None

		if adapter_kind == "bottleneck" and self.is_multimodal:
			self.down_sample_size = input_dim // reduction_factor


			self.my_tokens = nn.Parameter(torch.zeros((self.opt.num_tokens, input_dim)))
			self.gate_av = nn.Parameter(torch.zeros(1))

			### <------

			self.activation = nn.ReLU(inplace=True)
			self.down_sampler = nn.Conv2d(input_dim, self.down_sample_size, 1, groups=self.opt.num_conv_group, bias=False)
			self.up_sampler = nn.Conv2d(self.down_sample_size, output_dim, 1, groups=self.opt.num_conv_group, bias=False)

			if use_bn:
				self.bn1 = nn.BatchNorm2d(self.down_sample_size)
				self.bn2 = nn.BatchNorm2d(output_dim)
			
			if self.opt.is_before_layernorm:
				self.ln_before = nn.LayerNorm(output_dim)
			if self.opt.is_post_layernorm:
				self.ln_post = nn.LayerNorm(output_dim)
			### <---------

		elif adapter_kind == "bottleneck":
			self.down_sample_size = input_dim // reduction_factor
			self.activation = nn.ReLU(inplace=True)
			
			self.down_sampler = nn.Conv2d(input_dim, self.down_sample_size, 1, groups=self.opt.num_conv_group, bias=False)

			self.up_sampler = nn.Conv2d(self.down_sample_size, output_dim, 1, groups=self.opt.num_conv_group, bias=False)

			if use_bn:
				self.bn1 = nn.BatchNorm2d(self.down_sample_size)
				self.bn2 = nn.BatchNorm2d(output_dim)

			if self.opt.is_before_layernorm:
				self.ln_before = nn.LayerNorm(output_dim)
			if self.opt.is_post_layernorm:
				self.ln_post = nn.LayerNorm(output_dim)
			### <---------

		elif adapter_kind == "basic":
			self.activation = nn.ReLU(inplace=True)
			self.conv = nn.Linear(input_dim, output_dim, bias=False)

			if use_bn:
				self.bn = nn.BatchNorm1d(output_dim)

		else:
			raise NotImplementedError

	def forward(self, x, vis_token=None):
		vis_token = self.conv_adapter(vis_token.transpose(2, 1))
		vis_token = self.fc(vis_token.squeeze(-1))
		vis_token = vis_token.permute(0, 2, 1).unsqueeze(-1)
        
		spatial_att_maps = None
		if self.adapter_kind == "bottleneck" and self.is_multimodal:
			
			

			### -------> high dim att
			rep_token = repeat(self.my_tokens, 't d -> b t d', b=x.size(0))


			att_v2tk = torch.bmm(rep_token, vis_token.squeeze(-1))

			att_v2tk = F.softmax(att_v2tk, dim=-1)
			rep_token_res = torch.bmm(att_v2tk, vis_token.squeeze(-1).permute(0,2,1))

			rep_token = rep_token + rep_token_res
			

			att_tk2x = torch.bmm(x.squeeze(-1).permute(0,2,1), rep_token.permute(0,2,1))

			att_tk2x = F.softmax(att_tk2x, dim=-1)
			x_res = torch.bmm(att_tk2x, rep_token).permute(0,2,1).unsqueeze(-1)



			x = x + self.gate_av*x_res.contiguous()
            
			# ============================== Channel Attention ====================================    
			audio = vis_token.mean(dim=2).squeeze(-1) # [B*10, dim]
			audio_query_1 = F.relu(self.fc_affine_audio_1(audio)).unsqueeze(-2)  
			video_query_1 = F.relu(self.fc_affine_video_1(x.squeeze(-1).permute(0, 2, 1))) # [*, grid ** 2, width]       
			audio_video_query_raw = (audio_query_1 * video_query_1).mean(-2) #  [*, width] 
			audio_video_query = F.relu(self.fc_affine_bottleneck(audio_video_query_raw))
			channel_att_maps = self.fc_affine_v_c_att(audio_video_query).sigmoid().reshape(x.size(0), 1, -1)      
			c_att_visual_feat = (x.squeeze(-1).permute(0, 2, 1) * (channel_att_maps + 1)) # [B*10, 36, 768]  

			# ============================== Spatial Attention =====================================
			# channel attended visual feature: [batch * 10, 36, v_dim]
			c_att_visual_query = F.relu(self.fc_affine_video_2(c_att_visual_feat))
			audio_query_2 = F.relu(self.fc_affine_audio_2(audio)).unsqueeze(-2)
			audio_video_query_2 = c_att_visual_query * audio_query_2
			spatial_att_maps_tmp = self.fc_affine_v_s_att(audio_video_query_2) 
			spatial_att_maps_sigmoid = spatial_att_maps_tmp.transpose(2, 1).sigmoid()
			spatial_att_maps_sigmoid = spatial_att_maps_sigmoid.transpose(2, 1)
			spatial_att_maps = self.fc_softmax(self.fc_tanh(spatial_att_maps_tmp).transpose(2, 1))
			c_s_att_visual_feat = torch.bmm(spatial_att_maps, c_att_visual_feat)


# 			beta = 0.3
# 			x = x.squeeze(-1).permute(0, 2, 1) * (beta * channel_att_maps + 1 - beta)
# 			x = x.permute(0, 2, 1).unsqueeze(-1)

			alpha, beta = 0.3, 0.05            
			x = x.squeeze(-1).permute(0, 2, 1) * (alpha * channel_att_maps + beta * spatial_att_maps_sigmoid + 1 - alpha)
			x = x.permute(0, 2, 1).unsqueeze(-1)

			# =======================================================================================

			### <----------
			if self.opt.is_before_layernorm:
				x = self.ln_before(x.squeeze(-1).permute(0,2,1)).permute(0,2,1).unsqueeze(-1)

			z = self.down_sampler(x)
		
			## <----

			if self.use_bn:
				z = self.bn1(z)

			z = self.activation(z)
			output = self.up_sampler(z)
			if self.use_bn:
				output = self.bn2(output)
	
		elif self.adapter_kind == "bottleneck":
			
			if self.opt.is_before_layernorm:
				x = self.ln_before(x.squeeze(-1).permute(0,2,1)).permute(0,2,1).unsqueeze(-1)
	
			z = self.down_sampler(x)

			if self.use_bn:
				z = self.bn1(z)

			z = self.activation(z)
			output = self.up_sampler(z)
			if self.use_bn:
				output = self.bn2(output)
			

		elif self.adapter_kind == "basic":
			output = self.conv(x)
			if self.use_bn:
				output = self.bn(rearrange(output, 'N C L -> N L C') )
				output = rearrange(output, 'N L C -> N C L')


		if self.opt.is_post_layernorm:
			output = self.ln_post(output.squeeze(-1).permute(0,2,1)).permute(0,2,1).unsqueeze(-1)

		if self.gate is not None:
			output = self.gate * output



	

		return output, spatial_att_maps



def batch_organize(out_match_posi, out_match_nega):
	# audio B 512
	# posi B 512
	# nega B 512

	out_match = torch.zeros(out_match_posi.shape[0] * 2, out_match_posi.shape[1])
	batch_labels = torch.zeros(out_match_posi.shape[0] * 2)
	for i in range(out_match_posi.shape[0]):
		out_match[i * 2, :] = out_match_posi[i, :]
		out_match[i * 2 + 1, :] = out_match_nega[i, :]
		batch_labels[i * 2] = 1
		batch_labels[i * 2 + 1] = 0
	
	return out_match, batch_labels

# Question
class QstEncoder(nn.Module):

	def __init__(self, qst_vocab_size, word_embed_size, embed_size, num_layers, hidden_size):

		super(QstEncoder, self).__init__()
		self.word2vec = nn.Embedding(qst_vocab_size, word_embed_size)
		self.tanh = nn.Tanh()
		self.lstm = nn.LSTM(word_embed_size, hidden_size, num_layers)
		self.fc = nn.Linear(2*num_layers*hidden_size, embed_size)     # 2 for hidden and cell states

	def forward(self, question):

		qst_vec = self.word2vec(question)                             # [batch_size, max_qst_length=30, word_embed_size=300]
		qst_vec = self.tanh(qst_vec)
		qst_vec = qst_vec.transpose(0, 1)                             # [max_qst_length=30, batch_size, word_embed_size=300]
		self.lstm.flatten_parameters()
		_, (hidden, cell) = self.lstm(qst_vec)                        # [num_layers=2, batch_size, hidden_size=512]
		qst_feature = torch.cat((hidden, cell), 2)                    # [num_layers=2, batch_size, 2*hidden_size=1024]
		qst_feature = qst_feature.transpose(0, 1)                     # [batch_size, num_layers=2, 2*hidden_size=1024]
		qst_feature = qst_feature.reshape(qst_feature.size()[0], -1)  # [batch_size, 2*num_layers*hidden_size=2048]
		qst_feature = self.tanh(qst_feature)
		qst_feature = self.fc(qst_feature)                            # [batch_size, embed_size]

		return qst_feature


import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from einops import rearrange
from models.q_encoder import QstEncoder  # 기존과 동일한 질문 인코더 사용 가정


class AVQA_Fusion_Net(nn.Module):
    def __init__(self, opt):
        super(AVQA_Fusion_Net, self).__init__()
        self.opt = opt

        # InternVideo2 Stage2 (video/audio encoders)
        self.internvideo2 = AutoModel.from_pretrained("OpenGVLab/InternVideo2-Stage2_6B")
        self.swin = self.internvideo2.video_encoder
        self.htsat = self.internvideo2.audio_encoder

        # Feature projection
        self.fc_a1 = nn.Linear(768, 1024)
        self.fc_a2 = nn.Linear(1024, 1024)
        self.fc_gl = nn.Linear(1024 + 1024, 1024)

        self.fc1 = nn.Linear(1024 + 1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 2)

        self.question_encoder = QstEncoder(93, 1024, 1024, 1, 1024)
        self.fc_ans = nn.Linear(1024, 42)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.tanh = nn.Tanh()

        self.linear11 = nn.Linear(1024, 1024)
        self.linear12 = nn.Linear(1024, 1024)
        self.linear21 = nn.Linear(1024, 1024)
        self.linear22 = nn.Linear(1024, 1024)

        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)
        self.dropout3 = nn.Dropout(0.1)
        self.dropout4 = nn.Dropout(0.1)

        self.norm1 = nn.LayerNorm(1024)
        self.norm2 = nn.LayerNorm(1024)
        self.norm3 = nn.LayerNorm(1024)

        self.attn_a = nn.MultiheadAttention(1024, 4, dropout=0.1)
        self.attn_v = nn.MultiheadAttention(1024, 4, dropout=0.1)

    def forward(self, audio, visual_posi, visual_nega, question, mixup_lambda=None, stage='eval'):
        bs, t, c, h, w = visual_posi.shape
        visual_posi = rearrange(visual_posi, 'b t c h w -> b c t h w')
        visual_nega = rearrange(visual_nega, 'b t c h w -> b c t h w')
        audio = rearrange(audio, 'b t d -> (b t) d')

        f_v = self.swin(visual_posi)  # [B, L, C]
        f_v_neg = self.swin(visual_nega)

        f_a = self.htsat(audio)  # [B*T, D]
        f_a = f_a.mean(dim=1) if f_a.ndim == 3 else f_a  # 평균 pooling if needed
        audio = rearrange(f_a, '(b t) c -> b t c', b=bs, t=t)

        audio_feat = F.relu(self.fc_a1(audio))
        audio_feat = self.fc_a2(audio_feat)
        audio_feat_pure = audio_feat
        B, T, C = audio_feat.size()
        audio_feat = audio_feat.view(B*T, C)

        def grounding(visual_input):
            temp_visual = visual_input.view(B*T, C, h, w)
            visual_feat_before = self.avgpool(temp_visual).squeeze()
            v_feat = temp_visual.view(B*T, C, h * w).permute(0, 2, 1)
            visual_feat_norm = F.normalize(v_feat, dim=2)
            audio_feat_aa = F.normalize(audio_feat.unsqueeze(-1), dim=1)
            x2_va = torch.matmul(visual_feat_norm, audio_feat_aa).squeeze()
            x2_p = F.softmax(x2_va, dim=-1).unsqueeze(-2)
            visual_feat_grd = torch.matmul(x2_p, visual_feat_norm).squeeze()
            visual_gl = torch.cat((visual_feat_before, visual_feat_grd), dim=-1)
            visual_feat_fused = self.fc_gl(self.tanh(visual_gl))
            return visual_feat_fused

        visual_feat_grd_posi = grounding(f_v)
        visual_feat_grd_nega = grounding(f_v_neg)

        feat_posi = F.relu(self.fc1(torch.cat((audio_feat, visual_feat_grd_posi), dim=-1)))
        feat_posi = F.relu(self.fc2(feat_posi))
        feat_posi = F.relu(self.fc3(feat_posi))
        out_match_posi = self.fc4(feat_posi)

        feat_nega = F.relu(self.fc1(torch.cat((audio_feat, visual_feat_grd_nega), dim=-1)))
        feat_nega = F.relu(self.fc2(feat_nega))
        feat_nega = F.relu(self.fc3(feat_nega))
        out_match_nega = self.fc4(feat_nega)

        qst_feature = self.question_encoder(question)
        xq = qst_feature.unsqueeze(0)

        visual_feat_att = self.attn_v(xq, visual_feat_grd_posi.unsqueeze(0), visual_feat_grd_posi.unsqueeze(0))[0].squeeze(0)
        src = self.linear12(self.dropout1(F.relu(self.linear11(visual_feat_att))))
        visual_feat_att = self.norm1(visual_feat_att + self.dropout2(src))

        audio_feat_att = self.attn_a(xq, audio_feat_pure.view(B, T, -1).permute(1, 0, 2), audio_feat_pure.view(B, T, -1).permute(1, 0, 2))[0].squeeze(0)
        src = self.linear22(self.dropout3(F.relu(self.linear21(audio_feat_att))))
        audio_feat_att = self.norm2(audio_feat_att + self.dropout4(src))

        feat = self.tanh(torch.cat((audio_feat_att + audio_feat_pure.mean(dim=1),
                                    visual_feat_att + visual_feat_grd_posi.view(B, T, -1).mean(dim=1)), dim=-1))
        feat = self.fc_fusion(feat)

        combined_feature = self.tanh(torch.mul(feat, qst_feature))
        out_qa = self.fc_ans(combined_feature)

        return out_qa, out_match_posi, out_match_nega
