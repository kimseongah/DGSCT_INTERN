from types import SimpleNamespace
import torch
# import torchvision
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from visual_net import resnet18

from ipdb import set_trace
import timm
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
from InternVideo.InternVideo2.multi_modality.models.internvideo2_stage2 import InternVideo2_Stage2
from InternVideo.InternVideo2.multi_modality.configs.model import TextEncoders
from transformers import BertTokenizer
from omegaconf import OmegaConf
from InternVideo.InternVideo2.multi_modality.models.backbones.beats.BEATs import BEATs, BEATsConfig



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
		qst_feature = torch.cat((hidden, cell), 2)                    # [num_layers=2, batch_size, 2*hidden_size=3200]
		qst_feature = qst_feature.transpose(0, 1)                     # [batch_size, num_layers=2, 2*hidden_size=3200]
		qst_feature = qst_feature.reshape(qst_feature.size()[0], -1)  # [batch_size, 2*num_layers*hidden_size=2048]
		qst_feature = self.tanh(qst_feature)
		qst_feature = self.fc(qst_feature)                            # [batch_size, embed_size]

		return qst_feature


# class AVQA_Fusion_Net(nn.Module):

# 	def __init__(self, opt):
# 		super(AVQA_Fusion_Net, self).__init__()

# 		self.opt = opt

# 		internvideo2 = AutoModel.from_pretrained("OpenGVLab/InternVideo2-Stage2_6B", torch_dtype=torch.bfloat16, trust_remote_code=True)
# 		self.swin = internvideo2.vision_encoder  # 영상 인코더 추출
# 		del internvideo2
# 		torch.cuda.empty_cache() 
# 		audio_cfg = BEATsConfig()
# 		audio_cfg.input_patch_size = 4
# 		self.htsat = BEATs(audio_cfg)

# 		# for features
# 		self.fc_a1 =  nn.Linear(768, 3200)
# 		self.fc_a2=nn.Linear(3200, 3200)

# 		self.fc_a1_pure =  nn.Linear(768, 3200)
# 		self.fc_a2_pure=nn.Linear(3200, 3200)
        
# 		self.fc_fusion = nn.Linear(3200+3200, 3200)

# 		self.linear11 = nn.Linear(3200, 3200)
# 		self.dropout1 = nn.Dropout(0.1)
# 		self.linear12 = nn.Linear(3200, 3200)

# 		self.linear21 = nn.Linear(3200, 3200)
# 		self.dropout2 = nn.Dropout(0.1)
# 		self.linear22 = nn.Linear(3200, 3200)
# 		self.norm1 = nn.LayerNorm(3200)
# 		self.norm2 = nn.LayerNorm(3200)
# 		self.dropout3 = nn.Dropout(0.1)
# 		self.dropout4 = nn.Dropout(0.1)
# 		self.norm3 = nn.LayerNorm(3200)

# 		self.attn_a = nn.MultiheadAttention(3200, 4, dropout=0.1)
# 		self.attn_v = nn.MultiheadAttention(3200, 4, dropout=0.1)

# 		# question
# 		self.question_encoder = QstEncoder(93, 3200, 3200, 1, 3200)

# 		self.tanh = nn.Tanh()
# 		self.dropout = nn.Dropout(0.5)
# 		self.fc_ans = nn.Linear(3200, 42)

# 		self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
# 		self.fc_gl=nn.Linear(3200+3200, 3200)

# 		# combine
# 		self.fc1 = nn.Linear(3200+3200, 512)
# 		self.relu1 = nn.ReLU()
# 		self.fc2 = nn.Linear(512, 256)
# 		self.relu2 = nn.ReLU()
# 		self.fc3 = nn.Linear(256, 128)
# 		self.relu3 = nn.ReLU()
# 		self.fc4 = nn.Linear(128, 2)
# 		self.relu4 = nn.ReLU()

# 		hidden_list, hidden_list_a = [], []
# 		down_in_dim, down_in_dim_a = [], []
# 		down_out_dim, down_out_dim_a = [], []
# 		conv_dim, conv_dim_a = [], []
        
# 		### ------------> for swin and htsat
# 		for idx_layer, (my_blk, my_blk_a) in enumerate(zip(self.swin.layers, self.htsat.layers)):
# 			conv_dim_tmp = (my_blk.input_resolution[0]*my_blk.input_resolution[1])
# 			conv_dim_tmp_a = (my_blk_a.input_resolution[0]*my_blk_a.input_resolution[1])
# 			if not isinstance(my_blk.downsample, nn.Identity):
# 				down_in_dim.append(my_blk.downsample.reduction.in_features)
# 				down_out_dim.append(my_blk.downsample.reduction.out_features)
# 			if my_blk_a.downsample is not None:
# 				down_in_dim_a.append(my_blk_a.downsample.reduction.in_features)
# 				down_out_dim_a.append(my_blk_a.downsample.reduction.out_features)
            
# 			for blk, blk_a in zip(my_blk.blocks, my_blk_a.blocks):
# 				hidden_d_size = blk.norm1.normalized_shape[0]
# 				hidden_list.append(hidden_d_size)
# 				conv_dim.append(conv_dim_tmp)
# 				hidden_d_size_a = blk_a.norm1.normalized_shape[0]
# 				hidden_list_a.append(hidden_d_size_a)
# 				conv_dim_a.append(conv_dim_tmp_a)
# 		### <--------------


# 		self.audio_adapter_blocks_p1 = nn.ModuleList([
# 			VisualAdapter(input_dim=hidden_list_a[i], output_dim=hidden_list_a[i], adapter_kind="bottleneck",dim_list=hidden_list_a, layer_idx=i,reduction_factor=self.opt.Adapter_downsample, opt=opt, use_bn=self.opt.is_bn, use_gate=self.opt.is_gate, conv_dim_in=conv_dim[i], conv_dim_out=conv_dim_a[i],
# 				linear_in=hidden_list[i], linear_out=hidden_list_a[i])
# 			for i in range(len(hidden_list_a))])

# 		self.vis_adapter_blocks_p1 = nn.ModuleList([
# 			VisualAdapter(input_dim=hidden_list[i], output_dim=hidden_list[i], adapter_kind="bottleneck", dim_list=hidden_list, layer_idx=i, reduction_factor=self.opt.Adapter_downsample, opt=opt, use_bn=self.opt.is_bn, use_gate=True, conv_dim_in=conv_dim_a[i], conv_dim_out=conv_dim[i],
# 				linear_in=hidden_list_a[i], linear_out=hidden_list[i])
# 			for i in range(len(hidden_list))])

# 		self.audio_adapter_blocks_p2 = nn.ModuleList([
# 			VisualAdapter(input_dim=hidden_list_a[i], output_dim=hidden_list_a[i], adapter_kind="bottleneck", dim_list=hidden_list_a, layer_idx=i, reduction_factor=self.opt.Adapter_downsample, opt=opt, use_bn=self.opt.is_bn, use_gate=self.opt.is_gate, conv_dim_in=conv_dim[i], conv_dim_out=conv_dim_a[i],
# 				linear_in=hidden_list[i], linear_out=hidden_list_a[i])
# 			for i in range(len(hidden_list))])

# 		self.vis_adapter_blocks_p2 = nn.ModuleList([
# 			VisualAdapter(input_dim=hidden_list[i], output_dim=hidden_list[i], adapter_kind="bottleneck", dim_list=hidden_list, layer_idx=i, reduction_factor=self.opt.Adapter_downsample, opt=opt, use_bn=self.opt.is_bn, use_gate=True, conv_dim_in=conv_dim_a[i], conv_dim_out=conv_dim[i],
# 				linear_in=hidden_list_a[i], linear_out=hidden_list[i])
# 			for i in range(len(hidden_list_a))])
# 	def to(self, device):
# 		self.swin = self.swin.to(device)
# 		self.audio_encoder = self.audio_encoder.to(device)
# 		self.fc_a1 = self.fc_a1.to(device)
# 		self.fc_a2 = self.fc_a2.to(device)
# 		self.fc_gl = self.fc_gl.to(device)

# 		self.fc1 = self.fc1.to(device)
# 		self.fc2 = self.fc2.to(device)
# 		self.fc3 = self.fc3.to(device)
# 		self.fc4 = self.fc4.to(device)

# 		self.question_encoder = self.question_encoder.to(device)
# 		self.fc_ans = self.fc_ans.to(device)
# 		self.fc_fusion = self.fc_fusion.to(device)

# 		self.avgpool = self.avgpool.to(device)
# 		self.tanh = self.tanh.to(device)

# 		self.linear11 = self.linear11.to(device)
# 		self.linear12 = self.linear12.to(device)
# 		self.linear21 = self.linear21.to(device)
# 		self.linear22 = self.linear22.to(device)

# 		self.dropout1 = self.dropout1.to(device)
# 		self.dropout2 = self.dropout2.to(device)
# 		self.dropout3 = self.dropout3.to(device)
# 		self.dropout4 = self.dropout4.to(device)

# 		self.norm1 = self.norm1.to(device)
# 		self.norm2 = self.norm2.to(device)
# 		self.norm3 = self.norm3.to(device)

# 		self.attn_a = self.attn_a.to(device)
# 		self.attn_v = self.attn_v.to(device)

# 		return self

# 	def forward(self, audio, visual_posi, visual_nega, question, mixup_lambda, stage='eval'):
# 		'''
# 			input question shape:    [B, T]
# 			input audio shape:       [B, T, C]
# 			input visual_posi shape: [B, T, C, H, W]
# 			input visual_nega shape: [B, T, C, H, W]
# 		'''
		
	
# 		bs,t,c,h,w = visual_posi.shape



# 		audio = audio.view(audio.size(0)*audio.size(1), -1)
# 		waveform = audio
# 		bs = visual_posi.size(0)


# 		# [B, T, C, H, W] → [B, C, T, H, W]
# 		visual_posi = visual_posi.permute(0, 2, 1, 3, 4)  # (B, C, T, H, W)
# 		f_v = self.swin(visual_posi)        # → shape: (B, N, D)

# 		# f_v_neg = self.swin.patch_embed(visual_nega)
        
# 		audio = self.htsat.spectrogram_extractor(audio)
# 		audio = self.htsat.logmel_extractor(audio)        
# 		audio = audio.transpose(1, 3)
# 		audio = self.htsat.bn0(audio)
# 		audio = audio.transpose(1, 3)
# 		if self.htsat.training:
# 			audio = self.htsat.spec_augmenter(audio)
# 		if self.htsat.training and mixup_lambda is not None:
# 			audio = do_mixup(audio, mixup_lambda)

# 		if audio.shape[2] > self.htsat.freq_ratio * self.htsat.spec_size:
# 			audio = self.htsat.crop_wav(audio, crop_size=self.htsat.freq_ratio * self.htsat.spec_size)
# 			audio = self.htsat.reshape_wav2img(audio)
# 		else: # this part is typically used, and most easy one
# 			audio = self.htsat.reshape_wav2img(audio)
# 		frames_num = audio.shape[2]
# 		f_a = self.htsat.patch_embed(audio)
# 		if self.htsat.ape:
# 			f_a = f_a + self.htsat.absolute_pos_embed
# 		f_a = self.htsat.pos_drop(f_a)
        
# 		for blk in self.htsat.layers:
# 			for b in blk.blocks:
# 				f_a, _ = b(f_a)
# 			if blk.downsample is not None:
# 				f_a = blk.downsample(f_a)


# 		with torch.no_grad():

# 			visual_nega = visual_nega.permute(0, 2, 1, 3, 4)  # (B, C, T, H, W)
# 			visual_nega = self.swin(visual_nega, task="video")  # (B, N, D)



# 		############## <----------

# 		visual_posi = rearrange(f_v, '(b t) (h w) c -> b t c h w', b=bs ,t=t, h=6 ,w=6)
# 		visual_nega = rearrange(visual_nega, '(b t) (h w) c -> b t c h w', b=bs ,t=t, h=6 ,w=6)


# 		f_a = f_a.mean(dim=1)
# 		# f_a = torch.bmm(f_a_spatial_att_maps, f_a).squeeze(dim=1)     
# 		audio = rearrange(f_a, '(b t) c -> b t c', b=bs ,t=t)
# 		### <-----

# 		# visual_posi, f_a = self.temporal_attn(visual_posi, f_a)
        
# 		## question features
# 		qst_feature = self.question_encoder(question)
# 		xq = qst_feature.unsqueeze(0)

# 		## audio features  [2*B*T, 128]
# 		audio_feat = F.relu(self.fc_a1(audio))
# 		audio_feat = self.fc_a2(audio_feat)  
# 		audio_feat_pure = audio_feat
# 		B, T, C = audio_feat.size()             # [B, T, C]
# 		audio_feat = audio_feat.view(B*T, C)    # [B*T, C]

# 		## visual posi [2*B*T, C, H, W]
# 		B, T, C, H, W = visual_posi.size()
# 		temp_visual = visual_posi.view(B*T, C, H, W)            # [B*T, C, H, W]
# 		v_feat = self.avgpool(temp_visual)                      # [B*T, C, 1, 1]
# 		visual_feat_before_grounding_posi = v_feat.squeeze()    # [B*T, C]

# 		(B, C, H, W) = temp_visual.size()
# 		v_feat = temp_visual.view(B, C, H * W)                      # [B*T, C, HxW]
# 		v_feat = v_feat.permute(0, 2, 1)                            # [B, HxW, C]
# 		visual_feat_posi = nn.functional.normalize(v_feat, dim=2)   # [B, HxW, C]

# 		## audio-visual grounding posi
# 		audio_feat_aa = audio_feat.unsqueeze(-1)                        # [B*T, C, 1]
# 		audio_feat_aa = nn.functional.normalize(audio_feat_aa, dim=1)   # [B*T, C, 1]
	   
# 		x2_va = torch.matmul(visual_feat_posi, audio_feat_aa).squeeze() # [B*T, HxW]

# 		x2_p = F.softmax(x2_va, dim=-1).unsqueeze(-2)                       # [B*T, 1, HxW]
# 		visual_feat_grd = torch.matmul(x2_p, visual_feat_posi)
# 		visual_feat_grd_after_grounding_posi = visual_feat_grd.squeeze()    # [B*T, C]   

# 		visual_gl = torch.cat((visual_feat_before_grounding_posi, visual_feat_grd_after_grounding_posi),dim=-1)
# 		visual_feat_grd = self.tanh(visual_gl)
# 		visual_feat_grd_posi = self.fc_gl(visual_feat_grd)              # [B*T, C]

# 		feat = torch.cat((audio_feat, visual_feat_grd_posi), dim=-1)    # [B*T, C*2], [B*T, 3072]

# 		feat = F.relu(self.fc1(feat))       # (3072, 512)
# 		feat = F.relu(self.fc2(feat))       # (512, 256)
# 		feat = F.relu(self.fc3(feat))       # (256, 128)
# 		out_match_posi = self.fc4(feat)     # (128, 2)

# 		###############################################################################################
# 		# visual nega
# 		B, T, C, H, W = visual_nega.size()
# 		temp_visual = visual_nega.view(B*T, C, H, W)
# 		v_feat = self.avgpool(temp_visual)
# 		visual_feat_before_grounding_nega = v_feat.squeeze() # [B*T, C]

# 		(B, C, H, W) = temp_visual.size()
# 		v_feat = temp_visual.view(B, C, H * W)  # [B*T, C, HxW]
# 		v_feat = v_feat.permute(0, 2, 1)        # [B, HxW, C]
# 		visual_feat_nega = nn.functional.normalize(v_feat, dim=2)

# 		##### av grounding nega
# 		x2_va = torch.matmul(visual_feat_nega, audio_feat_aa).squeeze()
# 		x2_p = F.softmax(x2_va, dim=-1).unsqueeze(-2)                       # [B*T, 1, HxW]
# 		visual_feat_grd = torch.matmul(x2_p, visual_feat_nega)
# 		visual_feat_grd_after_grounding_nega = visual_feat_grd.squeeze()    # [B*T, C]   

# 		visual_gl=torch.cat((visual_feat_before_grounding_nega,visual_feat_grd_after_grounding_nega),dim=-1)
# 		visual_feat_grd=self.tanh(visual_gl)
# 		visual_feat_grd_nega=self.fc_gl(visual_feat_grd)    # [B*T, C]

# 		# combine a and v
# 		feat = torch.cat((audio_feat, visual_feat_grd_nega), dim=-1)   # [B*T, C*2], [B*T, 3200]

# 		feat = F.relu(self.fc1(feat))       # (3200, 512)
# 		feat = F.relu(self.fc2(feat))       # (512, 256)
# 		feat = F.relu(self.fc3(feat))       # (256, 128)
# 		out_match_nega = self.fc4(feat)     # (128, 2)

# 		###############################################################################################

# 		# out_match=None
# 		# match_label=None

# 		B = xq.shape[1]
# 		visual_feat_grd_be = visual_feat_grd_posi.view(B, -1, 3200)   # [B, T, 512]
# 		visual_feat_grd=visual_feat_grd_be.permute(1,0,2)
		
# 		## attention, question as query on visual_feat_grd
# 		visual_feat_att = self.attn_v(xq, visual_feat_grd, visual_feat_grd, attn_mask=None, key_padding_mask=None)[0].squeeze(0)
# 		src = self.linear12(self.dropout1(F.relu(self.linear11(visual_feat_att))))
# 		visual_feat_att = visual_feat_att + self.dropout2(src)
# 		visual_feat_att = self.norm1(visual_feat_att)
	
# 		# attention, question as query on audio
# 		audio_feat_be=audio_feat_pure.view(B, -1, 3200)
# 		audio_feat = audio_feat_be.permute(1, 0, 2)
# 		audio_feat_att = self.attn_a(xq, audio_feat, audio_feat, attn_mask=None,key_padding_mask=None)[0].squeeze(0)
# 		src = self.linear22(self.dropout3(F.relu(self.linear21(audio_feat_att))))
# 		audio_feat_att = audio_feat_att + self.dropout4(src)
# 		audio_feat_att = self.norm2(audio_feat_att)
		
# 		feat = torch.cat((audio_feat_att+audio_feat_be.mean(dim=-2).squeeze(), visual_feat_att+visual_feat_grd_be.mean(dim=-2).squeeze()), dim=-1)
# 		feat = self.tanh(feat)
# 		feat = self.fc_fusion(feat)

# 		## fusion with question
# 		combined_feature = torch.mul(feat, qst_feature)
# 		combined_feature = self.tanh(combined_feature)
# 		out_qa = self.fc_ans(combined_feature)              # [batch_size, ans_vocab_size]

# 		return out_qa, out_match_posi, out_match_nega

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from einops import rearrange

class AVQA_Fusion_Net(nn.Module):
    def __init__(self, opt):
        super(AVQA_Fusion_Net, self).__init__()
        self.opt = opt

        # InternVideo2 Stage2 전체 모델 로드 (audio + video encoder 포함)
        internvideo2 = AutoModel.from_pretrained("OpenGVLab/InternVideo2-Stage2_6B", torch_dtype=torch.bfloat16, trust_remote_code=True)
        self.swin = internvideo2.vision_encoder  # 영상 인코더 추출
        del internvideo2
        torch.cuda.empty_cache() 
        audio_cfg = BEATsConfig()
        audio_cfg.input_patch_size = 16
        self.audio_encoder = BEATs(audio_cfg)
        missing, unexpected = self.audio_encoder.load_state_dict(
            torch.load('/workspace/DGSCT_INTERN/data/AVQA/BEATs_iter3.pt')['model'],
            strict=False)
        print('missing', missing[:5], '...')      # 로드 안 된 층
        print('unexpected', unexpected[:5], '...')
        # Feature projection
        self.fc_a1 = nn.Linear(768, 2048)
        self.fc_a2 = nn.Linear(2048, 3200)
        self.fc_gl = nn.Linear(3200 + 3200, 3200)

        self.fc1 = nn.Linear(3200 + 3200, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 2)

        self.question_encoder = QstEncoder(93, 3200, 3200, 1, 3200)
        self.fc_ans = nn.Linear(3200, 42)
        self.fc_fusion = nn.Linear(3200 * 2, 3200)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.tanh = nn.Tanh()

        self.linear11 = nn.Linear(3200, 3200)
        self.linear12 = nn.Linear(3200, 3200)
        self.linear21 = nn.Linear(3200, 3200)
        self.linear22 = nn.Linear(3200, 3200)

        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)
        self.dropout3 = nn.Dropout(0.1)
        self.dropout4 = nn.Dropout(0.1)

        self.norm1 = nn.LayerNorm(3200)
        self.norm2 = nn.LayerNorm(3200)
        self.norm3 = nn.LayerNorm(3200)

        self.attn_a = nn.MultiheadAttention(3200, 4, dropout=0.1)
        self.attn_v = nn.MultiheadAttention(3200, 4, dropout=0.1)
    def to(self, device):
        self.swin = self.swin.to(device)
        self.audio_encoder = self.audio_encoder.to(device)
        self.fc_a1 = self.fc_a1.to(device)
        self.fc_a2 = self.fc_a2.to(device)
        self.fc_gl = self.fc_gl.to(device)

        self.fc1 = self.fc1.to(device)
        self.fc2 = self.fc2.to(device)
        self.fc3 = self.fc3.to(device)
        self.fc4 = self.fc4.to(device)

        self.question_encoder = self.question_encoder.to(device)
        self.fc_ans = self.fc_ans.to(device)
        self.fc_fusion = self.fc_fusion.to(device)

        self.avgpool = self.avgpool.to(device)
        self.tanh = self.tanh.to(device)

        self.linear11 = self.linear11.to(device)
        self.linear12 = self.linear12.to(device)
        self.linear21 = self.linear21.to(device)
        self.linear22 = self.linear22.to(device)

        self.dropout1 = self.dropout1.to(device)
        self.dropout2 = self.dropout2.to(device)
        self.dropout3 = self.dropout3.to(device)
        self.dropout4 = self.dropout4.to(device)

        self.norm1 = self.norm1.to(device)
        self.norm2 = self.norm2.to(device)
        self.norm3 = self.norm3.to(device)

        self.attn_a = self.attn_a.to(device)
        self.attn_v = self.attn_v.to(device)

        return self

    def forward(self, audio, visual_posi, visual_nega, question, mixup_lambda=None, stage='eval'):
        def resize_and_sample_frames_uniform(x, target_size=(224, 224), target_frames=10):
            """
            균등 간격 프레임 샘플링 + H/W 리사이즈
            """
            B, T, C, H, W = x.shape
            x = rearrange(x, 'b t c h w -> b c t h w')

            # [B, C, T, H, W] → [B*T, C, H, W] → resize
            x = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
            x = F.interpolate(x, size=target_size, mode="bilinear", align_corners=False)
            x = x.view(B, T, C, *target_size).permute(0, 2, 1, 3, 4)  # [B, C, T, H, W]

            # 균등 샘플링
            if T > target_frames:
                indices = torch.linspace(0, T - 1, steps=target_frames).long()
                x = x[:, :, indices, :, :]
            elif T < target_frames:
                pad = x[:, :, -1:, :, :].expand(B, C, target_frames - T, *target_size)
                x = torch.cat([x, pad], dim=2)
            return x

        
        def sample_waveform(waveform, target_length=160000):
            """
            waveform: (B, T, L) or (T, L) shaped tensor
            target_length: 고정 입력 길이 (ex. 10초 → 160000 at 16kHz)
            """
            audio_len = waveform.shape[-1]
            if audio_len <= target_length:
                # padding (선택)
                pad_len = target_length - audio_len
                waveform = F.pad(waveform, (0, pad_len), mode='constant', value=0)
            else:
                start = torch.randint(0, audio_len - target_length + 1, (1,)).item()
                waveform = waveform[..., start:start + target_length]
            return waveform
        bs, t, c, h, w = visual_posi.shape
        visual_posi = resize_and_sample_frames_uniform(visual_posi, target_size=(224, 224), target_frames=4)
        visual_nega = resize_and_sample_frames_uniform(visual_nega, target_size=(224, 224), target_frames=4)

        f_v = self.swin(visual_posi)
        visual_nega = self.swin(visual_nega)

        # InternVideo2 audio encoder expects waveform
        audio = sample_waveform(audio, target_length=160000)
        B, T, L = audio.shape
        audio = audio.view(B * T, L)
        f_a, _ = self.audio_encoder.extract_features(audio)

        f_a = f_a.mean(dim=1) if f_a.ndim == 3 else f_a
        audio = rearrange(f_a, '(b t) c -> b t c', b=B, t=T)
        audio_feat = F.relu(self.fc_a1(audio))
        audio_feat = self.fc_a2(audio_feat)
        audio_feat_pure = audio_feat
        B, T, C = audio_feat.size()
        audio_feat = audio_feat.view(B*T, C)
        if isinstance(f_v, tuple):
            f_v = f_v[0]
        if isinstance(visual_nega, tuple):
            visual_nega = visual_nega[0]
        f_v = f_v[:,1:,:]
        visual_nega = visual_nega[:,1:,:]
        # visual_posi = rearrange(f_v, 'b (t h w) c -> b t c h w', b=bs, t=t, h=8, w=8)
        # visual_nega = rearrange(visual_nega, 'b (t h w) c -> b t c h w', b=bs, t=t, h=8, w=8)

        # f_a = f_a.mean(dim=1)
        # f_a = torch.bmm(f_a_spatial_att_maps, f_a).squeeze(dim=1)     
        audio = rearrange(f_a, '(b t) c -> b t c', b=bs ,t=t)
		### <-----

		# visual_posi, f_a = self.temporal_attn(visual_posi, f_a)
        
		## question features
        qst_feature = self.question_encoder(question)
        xq = qst_feature.unsqueeze(0)

        v_feat = f_v.mean(dim=1)
        visual_feat_before_grounding_posi = v_feat.squeeze(-1)    # [B*T, C]

        visual_feat_posi = nn.functional.normalize(f_v, dim=2)   # [B, HxW, C]

		## audio-visual grounding posi
        audio_feat = audio_feat.view(B, T, C).mean(dim=1)  # shape: [B, 1, C]
        audio_feat_aa = audio_feat.view(B, 3200, 1)
        visual_feat_posi = visual_feat_posi.float()
        audio_feat_aa = audio_feat_aa.float()
        audio_feat_aa = nn.functional.normalize(audio_feat_aa, dim=1)   # [B, C, 1]
	   
        x2_va = torch.matmul(visual_feat_posi, audio_feat_aa).squeeze(-1) # [B, HxW]
        visual_feat_flat = visual_feat_posi    # [B, 3200, C]
        x2_p = F.softmax(x2_va, dim=-1).unsqueeze(-2)                       # [B*T, 1, HxW]
        visual_feat_grd_after_grounding_posi = torch.bmm(x2_p, visual_feat_flat).squeeze(1)
        visual_gl = torch.cat((visual_feat_before_grounding_posi, visual_feat_grd_after_grounding_posi),dim=-1)
        visual_feat_grd = self.tanh(visual_gl)
        visual_feat_grd_posi = self.fc_gl(visual_feat_grd)              # [B*T, C]
        feat = torch.cat((audio_feat, visual_feat_grd_posi), dim=-1)    # [B*T, C*2], [B*T, 3072]

        feat = F.relu(self.fc1(feat))       # (3072, 512)
        feat = F.relu(self.fc2(feat))       # (512, 256)
        feat = F.relu(self.fc3(feat))       # (256, 128)
        out_match_posi = self.fc4(feat)     # (128, 2)

		###############################################################################################
        # visual_feat_before_grounding_nega: avg over all tokens
        v_feat = visual_nega.mean(dim=1)  # [B, C]
        visual_feat_before_grounding_nega = v_feat.squeeze(-1)

        # normalize visual tokens
        visual_feat_nega = nn.functional.normalize(visual_nega, dim=2)  # [B, HxW, C]
        visual_feat_nega = visual_feat_nega.float()
        audio_feat_aa = audio_feat_aa.float()

        # audio-visual attention
        x2_va = torch.matmul(visual_feat_nega, audio_feat_aa).squeeze(-1)  # [B, HxW]
        x2_p = F.softmax(x2_va, dim=-1).unsqueeze(-2)  # [B, 1, HxW]

        # weighted sum of visual features
        visual_feat_grd_after_grounding_nega = torch.bmm(x2_p, visual_feat_nega).squeeze(1)  # [B, C]

        # fuse raw visual + grounded visual
        visual_gl = torch.cat((visual_feat_before_grounding_nega, visual_feat_grd_after_grounding_nega), dim=-1)
        visual_feat_grd = self.tanh(visual_gl)
        visual_feat_grd_nega = self.fc_gl(visual_feat_grd)  # [B, C]

        # fuse with audio
        feat = torch.cat((audio_feat, visual_feat_grd_nega), dim=-1)  # [B, C*2], [B, 3200]

        feat = F.relu(self.fc1(feat))  # (3200, 512)
        feat = F.relu(self.fc2(feat))  # (512, 256)
        feat = F.relu(self.fc3(feat))  # (256, 128)
        out_match_nega = self.fc4(feat)  # (128, 2)


		###############################################################################################

		# out_match=None
		# match_label=None

        B = xq.shape[1]
        visual_feat_grd_be = visual_feat_grd_posi.view(B, -1, 3200)   # [B, T, 512]
        visual_feat_grd=visual_feat_grd_be.permute(1,0,2)
		
		## attention, question as query on visual_feat_grd
        visual_feat_att = self.attn_v(xq, visual_feat_grd, visual_feat_grd, attn_mask=None, key_padding_mask=None)[0].squeeze(0)
        src = self.linear12(self.dropout1(F.relu(self.linear11(visual_feat_att))))
        visual_feat_att = visual_feat_att + self.dropout2(src)
        visual_feat_att = self.norm1(visual_feat_att)
	
		# attention, question as query on audio
        audio_feat_be=audio_feat_pure.view(B, -1, 3200)
        audio_feat = audio_feat_be.permute(1, 0, 2)
        audio_feat_att = self.attn_a(xq, audio_feat, audio_feat, attn_mask=None,key_padding_mask=None)[0].squeeze(0)
        src = self.linear22(self.dropout3(F.relu(self.linear21(audio_feat_att))))
        audio_feat_att = audio_feat_att + self.dropout4(src)
        audio_feat_att = self.norm2(audio_feat_att)
		
        feat = torch.cat((audio_feat_att+audio_feat_be.mean(dim=-2).squeeze(), visual_feat_att+visual_feat_grd_be.mean(dim=-2).squeeze()), dim=-1)
        feat = self.tanh(feat)
        feat = self.fc_fusion(feat)

		## fusion with question
        combined_feature = torch.mul(feat, qst_feature)
        combined_feature = self.tanh(combined_feature)
        out_qa = self.fc_ans(combined_feature)              # [batch_size, ans_vocab_size]

        return out_qa, out_match_posi, out_match_nega
