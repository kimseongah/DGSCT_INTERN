from __future__ import print_function
import sys 

import argparse
from base_options import BaseOptions
from gpuinfo import GPUInfo 
from utils import do_mixup, get_mix_lambda, do_mixup_label
import os
args = BaseOptions().parse()

current_dir = os.getcwd()
print("current_dir", current_dir)

parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
grandparent_dir = os.path.abspath(os.path.join(parent_dir, os.pardir))
print("root_path", grandparent_dir)

args.root_path = grandparent_dir

mygpu = GPUInfo.get_info()[0]
gpu_source = {}

if 'N/A' in mygpu.keys():
	for info in mygpu['N/A']:
		if info in gpu_source.keys():
			gpu_source[info] +=1
		else:
			gpu_source[info] =1

for gpu_id in args.gpu:
	gpu_id = str(gpu_id)

	if gpu_id not in gpu_source.keys():
		print('go gpu:', gpu_id)
		os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
		break
	elif gpu_source[gpu_id] < 1:
		print('go gpu:', gpu_id)
		os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
		break
import torch
import torch.nn as nn
import torch.optim as optim
from ipdb import set_trace
from dataloader_avst import *
# from dataloader_avst_bk import *
from net_avst import AVQA_Fusion_Net
import ast
import json
import numpy as np
import pdb
# from .net_avst import AVQA_Fusion_Net

import warnings
from datetime import datetime
import wandb
TIMESTAMP = "{0:%Y-%m-%d-%H-%M-%S/}".format(datetime.now()) 
warnings.filterwarnings('ignore')
# from torch.utils.tensorboard import SummaryWriter
# writer = SummaryWriter('runs/net_avst/'+TIMESTAMP)

import certifi
os.environ['REQUESTS_CA_BUNDLE'] = os.path.join(os.path.dirname(sys.argv[0]), certifi.where())

print("\n--------------- Audio-Visual Spatial-Temporal Model --------------- \n")

def batch_organize(out_match_posi,out_match_nega):
	# audio B 512
	# posi B 512
	# nega B 512

	# print("audio data: ", audio_data.shape)
	out_match = torch.zeros(out_match_posi.shape[0] * 2, out_match_posi.shape[1])
	batch_labels = torch.zeros(out_match_posi.shape[0] * 2)
	for i in range(out_match_posi.shape[0]):
		out_match[i * 2, :] = out_match_posi[i, :]
		out_match[i * 2 + 1, :] = out_match_nega[i, :]
		batch_labels[i * 2] = 1
		batch_labels[i * 2 + 1] = 0
	
	return out_match, batch_labels


def train(args, model, train_loader, optimizer, criterion, epoch):
	model.train()
	total_qa = 0
	correct_qa = 0
	for batch_idx, sample in enumerate(train_loader):
		visual_posi,visual_nega, target, question, wave = sample['visual_posi'].to('cuda'),sample['visual_nega'].to('cuda'), sample['label'].to('cuda'), sample['question'].to('cuda'), sample['wave'].to('cuda')
        
		if args.backbone_type == "audioset":
			mixup_lambda = torch.from_numpy(get_mix_lambda(0.5, len(wave)*10)).to('cuda')
		else:
			mixup_lambda = None
            
		optimizer.zero_grad()
		out_qa, out_match_posi,out_match_nega = model(wave, visual_posi,visual_nega, question, mixup_lambda=mixup_lambda, stage='train')  
		out_match,match_label=batch_organize(out_match_posi,out_match_nega)  
		out_match,match_label = out_match.type(torch.FloatTensor).cuda(), match_label.type(torch.LongTensor).cuda()

		loss_match=criterion(out_match,match_label)
		loss_qa = criterion(out_qa, target)
		loss = loss_qa + 0.5*loss_match



		loss.backward()
		optimizer.step()
		if batch_idx % args.log_interval == 0:
			print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
				epoch, batch_idx * len(visual_posi), len(train_loader.dataset),
					   100. * batch_idx / len(train_loader), loss.item()))


def eval(model, val_loader,epoch):
	model.eval()
	total_qa = 0
	total_match=0
	correct_qa = 0
	correct_match=0
	with torch.no_grad():
		for batch_idx, sample in enumerate(val_loader):
			visual_posi,visual_nega, target, question, wave = sample['visual_posi'].to('cuda'),sample['visual_nega'].to('cuda'), sample['label'].to('cuda'), sample['question'].to('cuda'), sample['wave'].to('cuda')
			if args.backbone_type == "audioset":
				mixup_lambda = torch.from_numpy(get_mix_lambda(0.5, len(wave)*10)).to('cuda')
			else:
				mixup_lambda = None
        
			preds_qa, out_match_posi,out_match_nega = model(wave, visual_posi,visual_nega, question, mixup_lambda=mixup_lambda)

			_, predicted = torch.max(preds_qa.data, 1)
			total_qa += preds_qa.size(0)
			correct_qa += (predicted == target).sum().item()

	print('Accuracy qa: %.2f %%' % (100 * correct_qa / total_qa))
	# writer.add_scalar('metric/acc_qa',100 * correct_qa / total_qa, epoch)

	return 100 * correct_qa / total_qa


def test(model, val_loader):
	model.eval()
	total = 0
	correct = 0
	samples = json.load(open(os.path.join(args.root_path, args.label_test), 'r'))
	A_count = []
	A_cmp = []
	V_count = []
	V_loc = []
	AV_ext = []
	AV_count = []
	AV_loc = []
	AV_cmp = []
	AV_temp = []
	with torch.no_grad():
		for batch_idx, sample in enumerate(val_loader):
			visual_posi,visual_nega, target, question, wave = sample['visual_posi'].to('cuda'),sample['visual_nega'].to('cuda'), sample['label'].to('cuda'), sample['question'].to('cuda'), sample['wave'].to('cuda')
			if args.backbone_type == "audioset":
				mixup_lambda = torch.from_numpy(get_mix_lambda(0.5, len(wave)*10)).to('cuda')
			else:
				mixup_lambda = None
                
			preds_qa,out_match_posi,out_match_nega = model(wave, visual_posi,visual_nega, question, mixup_lambda=mixup_lambda)
			preds = preds_qa
			_, predicted = torch.max(preds.data, 1)

			total += preds.size(0)
			correct += (predicted == target).sum().item()

			x = samples[batch_idx]
			type =ast.literal_eval(x['type'])
			if type[0] == 'Audio':
				if type[1] == 'Counting':
					A_count.append((predicted == target).sum().item())
				elif type[1] == 'Comparative':
					A_cmp.append((predicted == target).sum().item())
			elif type[0] == 'Visual':
				if type[1] == 'Counting':
					V_count.append((predicted == target).sum().item())
				elif type[1] == 'Location':
					V_loc.append((predicted == target).sum().item())
			elif type[0] == 'Audio-Visual':
				if type[1] == 'Existential':
					AV_ext.append((predicted == target).sum().item())
				elif type[1] == 'Counting':
					AV_count.append((predicted == target).sum().item())
				elif type[1] == 'Location':
					AV_loc.append((predicted == target).sum().item())
				elif type[1] == 'Comparative':
					AV_cmp.append((predicted == target).sum().item())
				elif type[1] == 'Temporal':
					AV_temp.append((predicted == target).sum().item())

	print('Audio Counting Accuracy: %.2f %%' % (
			100 * sum(A_count)/len(A_count)))
	print('Audio Cmp Accuracy: %.2f %%' % (
			100 * sum(A_cmp) / len(A_cmp)))
	print('Audio Accuracy: %.2f %%' % (
			100 * (sum(A_count) + sum(A_cmp)) / (len(A_count) + len(A_cmp))))
	print('Visual Counting Accuracy: %.2f %%' % (
			100 * sum(V_count) / len(V_count)))
	print('Visual Loc Accuracy: %.2f %%' % (
			100 * sum(V_loc) / len(V_loc)))
	print('Visual Accuracy: %.2f %%' % (
			100 * (sum(V_count) + sum(V_loc)) / (len(V_count) + len(V_loc))))
	print('AV Ext Accuracy: %.2f %%' % (
			100 * sum(AV_ext) / len(AV_ext)))
	print('AV counting Accuracy: %.2f %%' % (
			100 * sum(AV_count) / len(AV_count)))
	print('AV Loc Accuracy: %.2f %%' % (
			100 * sum(AV_loc) / len(AV_loc)))
	print('AV Cmp Accuracy: %.2f %%' % (
			100 * sum(AV_cmp) / len(AV_cmp)))
	print('AV Temporal Accuracy: %.2f %%' % (
			100 * sum(AV_temp) / len(AV_temp)))

	print('AV Accuracy: %.2f %%' % (
			100 * (sum(AV_count) + sum(AV_loc)+sum(AV_ext)+sum(AV_temp)
				   +sum(AV_cmp)) / (len(AV_count) + len(AV_loc)+len(AV_ext)+len(AV_temp)+len(AV_cmp))))

	print('Overall Accuracy: %.2f %%' % (
			100 * correct / total))

	return 100 * correct / total

def main():
	# Training settings
	if args.wandb:
		wandb.init(config=args, project="AVQA",name=args.model_name)


	torch.manual_seed(args.seed)

	if args.model == 'AVQA_Fusion_Net':
		model = AVQA_Fusion_Net(args)
		# model = nn.DataParallel(model)
		model = model.to('cuda')
	else:
		raise ('not recognized')

	if args.mode == 'train':
		train_dataset = AVQA_dataset(root_path=args.root_path, label=args.label_train, audio_dir=args.audio_dir, video_res14x14_dir=args.video_res14x14_dir,
									transform=transforms.Compose([ToTensor()]), mode_flag='train')
		train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
		val_dataset = AVQA_dataset(root_path=args.root_path, label=args.label_val, audio_dir=args.audio_dir, video_res14x14_dir=args.video_res14x14_dir,
									transform=transforms.Compose([ToTensor()]), mode_flag='test')
		val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)


		# ===================================== load pretrained model ===============================================
		####### concat model
		pretrained_file = "grounding_gen/models_grounding_gen/lavish_grounding_gen_best.pt"
		checkpoint = torch.load(pretrained_file)
		print("\n-------------- loading pretrained models --------------")
		model_dict = model.state_dict()
		tmp = ['module.fc_a1.weight', 'module.fc_a1.bias','module.fc_a2.weight','module.fc_a2.bias','module.fc_gl.weight','module.fc_gl.bias','module.fc1.weight', 'module.fc1.bias','module.fc2.weight', 'module.fc2.bias','module.fc3.weight', 'module.fc3.bias','module.fc4.weight', 'module.fc4.bias']
		tmp2 = ['module.fc_a1.weight', 'module.fc_a1.bias','module.fc_a2.weight','module.fc_a2.bias']
		pretrained_dict1 = {k: v for k, v in checkpoint.items() if k in tmp}
		pretrained_dict2 = {str(k).split('.')[0]+'.'+str(k).split('.')[1]+'_pure.'+str(k).split('.')[-1]: v for k, v in checkpoint.items() if k in tmp2}

		model_dict.update(pretrained_dict1) #利用预训练模型的参数，更新模型
		model_dict.update(pretrained_dict2) #利用预训练模型的参数，更新模型
		model.load_state_dict(model_dict, strict=False)

		print("\n-------------- load pretrained models --------------")
		# ===================================== load pretrained model ===============================================


		# ===================================== for fast training ===============================================
		# checkpoint = torch.load(args.model_save_dir + args.checkpoint + ".pt")

		# model.load_state_dict(checkpoint['model_state_dict'])
		# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
		# if scheduler is not None and checkpoint['scheduler_state_dict'] is not None:
		# 	scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
		# print("\n-------------- load checkpoint models --------------")
		# ===================================== for fast training ===============================================
		k_last_blocks=4
		depth = len(model.swin.blocks)
		tune_ids = set(range(depth - k_last_blocks, depth))          # {44,45,46,47}
		always_open = ("swin.norm", "swin.attn_pool", "swin.clip_proj")
		param_group = []
		train_params = 0
		total_params = 0
		additional_params = 0
		for name, param in model.named_parameters():
			
			param.requires_grad = True
			### ---> compute params
			tmp = 1
			for num in param.shape:
				tmp *= num

			if 'ViT'in name or 'swin' in name or 'Resnet' in name:
				parts = name.split('.')
				if len(parts) > 2 and parts[1] == "blocks" and parts[2].isdigit() and int(parts[2]) in tune_ids:
					param.requires_grad = True
					total_params += tmp
					train_params += tmp
				if any(name.startswith(tok) for tok in always_open):
					param.requires_grad = True
					total_params += tmp
					train_params += tmp
				elif 'norm' in name:
					param.requires_grad = bool(args.is_vit_ln)
					total_params += tmp
					train_params += tmp
				# elif 'prompt' in name:
				# 	param.requires_grad = True
				# 	train_params += tmp
				# 	additional_params += tmp
				# 	total_params += tmp
				else:
					param.requires_grad = False
					total_params += tmp
				
			elif name == "audio_encoder.encoder.pos_conv.0.weight":
				param.requires_grad = True
				train_params += tmp
				total_params += tmp
				print('########### train layer:', name)
			elif "audio_encoder" in name:
				param.requires_grad = False
				total_params += tmp
			# ### <----
			
			elif 'adapter_blocks' in name:
				param.requires_grad = True
				train_params += tmp
				additional_params += tmp
				total_params += tmp
				print('########### train layer:', name)
			else:
				param.requires_grad = True
				train_params += tmp
				total_params += tmp
	
			if 'adapter_blocks' in name:
				param_group.append({"params": param, "lr":args.lr_block})
			else:
				param_group.append({"params": param, "lr":args.lr})
		print('####### Trainable params: %0.4f  #######'%(train_params*100/total_params))
		print('####### Additional params: %0.4f  ######'%(additional_params*100/(total_params-additional_params)))
		print('####### Total params in M: %0.1f M  #######'%(total_params/1000000))


		optimizer = optim.Adam(model.parameters(), lr=args.lr)
		scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.1)
		criterion = nn.CrossEntropyLoss()
		best_F = 0
		count = 0
		for epoch in range(1, args.epochs + 1):
			train(args, model, train_loader, optimizer, criterion, epoch=epoch)
			scheduler.step(epoch)
			F = eval(model, val_loader, epoch)
			# F = test(model, val_loader)
			count +=1
			if F >= best_F:
				count = 0
				best_F = F
				if args.wandb:
					wandb.log({"val-best": best_F})
				# torch.save(model.state_dict(), args.model_save_dir + args.checkpoint + ".pt")
				checkpoint = {
					'model_state_dict': model.state_dict(),
					'optimizer_state_dict': optimizer.state_dict(),
					'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
				}
				score_str = f"{best_F:.2f}"  # 소수점 2자리까지 formatting
				save_path = os.path.join(args.model_save_dir, f"{args.checkpoint}_{score_str}.pt")
				torch.save(checkpoint, save_path)
			if count == args.early_stop:
				exit()

	else:
		test_dataset = AVQA_dataset(root_path=args.root_path, label=args.label_test, audio_dir=args.audio_dir, video_res14x14_dir=args.video_res14x14_dir,
								   transform=transforms.Compose([ToTensor()]), mode_flag='test')
		print(test_dataset.__len__())
		test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=16, pin_memory=True)
		model.load_state_dict(torch.load(args.model_save_dir + args.checkpoint + ".pt"), strict=False)
		test(model, test_loader)


if __name__ == '__main__':
	main()