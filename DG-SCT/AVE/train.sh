CUDA_VISIBLE_DEVICES=0 python3 main_trans.py --Adapter_downsample=8 --accum_itr=2 \
	--batch_size=2 --decay=0.35 --decay_epoch=3 --early_stop=50 --epochs=50 --is_audio_adapter_p1=0 --is_audio_adapter_p2=0 \
	--is_audio_adapter_p3=0 --is_before_layernorm=1 --is_bn=1 --is_fusion_before=1 --is_gate=1  \
	--is_post_layernorm=1 --is_vit_ln=0 --lr=5e-04 --lr_mlp=5e-06 --mode=train \
	--model=Prompted_MMIL_Net --num_conv_group=2 --num_tokens=32 --num_workers=16 --seed 43 --gpu=0 --wandb=1 --init_best=0 \
	--backbone_type audioset