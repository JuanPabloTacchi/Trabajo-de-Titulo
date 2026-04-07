
#creates a embedding dataset with name of pretrain_dataset_name, please use shapetrain and change de directories

#!/bin/bash
#command bash scripts/pretrain_memoria.sh
WORLD_SIZE=1

model=create_uni3d


clip_model="EVA02-E-14-plus" 
pretrained="/path/to/clip_model" # or "laion2b_s9b_b144k"  
embed_dim=1024

#actually this isnt necesary but im leaving it here because i dont have time to test it without
#modelo tiny
pc_model="eva02_tiny_patch14_224"
pc_feat_dim=192

pc_encoder_dim=512 
#checkpoint path
ckpt_path="/path/to/t_model"



CUDA_VISIBLE_DEVICES=1 torchrun --nnodes=$WORLD_SIZE \
    --nproc-per-node=1 \
    main.py \
    --model $model \
    --pretrain_dataset_name NAME_OF_YML_(Example_Shapetrain) \
    --npoints 10000 \
    --num-group 512 \
    --group-size 64 \
    --clip-model $clip_model \
    --pc-model $pc_model \
    --pretrained $pretrained \
    --warmup 1000 \
    --batch-size=1 \
    --epochs 100 \
    --pc-feat-dim=$pc_feat_dim \
    --pc-encoder-dim=$pc_encoder_dim \
    --embed-dim=$embed_dim \
    --lr=1e-3 \
    --point-lr=1e-3 \
    --drop-path-rate=0.20 \
    --wd=0.1 \
    --point-wd=0.1 \
    --ld=1.0 \
    --point-ld=0.95 \
    --grad-clip-norm=5.0 \
    --smoothing=0. \
    --seed 4096 \
    --patch-dropout=0.5 \
    --optimizer="adamw" \
    --zero-stage=1 \
    --validate_dataset_name NAME_OF_YML_(Example_SHAPEVAL) \
    --validate_dataset_name_lvis objaverse_lvis_openshape \
    --validate_dataset_name_scanobjnn scanobjnn_openshape \
    --save-frequency -1 \
    --enable-deepspeed \
    --embedd_Dataset \
    #--embedd_Dataset # creates a embedding dataset with name of pretrain_dataset_name
    
   