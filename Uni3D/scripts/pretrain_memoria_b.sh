#!/bin/bash
#command bash scripts/pretrain_memoria.sh
WORLD_SIZE=1

model=create_uni3d

clip_model="EVA02-E-14-plus" 
pretrained="/path/to/clip_model" # or "laion2b_s9b_b144k"  
embed_dim=1024



#modelo base

pc_model="eva02_base_patch14_448"
pc_feat_dim=768



pc_encoder_dim=512 

ckpt_path="/path/to/model/model_b.pt"

CUDA_VISIBLE_DEVICES=0,1 torchrun --nnodes=$WORLD_SIZE \
    --nproc-per-node=2 \
    main.py \
    --model $model \
    --pretrain_dataset_name ShapeTrain_Embed \
    --Experiment_detail "Base model finetuning"\
    --npoints 10000 \
    --num-group 512 \
    --group-size 64 \
    --clip-model $clip_model \
    --pc-model $pc_model \
    --pretrained $pretrained \
    --warmup 200 \
    --batch-size=30 \
    --epochs 30 \
    --pc-feat-dim=$pc_feat_dim \
    --pc-encoder-dim=$pc_encoder_dim \
    --embed-dim=$embed_dim \
    --lr=0.0005 \
    --point-lr=0.0001 \
    --drop-path-rate=0.1 \
    --wd=0.01 \
    --point-wd=0.01 \
    --ld=1.0 \
    --point-ld=0.95 \
    --grad-clip-norm=5.0 \
    --smoothing=0. \
    --seed 4096 \
    --patch-dropout=0.1 \
    --optimizer="adamw" \
    --zero-stage=1 \
    --validate_dataset_name ShapeVal_Embed \
    --validate_dataset_name_lvis objaverse_lvis_openshape \
    --validate_dataset_name_scanobjnn scanobjnn_openshape \
    --save-frequency 5 \
    --enable-deepspeed \
    --use-embed \
    --ckpt_path $ckpt_path \

    
    #added params
    #--embedd_Dataset # creates a embedded dataset with name of pretrain_dataset_name
    #--use-embed \ #uses a embedded dataset (if not it has to calculate the embeddings all over again)
    #--is_shard "./outputs/checkpoint_best" \
    #--resume "./outputs/checkpoint_last" \ resumes the training (dont confuse with ckpt_path), it is for training only because the weights are distributed
    #--ckpt_path $ckpt_path \ tells where to find an checkpoint of the model (not distributed)
