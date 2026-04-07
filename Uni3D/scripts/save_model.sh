#This is just a script i use to undistribute the model into a single .pt file


#!/bin/bash
#command bash scripts/pretrain_memoria.sh
WORLD_SIZE=1

model=create_uni3d

clip_model="EVA02-E-14-plus" 
pretrained="/path/to/clip_model" # or "laion2b_s9b_b144k"  
embed_dim=1024


#modelo tiny
pc_model="eva02_tiny_patch14_224"
pc_feat_dim=192

pc_encoder_dim=512 
#as an example is the model_t but it doesnt matter if it exist because it is not loaded actually
ckpt_path="/path/to/model/model_t.pt"

CUDA_VISIBLE_DEVICES=0,1 torchrun --nnodes=$WORLD_SIZE \
    --nproc-per-node=2 \
    main.py \
    --model $model \
    --pretrain_dataset_name ShapeTrain_Embed \
    --npoints 10000 \
    --num-group 512 \
    --group-size 64 \
    --clip-model $clip_model \
    --pc-model $pc_model \
    --pretrained $pretrained \
    --warmup 1000 \
    --batch-size=100 \
    --epochs 20 \
    --pc-feat-dim=$pc_feat_dim \
    --pc-encoder-dim=$pc_encoder_dim \
    --embed-dim=$embed_dim \
    --lr=0.001 \
    --point-lr=0.0003 \
    --drop-path-rate=0.1 \
    --wd=0.1 \
    --point-wd=0.1 \
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
    --resume "./outputs/checkpoint_to_undistribute" \
    --save_pt "/path/to/output_model/model_t_new.pt" \
    
    
     #added params
    #--embedd_Dataset # creates a embedded dataset with name of pretrain_dataset_name
    #--use-embed \ #uses a embedded dataset (if not it has to calculate the embeddings all over again)
    #--is_shard "./outputs/checkpoint_best" \
    #--resume "./outputs/checkpoint_last" \ resumes the training (dont confuse with ckpt_path), it is for training only because the weights are distributed
    #--ckpt_path $ckpt_path \ tells where to find an checkpoint of the model (not distributed)