#runs all the metrics script to test the model
#test top 1,3,5,10 accuracy and MRR

#!/bin/bash
#command bash scripts/pretrain_memoria.sh
WORLD_SIZE=1

model=create_uni3d

clip_model="EVA02-E-14-plus" 
pretrained="/path/to/clip_model/open_clip_pytorch_model.bin" # or "laion2b_s9b_b144k"  
embed_dim=1024

#modelo gigante
#pc_model="eva_giant_patch14_560.m30m_ft_in22k_in1k"
#pc_feat_dim=1408

#modelo large
#pc_model="eva02_large_patch14_448"
#pc_feat_dim=1024


#modelo base

#pc_model="eva02_base_patch14_448"
#pc_feat_dim=768

#modelo small
#pc_model="eva02_small_patch14_224"
#pc_feat_dim=384

#modelo tiny
pc_model="eva02_tiny_patch14_224"
pc_feat_dim=192

pc_encoder_dim=512 


ckpt_path="/path/to/model/model_t_new1" \

CUDA_VISIBLE_DEVICES=1 torchrun --nnodes=$WORLD_SIZE \
    --nproc-per-node=1 \
    main.py \
    --model $model \
    --Experiment_detail "Normal metrics for tiny finetuned"\
    --pretrain_dataset_name ShapeTest_Embed \
    --npoints 10000 \
    --num-group 512 \
    --group-size 64 \
    --clip-model $clip_model \
    --pc-model $pc_model \
    --pretrained $pretrained \
    --warmup 1000 \
    --batch-size=50 \
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
    --patch-dropout=0.1 \
    --optimizer="adamw" \
    --zero-stage=1 \
    --validate_dataset_name ShapeTest_Embed \
    --validate_dataset_name_lvis objaverse_lvis_openshape \
    --validate_dataset_name_scanobjnn scanobjnn_openshape \
    --save-frequency -1 \
    --enable-deepspeed \
    --evaluate_memoria \
    --ckpt_path $ckpt_path \
    --use-embed \
    --saved_before

     #added params
    #--saved_before it just a param to tell the script to use a previously trained .pt model in testing
    #--embedd_Dataset # creates a embedded dataset with name of pretrain_dataset_name
    #--use-embed \ #uses a embedded dataset (if not it has to calculate the embeddings all over again)
    #--is_shard "./outputs/checkpoint_best" \
    #--resume "./outputs/checkpoint_last" \ resumes the training (dont confuse with ckpt_path), it is for training only because the weights are distributed
    #--ckpt_path $ckpt_path \ tells where to find an checkpoint of the model (not distributed)