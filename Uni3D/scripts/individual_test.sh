#individual test script for comparing against a particular test the embedding, sadly you have to change the text manualy
#on the individual_test() function in main.py

#!/bin/bash
WORLD_SIZE=1

model=create_uni3d

clip_model="EVA02-E-14-plus" 
pretrained="/path/to/clip_model" # or "laion2b_s9b_b144k"  
embed_dim=1024

#choose any model you want, base model for example
#model
pc_model="eva02_base_patch14_448"
pc_feat_dim=768


pc_encoder_dim=512 

ckpt_path="/path/to/checkpoint/model_b.pt"

#change ShapeTest_Embed for the dataset you want to test
CUDA_VISIBLE_DEVICES=0 torchrun --nnodes=$WORLD_SIZE \
    --nproc-per-node=1 \
    main.py \
    --model $model \
    --Experiment_detail "text analisis total description"\
    --pretrain_dataset_name ShapeTest_Embed \
    --npoints 10000 \
    --num-group 512 \
    --group-size 64 \
    --clip-model $clip_model \
    --pc-model $pc_model \
    --pretrained $pretrained \
    --warmup 1000 \
    --batch-size=100 \
    --epochs 10 \
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
    --individual_test \
    --ckpt_path $ckpt_path \
    --use-embed \
    --saved_before \

    #added params
    #--saved_before its just a param to tell the script to use a previously trained .pt model in testing
    #--embedd_Dataset # creates a embedded dataset with name of pretrain_dataset_name
    #--use-embed \ #uses a embedded dataset (if not it has to calculate the embeddings all over again)
    #--is_shard "./outputs/checkpoint_best" \
    #--resume "./outputs/checkpoint_last" \ resumes the training (dont confuse with ckpt_path), it is for training only because the weights are distributed
    #--ckpt_path $ckpt_path \ tells where to find an checkpoint of the model (not distributed)
