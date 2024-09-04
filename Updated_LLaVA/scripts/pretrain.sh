#!/bin/bash

# IMPORTANT: this is the training script for the original LLaVA, NOT FOR LLaVA V1.5!

# Uncomment and set the following variables correspondingly to run this script:

# MODEL_VERSION=vicuna-v1-3-7b
# MODEL_VERSION=llama-2-7b-chat

########### DO NOT CHANGE ###########
########### USE THIS FOR BOTH ###########
PROMPT_VERSION=plain
########### DO NOT CHANGE ###########

 deepspeed --master_port=25633 --include localhost:4,5 ../llava/train/train_mem.py \
    --deepspeed /data/user2/LLaVA_DWW/scripts/zero2.json \
    --model_name_or_path /data/Instruct_tuning/GeoChat/LLAVA/llavav1.5-7b \
    --version $PROMPT_VERSION \
    --data_path /data/Instruct_tuning/GeoChat/TEST/train_data/LR+HR10_train_images_geoinstruct_prompt_full.json \
    --image_folder /data/user3/gzqy/CoCaVQA/RSVQA_Images  \
    --vision_tower /data/Instruct_tuning/GeoChat/openai/clip-vit-large-patch14-336 \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir /data/user2/LLaVA_DWW/checkpoint/pretrain/llava_lora_LR_HR10_full_LLAVA336_pretrain \
    --num_train_epochs 1 \
    --per_device_train_batch_size 6 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 6 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 24000 \
    --save_total_limit 1 \
    --learning_rate 2e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb