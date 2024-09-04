#!/bin/bash
#set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:6144
################## VICUNA ##################
PROMPT_VERSION=v1
MODEL_VERSION="vicuna-v1.5-7b"
################## VICUNA ##################
#export CUDA_VISIBLE_DEVICES='4'
#
#GPUS_PER_NODE=1
#export NCCL_IB_DISABLE=1
#export NCCL_IBEXT_DISABLE=1
 deepspeed --master_port=25643 --include localhost:0,1 train_mem.py \
    --deepspeed zero2.json \
    --lora_enable True \
    --model_name_or_path llavav1.5-7b \
    --version $PROMPT_VERSION \
    --data_path xxx \
    --image_folder xx  \
    --vision_tower xx/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --pretrain_mm_mlp_adapter xx/llava-v1.5-mlp2x-336px-pretrain-vicuna-7b-v1.5/mm_projector.bin \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --bf16 True \
    --output_dir xx \
    --num_train_epochs 1 \
    --per_device_train_batch_size 6 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 6 \
    --evaluation_strategy "no" \
    --save_strategy "epoch" \
    --save_steps 1000 \
    --save_total_limit 1 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --dataloader_num_workers 4 \
    --report_to wandb
