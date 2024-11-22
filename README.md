### Co-LLaVA: Efficient Remote Sensing Visual Question Answering via Model Collaboration
[whole pipeline.pdf](https://github.com/user-attachments/files/17869075/whole.pipeline.pdf)


### Preparing environment for CoCa
```bash
pip install open_clip_torch
'''
### Preparing environment for LLaVA
```bash
conda create -n llava python=3.10 -y
conda activate llava
pip install --upgrade pip  # enable PEP 660 support
pip install -e .
```

###  Running training code of CoCa
```bash
python -m training/main.py \
    --dataset-type "instruct" \
    --train-data path-to-traindatajson \
    --val-data path-to-valdatajson \
    --warmup 1000 \
    --batch-size 10 \
    --lr 1e-5 \
    --wd 0.1 \
    --beta1 0.9 \
    --beta2 0.98 \
    --epochs 6 \
    --workers 8 \
    --image-mean  0.48145466 0.4578275 0.40821073 \
    --image-std 0.26862954 0.26130258 0.27577711 \
    --pretrained laion2b_s13b_b90k \
    --csv-img-key filename \
    --model "coca_ViT-L-14" \
    --coca-contrastive-loss-weight 1 \
    --coca-caption-loss-weight 4 \
    --log-every-n-steps 500 \
    --logs path-to-logs
```

### Running fine-tuning code of Updated-LLaVA
```bash
python -m train_mem.py \
    --deepspeed zero2.json \
    --lora_enable True \
    --model_name_or_path llavav1.5-7b \
    --version v1 \
    --data_path path-to-datajson \
    --image_folder path-to-image  \
    --vision_tower path-to-clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --pretrain_mm_mlp_adapter path-to-mm_projector.bin \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --bf16 True \
    --output_dir path-to-output_dir \
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
```

