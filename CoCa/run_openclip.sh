export CUDA_VISIBLE_DEVICES=3,2
cd src
export PYTHONPATH="$PWD"
echo "PYTHONPATH=$PYTHONPATH"

torchrun --nproc_per_node=2 --master_port='29501' training/main.py \
    --dataset-type "instruct" \
    --train-data "xxx" \
    --val-data "xxx" \
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
    --logs xxx
