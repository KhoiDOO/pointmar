#!/bin/bash

# Training script for Mesh500 dataset with MAR model and Diffusion Loss
# Point-by-Point Generation

# Set environment
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Dataset configuration
DATASET_NAME="shapenet"
DATA_PATH="./.cache"
NUM_POINTS=1024

# Model configuration
MODEL="mar_base"
TOKEN_EMBED_DIM=3

# Training configuration
BATCH_SIZE=32
EPOCHS=400
NUM_WORKERS=10

# Learning rate configuration
BASE_LR=1e-4
WEIGHT_DECAY=0.02
WARMUP_EPOCHS=100
LR_SCHEDULE="constant"
MIN_LR=0.0

# MAR specific parameters
MASK_RATIO_MIN=0.7
GRAD_CLIP=3.0
ATTN_DROPOUT=0.1
PROJ_DROPOUT=0.1
BUFFER_SIZE=64

# Diffusion Loss parameters
NUM_SAMPLING_STEPS="100"
DIFFUSION_BATCH_MUL=1
TEMPERATURE=1.0

# Generation parameters
NUM_ITER=64
CFG=1.0
CFG_SCHEDULE="linear"

# EMA parameters
EMA_RATE=0.9999

# Evaluation parameters
EVAL_FREQ=40
SAVE_LAST_FREQ=5
EVAL_BSZ=64

# Output directories
OUTPUT_DIR="./outputs/shapenet/$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$OUTPUT_DIR"

# Create output directory
mkdir -p $OUTPUT_DIR

# Generate random port for distributed training
RANDOM_PORT=$((10000 + RANDOM % 90000))

echo "Starting Mesh500 training..."
echo "Output directory: $OUTPUT_DIR"
echo "Dataset path: $DATA_PATH"
echo "Using port: $RANDOM_PORT"

# Training command
torchrun \
    --nproc_per_node=1 \
    --nnodes=1 \
    --master_port $RANDOM_PORT \
    main.py \
    --batch_size $BATCH_SIZE \
    --epochs $EPOCHS \
    \
    --model $MODEL \
    \
    --num_points $NUM_POINTS \
    --token_embed_dim $TOKEN_EMBED_DIM \
    \
    --num_iter $NUM_ITER \
    --cfg $CFG \
    --cfg_schedule $CFG_SCHEDULE \
    --eval_freq $EVAL_FREQ \
    --save_last_freq $SAVE_LAST_FREQ \
    --eval_bsz $EVAL_BSZ \
    --checkpoint_key loss \
    --checkpoint_mode min \
    \
    --weight_decay $WEIGHT_DECAY \
    \
    --grad_checkpointing \
    --blr $BASE_LR \
    --min_lr $MIN_LR \
    --lr_schedule $LR_SCHEDULE \
    --warmup_epochs $WARMUP_EPOCHS \
    --ema_rate $EMA_RATE \
    \
    --mask_ratio_min $MASK_RATIO_MIN \
    --grad_clip $GRAD_CLIP \
    --attn_dropout $ATTN_DROPOUT \
    --proj_dropout $PROJ_DROPOUT \
    --buffer_size $BUFFER_SIZE \
    \
    --num_sampling_steps $NUM_SAMPLING_STEPS \
    --diffusion_batch_mul $DIFFUSION_BATCH_MUL \
    --temperature $TEMPERATURE \
    \
    --dataset_name $DATASET_NAME \
    --data_path $DATA_PATH \
    \
    --output_dir $OUTPUT_DIR \
    --log_dir $LOG_DIR \
    --seed 1 \
    --num_workers $NUM_WORKERS \
    --pin_mem \
    \
    --world_size 1
    

echo "Training completed!"
echo "Results saved in: $OUTPUT_DIR"
