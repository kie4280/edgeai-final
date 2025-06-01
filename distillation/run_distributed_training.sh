#!/bin/bash

# Distributed Knowledge Distillation Training Script
# Usage: ./run_distributed_training.sh [number_of_gpus] [additional_args...]

# Set default number of GPUs if not specified
NPROC_PER_NODE=${1:-4}
shift  # Remove first argument so $@ contains only the remaining arguments

# Set distributed training environment variables
export MASTER_ADDR=localhost
export MASTER_PORT=29500
export OMP_NUM_THREADS=1

echo "Starting distributed training on $NPROC_PER_NODE GPUs..."
echo "Additional arguments: $@"

# Run distributed training with torchrun
torchrun \
    --standalone \
    --nproc_per_node=$NPROC_PER_NODE \
    --nnodes=1 \
    --master_port=$MASTER_PORT \
    distillation_trainer_hf.py \
    --output_dir ./distilled_model_distributed \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --learning_rate 1e-4 \
    --warmup_steps 100 \
    --eval_steps 250 \
    --save_steps 500 \
    --logging_steps 10 \
    --wandb_project llama-distillation-distributed \
    --run_name 3b-to-1b-dist-$(date +%Y%m%d_%H%M%S) \
    "$@"

echo "Training completed!"
