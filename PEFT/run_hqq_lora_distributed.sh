#!/bin/bash

# Distributed HQQ + LoRA Training Script
# Usage: ./run_hqq_lora_distributed.sh [num_gpus] [additional_args...]

set -e


# Default values
NUM_GPUS=${1:-2}
shift || true  # Remove first argument if it exists

# Additional arguments passed to the script
ADDITIONAL_ARGS="$@"

# Check if CUDA is available
if ! command -v nvidia-smi &> /dev/null; then
    echo "Error: CUDA/nvidia-smi not found. Please ensure CUDA is installed."
    exit 1
fi

# Check available GPUs
AVAILABLE_GPUS=$(nvidia-smi --list-gpus | wc -l)
echo "Available GPUs: $AVAILABLE_GPUS"

if [ "$NUM_GPUS" -gt "$AVAILABLE_GPUS" ]; then
    echo "Warning: Requested $NUM_GPUS GPUs but only $AVAILABLE_GPUS available. Using $AVAILABLE_GPUS GPUs."
    NUM_GPUS=$AVAILABLE_GPUS
fi

echo "Running distributed training on $NUM_GPUS GPUs..."

# Set environment variables for distributed training
export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NUM_GPUS-1)))
# export NCCL_DEBUG=INFO
export NCCL_TREE_THRESHOLD=0
export OMP_NUM_THREADS=1

# Default training arguments
DEFAULT_ARGS=(
    --model_name "meta-llama/Llama-3.2-1B"
    --output_dir "./hqq_lora_output"
    --hqq_bits 4
    --hqq_group_size 128
    --hqq_config "uniform"
    --lora_r 16
    --lora_alpha 32
    --lora_dropout 0.1
    --num_epochs 3
    --batch_size 4
    --eval_batch_size 4
    --gradient_accumulation_steps 4
    --learning_rate 2e-4
    --weight_decay 0.01
    --warmup_steps 100
    --max_length 512
    --logging_steps 10
    --eval_steps 100
    --save_steps 500
    --save_total_limit 3
    # --use_wandb
    --run_name "hqq_lora_llama_1b_$(date +%Y%m%d_%H%M%S)"
    --seed 42
)

# Combine default and additional arguments
ALL_ARGS=("${DEFAULT_ARGS[@]}" $ADDITIONAL_ARGS)

# Create output directory
mkdir -p ./hqq_lora_output

# Run distributed training
if [ "$NUM_GPUS" -eq 1 ]; then
    echo "Running single GPU training..."
    python hqq_lora_distributed_training.py "${ALL_ARGS[@]}"
else
    echo "Running multi-GPU distributed training..."
    torchrun \
        --nproc_per_node=$NUM_GPUS \
        --master_port=29500 \
        hqq_lora_distributed_training.py \
        "${ALL_ARGS[@]}"
fi

echo "Training completed!"
echo "Results saved to: ./hqq_lora_output"
echo "LoRA adapters saved to: ./hqq_lora_output/lora_adapters"
