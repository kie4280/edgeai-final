#!/bin/bash

# Memory-Optimized Distributed Knowledge Distillation Training Script
# This script supports both single-GPU and multi-GPU training with maximum memory optimization

set -e

# Default values
NUM_GPUS=${NUM_GPUS:-1}
OUTPUT_DIR=${OUTPUT_DIR:-"./distilled_model_memory_opt"}
TEACHER_MODEL=${TEACHER_MODEL:-"meta-llama/Llama-3.2-3B-Instruct"}
STUDENT_MODEL=${STUDENT_MODEL:-"meta-llama/Llama-3.2-1B-Instruct"}
WANDB_PROJECT=${WANDB_PROJECT:-""}
RUN_NAME=${RUN_NAME:-"memory-opt-distillation-$(date +%Y%m%d-%H%M%S)"}

# Memory-optimized training parameters
BATCH_SIZE=${BATCH_SIZE:-1}
GRAD_ACCUMULATION=${GRAD_ACCUMULATION:-32}
MAX_LENGTH=${MAX_LENGTH:-256}
LEARNING_RATE=${LEARNING_RATE:-2e-4}
EPOCHS=${EPOCHS:-1}

# LoRA parameters (memory-optimized)
LORA_R=${LORA_R:-8}
LORA_ALPHA=${LORA_ALPHA:-16}
LORA_DROPOUT=${LORA_DROPOUT:-0.1}
LORA_TARGET_MODULES=${LORA_TARGET_MODULES:-"q_proj,v_proj,gate_proj,down_proj"}

# Distillation parameters
TEMPERATURE=${TEMPERATURE:-4.0}
ALPHA=${ALPHA:-0.7}
BETA=${BETA:-0.3}

echo "Starting Memory-Optimized Knowledge Distillation Training"
echo "=============================================="
echo "Teacher Model: $TEACHER_MODEL"
echo "Student Model: $STUDENT_MODEL"
echo "Output Directory: $OUTPUT_DIR"
echo "Number of GPUs: $NUM_GPUS"
echo "Batch Size: $BATCH_SIZE"
echo "Gradient Accumulation: $GRAD_ACCUMULATION"
echo "Max Length: $MAX_LENGTH"
echo "Learning Rate: $LEARNING_RATE"
echo "Epochs: $EPOCHS"
echo "LoRA r: $LORA_R, alpha: $LORA_ALPHA"
echo "Temperature: $TEMPERATURE, Alpha: $ALPHA, Beta: $BETA"
echo "=============================================="

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Common training arguments
COMMON_ARGS="
    --teacher_model_name $TEACHER_MODEL \
    --student_model_name $STUDENT_MODEL \
    --output_dir $OUTPUT_DIR \
    --per_device_train_batch_size $BATCH_SIZE \
    --per_device_eval_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUMULATION \
    --learning_rate $LEARNING_RATE \
    --num_train_epochs $EPOCHS \
    --max_length $MAX_LENGTH \
    --lora_r $LORA_R \
    --lora_alpha $LORA_ALPHA \
    --lora_dropout $LORA_DROPOUT \
    --lora_target_modules $LORA_TARGET_MODULES \
    --temperature $TEMPERATURE \
    --alpha $ALPHA \
    --beta $BETA \
    --warmup_steps 50 \
    --eval_steps 500 \
    --save_steps 1000 \
    --logging_steps 25 \
    --run_name $RUN_NAME"

# Add wandb project if specified
if [ -n "$WANDB_PROJECT" ]; then
    COMMON_ARGS="$COMMON_ARGS --wandb_project $WANDB_PROJECT"
fi

# Run training based on number of GPUs
if [ "$NUM_GPUS" -eq 1 ]; then
    echo "Running single-GPU memory-optimized training..."
    python distillation_trainer_memory_optimized.py $COMMON_ARGS
else
    echo "Running multi-GPU memory-optimized training with $NUM_GPUS GPUs..."
    torchrun \
        --standalone \
        --nproc_per_node=$NUM_GPUS \
        distillation_trainer_memory_optimized.py \
        $COMMON_ARGS
fi

echo "Training completed! Model saved to: $OUTPUT_DIR"

# Display memory usage if nvidia-smi is available
if command -v nvidia-smi &> /dev/null; then
    echo "Final GPU memory usage:"
    nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits
fi
