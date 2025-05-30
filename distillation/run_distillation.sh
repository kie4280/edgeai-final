#!/bin/bash
# Simple script to run the knowledge distillation training

echo "Starting Llama-3.2 3B -> 1B Knowledge Distillation Training"
echo "============================================================"

# Default parameters
TEACHER_MODEL="meta-llama/Llama-3.2-3B-Instruct"
STUDENT_MODEL="meta-llama/Llama-3.2-1B-Instruct"
OUTPUT_DIR="./distilled_llama_1b"
EPOCHS=3
BATCH_SIZE=4
GRAD_ACCUM_STEPS=4
LEARNING_RATE=1e-4
MAX_LENGTH=512

# LoRA parameters
LORA_R=16
LORA_ALPHA=32
LORA_DROPOUT=0.05

# Distillation parameters
TEMPERATURE=4.0
ALPHA=0.7  # Weight for distillation loss
BETA=0.3   # Weight for task loss

# Wandb settings
WANDB_PROJECT="llama-distillation"
RUN_NAME="3b-to-1b-wikitext-$(date +%Y%m%d_%H%M%S)"

python distillation_trainer_hf.py \
    --teacher_model_name "$TEACHER_MODEL" \
    --student_model_name "$STUDENT_MODEL" \
    --output_dir "$OUTPUT_DIR" \
    --num_train_epochs $EPOCHS \
    --per_device_train_batch_size $BATCH_SIZE \
    --per_device_eval_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
    --learning_rate $LEARNING_RATE \
    --max_length $MAX_LENGTH \
    --lora_r $LORA_R \
    --lora_alpha $LORA_ALPHA \
    --lora_dropout $LORA_DROPOUT \
    --temperature $TEMPERATURE \
    --alpha $ALPHA \
    --beta $BETA \
    --warmup_steps 100 \
    --eval_steps 250 \
    --save_steps 500 \
    --logging_steps 10 \
    --wandb_project "$WANDB_PROJECT" \
    --run_name "$RUN_NAME" \
    --seed 42

echo "Training completed!"
echo "Model saved to: $OUTPUT_DIR"
echo "LoRA adapters saved to: $OUTPUT_DIR/lora_adapters"
