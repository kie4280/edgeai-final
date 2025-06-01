# Memory Optimization Strategies for Knowledge Distillation

## Quick Fixes (apply immediately):

### 1. Reduce Batch Size and Sequence Length
```bash
python distillation_trainer_hf.py \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --max_length 256
```

### 2. Use the Memory-Optimized Script
```bash
bash run_memory_optimized_training.sh
```

## Advanced Memory Optimization Techniques:

### 1. Environment Variables
Add these to your shell or script:
```bash
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export CUDA_VISIBLE_DEVICES=0  # Use only one GPU
```

### 2. CPU Offloading for Teacher Model
The memory-optimized script moves the teacher model to CPU between forward passes.

### 3. Gradient Checkpointing
Already enabled in the optimized versions - trades compute for memory.

### 4. 8-bit Quantization
The teacher model is loaded with 8-bit quantization to reduce memory footprint.

### 5. Reduced LoRA Parameters
- Smaller rank (r=8 instead of 16)
- Fewer target modules
- Lower alpha values

## If Still Running Out of Memory:

### Option 1: Ultra-Low Memory Settings
```bash
python distillation_trainer_memory_optimized.py \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 64 \
    --max_length 128 \
    --lora_r 4 \
    --lora_alpha 8
```

### Option 2: CPU-Only Training (Slower but Works)
Modify the script to load models with `device_map="cpu"` for both teacher and student.

### Option 3: Use Smaller Models
- Teacher: "microsoft/DialoGPT-medium" (345M parameters)
- Student: "microsoft/DialoGPT-small" (117M parameters)

### Option 4: Sequential Training
Train without the teacher model loaded simultaneously - cache teacher outputs first.

## Monitoring Memory Usage:
```bash
nvidia-smi -l 1  # Monitor GPU memory in real-time
```

## Distributed Training on Multiple GPUs:
```bash
torchrun --standalone --nproc_per_node=2 distillation_trainer_hf.py \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8
```

This distributes the memory load across multiple GPUs.
