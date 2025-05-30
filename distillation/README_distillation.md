# Llama Knowledge Distillation with Hugging Face Trainer

This directory contains a complete implementation for distilling knowledge from a Llama-3.2-3B-Instruct model into a Llama-3.2-1B-Instruct model using:

- **Hugging Face Trainer API** for training orchestration
- **LoRA adapters with PEFT** for efficient fine-tuning
- **WikiText-v2 dataset** for training data
- **Knowledge distillation** with temperature scaling

## Files Overview

- `distillation_trainer_hf.py` - Main training script using Hugging Face Trainer
- `run_distillation.sh` - Convenient bash script to run training with default parameters
- `requirements_distillation.txt` - Python dependencies
- `hqq_distillation_training.py` - Alternative implementation with custom training loop (existing)

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements_distillation.txt
```

### 2. Run Training (Easy Method)

```bash
./run_distillation.sh
```

### 3. Run Training (Custom Parameters)

```bash
python distillation_trainer_hf.py \
    --teacher_model_name "meta-llama/Llama-3.2-3B-Instruct" \
    --student_model_name "meta-llama/Llama-3.2-1B-Instruct" \
    --output_dir "./my_distilled_model" \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --learning_rate 1e-4 \
    --lora_r 16 \
    --temperature 4.0 \
    --alpha 0.7 \
    --beta 0.3
```

## Key Features

### 🔥 Knowledge Distillation
- **Temperature Scaling**: Uses temperature=4.0 for soft target generation
- **Combined Loss**: Weighted combination of distillation loss (α=0.7) and task loss (β=0.3)
- **KL Divergence**: Soft targets from teacher guide student learning

### 🎯 LoRA Fine-tuning
- **Parameter Efficient**: Only fine-tunes ~1% of model parameters
- **Target Modules**: Attention and MLP layers (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`)
- **Configurable**: Rank (r=16), alpha (32), dropout (0.05)

### 📊 Training Features
- **Automatic Mixed Precision**: FP16 training for efficiency
- **Gradient Accumulation**: Effective batch size scaling
- **Learning Rate Scheduling**: Warmup + linear decay
- **Evaluation Strategy**: Regular evaluation on validation set
- **Checkpointing**: Best model saving based on validation loss

### 📈 Monitoring
- **Weights & Biases Integration**: Automatic experiment tracking
- **Detailed Logging**: Separate tracking of distillation and task losses
- **Perplexity Metrics**: Automatic perplexity calculation

## Configuration Options

### Model Arguments
- `--teacher_model_name`: Hugging Face model ID for teacher (default: Llama-3.2-3B-Instruct)
- `--student_model_name`: Hugging Face model ID for student (default: Llama-3.2-1B-Instruct)

### LoRA Configuration
- `--lora_r`: LoRA rank (default: 16)
- `--lora_alpha`: LoRA alpha scaling (default: 32)
- `--lora_dropout`: LoRA dropout rate (default: 0.05)
- `--lora_target_modules`: Comma-separated list of target modules

### Distillation Parameters
- `--temperature`: Softmax temperature for distillation (default: 4.0)
- `--alpha`: Weight for distillation loss (default: 0.7)
- `--beta`: Weight for task loss (default: 0.3)

### Training Arguments
- `--num_train_epochs`: Number of training epochs (default: 3)
- `--per_device_train_batch_size`: Batch size per device (default: 4)
- `--gradient_accumulation_steps`: Steps to accumulate gradients (default: 4)
- `--learning_rate`: Learning rate (default: 1e-4)
- `--max_length`: Maximum sequence length (default: 512)

## Output Structure

After training, you'll find:

```
./distilled_model/
├── pytorch_model.bin              # Full fine-tuned model
├── config.json                    # Model configuration
├── generation_config.json         # Generation settings
├── training_args.bin              # Training arguments
├── tokenizer.json                 # Tokenizer files
├── tokenizer_config.json
├── special_tokens_map.json
└── lora_adapters/                 # LoRA-specific files
    ├── adapter_config.json        # LoRA configuration
    ├── adapter_model.bin          # LoRA weights only
    └── tokenizer files...
```

## Memory Requirements

- **Minimum GPU Memory**: 16GB (with gradient checkpointing)
- **Recommended GPU Memory**: 24GB or higher
- **CPU Memory**: 32GB+ recommended for dataset loading

## Performance Tips

1. **Batch Size**: Increase `per_device_train_batch_size` if you have more GPU memory
2. **Gradient Accumulation**: Use higher `gradient_accumulation_steps` for effective larger batch sizes
3. **Sequence Length**: Reduce `max_length` if facing memory issues
4. **LoRA Rank**: Lower `lora_r` reduces memory but may hurt performance

## Example Results

After 3 epochs of training, you should expect:
- **Distillation Loss**: Decreasing KL divergence between teacher and student
- **Task Loss**: Standard language modeling loss improvement
- **Perplexity**: Validation perplexity around 15-25 on WikiText-v2

## Troubleshooting

### Out of Memory (OOM)
- Reduce `per_device_train_batch_size`
- Reduce `max_length`
- Increase `gradient_accumulation_steps` to maintain effective batch size

### Slow Training
- Increase `per_device_train_batch_size`
- Use `--fp16` for mixed precision training
- Consider using `flash-attn` for faster attention computation

### Poor Performance
- Increase `num_train_epochs`
- Tune `temperature`, `alpha`, and `beta` parameters
- Increase `lora_r` for more capacity

## Advanced Usage

### Custom Dataset
To use your own dataset, modify the `prepare_dataset` function in `distillation_trainer_hf.py`:

```python
# Replace the load_dataset call with your custom data loading
dataset = load_dataset("your_dataset_name")
```

### Different Models
You can distill between any compatible Hugging Face models:

```bash
python distillation_trainer_hf.py \
    --teacher_model_name "your-teacher-model" \
    --student_model_name "your-student-model"
```

### Hyperparameter Tuning
Key hyperparameters to tune:
- `temperature`: Higher values (4-8) for more diverse soft targets
- `alpha/beta`: Balance between distillation and task loss
- `lora_r`: Higher rank for more model capacity
- `learning_rate`: Typically 1e-4 to 5e-4 for LoRA

## Citation

If you use this code, please cite the relevant papers:
- LoRA: https://arxiv.org/abs/2106.09685
- Knowledge Distillation: https://arxiv.org/abs/1503.02531
