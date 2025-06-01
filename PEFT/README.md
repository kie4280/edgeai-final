# HQQ + LoRA Distributed Training for Llama-3.2-1B

This repository contains scripts for distributed training of Llama-3.2-1B model using HQQ (Hierarchical Quantization) and LoRA (Low-Rank Adaptation) on the WikiText-v2 dataset.

## Features

- **HQQ Quantization**: Efficient 4-bit/8-bit quantization for memory reduction
- **LoRA Adapters**: Parameter-efficient fine-tuning
- **Distributed Training**: Multi-GPU support using PyTorch DDP
- **Hugging Face Trainer**: Easy-to-use training pipeline
- **Configurable**: Multiple predefined configurations for different scenarios
- **Monitoring**: Weights & Biases integration for experiment tracking

## Requirements

Install the required dependencies:

```bash
pip install -r requirements_hqq_lora.txt
```

### Key Dependencies

- PyTorch >= 2.0.0
- Transformers >= 4.36.0
- PEFT >= 0.7.0
- HQQ >= 0.1.0
- Datasets >= 2.14.0

## Quick Start

### 1. Basic Training

Run distributed training with default settings:

```bash
# Make the script executable
chmod +x run_hqq_lora_distributed.sh

# Run on 2 GPUs
./run_hqq_lora_distributed.sh 2

# Run on 4 GPUs
./run_hqq_lora_distributed.sh 4
```

### 2. Using Predefined Configurations

List available configurations:

```bash
python hqq_lora_distributed_training.py --list_configs
```

Run with a specific configuration:

```bash
# Fast test configuration (1 epoch, small model)
./run_hqq_lora_distributed.sh 2 --config fast_test

# Balanced configuration (default)
./run_hqq_lora_distributed.sh 2 --config balanced

# High quality configuration (8-bit, higher rank)
./run_hqq_lora_distributed.sh 2 --config high_quality

# Memory efficient configuration (3-bit, low rank)
./run_hqq_lora_distributed.sh 2 --config memory_efficient
```

### 3. Custom Parameters

Override specific parameters:

```bash
./run_hqq_lora_distributed.sh 2 \
    --model_name "meta-llama/Llama-3.2-1B" \
    --hqq_bits 4 \
    --lora_r 32 \
    --num_epochs 5 \
    --batch_size 2 \
    --learning_rate 1e-4
```

## Configuration Options

### HQQ Quantization

- `--hqq_bits`: Number of bits (3, 4, 8)
- `--hqq_group_size`: Group size for quantization (64, 128, 256)
- `--hqq_config`: Configuration type ("uniform" or "custom")

### LoRA Parameters

- `--lora_r`: LoRA rank (8, 16, 32, 64)
- `--lora_alpha`: LoRA alpha parameter
- `--lora_dropout`: Dropout rate for LoRA layers
- `--lora_target_modules`: Target modules for LoRA adaptation

### Training Parameters

- `--num_epochs`: Number of training epochs
- `--batch_size`: Training batch size per device
- `--gradient_accumulation_steps`: Gradient accumulation steps
- `--learning_rate`: Learning rate
- `--max_length`: Maximum sequence length

## Predefined Configurations

### 1. Fast Test (`fast_test`)
- Purpose: Quick testing and debugging
- HQQ: 4-bit, group_size=64
- LoRA: r=8, alpha=16, limited target modules
- Training: 1 epoch, small batch size

### 2. Balanced (`balanced`)
- Purpose: Good balance of quality and efficiency
- HQQ: 4-bit, group_size=128
- LoRA: r=16, alpha=32
- Training: 3 epochs, moderate batch size

### 3. High Quality (`high_quality`)
- Purpose: Best quality results
- HQQ: 8-bit, group_size=64, custom config
- LoRA: r=32, alpha=64
- Training: 5 epochs, larger context

### 4. Memory Efficient (`memory_efficient`)
- Purpose: Minimal memory usage
- HQQ: 3-bit, group_size=128
- LoRA: r=8, alpha=16, limited modules
- Training: Small batch, high accumulation

## Inference

After training, use the inference script to test your model:

```bash
# Interactive mode
python inference_hqq_lora.py \
    --model_path ./hqq_lora_output \
    --adapter_path ./hqq_lora_output/lora_adapters \
    --mode interactive

# Single prompt
python inference_hqq_lora.py \
    --model_path ./hqq_lora_output \
    --adapter_path ./hqq_lora_output/lora_adapters \
    --mode single \
    --prompt "Explain machine learning in simple terms."

# Benchmark mode
python inference_hqq_lora.py \
    --model_path ./hqq_lora_output \
    --adapter_path ./hqq_lora_output/lora_adapters \
    --mode benchmark
```

## Output Structure

After training, the output directory contains:

```
hqq_lora_output/
├── config.json                 # Model configuration
├── pytorch_model.bin           # Trained model weights
├── tokenizer.json              # Tokenizer files
├── training_args.bin           # Training arguments
├── eval_results.json           # Final evaluation results
├── lora_adapters/              # LoRA adapter weights
│   ├── adapter_config.json
│   └── adapter_model.safetensors
└── runs/                       # TensorBoard logs
```

## Monitoring with Weights & Biases

To enable W&B logging:

1. Install wandb: `pip install wandb`
2. Login: `wandb login`
3. Add `--use_wandb` flag to your training command

```bash
./run_hqq_lora_distributed.sh 2 --use_wandb --run_name "my_experiment"
```

## Memory Requirements

Approximate memory usage for Llama-3.2-1B:

| Configuration | GPU Memory | Training Time (3 epochs) |
|---------------|------------|--------------------------|
| Memory Efficient | ~4GB | ~2 hours |
| Balanced | ~6GB | ~3 hours |
| High Quality | ~8GB | ~5 hours |

*Note: Times are approximate and depend on hardware*

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce `--batch_size`
   - Increase `--gradient_accumulation_steps`
   - Use `memory_efficient` configuration

2. **Slow Training**
   - Ensure CUDA is properly installed
   - Check GPU utilization with `nvidia-smi`
   - Use mixed precision training (enabled by default)

3. **Import Errors**
   - Install all requirements: `pip install -r requirements_hqq_lora.txt`
   - Check HQQ installation: `pip install hqq`

### Performance Tips

1. **Optimize Batch Size**: Find the largest batch size that fits in memory
2. **Use Multiple GPUs**: Distributed training scales well
3. **Enable Compilation**: PyTorch compilation is enabled by default
4. **Monitor GPU Usage**: Use `nvidia-smi` to check utilization

## Advanced Usage

### Custom HQQ Configuration

For custom layer-wise quantization:

```python
# In training_config.py, modify the custom configuration
q4_config = BaseQuantizeConfig(nbits=4, group_size=64)
q3_config = BaseQuantizeConfig(nbits=3, group_size=128)

custom_quant_config = {
    'self_attn.q_proj': q4_config,
    'self_attn.k_proj': q4_config,
    'self_attn.v_proj': q4_config,
    'self_attn.o_proj': q4_config,
    'mlp.gate_proj': q3_config,  # Lower precision for MLP
    'mlp.up_proj': q3_config,
    'mlp.down_proj': q3_config,
}
```

### Custom LoRA Targets

To target specific modules:

```bash
./run_hqq_lora_distributed.sh 2 \
    --lora_target_modules q_proj k_proj v_proj o_proj \
    --lora_r 16
```

## Contributing

Feel free to submit issues and enhancement requests!

## License

This project is licensed under the MIT License - see the LICENSE file for details.
