"""
Configuration file for HQQ + LoRA distributed training
This file contains predefined configurations for different scenarios
"""

import argparse
from dataclasses import dataclass
from typing import List, Dict, Any


@dataclass
class HQQConfig:
    """HQQ quantization configuration"""
    bits: int = 4
    group_size: int = 128
    config_type: str = "uniform"  # "uniform" or "custom"


@dataclass
class LoRAConfig:
    """LoRA configuration"""
    r: int = 16
    alpha: int = 32
    dropout: float = 0.1
    target_modules: List[str] = None
    
    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj", 
                "gate_proj", "up_proj", "down_proj"
            ]


@dataclass
class TrainingConfig:
    """Training configuration"""
    num_epochs: int = 3
    batch_size: int = 4
    eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_steps: int = 100
    max_length: int = 512
    logging_steps: int = 10
    eval_steps: int = 100
    save_steps: int = 500
    save_total_limit: int = 3


# Predefined configurations for different scenarios
CONFIGS = {
    "fast_test": {
        "hqq": HQQConfig(bits=4, group_size=64),
        "lora": LoRAConfig(r=8, alpha=16, target_modules=["q_proj", "v_proj"]),
        "training": TrainingConfig(
            num_epochs=1, 
            batch_size=2, 
            max_length=256,
            eval_steps=50,
            save_steps=100
        )
    },
    
    "balanced": {
        "hqq": HQQConfig(bits=4, group_size=128),
        "lora": LoRAConfig(r=16, alpha=32),
        "training": TrainingConfig(
            num_epochs=3,
            batch_size=4,
            max_length=512
        )
    },
    
    "high_quality": {
        "hqq": HQQConfig(bits=8, group_size=64, config_type="custom"),
        "lora": LoRAConfig(r=32, alpha=64, dropout=0.05),
        "training": TrainingConfig(
            num_epochs=5,
            batch_size=2,
            gradient_accumulation_steps=8,
            learning_rate=1e-4,
            max_length=1024
        )
    },
    
    "memory_efficient": {
        "hqq": HQQConfig(bits=3, group_size=128),
        "lora": LoRAConfig(r=8, alpha=16, target_modules=["q_proj", "v_proj"]),
        "training": TrainingConfig(
            num_epochs=3,
            batch_size=1,
            gradient_accumulation_steps=16,
            max_length=256
        )
    }
}


def get_config(config_name: str) -> Dict[str, Any]:
    """Get configuration by name"""
    if config_name not in CONFIGS:
        raise ValueError(f"Unknown config: {config_name}. Available: {list(CONFIGS.keys())}")
    
    return CONFIGS[config_name]


def apply_config_to_args(args: argparse.Namespace, config_name: str) -> argparse.Namespace:
    """Apply configuration to command line arguments"""
    config = get_config(config_name)
    
    # Apply HQQ config
    hqq_config = config["hqq"]
    args.hqq_bits = hqq_config.bits
    args.hqq_group_size = hqq_config.group_size
    args.hqq_config = hqq_config.config_type
    
    # Apply LoRA config
    lora_config = config["lora"]
    args.lora_r = lora_config.r
    args.lora_alpha = lora_config.alpha
    args.lora_dropout = lora_config.dropout
    args.lora_target_modules = lora_config.target_modules
    
    # Apply training config
    training_config = config["training"]
    args.num_epochs = training_config.num_epochs
    args.batch_size = training_config.batch_size
    args.eval_batch_size = training_config.eval_batch_size
    args.gradient_accumulation_steps = training_config.gradient_accumulation_steps
    args.learning_rate = training_config.learning_rate
    args.weight_decay = training_config.weight_decay
    args.warmup_steps = training_config.warmup_steps
    args.max_length = training_config.max_length
    args.logging_steps = training_config.logging_steps
    args.eval_steps = training_config.eval_steps
    args.save_steps = training_config.save_steps
    args.save_total_limit = training_config.save_total_limit
    
    return args


def print_available_configs():
    """Print all available configurations"""
    print("Available configurations:")
    print("=" * 50)
    
    for name, config in CONFIGS.items():
        print(f"\n{name}:")
        print(f"  HQQ: {config['hqq'].bits}-bit, group_size={config['hqq'].group_size}")
        print(f"  LoRA: r={config['lora'].r}, alpha={config['lora'].alpha}")
        print(f"  Training: {config['training'].num_epochs} epochs, batch_size={config['training'].batch_size}")


if __name__ == "__main__":
    print_available_configs()
