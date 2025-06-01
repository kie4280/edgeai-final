#!/usr/bin/env python3
"""
Distributed Training Script for Llama-3.2-1B with HQQ Quantization and PEFT LoRA
Dataset: WikiText-v2
"""

import os
import sys
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling,
    set_seed
)
from datasets import load_dataset
from peft import (
    LoraConfig, 
    get_peft_model, 
    TaskType,
    PeftModel
)
from hqq.models.hf.base import AutoHQQHFModel
from hqq.models.base import HQQLinear
from hqq.core.quantize import BaseQuantizeConfig, HQQBackend
from hqq.utils.patching import prepare_for_inference, recommended_inductor_config_setter
import wandb
import argparse
from typing import Dict, Any
import json
import logging
from tqdm.auto import tqdm
from training_config import apply_config_to_args, print_available_configs


def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('hqq_lora_training.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def setup_distributed():
    """Initialize distributed training"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
        
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)
        
        return rank, world_size, local_rank
    else:
        # Single GPU training
        return 0, 1, 0


def get_hqq_config(args):
    """Get HQQ quantization configuration"""
    if args.hqq_config == "custom":
        # Custom configuration for different layers
        q4_config = BaseQuantizeConfig(nbits=4, group_size=64)
        q3_config = BaseQuantizeConfig(nbits=3, group_size=64)
        
        quant_config = {
            'self_attn.q_proj': q4_config,
            'self_attn.k_proj': q4_config,
            'self_attn.v_proj': q4_config,
            'self_attn.o_proj': q4_config,
            'mlp.gate_proj': q3_config,
            'mlp.up_proj': q3_config,
            'mlp.down_proj': q3_config,
            'offload_meta': False
        }
    else:
        # Uniform quantization
        quant_config = BaseQuantizeConfig(
            nbits=args.hqq_bits, 
            group_size=args.hqq_group_size
        )
    
    return quant_config


def get_lora_config(args):
    """Get LoRA configuration"""
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.lora_target_modules,
        bias="none",
        inference_mode=False,
    )
    return lora_config


def load_and_prepare_model(args, logger):
    """Load model, apply HQQ quantization, then LoRA adapters"""
    logger.info(f"Loading model: {args.model_name}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )
    
    # Apply HQQ quantization first
    logger.info("Applying HQQ quantization to base model...")
    quant_config = get_hqq_config(args)
    HQQLinear.set_backend(HQQBackend.PYTORCH_COMPILE)
    AutoHQQHFModel.quantize_model(
        model,
        quant_config=quant_config,
        compute_dtype=torch.float16,
        device='cuda'
    )
    
    # Now apply LoRA adapters on quantized model
    logger.info("Applying LoRA adapters...")
    lora_config = get_lora_config(args)
    model = get_peft_model(model, lora_config)
    
    # Confirm LoRA parameters are trainable
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total trainable LoRA parameters: {trainable:,}")
    if trainable == 0:
        logger.error("No LoRA parameters detected as trainable!")
        raise ValueError("LoRA parameters were not set as trainable.")

    model.print_trainable_parameters()
    
    return model, tokenizer


def prepare_dataset(tokenizer, args, logger):
    """Prepare WikiText-v2 dataset for training"""
    logger.info("Loading WikiText-v2 dataset...")
    
    # Load dataset
    dataset = load_dataset("wikitext", "wikitext-2-v1")
    
    def tokenize_function(examples):
        # Filter out empty texts
        texts = [text.strip() for text in examples["text"] if text.strip()]
        
        if not texts:
            return {"input_ids": [], "attention_mask": [], "labels": []}
        
        # Tokenize each text individually to avoid length mismatches
        all_input_ids = []
        all_attention_masks = []
        all_labels = []
        
        for text in texts:
            # Tokenize individual text
            encoded = tokenizer(
                text,
                truncation=True,
                padding=False,
                max_length=args.max_length,
                return_attention_mask=True,
            )
            
            input_ids = encoded["input_ids"]
            attention_mask = encoded["attention_mask"]
            
            # Skip very short sequences
            if len(input_ids) < 10:
                continue
            
            # Ensure exact length
            if len(input_ids) > args.max_length:
                input_ids = input_ids[:args.max_length]
                attention_mask = attention_mask[:args.max_length]
            elif len(input_ids) < args.max_length:
                # Pad to max_length
                pad_length = args.max_length - len(input_ids)
                input_ids = input_ids + [tokenizer.pad_token_id] * pad_length
                attention_mask = attention_mask + [0] * pad_length
            
            all_input_ids.append(input_ids)
            all_attention_masks.append(attention_mask)
            all_labels.append(input_ids.copy())  # Labels same as input_ids for causal LM
        
        return {
            "input_ids": all_input_ids,
            "attention_mask": all_attention_masks,
            "labels": all_labels
        }
    
    # Process dataset with smaller batches to avoid memory issues
    logger.info("Tokenizing train dataset...")
    train_dataset = dataset["train"].map(
        tokenize_function,
        batched=True,
        remove_columns=dataset["train"].column_names,
        batch_size=50,
        desc="Tokenizing train dataset"
    )
    
    logger.info("Tokenizing validation dataset...")
    val_dataset = dataset["validation"].map(
        tokenize_function,
        batched=True,
        remove_columns=dataset["validation"].column_names,
        batch_size=50,
        desc="Tokenizing validation dataset"
    )
    
    # Filter out empty examples and flatten if needed
    def flatten_dataset(example):
        # Ensure all fields are lists of the same length
        if isinstance(example["input_ids"][0], list):
            # Flatten nested lists
            flat_input_ids = []
            flat_attention_masks = []
            flat_labels = []
            
            for input_ids, attention_mask, labels in zip(
                example["input_ids"], 
                example["attention_mask"], 
                example["labels"]
            ):
                if isinstance(input_ids, list) and len(input_ids) > 0:
                    flat_input_ids.extend(input_ids if isinstance(input_ids[0], list) else [input_ids])
                    flat_attention_masks.extend(attention_mask if isinstance(attention_mask[0], list) else [attention_mask])
                    flat_labels.extend(labels if isinstance(labels[0], list) else [labels])
            
            return {
                "input_ids": flat_input_ids,
                "attention_mask": flat_attention_masks,
                "labels": flat_labels
            }
        return example
    
    train_dataset = train_dataset.map(flatten_dataset, batched=True)
    val_dataset = val_dataset.map(flatten_dataset, batched=True)
    
    # Filter out empty examples
    train_dataset = train_dataset.filter(lambda x: len(x["input_ids"]) > 0)
    val_dataset = val_dataset.filter(lambda x: len(x["input_ids"]) > 0)
    
    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Validation dataset size: {len(val_dataset)}")
    
    # Verify dataset format
    if len(train_dataset) > 0:
        sample = train_dataset[0]
        logger.info(f"Sample input_ids length: {len(sample['input_ids'])}")
        logger.info(f"Sample attention_mask length: {len(sample['attention_mask'])}")
        logger.info(f"Sample labels length: {len(sample['labels'])}")
    
    return train_dataset, val_dataset


def create_trainer(model, tokenizer, train_dataset, val_dataset, args, rank):
    """Create Hugging Face Trainer"""
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8,  # For better performance
        return_tensors="pt",
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="wandb" if args.use_wandb and rank == 0 else None,
        run_name=args.run_name,
        ddp_find_unused_parameters=False,
        dataloader_pin_memory=False,
        fp16=True,
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        ddp_backend="nccl",
        local_rank=rank,
        remove_unused_columns=False,
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        processing_class=tokenizer,
    )
    
    return trainer


def main():
    parser = argparse.ArgumentParser(description="HQQ + LoRA Distributed Training")
    
    # Model arguments
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-1B",
                        help="Model name or path")
    parser.add_argument("--output_dir", type=str, default="./hqq_lora_output",
                        help="Output directory")
    
    # HQQ arguments
    parser.add_argument("--hqq_bits", type=int, default=4,
                        help="Number of bits for HQQ quantization")
    parser.add_argument("--hqq_group_size", type=int, default=128,
                        help="Group size for HQQ quantization")
    parser.add_argument("--hqq_config", type=str, default="uniform",
                        choices=["uniform", "custom"],
                        help="HQQ configuration type")
    
    # LoRA arguments
    parser.add_argument("--lora_r", type=int, default=16,
                        help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32,
                        help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.1,
                        help="LoRA dropout")
    parser.add_argument("--lora_target_modules", type=str, nargs="+",
                        default=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                        help="LoRA target modules")
    
    # Training arguments
    parser.add_argument("--num_epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Training batch size per device")
    parser.add_argument("--eval_batch_size", type=int, default=4,
                        help="Evaluation batch size per device")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                        help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=2e-4,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Weight decay")
    parser.add_argument("--warmup_steps", type=int, default=100,
                        help="Warmup steps")
    parser.add_argument("--max_length", type=int, default=512,
                        help="Maximum sequence length")
    
    # Logging and saving
    parser.add_argument("--logging_steps", type=int, default=10,
                        help="Logging steps")
    parser.add_argument("--eval_steps", type=int, default=100,
                        help="Evaluation steps")
    parser.add_argument("--save_steps", type=int, default=500,
                        help="Save steps")
    parser.add_argument("--save_total_limit", type=int, default=3,
                        help="Total save limit")
    
    # Experiment tracking
    parser.add_argument("--use_wandb", action="store_true",
                        help="Use Weights & Biases for logging")
    parser.add_argument("--run_name", type=str, default="hqq_lora_llama_1b",
                        help="Run name for experiment tracking")
    
    # Other
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--config", type=str, default=None,
                        choices=["fast_test", "balanced", "high_quality", "memory_efficient"],
                        help="Use predefined configuration")
    parser.add_argument("--list_configs", action="store_true",
                        help="List available configurations and exit")
    
    args = parser.parse_args()
    
    # List configurations if requested
    if args.list_configs:
        print_available_configs()
        return
    
    # Apply predefined configuration if specified
    if args.config:
        args = apply_config_to_args(args, args.config)
        print(f"Applied configuration: {args.config}")
    
    # Setup logging
    logger = setup_logging()
    
    # Setup distributed training
    rank, world_size, local_rank = setup_distributed()
    
    # Set seed
    set_seed(args.seed)
    
    # Setup inductor config for HQQ
    recommended_inductor_config_setter()
    
    logger.info(f"Rank: {rank}, World Size: {world_size}, Local Rank: {local_rank}")
    
    # Initialize wandb only on rank 0
    if args.use_wandb and rank == 0:
        wandb.init(
            project="hqq-lora-llama",
            name=args.run_name,
            config=vars(args)
        )
    
    try:
        # Load and prepare model
        model, tokenizer = load_and_prepare_model(args, logger)
        
        # Move model to GPU
        device = torch.device(f"cuda:{local_rank}")
        model = model.to(device)
        
        # Verify gradients are still enabled after moving to GPU
        trainable_params_after_gpu = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Trainable parameters after GPU move: {trainable_params_after_gpu:,}")
        
        if trainable_params_after_gpu == 0:
            logger.warning("No trainable parameters after GPU move! Re-enabling LoRA gradients...")
            for name, param in model.named_parameters():
                if any(lora_key in name for lora_key in ['lora_A', 'lora_B', 'lora_embedding_A', 'lora_embedding_B']):
                    param.requires_grad = True
                    logger.info(f"Re-enabled gradients for: {name}")
        
        # # Enable gradient checkpointing before wrapping with DDP
        # if hasattr(model, 'gradient_checkpointing_enable'):
        #     model.gradient_checkpointing_enable()
        # Disable caching when using gradient checkpointing
        if hasattr(model.config, 'use_cache'):
            model.config.use_cache = False
    
        # Wrap model with DDP if distributed
        if world_size > 1:
            model = DDP(model, device_ids=[local_rank], output_device=local_rank)
        
        # Prepare dataset
        train_dataset, val_dataset = prepare_dataset(tokenizer, args, logger)
        
        # Create trainer
        trainer = create_trainer(model, tokenizer, train_dataset, val_dataset, args, rank)
        
        # Start training
        logger.info("Starting training...")
        trainer.train()
        
        # Save model (only on rank 0)
        if rank == 0:
            logger.info("Saving final model...")
            trainer.save_model()
            tokenizer.save_pretrained(args.output_dir)
            
            # Save LoRA adapters separately
            lora_output_dir = os.path.join(args.output_dir, "lora_adapters")
            if hasattr(model, 'module'):
                model.module.save_pretrained(lora_output_dir)
            else:
                model.save_pretrained(lora_output_dir)
            
            logger.info(f"Model and LoRA adapters saved to {args.output_dir}")
        
        # Final evaluation
        logger.info("Running final evaluation...")
        eval_results = trainer.evaluate()
        
        if rank == 0:
            logger.info(f"Final evaluation results: {eval_results}")
            
            # Save evaluation results
            with open(os.path.join(args.output_dir, "eval_results.json"), "w") as f:
                json.dump(eval_results, f, indent=2)
    
    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}")
        raise
    
    finally:
        # Cleanup
        if world_size > 1:
            dist.destroy_process_group()
        
        if args.use_wandb and rank == 0:
            wandb.finish()


if __name__ == "__main__":
    main()
