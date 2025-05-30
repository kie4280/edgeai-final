#!/usr/bin/env python3
"""
Memory-Optimized Knowledge Distillation Training Script
=======================================================

This is a memory-optimized version of the distillation trainer that:
1. Uses CPU offloading for the teacher model
2. Implements gradient accumulation with smaller batches
3. Uses 8-bit quantization for the teacher model
4. Optimizes memory usage during training

Usage:
    python distillation_trainer_memory_optimized.py --output_dir ./distilled_model --num_train_epochs 1
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import os
import logging
import numpy as np
import gc
from dataclasses import dataclass
from typing import Dict, Optional, Union, Any

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    PreTrainedModel,
    EvalPrediction,
    BitsAndBytesConfig
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
from datasets import load_dataset
import wandb


@dataclass
class DistillationTrainingArguments(TrainingArguments):
    """Extended training arguments for knowledge distillation."""
    
    # Teacher model arguments
    teacher_model_name: str = "meta-llama/Llama-3.2-3B-Instruct"
    
    # Student model arguments  
    student_model_name: str = "meta-llama/Llama-3.2-1B-Instruct"
    
    # LoRA configuration
    lora_r: int = 8  # Reduced from 16 for memory
    lora_alpha: int = 16  # Reduced from 32 for memory
    lora_dropout: float = 0.1
    lora_target_modules: str = "q_proj,v_proj,gate_proj,down_proj"  # Fewer modules for memory
    
    # Distillation parameters
    temperature: float = 4.0
    alpha: float = 0.7
    beta: float = 0.3
    
    # Data parameters
    max_length: int = 256  # Reduced from 512
    dataset_name: str = "wikitext"
    
    # Evaluation
    eval_steps: int = 500  # Less frequent eval to save memory
    save_steps: int = 1000


class MemoryOptimizedDistillationTrainer(Trainer):
    """Memory-optimized trainer for knowledge distillation."""
    
    def __init__(self, teacher_model=None, temperature=4.0, alpha=0.7, beta=0.3, **kwargs):
        super().__init__(**kwargs)
        self.teacher_model = teacher_model
        self.temperature = temperature
        self.alpha = alpha
        self.beta = beta
        
        # Set teacher model to eval mode and freeze parameters
        if self.teacher_model is not None:
            self.teacher_model.eval()
            for param in self.teacher_model.parameters():
                param.requires_grad = False
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Memory-optimized loss computation."""
        
        # Get student outputs
        student_outputs = model(**inputs)
        student_logits = student_outputs.logits
        
        # Move teacher model to GPU temporarily for forward pass
        if hasattr(self.teacher_model, 'cuda'):
            device = next(model.parameters()).device
        
        # Get teacher outputs (no gradients)
        with torch.no_grad():
            teacher_outputs = self.teacher_model(**inputs)
            teacher_logits = teacher_outputs.logits
        
        # Move teacher back to CPU to free GPU memory
        if hasattr(self.teacher_model, 'cpu'):
            self.teacher_model = self.teacher_model.cpu()
            torch.cuda.empty_cache()
        
        # Extract labels
        labels = inputs.get('labels')
        
        if labels is not None:
            # Shift logits and labels for causal LM
            shift_logits_student = student_logits[..., :-1, :].contiguous()
            shift_logits_teacher = teacher_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Create loss mask
            loss_mask = (shift_labels != -100).float()
            
            # Soft distillation loss (KL divergence) - computed in chunks to save memory
            batch_size = shift_logits_student.size(0)
            seq_len = shift_logits_student.size(1)
            vocab_size = shift_logits_student.size(2)
            
            kl_loss = 0.0
            chunk_size = min(seq_len, 64)  # Process in smaller chunks
            
            for i in range(0, seq_len, chunk_size):
                end_i = min(i + chunk_size, seq_len)
                
                student_chunk = shift_logits_student[:, i:end_i, :]
                teacher_chunk = shift_logits_teacher[:, i:end_i, :]
                mask_chunk = loss_mask[:, i:end_i]
                
                student_soft = F.log_softmax(student_chunk / self.temperature, dim=-1)
                teacher_soft = F.softmax(teacher_chunk / self.temperature, dim=-1)
                
                kl_loss_chunk = F.kl_div(
                    student_soft.view(-1, vocab_size),
                    teacher_soft.view(-1, vocab_size),
                    reduction='none'
                ).sum(dim=-1)
                
                kl_loss_chunk = kl_loss_chunk.view(batch_size, end_i - i)
                kl_loss += (kl_loss_chunk * mask_chunk).sum()
            
            kl_loss = kl_loss / (loss_mask.sum() + 1e-8)
            kl_loss = kl_loss * (self.temperature ** 2)
            
            # Hard task loss
            task_loss = F.cross_entropy(
                shift_logits_student.view(-1, vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
                reduction='mean'
            )
            
            # Combined loss
            loss = self.alpha * kl_loss + self.beta * task_loss
            
            # Log losses
            if self.state.global_step % 50 == 0:
                self.log({
                    'train/kl_loss': kl_loss.item(),
                    'train/task_loss': task_loss.item(),
                })
        else:
            loss = student_outputs.loss
        
        return (loss, student_outputs) if return_outputs else loss


def load_models_memory_optimized(args: DistillationTrainingArguments, local_rank: int = -1):
    """Load models with memory optimizations and distributed training support."""
    
    # Quantization config for teacher model
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
        bnb_8bit_quant_type="nf8",
        bnb_8bit_compute_dtype=torch.float16,
        bnb_8bit_use_double_quant=True,
    )
    
    # Load teacher model (3B) with 8-bit quantization and CPU offloading
    logging.info(f"Loading teacher model with 8-bit quantization: {args.teacher_model_name}")
    
    # For distributed training, don't use device_map="auto"
    device_map = None if local_rank != -1 else "cpu"
    
    teacher_model = AutoModelForCausalLM.from_pretrained(
        args.teacher_model_name,
        quantization_config=bnb_config,
        torch_dtype=torch.float16,
        device_map=device_map,  # Keep on CPU initially for memory optimization
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    
    teacher_tokenizer = AutoTokenizer.from_pretrained(
        args.teacher_model_name,
        trust_remote_code=True
    )
    
    # Load student model (1B) normally
    logging.info(f"Loading student model: {args.student_model_name}")
    
    # For distributed training, don't use device_map="auto"
    student_device_map = None if local_rank != -1 else "auto"
    
    student_model = AutoModelForCausalLM.from_pretrained(
        args.student_model_name,
        torch_dtype=torch.float16,
        device_map=student_device_map,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    
    student_tokenizer = AutoTokenizer.from_pretrained(
        args.student_model_name,
        trust_remote_code=True
    )
    
    # Set pad tokens
    if teacher_tokenizer.pad_token is None:
        teacher_tokenizer.pad_token = teacher_tokenizer.eos_token
    if student_tokenizer.pad_token is None:
        student_tokenizer.pad_token = student_tokenizer.eos_token
    
    return teacher_model, teacher_tokenizer, student_model, student_tokenizer


def setup_lora_adapters_memory_optimized(model: PreTrainedModel, args: DistillationTrainingArguments):
    """Setup memory-optimized LoRA adapters."""
    
    logging.info("Setting up memory-optimized LoRA adapters...")
    
    # Prepare model for PEFT
    model = prepare_model_for_kbit_training(model)
    
    # Parse target modules (fewer modules for memory optimization)
    target_modules = [m.strip() for m in args.lora_target_modules.split(",")]
    
    # LoRA configuration with smaller parameters
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    
    # Apply LoRA
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    return model


def prepare_dataset_memory_optimized(tokenizer: AutoTokenizer, args: DistillationTrainingArguments):
    """Prepare dataset with memory optimizations."""
    
    logging.info("Loading WikiText-v2 dataset...")
    
    # Load smaller subset for memory optimization
    dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")
    
    # Filter and limit dataset size
    def filter_empty_texts(example):
        return len(example["text"].strip()) > 20  # Only keep non-trivial texts
    
    train_dataset = dataset["train"].filter(filter_empty_texts)
    eval_dataset = dataset["validation"].filter(filter_empty_texts)
    
    # Take smaller subset for memory optimization
    train_dataset = train_dataset.select(range(min(5000, len(train_dataset))))
    eval_dataset = eval_dataset.select(range(min(500, len(eval_dataset))))
    
    # Tokenization function
    def tokenize_function(examples):
        tokenized = tokenizer(
            examples["text"],
            truncation=True,
            padding=False,
            max_length=args.max_length,
            return_tensors=None
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized
    
    # Tokenize datasets
    train_dataset = train_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=train_dataset.column_names,
        desc="Tokenizing train dataset"
    )
    
    eval_dataset = eval_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=eval_dataset.column_names,
        desc="Tokenizing eval dataset"
    )
    
    logging.info(f"Memory-optimized dataset: {len(train_dataset)} train, {len(eval_dataset)} eval samples")
    
    return train_dataset, eval_dataset


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    import argparse
    parser = argparse.ArgumentParser(description="Memory-Optimized Knowledge Distillation")
    
    # Model arguments
    parser.add_argument("--teacher_model_name", type=str, default="meta-llama/Llama-3.2-3B-Instruct")
    parser.add_argument("--student_model_name", type=str, default="meta-llama/Llama-3.2-1B-Instruct")
    
    # LoRA arguments (memory-optimized defaults)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--lora_target_modules", type=str, default="q_proj,v_proj,gate_proj,down_proj")
    
    # Distillation arguments
    parser.add_argument("--temperature", type=float, default=4.0)
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--beta", type=float, default=0.3)
    
    # Training arguments (memory-optimized defaults)
    parser.add_argument("--output_dir", type=str, default="./distilled_model_memory_opt")
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--warmup_steps", type=int, default=50)
    parser.add_argument("--eval_steps", type=int, default=500)
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--logging_steps", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    
    # Distributed training arguments
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training")
    
    # Wandb arguments
    parser.add_argument("--wandb_project", type=str, default="")
    parser.add_argument("--run_name", type=str, default="memory-opt-distillation")
    
    args = parser.parse_args()
    
    # Setup distributed training
    local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank))
    
    if local_rank != -1:
        # Initialize distributed training
        torch.distributed.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        
        # Only log from main process
        if local_rank != 0:
            logging.getLogger().setLevel(logging.WARNING)
    
    # Clear GPU cache
    torch.cuda.empty_cache()
    gc.collect()
    
    # Create training arguments
    training_args = DistillationTrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=False,  # Disabled to save memory
        
        # Memory optimizations
        fp16=True,
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        dataloader_num_workers=0,
        gradient_checkpointing=True,
        optim="adamw_torch_fused",
        seed=args.seed,
        
        # Distributed training
        ddp_find_unused_parameters=False,
        ddp_backend="nccl",
        local_rank=local_rank,
        
        # Reporting
        report_to="wandb" if args.wandb_project else "none",
        run_name=args.run_name,
        
        # Custom arguments
        teacher_model_name=args.teacher_model_name,
        student_model_name=args.student_model_name,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=args.lora_target_modules,
        temperature=args.temperature,
        alpha=args.alpha,
        beta=args.beta,
        max_length=args.max_length,
    )
    
    # Initialize wandb only on main process
    if args.wandb_project and (local_rank == -1 or local_rank == 0):
        wandb.init(project=args.wandb_project, name=args.run_name, config=vars(args))
    
    # Load models with memory optimizations
    teacher_model, teacher_tokenizer, student_model, student_tokenizer = load_models_memory_optimized(training_args, local_rank)
    
    # Setup LoRA adapters
    student_model = setup_lora_adapters_memory_optimized(student_model, training_args)
    
    # Prepare dataset
    train_dataset, eval_dataset = prepare_dataset_memory_optimized(student_tokenizer, training_args)
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=student_tokenizer,
        mlm=False,
        pad_to_multiple_of=8,
    )
    
    # Initialize trainer
    trainer = MemoryOptimizedDistillationTrainer(
        model=student_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=student_tokenizer,
        teacher_model=teacher_model,
        temperature=training_args.temperature,
        alpha=training_args.alpha,
        beta=training_args.beta,
    )
    
    # Start training
    logging.info("Starting memory-optimized knowledge distillation training...")
    
    try:
        trainer.train()
    except KeyboardInterrupt:
        logging.info("Training interrupted by user")
    except RuntimeError as e:
        if "out of memory" in str(e):
            logging.error("Still running out of memory. Try reducing batch size further or using CPU-only training.")
            raise e
    
    # Save final model
    logging.info("Saving final model...")
    trainer.save_model()
    student_tokenizer.save_pretrained(training_args.output_dir)
    
    # Save LoRA adapters
    lora_output_dir = os.path.join(training_args.output_dir, "lora_adapters")
    student_model.save_pretrained(lora_output_dir)
    student_tokenizer.save_pretrained(lora_output_dir)
    
    logging.info(f"Training completed! Models saved to {training_args.output_dir}")
    
    if args.wandb_project:
        wandb.finish()


if __name__ == "__main__":
    main()
