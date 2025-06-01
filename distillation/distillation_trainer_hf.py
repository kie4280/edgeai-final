#!/usr/bin/env python3
"""
Knowledge Distillation Training Script using Hugging Face Trainer
================================================================

This script distills a Llama-3.2-3B-Instruct model into a 1B model using:
1. Hugging Face Trainer API
2. LoRA adapters with PEFT
3. WikiText-v2 dataset
4. Knowledge distillation with temperature scaling

Usage:
    python distillation_trainer_hf.py --output_dir ./distilled_model --num_train_epochs 3
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import os
import logging
import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional, Union, Any

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    PreTrainedModel,
    EvalPrediction
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
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: str = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
    
    # Distillation parameters
    temperature: float = 4.0
    alpha: float = 0.7  # Weight for distillation loss
    beta: float = 0.3   # Weight for task loss
    
    # Data parameters
    max_length: int = 512
    dataset_name: str = "wikitext"
    
    # Evaluation
    eval_steps: int = 250
    save_steps: int = 500


class DistillationTrainer(Trainer):
    """Custom Trainer for knowledge distillation."""
    
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
            
            # Move teacher to same device as student
            if hasattr(self.model, 'device'):
                self.teacher_model = self.teacher_model.to(self.model.device)
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Compute knowledge distillation loss combining:
        1. Soft distillation loss (KL divergence between teacher and student)
        2. Hard task loss (standard language modeling loss)
        """
        # Get student outputs
        student_outputs = model(**inputs)
        student_logits = student_outputs.logits
        
        # Get teacher outputs (no gradients)
        with torch.no_grad():
            teacher_outputs = self.teacher_model(**inputs)
            teacher_logits = teacher_outputs.logits
        
        # Extract labels and attention mask
        labels = inputs.get('labels')
        attention_mask = inputs.get('attention_mask')
        
        if labels is not None:
            # Shift logits and labels for causal LM
            shift_logits_student = student_logits[..., :-1, :].contiguous()
            shift_logits_teacher = teacher_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            shift_attention_mask = attention_mask[..., :-1].contiguous() if attention_mask is not None else None
            
            # Create loss mask (ignore padding tokens and -100 labels)
            if shift_attention_mask is not None:
                loss_mask = shift_attention_mask.float()
            else:
                loss_mask = (shift_labels != -100).float()
            
            # Soft distillation loss (KL divergence)
            student_soft = F.log_softmax(shift_logits_student / self.temperature, dim=-1)
            teacher_soft = F.softmax(shift_logits_teacher / self.temperature, dim=-1)
            
            # Compute KL divergence per token
            kl_loss_per_token = F.kl_div(
                student_soft.view(-1, student_soft.size(-1)),
                teacher_soft.view(-1, teacher_soft.size(-1)),
                reduction='none'
            ).sum(dim=-1)
            
            # Reshape and apply mask
            kl_loss_per_token = kl_loss_per_token.view(shift_logits_student.size(0), shift_logits_student.size(1))
            kl_loss = (kl_loss_per_token * loss_mask).sum() / (loss_mask.sum() + 1e-8)
            kl_loss = kl_loss * (self.temperature ** 2)
            
            # Hard task loss (standard language modeling loss)
            task_loss_per_token = F.cross_entropy(
                shift_logits_student.view(-1, shift_logits_student.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
                reduction='none'
            )
            
            # Reshape and apply mask (only count non-ignored tokens)
            task_loss_per_token = task_loss_per_token.view(shift_labels.size())
            valid_tokens = (shift_labels != -100).float()
            task_loss = (task_loss_per_token * valid_tokens).sum() / (valid_tokens.sum() + 1e-8)
            
            # Combined loss
            loss = self.alpha * kl_loss + self.beta * task_loss
            
            # Log individual losses
            if self.state.global_step % 10 == 0:  # Log every 10 steps
                self.log({
                    'train/kl_loss': kl_loss.item(),
                    'train/task_loss': task_loss.item(),
                    'train/alpha': self.alpha,
                    'train/beta': self.beta,
                })
        else:
            # If no labels, just return student loss
            loss = student_outputs.loss
        
        return (loss, student_outputs) if return_outputs else loss


def load_models(args: DistillationTrainingArguments, local_rank: int = -1):
    """Load teacher and student models with appropriate configurations."""
    
    # For distributed training, don't use device_map="auto"
    device_map = None if local_rank != -1 else "auto"
    
    # Load teacher model (3B) with quantization to save memory
    logging.info(f"Loading teacher model: {args.teacher_model_name}")
    teacher_model = AutoModelForCausalLM.from_pretrained(
        args.teacher_model_name,
        torch_dtype=torch.float16,
        device_map=device_map,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        # Use 8-bit quantization for teacher to save memory
        load_in_8bit=True if device_map == "auto" else False,
    )
    
    teacher_tokenizer = AutoTokenizer.from_pretrained(
        args.teacher_model_name,
        trust_remote_code=True
    )
    
    # Load student model (1B)
    logging.info(f"Loading student model: {args.student_model_name}")
    student_model = AutoModelForCausalLM.from_pretrained(
        args.student_model_name,
        torch_dtype=torch.float16,
        device_map=device_map,
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
    
    # Ensure both tokenizers have the same vocabulary
    assert teacher_tokenizer.vocab_size == student_tokenizer.vocab_size, \
        "Teacher and student tokenizers must have the same vocabulary size"
    
    return teacher_model, teacher_tokenizer, student_model, student_tokenizer


def setup_lora_adapters(model: PreTrainedModel, args: DistillationTrainingArguments):
    """Setup LoRA adapters for the student model."""
    
    logging.info("Setting up LoRA adapters for student model...")
    
    # Prepare model for PEFT
    model = prepare_model_for_kbit_training(model)
    
    # Parse target modules
    target_modules = [m.strip() for m in args.lora_target_modules.split(",")]
    
    # LoRA configuration
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
    
    logging.info("LoRA adapters configured successfully")
    return model


def prepare_dataset(tokenizer: AutoTokenizer, args: DistillationTrainingArguments):
    """Prepare WikiText-v2 dataset for training."""
    
    logging.info("Loading WikiText-v2 dataset...")
    
    # Load WikiText-v2 dataset
    dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")
    
    # Filter out empty texts
    def filter_empty_texts(example):
        return len(example["text"].strip()) > 0
    
    train_dataset = dataset["train"].filter(filter_empty_texts)
    eval_dataset = dataset["validation"].filter(filter_empty_texts)
    
    # Tokenization function
    def tokenize_function(examples):
        # Tokenize texts
        tokenized = tokenizer(
            examples["text"],
            truncation=True,
            padding=False,
            max_length=args.max_length,
            return_tensors=None
        )
        
        # For causal LM, labels are the same as input_ids
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
    
    logging.info(f"Dataset prepared: {len(train_dataset)} train, {len(eval_dataset)} eval samples")
    
    return train_dataset, eval_dataset


def compute_metrics(eval_pred: EvalPrediction) -> Dict[str, float]:
    """Compute perplexity metric for evaluation."""
    predictions, labels = eval_pred
    
    # Flatten predictions and labels
    predictions = predictions.reshape(-1, predictions.shape[-1])
    labels = labels.reshape(-1)
    
    # Calculate cross entropy loss
    loss = F.cross_entropy(
        torch.tensor(predictions), 
        torch.tensor(labels), 
        ignore_index=-100,
        reduction='mean'
    )
    
    # Calculate perplexity
    perplexity = torch.exp(loss).item()
    
    return {"perplexity": perplexity}


def main():
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser(description="Knowledge Distillation with Hugging Face Trainer")
    
    # Model arguments
    parser.add_argument("--teacher_model_name", type=str, default="meta-llama/Llama-3.2-3B-Instruct")
    parser.add_argument("--student_model_name", type=str, default="meta-llama/Llama-3.2-1B-Instruct")
    
    # LoRA arguments
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_target_modules", type=str, default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
    
    # Distillation arguments
    parser.add_argument("--temperature", type=float, default=4.0)
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--beta", type=float, default=0.3)
    
    # Training arguments
    parser.add_argument("--output_dir", type=str, default="./distilled_model")
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)  # Reduced from 4
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)   # Reduced from 4
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16) # Increased to maintain effective batch size
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_length", type=int, default=256)  # Reduced from 512
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--eval_steps", type=int, default=250)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    
    # Wandb arguments
    parser.add_argument("--wandb_project", type=str, default="llama-distillation")
    parser.add_argument("--run_name", type=str, default="3b-to-1b-distillation")
    
    args = parser.parse_args()
    
    # Create training arguments
    training_args = DistillationTrainingArguments(
        # Basic training arguments
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        
        # Evaluation and logging
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        
        # Optimization
        fp16=True,
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        seed=args.seed,
        dataloader_num_workers=0,  # Reduce memory usage
        
        # Memory optimization
        gradient_checkpointing=True,  # Enable gradient checkpointing
        optim="adamw_torch_fused",    # Use fused optimizer for better memory efficiency
        
        # Distributed training (these will be automatically set by torchrun)
        ddp_find_unused_parameters=False,
        ddp_backend="nccl",
        
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
    
    # Get local rank for distributed training
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    
    # Initialize wandb only on main process
    if args.wandb_project and (local_rank == -1 or local_rank == 0):
        wandb.init(
            project=args.wandb_project,
            name=args.run_name,
            config=vars(args)
        )
    
    # Load models
    teacher_model, teacher_tokenizer, student_model, student_tokenizer = load_models(training_args, local_rank)
    
    # Setup LoRA adapters for student
    student_model = setup_lora_adapters(student_model, training_args)
    
    # Prepare dataset
    train_dataset, eval_dataset = prepare_dataset(student_tokenizer, training_args)
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=student_tokenizer,
        mlm=False,  # We're doing causal language modeling
        pad_to_multiple_of=8,
    )
    
    # Initialize trainer
    trainer = DistillationTrainer(
        model=student_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=student_tokenizer,
        compute_metrics=compute_metrics,
        teacher_model=teacher_model,
        temperature=training_args.temperature,
        alpha=training_args.alpha,
        beta=training_args.beta,
    )
    
    # Start training
    logging.info("Starting knowledge distillation training...")
    
    try:
        trainer.train()
    except KeyboardInterrupt:
        logging.info("Training interrupted by user")
    
    # Save final model
    logging.info("Saving final model...")
    trainer.save_model()
    student_tokenizer.save_pretrained(training_args.output_dir)
    
    # Save LoRA adapters separately
    lora_output_dir = os.path.join(training_args.output_dir, "lora_adapters")
    student_model.save_pretrained(lora_output_dir)
    student_tokenizer.save_pretrained(lora_output_dir)
    
    logging.info(f"Training completed! Models saved to {training_args.output_dir}")
    logging.info(f"LoRA adapters saved to {lora_output_dir}")
    
    # Final evaluation
    if eval_dataset:
        logging.info("Running final evaluation...")
        eval_results = trainer.evaluate()
        logging.info(f"Final evaluation results: {eval_results}")
    
    if args.wandb_project:
        wandb.finish()


if __name__ == "__main__":
    main()
