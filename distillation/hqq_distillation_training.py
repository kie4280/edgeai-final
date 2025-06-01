#!/usr/bin/env python3
"""
HQQ Quantization + PEFT Training Script for Knowledge Distillation
================================================================

This script:
1. Quantizes a LLaMA-3.2-1B-Instruct model using HQQ
2. Sets up KV cache for efficient inference
3. Performs PEFT training using LoRA adapters
4. Distills knowledge from LLaMA-3.2-3B-Instruct to the quantized 1B model

Usage:
    python hqq_distillation_training.py --epochs 3 --batch_size 4 --lr 1e-4
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    StaticCache,
    get_linear_schedule_with_warmup
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
from tqdm.auto import tqdm
import random
import numpy as np
import argparse
import logging
import os
from typing import Dict, List, Tuple, Optional
import wandb

# HQQ imports
from hqq.utils.patching import prepare_for_inference, recommended_inductor_config_setter
from hqq.models.hf.base import AutoHQQHFModel
from hqq.models.base import HQQLinear
from hqq.core.quantize import BaseQuantizeConfig, HQQBackend


class DistillationTrainer:
  """Knowledge Distillation trainer for HQQ quantized student and teacher models"""

  def __init__(
      self,
      student_model,
      teacher_model,
      student_tokenizer,
      teacher_tokenizer,
      args,
      device='cuda:0'
  ):
    self.student_model = student_model
    self.teacher_model = teacher_model
    self.student_tokenizer = student_tokenizer
    self.teacher_tokenizer = teacher_tokenizer
    self.args = args
    self.device = device

    # Setup teacher model for evaluation mode
    self.teacher_model.eval()
    for param in self.teacher_model.parameters():
      param.requires_grad = False

    # Setup distillation parameters
    self.temperature = args.temperature
    self.alpha = args.alpha  # weight for distillation loss
    self.beta = args.beta    # weight for task loss

    # Setup optimizer for student model
    self.setup_optimizer()

    logging.info(
        f"Distillation trainer initialized with T={
            self.temperature}, alpha={
            self.alpha}, beta={
            self.beta}"
    )

  def setup_optimizer(self):
    """Setup optimizer and scheduler for student model"""
    # Only optimize LoRA parameters
    trainable_params = [
        p for p in self.student_model.parameters() if p.requires_grad]

    self.optimizer = torch.optim.AdamW(
        trainable_params,
        lr=self.args.learning_rate,
        weight_decay=self.args.weight_decay,
        betas=(0.9, 0.999)
    )

    # Calculate total steps for scheduler
    total_steps = self.args.num_epochs * self.args.steps_per_epoch
    warmup_steps = int(0.1 * total_steps)

    self.scheduler = get_linear_schedule_with_warmup(
        self.optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    logging.info(
        f"Optimizer setup: {
            len(trainable_params)} trainable parameters"
    )

  def compute_distillation_loss(
      self,
      student_logits: torch.Tensor,
      teacher_logits: torch.Tensor,
      labels: torch.Tensor,
      attention_mask: torch.Tensor
  ) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute knowledge distillation loss combining:
    1. Soft distillation loss (KL divergence between teacher and student)
    2. Hard task loss (CrossEntropy with ground truth labels)
    """
    # For causal LM, we shift the logits and labels
    shift_logits_student = student_logits[..., :-1, :].contiguous()
    shift_logits_teacher = teacher_logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    shift_attention_mask = attention_mask[..., :-1].contiguous()
    
    # Create loss mask (ignore padding tokens)
    loss_mask = shift_attention_mask.float()
    
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
    kl_loss = (kl_loss_per_token * loss_mask).sum() / loss_mask.sum()
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
    task_loss = (task_loss_per_token * valid_tokens).sum() / valid_tokens.sum()

    # Combined loss
    total_loss = self.alpha * kl_loss + self.beta * task_loss

    loss_dict = {
        'total_loss': total_loss.item(),
        'kl_loss': kl_loss.item(),
        'task_loss': task_loss.item()
    }

    return total_loss, loss_dict

  def forward_student(
          self,
          input_ids: torch.Tensor,
          attention_mask: torch.Tensor) -> torch.Tensor:
    """Forward pass through student model"""
    outputs = self.student_model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,  # Disable cache for training to avoid gradient checkpointing conflicts
        return_dict=True
    )

    return outputs.logits

  @torch.no_grad()
  def forward_teacher(
          self,
          input_ids: torch.Tensor,
          attention_mask: torch.Tensor) -> torch.Tensor:
    """Forward pass through teacher model"""
    outputs = self.teacher_model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,  # Disable cache for consistency
        return_dict=True
    )

    return outputs.logits

  def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
    """Single training step"""
    self.student_model.train()

    input_ids = batch['input_ids'].to(self.device)
    attention_mask = batch['attention_mask'].to(self.device)
    labels = batch['labels'].to(self.device)

    # Forward passes
    student_logits = self.forward_student(input_ids, attention_mask)
    teacher_logits = self.forward_teacher(input_ids, attention_mask)

    # Compute distillation loss
    loss, loss_dict = self.compute_distillation_loss(
        student_logits, teacher_logits, labels, attention_mask
    )

    # Backward pass
    self.optimizer.zero_grad()
    loss.backward()

    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(
        [p for p in self.student_model.parameters() if p.requires_grad],
        max_norm=1.0
    )

    self.optimizer.step()
    self.scheduler.step()

    return loss_dict

  def evaluate(self, eval_dataloader: DataLoader) -> Dict[str, float]:
    """Evaluate the student model"""
    self.student_model.eval()

    total_loss = 0.0
    total_kl_loss = 0.0
    total_task_loss = 0.0
    num_batches = 0

    with torch.no_grad():
      for batch in eval_dataloader:
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        labels = batch['labels'].to(self.device)

        student_logits = self.forward_student(input_ids, attention_mask)
        teacher_logits = self.forward_teacher(input_ids, attention_mask)

        _, loss_dict = self.compute_distillation_loss(
            student_logits, teacher_logits, labels, attention_mask
        )

        total_loss += loss_dict['total_loss']
        total_kl_loss += loss_dict['kl_loss']
        total_task_loss += loss_dict['task_loss']
        num_batches += 1

    return {
        'eval_loss': total_loss / num_batches,
        'eval_kl_loss': total_kl_loss / num_batches,
        'eval_task_loss': total_task_loss / num_batches
    }


def get_quantization_config() -> BaseQuantizeConfig:
  """Get HQQ quantization configuration for 1B model"""
  # Aggressive quantization for the smaller student model
  q3_config = BaseQuantizeConfig(nbits=3, group_size=128)
  q4_config = BaseQuantizeConfig(nbits=4, group_size=64)

  quant_config = {
      'self_attn.q_proj': q4_config,
      'self_attn.k_proj': q4_config,
      'self_attn.v_proj': q4_config,
      'self_attn.o_proj': q4_config,
      'mlp.gate_proj': q3_config,
      'mlp.up_proj': q3_config,
      'mlp.down_proj': q3_config,
  }

  return quant_config


def load_and_quantize_student_model(
        model_name: str, device: str) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
  """Load and quantize the student model (1B) using HQQ"""
  logging.info(f"Loading student model: {model_name}")

  # Load model and tokenizer
  model = AutoModelForCausalLM.from_pretrained(
      model_name,
      torch_dtype=torch.float16,
      device_map=device,
      trust_remote_code=True
  )

  tokenizer = AutoTokenizer.from_pretrained(
      model_name,
      trust_remote_code=True
  )

  if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

  # Apply HQQ quantization
  quant_config = get_quantization_config()

  logging.info("Applying HQQ quantization...")
  AutoHQQHFModel.quantize_model(
      model,
      quant_config=quant_config,
      compute_dtype=torch.float16,
      device=device
  )

  # Prepare for inference
  prepare_for_inference(model, backend="pytorch")

  logging.info("Student model quantized successfully")
  return model, tokenizer


def load_teacher_model(model_name: str,
                       device: str) -> Tuple[AutoModelForCausalLM,
                                             AutoTokenizer]:
  """Load the teacher model (3B) without quantization"""
  logging.info(f"Loading teacher model: {model_name}")

  model = AutoModelForCausalLM.from_pretrained(
      model_name,
      torch_dtype=torch.float16,
      device_map=device,
      trust_remote_code=True
  )

  tokenizer = AutoTokenizer.from_pretrained(
      model_name,
      trust_remote_code=True
  )

  if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

  logging.info("Teacher model loaded successfully")
  return model, tokenizer


def setup_lora_adapters(
        model: AutoModelForCausalLM,
        args) -> AutoModelForCausalLM:
  """Setup LoRA adapters for the quantized student model"""
  logging.info("Setting up LoRA adapters...")

  # Prepare model for k-bit training (works with HQQ as well)
  model = prepare_model_for_kbit_training(model)

  # LoRA configuration
  lora_config = LoraConfig(
      r=args.lora_r,
      lora_alpha=args.lora_alpha,
      target_modules=[
          "q_proj", "k_proj", "v_proj", "o_proj",
          "gate_proj", "up_proj", "down_proj"
      ],
      lora_dropout=args.lora_dropout,
      bias="none",
      task_type="CAUSAL_LM",
  )

  # Apply LoRA
  model = get_peft_model(model, lora_config)
  model.print_trainable_parameters()

  logging.info("LoRA adapters configured successfully")
  return model


def prepare_dataset(tokenizer: AutoTokenizer,
                    args) -> Tuple[DataLoader, DataLoader]:
  """Prepare training and validation datasets"""
  logging.info("Loading and preparing dataset...")

  # Load dataset
  if args.dataset_name == "wikitext":
    dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")
    train_data = dataset["train"]
    eval_data = dataset["validation"]
    text_column = "text"
  elif args.dataset_name == "alpaca":
    dataset = load_dataset("yahma/alpaca-cleaned")
    train_data = dataset["train"]
    eval_data = train_data.train_test_split(test_size=0.1, seed=42)["test"]
    train_data = train_data.train_test_split(test_size=0.1, seed=42)["train"]
    text_column = "text"
  else:
    raise ValueError(f"Unsupported dataset: {args.dataset_name}")

  def tokenize_function(examples):
    # For instruction datasets, combine instruction and output
    if args.dataset_name == "alpaca":
      texts = []
      for inst, inp, out in zip(
              examples["instruction"], examples["input"], examples["output"]):
        if inp.strip():
          text = f"### Instruction:\n{inst}\n\n### Input:\n{inp}\n\n### Response:\n{out}"
        else:
          text = f"### Instruction:\n{inst}\n\n### Response:\n{out}"
        texts.append(text)
    else:
      texts = examples[text_column]

    # Tokenize
    tokenized = tokenizer(
        texts,
        truncation=True,
        padding=False,
        max_length=args.max_length,
        return_tensors=None
    )

    # For causal LM, labels are the same as input_ids
    tokenized["labels"] = tokenized["input_ids"].copy()

    return tokenized

  # Tokenize datasets
  train_dataset = train_data.map(
      tokenize_function,
      batched=True,
      remove_columns=train_data.column_names
  )

  eval_dataset = eval_data.map(
      tokenize_function,
      batched=True,
      remove_columns=eval_data.column_names
  )

  # Data collator with padding
  def collate_fn(batch):
    # Pad sequences
    input_ids = [torch.tensor(item["input_ids"]) for item in batch]
    labels = [torch.tensor(item["labels"]) for item in batch]

    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids, batch_first=True, padding_value=tokenizer.pad_token_id
    )
    labels = torch.nn.utils.rnn.pad_sequence(
        labels, batch_first=True, padding_value=-100
    )
    attention_mask = (input_ids != tokenizer.pad_token_id).long()

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

  # Create dataloaders
  train_dataloader = DataLoader(
      train_dataset,
      batch_size=args.batch_size,
      shuffle=True,
      collate_fn=collate_fn,
      num_workers=0,  # Reduce to 0 to avoid multiprocessing issues
      pin_memory=False  # Disable pin_memory to avoid CUDA context issues
  )

  eval_dataloader = DataLoader(
      eval_dataset,
      batch_size=args.batch_size,
      shuffle=False,
      collate_fn=collate_fn,
      num_workers=0,  # Reduce to 0 to avoid multiprocessing issues
      pin_memory=False  # Disable pin_memory to avoid CUDA context issues
  )

  logging.info(
      f"Dataset prepared: {
          len(train_dataset)} train, {
          len(eval_dataset)} eval")
  return train_dataloader, eval_dataloader


def main():
  # Set environment variables to avoid warnings
  os.environ["TOKENIZERS_PARALLELISM"] = "false"
  
  parser = argparse.ArgumentParser(
      description="HQQ + PEFT Knowledge Distillation Training")

  # Model arguments
  parser.add_argument(
      "--student_model",
      type=str,
      default="meta-llama/Llama-3.2-1B-Instruct")
  parser.add_argument(
      "--teacher_model",
      type=str,
      default="meta-llama/Llama-3.2-3B-Instruct")
  parser.add_argument("--device", type=str, default="cuda:0")

  # Training arguments
  parser.add_argument("--num_epochs", type=int, default=3)
  parser.add_argument("--batch_size", type=int, default=4)
  parser.add_argument("--learning_rate", type=float, default=1e-4)
  parser.add_argument("--weight_decay", type=float, default=0.01)
  parser.add_argument("--max_length", type=int, default=512)
  parser.add_argument("--steps_per_epoch", type=int, default=1000)

  # LoRA arguments
  parser.add_argument("--lora_r", type=int, default=16)
  parser.add_argument("--lora_alpha", type=int, default=32)
  parser.add_argument("--lora_dropout", type=float, default=0.05)

  # Distillation arguments
  parser.add_argument("--temperature", type=float, default=4.0)
  parser.add_argument("--alpha", type=float, default=0.7,
                      help="Weight for distillation loss")
  parser.add_argument(
      "--beta",
      type=float,
      default=0.3,
      help="Weight for task loss")

  # Dataset arguments
  parser.add_argument(
      "--dataset_name",
      type=str,
      default="wikitext",
      choices=[
          "wikitext",
          "alpaca"])

  # Logging and saving
  parser.add_argument("--output_dir", type=str, default="./hqq_distilled_model")
  parser.add_argument("--wandb_project", type=str, default="hqq-distillation")
  parser.add_argument("--save_steps", type=int, default=500)
  parser.add_argument("--eval_steps", type=int, default=250)
  parser.add_argument("--seed", type=int, default=42)

  args = parser.parse_args()

  # Setup logging
  logging.basicConfig(
      level=logging.INFO,
      format='%(asctime)s - %(levelname)s - %(message)s'
  )

  # Set seeds
  torch.manual_seed(args.seed)
  random.seed(args.seed)
  np.random.seed(args.seed)

  # Setup HQQ
  recommended_inductor_config_setter()
  HQQLinear.set_backend(HQQBackend.PYTORCH_COMPILE)

  # Initialize wandb
  if args.wandb_project:
    wandb.init(
        project=args.wandb_project,
        config=vars(args),
    )

  # Load models
  logging.info("Loading and setting up models...")
  student_model, student_tokenizer = load_and_quantize_student_model(
      args.student_model, args.device)
  teacher_model, teacher_tokenizer = load_teacher_model(
      args.teacher_model, args.device)

  # Setup LoRA adapters for student
  student_model = setup_lora_adapters(student_model, args)

  # Prepare dataset
  train_dataloader, eval_dataloader = prepare_dataset(student_tokenizer, args)
  args.steps_per_epoch = min(args.steps_per_epoch, len(train_dataloader))

  # Initialize trainer
  trainer = DistillationTrainer(
      student_model=student_model,
      teacher_model=teacher_model,
      student_tokenizer=student_tokenizer,
      teacher_tokenizer=teacher_tokenizer,
      args=args,
      device=args.device
  )

  # Training loop
  logging.info("Starting distillation training...")
  global_step = 0
  best_eval_loss = float('inf')

  for epoch in range(args.num_epochs):
    epoch_losses = []

    # Training
    progress_bar = tqdm(train_dataloader,
                        desc=f"Epoch {epoch + 1}/{args.num_epochs}")
    for step, batch in enumerate(progress_bar):
      if step >= args.steps_per_epoch:
        break

      loss_dict = trainer.train_step(batch)
      epoch_losses.append(loss_dict['total_loss'])

      # Update progress bar
      progress_bar.set_postfix({
          'loss': f"{loss_dict['total_loss']:.4f}",
          'kl': f"{loss_dict['kl_loss']:.4f}",
          'task': f"{loss_dict['task_loss']:.4f}"
      })

      # Log to wandb
      if args.wandb_project:
        wandb.log({
            'train/loss': loss_dict['total_loss'],
            'train/kl_loss': loss_dict['kl_loss'],
            'train/task_loss': loss_dict['task_loss'],
            'train/learning_rate': trainer.scheduler.get_last_lr()[0],
            'global_step': global_step
        })

      # Evaluation
      if (global_step + 1) % args.eval_steps == 0:
        eval_metrics = trainer.evaluate(eval_dataloader)
        logging.info(f"Step {global_step +
                             1} - Eval Loss: {eval_metrics['eval_loss']:.4f}")

        if args.wandb_project:
          wandb.log({
              'eval/loss': eval_metrics['eval_loss'],
              'eval/kl_loss': eval_metrics['eval_kl_loss'],
              'eval/task_loss': eval_metrics['eval_task_loss'],
              'global_step': global_step
          })

        # Save best model
        if eval_metrics['eval_loss'] < best_eval_loss:
          best_eval_loss = eval_metrics['eval_loss']
          best_model_path = os.path.join(args.output_dir, "best_model")
          os.makedirs(best_model_path, exist_ok=True)
          student_model.save_pretrained(best_model_path)
          student_tokenizer.save_pretrained(best_model_path)
          logging.info(f"Saved best model with eval loss: {best_eval_loss:.4f}")

      # Save checkpoint
      if (global_step + 1) % args.save_steps == 0:
        checkpoint_path = os.path.join(
            args.output_dir, f"checkpoint-{global_step + 1}")
        os.makedirs(checkpoint_path, exist_ok=True)
        student_model.save_pretrained(checkpoint_path)
        student_tokenizer.save_pretrained(checkpoint_path)
        logging.info(f"Saved checkpoint at step {global_step + 1}")

      global_step += 1

    # End of epoch evaluation
    eval_metrics = trainer.evaluate(eval_dataloader)
    avg_train_loss = np.mean(epoch_losses)

    logging.info(f"Epoch {epoch + 1} completed:")
    logging.info(f"  Average train loss: {avg_train_loss:.4f}")
    logging.info(f"  Eval loss: {eval_metrics['eval_loss']:.4f}")

  # Save final model
  final_model_path = os.path.join(args.output_dir, "final_model")
  os.makedirs(final_model_path, exist_ok=True)
  student_model.save_pretrained(final_model_path)
  student_tokenizer.save_pretrained(final_model_path)

  logging.info("Training completed!")
  logging.info(f"Models saved to: {args.output_dir}")

  if args.wandb_project:
    wandb.finish()


if __name__ == "__main__":
  main()
