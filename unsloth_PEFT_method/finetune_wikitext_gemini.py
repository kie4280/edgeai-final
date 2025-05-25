import torch
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments

# --- Configuration ---
# You can change these parameters based on your GPU memory and desired model
MODEL_NAME = "unsloth/llama-3-8b-bnb-4bit"  # Example: Llama-3-8B in 4-bit
MAX_SEQ_LENGTH = 256  # Max sequence length for training
DTYPE = None  # None for auto, or torch.float16, torch.bfloat16
LOAD_IN_4BIT = True  # Enable 4-bit quantization for memory efficiency

# Training parameters
PER_DEVICE_TRAIN_BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 4
LEARNING_RATE = 2e-4
# Adjust as needed. For full fine-tuning, you might want more epochs/steps.
MAX_STEPS = 100
# For quick testing, 100-500 steps is good.
OUTPUT_DIR = "unsloth_wikitext_v2_model_gemini"

print("Loading Wikitext-v2 dataset...")
# Load the 'train' split of the raw Wikitext-2 dataset
train_dataset = load_dataset(
    "Salesforce/wikitext",
    "wikitext-2-raw-v1",
    split="train")
# Load the 'validation' split for evaluation
eval_dataset = load_dataset(
    "Salesforce/wikitext",
    "wikitext-2-raw-v1",
    split="validation")
print("Wikitext-v2 dataset loaded.")
print(f"Training dataset size: {len(train_dataset)} examples")
print(f"Validation dataset size: {len(eval_dataset)} examples")

# Example of a text entry from the dataset
# print("\nExample from training dataset:")
# Print first 500 characters of the first example
# print(train_dataset['text'][:5])


def filter_empty_texts(example):
  # Filter out empty or whitespace-only strings
  text = example.get("text")
  # Ensure text exists and is not just whitespace
  return text is not None and len(text.strip()) > 0


print("Filtering empty text entries from training dataset...")
train_dataset = train_dataset.filter(filter_empty_texts)
print("Filtering empty text entries from validation dataset...")
eval_dataset = eval_dataset.filter(filter_empty_texts)

# --- Step 1: Load the Model and Tokenizer ---
print(f"Loading model: {MODEL_NAME}...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=DTYPE,
    load_in_4bit=LOAD_IN_4BIT,
)
print("Model and tokenizer loaded.")

# --- Step 2: Add LoRA Adapters ---
print("Adding LoRA adapters...")
model = FastLanguageModel.get_peft_model(
    model,
    r=16,  # LoRA attention dimension
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha=16,  # Alpha parameter for LoRA
    lora_dropout=0,  # Dropout for LoRA layers
    bias="none",
    # Enables Unsloth's memory-saving gradient checkpointing
    use_gradient_checkpointing="unsloth",
    random_state=3407,
    use_rslora=False,
    loftq_config=None,
)
print("LoRA adapters added.")


# --- Step 3: Configure and Run Training ---
print("Setting up SFTTrainer...")
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,  # Pass the evaluation dataset
    dataset_text_field="text",  # The column in Wikitext-v2 containing the text
    max_seq_length=MAX_SEQ_LENGTH,
    args=TrainingArguments(
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        warmup_steps=5,  # Short warmup for initial stability
        max_steps=MAX_STEPS,
        learning_rate=LEARNING_RATE,
        fp16=not torch.cuda.is_bf16_supported(),  # Use FP16 if BF16 is not supported
        bf16=torch.cuda.is_bf16_supported(),
        # Use BF16 if supported (recommended for newer GPUs)
        logging_steps=10,  # Log training progress every 10 steps
        output_dir=OUTPUT_DIR,
        optim="adamw_8bit",  # Memory-efficient AdamW optimizer
        seed=3407,  # For reproducibility
        save_steps=100,  # Save checkpoint every 100 steps
        eval_strategy="steps",  # Evaluate every 'eval_steps'
        eval_steps=100,  # Evaluate every 100 steps
        logging_dir="./logs",  # Directory for TensorBoard logs
        report_to="tensorboard",  # Report metrics to TensorBoard
    ),
)
print("SFTTrainer configured. Starting training...")

# Start training
trainer_stats = trainer.train()
print("\nTraining complete!")

# --- Step 4: Save and/or Push the Model ---
print("\nSaving LoRA adapters...")
# Save only the LoRA adapters
model.save_pretrained(f"{OUTPUT_DIR}_lora_adapters")
print(f"LoRA adapters saved to {OUTPUT_DIR}_lora_adapters")

# Optional: Merge LoRA adapters and save the full model (for direct inference)
print("\nMerging LoRA adapters and saving the full model (for inference)...")
# save_method="merged_4bit" or "merged_16bit" or "merged_bf16"
model.save_pretrained_merged(
    f"{OUTPUT_DIR}_merged",
    tokenizer,
    save_method="merged_4bit")
print(f"Full merged model saved to {OUTPUT_DIR}_merged")

# Optional: Convert and save the model in GGUF format (for llama.cpp/Ollama)
# Requires 'pip install -U optimum' and 'pip install -U accelerate'
try:
  print("\nConverting and saving model to GGUF format...")
  model.save_pretrained_gguf(f"{OUTPUT_DIR}_gguf", tokenizer)
  print(f"GGUF model saved to {OUTPUT_DIR}_gguf")
except Exception as e:
  print(
      f"Could not save GGUF model. Make sure optimum and accelerate are installed. Error: {e}")


# Optional: Push the model and/or GGUF to Hugging Face Hub
# You need to be logged in to Hugging Face Hub (`huggingface-cli login`)
# try:
#     print("\nPushing LoRA adapters to Hugging Face Hub...")
#     model.push_to_hub(f"your_username/{OUTPUT_DIR}_lora_adapters", tokenizer=tokenizer)
#     print("LoRA adapters pushed to Hub.")
# except Exception as e:
#     print(f"Could not push LoRA adapters to Hugging Face Hub. Error: {e}")

# try:
#     print("\nPushing merged model to Hugging Face Hub...")
#     model.push_to_hub_merged(f"your_username/{OUTPUT_DIR}_merged", tokenizer, save_method="merged_4bit")
#     print("Merged model pushed to Hub.")
# except Exception as e:
#     print(f"Could not push merged model to Hugging Face Hub. Error: {e}")

# try:
#     print("\nPushing GGUF model to Hugging Face Hub...")
#     model.push_to_hub_gguf(f"your_username/{OUTPUT_DIR}_gguf", tokenizer)
#     print("GGUF model pushed to Hub.")
# except Exception as e:
#     print(f"Could not push GGUF model to Hugging Face Hub. Error: {e}")

print("\nFine-tuning process complete!")
