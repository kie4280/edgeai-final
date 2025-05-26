import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
from trl import SFTTrainer
import os # For checking environment variables

# --- Configuration ---
# Choose your base model. We'll use a Llama 3 model (Meta's official or a community version)
# as an example, as you previously mentioned a Llama 3.2 model.
# Ensure you have access if it's a gated model.
BASE_MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct-QLORA_INT4_EO8" # Replace with your desired base model
# If you want to use the specific model you mentioned, replace:
# BASE_MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct-QLORA_INT4_EO8"
# Note: If this is already a QLoRA INT4 model, some quantization might be redundant but safe.

MAX_SEQ_LENGTH = 2048 # Adjust based on your needs and GPU memory
# For Llama 3 models, 8192 is common. Reduce if OOM.

OUTPUT_DIR = "hf_wikitext_v2_fine_tuned_model"

# Training parameters
PER_DEVICE_TRAIN_BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 4
LEARNING_RATE = 2e-4
MAX_STEPS = 100 # For quick testing; increase for full fine-tuning (e.g., 500-1000+)
# Or use num_train_epochs = 1 (typically 1-3 epochs for full pre-training/fine-tuning)

# LoRA configuration
LORA_R = 16
LORA_ALPHA = 16
LORA_DROPOUT = 0.05 # A small dropout is common for LoRA

# --- 1. Load Base Model and Tokenizer with Quantization ---
print(f"Loading base model: {BASE_MODEL_NAME} with 4-bit quantization...")

# Define 4-bit quantization configuration
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4", # "nf4" (Normalized Float 4) or "fp4" (Float 4)
    bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    bnb_4bit_use_double_quant=True, # Use double quantization for extra memory savings
)

# Check for Hugging Face token if needed
if os.getenv("HF_TOKEN") is None:
    print("WARNING: HF_TOKEN environment variable not found. If using gated models, please login via `huggingface-cli login`.")

try:
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto", # Automatically maps model to available devices
        torch_dtype=bnb_config.bnb_4bit_compute_dtype, # Set model dtype to compute dtype
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    # Set padding token if not already set, crucial for batching
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right" # Important for causal LMs

    print("Base model and tokenizer loaded with 4-bit quantization.")

except Exception as e:
    print(f"Error loading model {BASE_MODEL_NAME}: {e}")
    print("Please ensure the model exists, you have access (if gated), and `bitsandbytes` is correctly installed for your CUDA version.")
    exit() # Exit if model loading fails

# --- 2. Configure and Add LoRA Adapters ---
print("Configuring and adding LoRA adapters...")
lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type="CAUSAL_LM", # Specify task type for PEFT
    # Target modules usually include attention projections.
    # For Llama models, these are common:
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)

# Wrap the model with PEFT LoRA adapters
model = get_peft_model(model, lora_config)
model.print_trainable_parameters() # Shows how many parameters are trainable with LoRA
print("LoRA adapters added to the model.")

# --- 3. Prepare the Wikitext-v2 Dataset ---
print("Loading Wikitext-v2 dataset...")
raw_train_dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
raw_eval_dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="validation")

# Adapter function to join list of strings into a single string per example
def wikitext_adapter(example):
    text_content = example.get("text")
    if isinstance(text_content, list):
        joined_text = " ".join(text_content)
    elif isinstance(text_content, str):
        joined_text = text_content
    else:
        joined_text = "" # Handle unexpected types
    return {"text": joined_text}

print("Applying adapter to join text segments...")
train_dataset = raw_train_dataset.map(wikitext_adapter, batched=False)
eval_dataset = raw_eval_dataset.map(wikitext_adapter, batched=False)

# Filter out empty or whitespace-only strings after adaptation
def filter_empty_texts(example):
    text = example.get("text")
    return text is not None and len(text.strip()) > 0

print("Filtering empty text entries after adaptation...")
train_dataset = train_dataset.filter(filter_empty_texts)
eval_dataset = eval_dataset.filter(filter_empty_texts)

print("Wikitext-v2 dataset loaded, adapted, and filtered.")
print(f"Original training dataset size: {len(raw_train_dataset)} examples")
print(f"Adapted and filtered training dataset size: {len(train_dataset)} examples")
print(f"Original validation dataset size: {len(raw_eval_dataset)} examples")
print(f"Adapted and filtered validation dataset size: {len(eval_dataset)} examples")

if len(train_dataset) > 0:
    print("\nExample from adapted and filtered training dataset:")
    print(train_dataset[0]['text'][:500])
else:
    print("\nNo valid examples found in the adapted and filtered training dataset.")
    exit() # Exit if no valid data to train on

# --- 4. Configure and Run Training with SFTTrainer ---
print("Setting up SFTTrainer...")

# Determine compute_dtype for TrainingArguments
compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

training_args = TrainingArguments(
    per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    warmup_steps=5,
    max_steps=MAX_STEPS,
    learning_rate=LEARNING_RATE,
    fp16=not torch.cuda.is_bf16_supported(), # Use FP16 if BF16 not supported
    bf16=torch.cuda.is_bf16_supported(),     # Use BF16 if supported
    logging_steps=10,
    output_dir=OUTPUT_DIR,
    optim="paged_adamw_8bit", # Use paged AdamW 8-bit for memory efficiency
    seed=3407,
    save_steps=100,
    evaluation_strategy="steps",
    eval_steps=100,
    logging_dir="./logs",
    report_to="tensorboard",
    dataloader_num_workers=os.cpu_count() // 2, # Use half CPU cores for dataloader for faster loading
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LENGTH,
    args=training_args,
    # Packing is useful for continuous pre-training to fill context windows
    # Set to True for packing, False if you want individual examples per context window
    # packing=True,
)
print("SFTTrainer configured. Starting training...")

# Start training
trainer.train()
print("\nTraining complete!")

# --- 5. Save and/or Push the Model ---
print("\nSaving LoRA adapters...")
# Saves only the LoRA adapters (much smaller than the full model)
trainer.model.save_pretrained(f"{OUTPUT_DIR}_lora_adapters")
tokenizer.save_pretrained(f"{OUTPUT_DIR}_lora_adapters") # Save tokenizer with adapters

print(f"LoRA adapters saved to {OUTPUT_DIR}_lora_adapters")

# Optional: Push LoRA adapters to Hugging Face Hub
# Ensure you are logged in via `huggingface-cli login`
# try:
#     print(f"\nPushing LoRA adapters to Hugging Face Hub (user_name/{OUTPUT_DIR}_lora_adapters)...")
#     trainer.model.push_to_hub(f"your_username/{OUTPUT_DIR}_lora_adapters")
#     tokenizer.push_to_hub(f"your_username/{OUTPUT_DIR}_lora_adapters")
#     print("LoRA adapters pushed to Hub.")
# except Exception as e:
#     print(f"Could not push LoRA adapters to Hugging Face Hub. Error: {e}")


# Optional: Merge LoRA adapters into the base model and save (for direct inference without PEFT)
# This will be a larger file size.
print("\nMerging LoRA adapters and saving the full model (for direct inference)...")
# You typically need to load the base model again without quantization to merge fully
# However, you can save the merged model directly.
try:
    # Get the base model with adapters merged
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(f"{OUTPUT_DIR}_merged")
    tokenizer.save_pretrained(f"{OUTPUT_DIR}_merged")
    print(f"Full merged model saved to {OUTPUT_DIR}_merged")

    # Optional: Push merged model to Hugging Face Hub
    # try:
    #     print(f"\nPushing merged model to Hugging Face Hub (user_name/{OUTPUT_DIR}_merged)...")
    #     merged_model.push_to_hub(f"your_username/{OUTPUT_DIR}_merged")
    #     tokenizer.push_to_hub(f"your_username/{OUTPUT_DIR}_merged")
    #     print("Merged model pushed to Hub.")
    # except Exception as e:
    #     print(f"Could not push merged model to Hugging Face Hub. Error: {e}")

except Exception as e:
    print(f"Could not merge and save full model. This might require more GPU memory or specific PEFT versions. Error: {e}")

print("\nFine-tuning process complete!")