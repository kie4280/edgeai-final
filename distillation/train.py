from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import get_peft_model, LoraConfig, TaskType

# Load model and tokenizer (replace with the correct LLaMA 3.2B Instruct checkpoint)
model_name = "meta-llama/Llama-3.2-1B"  # Replace with 3.2B if available
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token

# Load dataset
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
train_texts = dataset["train"]
val_texts = dataset["validation"]

# Tokenization
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)

tokenized_train = train_texts.map(tokenize_function, batched=True, remove_columns=["text"])
tokenized_val = val_texts.map(tokenize_function, batched=True, remove_columns=["text"])

# LoRA Config
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],  # adjust based on model architecture
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

# Load model and apply LoRA
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_4bit=True,  # requires bitsandbytes
    device_map="auto"
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Training arguments
training_args = TrainingArguments(
    output_dir="./llama3-lora-wikitext2",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    evaluation_strategy="epoch",
    logging_dir="./logs",
    save_strategy="epoch",
    learning_rate=2e-4,
    fp16=True,
    report_to="none"
)

# Data collator
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    tokenizer=tokenizer,
    data_collator=data_collator
)

# Train
trainer.train()

# Save final model
model.save_pretrained("./llama3-lora-wikitext2-final")
