import os
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model
import json

def prepare_dataset(file_path, tokenizer, max_length=512):
    # Load and format the JSON dataset
    with open(file_path, "r", encoding="utf-8") as f:
        qa_pairs = json.load(f)
    
    formatted_data = []
    for qa in qa_pairs:
        # Format as conversation with instruction and response
        text = f"### Instruction: {qa['instruction']}\n### Response: {qa['response']}\n"
        formatted_data.append({"text": text})
    
    dataset = Dataset.from_list(formatted_data)
    
    # Tokenization function without fixed padding – dynamic padding will be applied by the collator
    def tokenize(examples):
        tokenized = tokenizer(
            examples['text'],
            truncation=True,
            max_length=max_length
        )
        tokenized['labels'] = tokenized['input_ids'].copy()
        return tokenized
    
    tokenized_dataset = dataset.map(
        tokenize,
        batched=True,
        remove_columns=dataset.column_names
    )
    
    return tokenized_dataset

def main():
    # Limit CPU threads to optimize performance on your system
    torch.set_num_threads(4)  # Adjust based on your CPU cores

    torch.manual_seed(42)
    
    model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_side="right"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True
    )
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    print("Configuring LoRA...")
    peft_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, peft_config)

    print("Preparing dataset...")
    train_dataset = prepare_dataset("deepseek_format_dataset.json", tokenizer)

    print("Creating data collator...")
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    training_args = TrainingArguments(
        output_dir="./finetuned_model",
        per_device_train_batch_size=1,         # Small batch size for low-memory systems
        gradient_accumulation_steps=4,           # Simulates a larger batch size without extra memory
        num_train_epochs=3,
        learning_rate=2e-4,
        logging_steps=50,
        save_steps=200,
        save_total_limit=3,
        optim="adamw_torch",
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        lr_scheduler_type="constant_with_warmup",
        no_cuda=True,                          # Force CPU training
        dataloader_num_workers=2,              # Adjust based on your system
        remove_unused_columns=False
    )

    print("Initializing trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator
    )

    print("Starting training...")
    trainer.train()

    print("Saving model...")
    trainer.save_model()
    tokenizer.save_pretrained("./finetuned_model")

    print("Fine-tuning completed successfully!")

if __name__ == "__main__":
    main()
