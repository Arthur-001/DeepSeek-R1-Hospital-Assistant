import os
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling,
    EvalPrediction,
    TrainerCallback,
    EarlyStoppingCallback
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel, PeftConfig
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import gc
import psutil
import argparse
import random

# ============= Global Hyperparameters =============
# Model Configuration
DEFAULT_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
MAX_SEQUENCE_LENGTH = 256

# Training Configuration
NUM_EPOCHS = 0.1  # Use fractional epochs for quick testing like 0.1
BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 4
LEARNING_RATE = 1e-4
WARMUP_RATIO = 0.03

# LoRA Configuration
LORA_RANK = 4  # r parameter
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "v_proj"]

# Evaluation and Saving Configuration
EVAL_RATIO = 0.25  # Evaluate every 1/4 of total steps
LOGGING_STEPS = 5
SAVE_TOTAL_LIMIT = 1
EARLY_STOPPING_PATIENCE = 2
EARLY_STOPPING_THRESHOLD = 0.0001

# Dataset Configuration
TRAIN_DATASET_PATH = os.path.join("..", "data-manipulation", "augmented_dataset.json")
EVAL_DATASET_PATH = os.path.join("..", "data-manipulation", "dataset.json")

# Model Saving Configuration
MODEL_SAVE_DIR = os.path.join("..", "Fine-Tuned Models")

# Quick Test Configuration
QUICK_TEST_EPOCHS = 0.01
QUICK_TEST_TRAIN_SIZE = 50
QUICK_TEST_EVAL_SIZE = 20
QUICK_TEST_MAX_LENGTH = 128
QUICK_TEST_EARLY_STOPPING_PATIENCE = 1

# Add argument parser for flexible testing
def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune LLM with optimized settings for CPU")
    parser.add_argument("--quick_test", action="store_true", help="Run in quick test mode with minimal data")
    parser.add_argument("--epochs", type=float, default=NUM_EPOCHS, help="Number of epochs (use fractional for quick tests)")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Model to fine-tune")
    parser.add_argument("--train_size", type=int, default=None, help="Limit training data size")
    parser.add_argument("--eval_size", type=int, default=None, help="Limit evaluation data size")
    parser.add_argument("--max_length", type=int, default=MAX_SEQUENCE_LENGTH, help="Max sequence length (smaller = faster)")
    parser.add_argument("--int8", action="store_true", help="Use int8 quantization")
    return parser.parse_args()

# Add memory monitoring
def print_memory_usage():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    print(f"Memory usage: {mem_info.rss / (1024 * 1024):.2f} MB")

# Custom callback to log metrics with memory monitoring
class MetricsLoggerCallback(TrainerCallback):
    def __init__(self, log_file_path, num_epochs):
        self.log_file_path = log_file_path
        self.num_epochs = num_epochs
        self.metrics = {
            'train_loss': [],
            'eval_loss': [],
            'eval_accuracy': [],
            'epochs': [],
            'steps': []
        }
        self.current_epoch = 0
        self.latest_train_loss = None
        self.latest_steps = {}
        self.start_time = datetime.now()
        
        # Create or clear the log file
        with open(self.log_file_path, 'w') as f:
            f.write(f"Training metrics log - Started at {self.start_time}\n")
            f.write("-" * 80 + "\n\n")
            f.write(f"{'Epoch':<8} | {'Step':<8} | {'Train Loss':<12} | {'Val Loss':<12} | {'Accuracy':<10} | {'Time':<12} | {'Mem (MB)':<10}\n")
            f.write("-" * 80 + "\n")
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        
        step = state.global_step
        current_time = datetime.now()
        elapsed = (current_time - self.start_time).total_seconds() / 60  # minutes
        
        # Calculate approximate epoch
        if args.num_train_epochs and state.max_steps:
            epoch = (step / state.max_steps) * args.num_train_epochs
            epoch_num = int(epoch) + 1
            
            if epoch_num != self.current_epoch:
                self.current_epoch = epoch_num
                self.latest_steps[epoch_num] = step
        else:
            epoch = 0
            epoch_num = 1
            
        self.metrics['steps'].append(step)
        self.metrics['epochs'].append(epoch)
        
        has_eval = 'eval_loss' in logs and 'eval_accuracy' in logs
        
        # Log training loss
        if 'loss' in logs:
            self.latest_train_loss = logs['loss']
            self.metrics['train_loss'].append((step, logs['loss']))
        
        # Get current memory usage
        mem_usage = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)  # MB
        
        # Log evaluation metrics
        if has_eval:
            self.metrics['eval_loss'].append((step, logs['eval_loss']))
            self.metrics['eval_accuracy'].append((step, logs['eval_accuracy']))
            
            # Print tabular format to terminal
            print("\n" + "-" * 80)
            print(f"{'Epoch':<8} | {'Step':<8} | {'Train Loss':<12} | {'Val Loss':<12} | {'Accuracy':<10} | {'Time':<12} | {'Mem (MB)':<10}")
            print("-" * 80)
            print(f"{epoch_num:<8} | {step:<8} | {self.latest_train_loss:<12.6f} | {logs['eval_loss']:<12.6f} | {logs['eval_accuracy']:<10.6f} | {elapsed:<12.2f} | {mem_usage:<10.1f}")
            print("-" * 80)
            
            # Write to file in tabular format
            with open(self.log_file_path, 'a') as f:
                f.write(f"{epoch_num:<8} | {step:<8} | {self.latest_train_loss:<12.6f} | {logs['eval_loss']:<12.6f} | {logs['eval_accuracy']:<10.6f} | {elapsed:<12.2f} | {mem_usage:<10.1f}\n")
        
        # Just print training loss without evaluation metrics
        elif 'loss' in logs and step % 10 == 0:  # Reduced to every 10 steps
            print(f"Epoch {epoch_num:<3} | Step {step:<6} | Train Loss: {logs['loss']:.6f} | Time: {elapsed:.2f}m | Mem: {mem_usage:.1f}MB")

def prepare_dataset(file_path, tokenizer, max_length=256, limit_size=None):
    """Prepare and tokenize dataset with optional size limit for faster testing"""
    print(f"Loading dataset from {file_path}...")
    
    # Load and format the JSON dataset
    with open(file_path, "r", encoding="utf-8") as f:
        qa_pairs = json.load(f)
    
    # Limit dataset size for quick testing if requested
    if limit_size is not None and limit_size < len(qa_pairs):
        print(f"Limiting dataset to {limit_size} examples (from {len(qa_pairs)})")
        qa_pairs = qa_pairs[:limit_size]
    
    formatted_data = []
    for qa in qa_pairs:
        # Format as conversation with instruction and response
        text = f"### Instruction: {qa['question']}\n### Response: {qa['answer']}\n"
        formatted_data.append({"text": text})
    
    dataset = Dataset.from_list(formatted_data)
    print(f"Dataset size: {len(dataset)} examples")
    
    # Tokenization function with explicit padding and truncation
    def tokenize(examples):
        return tokenizer(
            examples['text'],
            truncation=True,
            padding='max_length',
            max_length=max_length,  # Reduced max_length for faster processing
            return_tensors=None,
        )
    
    print(f"Tokenizing dataset with max_length={max_length}...")
    # Map the tokenization function over the dataset
    tokenized_dataset = dataset.map(
        tokenize,
        batched=True,
        batch_size=64,  # Process in larger batches for speed
        remove_columns=dataset.column_names
    )
    
    # Add labels for the language modeling task
    tokenized_dataset = tokenized_dataset.map(
        lambda examples: {'labels': examples['input_ids']},
        batched=True,
        batch_size=64
    )
    
    return tokenized_dataset

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    
    # Handle the case when labels are -100 (masked tokens)
    mask = labels != -100
    labels = labels[mask]
    
    # Get the predictions - handle memory efficiently by processing in chunks
    batch_size = 512
    predictions = []
    
    for i in range(0, labels.shape[0], batch_size):
        end_idx = min(i + batch_size, labels.shape[0])
        batch_mask = mask.reshape(-1)[i:end_idx]
        batch_logits = logits.reshape(-1, logits.shape[-1])[i:end_idx][batch_mask]
        
        # Get predictions for this batch
        batch_preds = np.argmax(batch_logits, axis=-1)
        predictions.append(batch_preds)
    
    # Combine predictions from all batches
    predictions = np.concatenate(predictions)
    
    # Calculate accuracy
    accuracy = np.mean(predictions == labels)
    
    # Release memory explicitly
    del logits, labels, mask, predictions
    gc.collect()
    
    return {
        'accuracy': accuracy,
    }

def plot_metrics(metrics_logger, output_dir):
    """Plot and save training metrics."""
    # Check if we have any metrics to plot
    if not metrics_logger.metrics['train_loss']:
        print("No metrics to plot - training was too short or no evaluations were performed.")
        return

    # Create plot directory if it doesn't exist
    plot_dir = os.path.join(output_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    # Extract data for plotting
    steps = [x[0] for x in metrics_logger.metrics['train_loss']]
    train_losses = [x[1] for x in metrics_logger.metrics['train_loss']]
    
    # Plot training loss
    plt.figure(figsize=(10, 5))
    plt.plot(steps, train_losses, label='Training Loss')
    
    # Only plot evaluation metrics if we have them
    if metrics_logger.metrics['eval_loss']:
        eval_steps = [x[0] for x in metrics_logger.metrics['eval_loss']]
        eval_losses = [x[1] for x in metrics_logger.metrics['eval_loss']]
        plt.plot(eval_steps, eval_losses, label='Evaluation Loss')
    
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.title('Training and Evaluation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plot_dir, 'loss_plot.png'))
    plt.close()

    # Plot accuracy if we have evaluation results
    if metrics_logger.metrics['eval_accuracy']:
        plt.figure(figsize=(10, 5))
        eval_steps = [x[0] for x in metrics_logger.metrics['eval_accuracy']]
        eval_accuracies = [x[1] for x in metrics_logger.metrics['eval_accuracy']]
        plt.plot(eval_steps, eval_accuracies, label='Evaluation Accuracy', color='green')
        plt.xlabel('Steps')
        plt.ylabel('Accuracy')
        plt.title('Evaluation Accuracy')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(plot_dir, 'accuracy_plot.png'))
        plt.close()

    print(f"Plots saved to {plot_dir}")

def generate_final_metrics_table(metrics_logger, output_dir):
    """Generate a final table summarizing metrics by epoch."""
    summary_path = os.path.join(output_dir, "metrics_summary.txt")
    
    # Organize metrics by epoch - simplified version
    epoch_metrics = {}
    
    # Calculate averages for each epoch
    for i, (step, loss) in enumerate(metrics_logger.metrics['train_loss']):
        # Find the corresponding epoch
        epoch_num = int(metrics_logger.metrics['epochs'][i]) + 1
        
        if epoch_num not in epoch_metrics:
            epoch_metrics[epoch_num] = {'train_loss': [], 'eval_loss': [], 'accuracy': []}
        
        epoch_metrics[epoch_num]['train_loss'].append(loss)
    
    # Add evaluation metrics
    for i, (step, loss) in enumerate(metrics_logger.metrics['eval_loss']):
        # Find the corresponding epoch
        for j, s in enumerate(metrics_logger.metrics['steps']):
            if s == step:
                epoch_num = int(metrics_logger.metrics['epochs'][j]) + 1
                break
        
        if epoch_num in epoch_metrics:
            epoch_metrics[epoch_num]['eval_loss'].append(loss)
            epoch_metrics[epoch_num]['accuracy'].append(metrics_logger.metrics['eval_accuracy'][i][1])
    
    # Calculate averages and final values for each epoch
    summary_metrics = {}
    for epoch_num, metrics in epoch_metrics.items():
        summary_metrics[epoch_num] = {
            'train_loss_avg': np.mean(metrics['train_loss']) if metrics['train_loss'] else None,
            'train_loss_final': metrics['train_loss'][-1] if metrics['train_loss'] else None,
            'eval_loss': metrics['eval_loss'][-1] if metrics['eval_loss'] else None,
            'accuracy': metrics['accuracy'][-1] if metrics['accuracy'] else None
        }
    
    # Write summary table to file
    with open(summary_path, 'w') as f:
        f.write(f"Metrics Summary by Epoch - Generated at {datetime.now()}\n")
        f.write("-" * 80 + "\n\n")
        
        # Write table header
        f.write(f"{'Epoch':<10} | {'Avg Train Loss':<20} | {'Final Train Loss':<20} | {'Validation Loss':<20} | {'Accuracy':<15}\n")
        f.write("-" * 80 + "\n")
        
        # Write each epoch's metrics
        for epoch_num in sorted(summary_metrics.keys()):
            metrics = summary_metrics[epoch_num]
            f.write(f"{epoch_num:<10} | ")
            f.write(f"{metrics['train_loss_avg']:<20.6f} | " if metrics['train_loss_avg'] is not None else f"{'N/A':<20} | ")
            f.write(f"{metrics['train_loss_final']:<20.6f} | " if metrics['train_loss_final'] is not None else f"{'N/A':<20} | ")
            f.write(f"{metrics['eval_loss']:<20.6f} | " if metrics['eval_loss'] is not None else f"{'N/A':<20} | ")
            f.write(f"{metrics['accuracy']:<15.6f}\n" if metrics['accuracy'] is not None else f"{'N/A':<15}\n")
    
    # Print the table to terminal
    print("\n" + "=" * 80)
    print("FINAL METRICS SUMMARY BY EPOCH")
    print("=" * 80)
    print(f"{'Epoch':<10} | {'Avg Train Loss':<20} | {'Final Train Loss':<20} | {'Validation Loss':<20} | {'Accuracy':<15}")
    print("-" * 80)
    
    for epoch_num in sorted(summary_metrics.keys()):
        metrics = summary_metrics[epoch_num]
        print(f"{epoch_num:<10} | ", end="")
        print(f"{metrics['train_loss_avg']:<20.6f} | " if metrics['train_loss_avg'] is not None else f"{'N/A':<20} | ", end="")
        print(f"{metrics['train_loss_final']:<20.6f} | " if metrics['train_loss_final'] is not None else f"{'N/A':<20} | ", end="")
        print(f"{metrics['eval_loss']:<20.6f} | " if metrics['eval_loss'] is not None else f"{'N/A':<20} | ", end="")
        print(f"{metrics['accuracy']:<15.6f}" if metrics['accuracy'] is not None else f"{'N/A':<15}")
    
    print("=" * 80)
    print(f"Detailed summary saved to: {summary_path}")
    
    return summary_path

def main():
    # Parse command line arguments
    args = parse_args()
    
    # Set smaller model if in quick test mode
    if args.quick_test:
        print("Running in QUICK TEST mode with minimal settings for fast verification")
        # Override settings for quick test
        args.epochs = QUICK_TEST_EPOCHS
        args.train_size = min(args.train_size or QUICK_TEST_TRAIN_SIZE, QUICK_TEST_TRAIN_SIZE)
        args.eval_size = min(args.eval_size or QUICK_TEST_EVAL_SIZE, QUICK_TEST_EVAL_SIZE)
        args.max_length = min(args.max_length, QUICK_TEST_MAX_LENGTH)
    
    # Optimize CPU threads for better performance
    cpu_count = os.cpu_count()
    optimal_threads = max(1, min(4, cpu_count - 1))  # Leave one core free for system
    print(f"Setting torch threads to {optimal_threads} (detected {cpu_count} cores)")
    torch.set_num_threads(optimal_threads)
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Create output directory with timestamp to avoid overwriting
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(MODEL_SAVE_DIR, f"finetuned_model_epochs_{args.epochs}_{timestamp}")
    if args.quick_test:
        output_dir = os.path.join(MODEL_SAVE_DIR, f"test_run_{timestamp}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Create metrics log file path
    metrics_log_path = os.path.join(output_dir, "training_metrics.txt")
    
    # Save configuration
    start_time = datetime.now()  # Add this at the beginning of main()
    initial_memory = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)  # Memory in MB

    with open(os.path.join(output_dir, "config.txt"), "w") as f:
        f.write("=" * 50 + "\n")
        f.write("TRAINING CONFIGURATION\n")
        f.write("=" * 50 + "\n\n")
        
        # Basic Training Info
        f.write("Training Information:\n")
        f.write("-" * 20 + "\n")
        f.write(f"Started at: {timestamp}\n")
        f.write(f"Model: {args.model}\n")
        f.write(f"Epochs: {args.epochs}\n")
        f.write(f"Quick test mode: {args.quick_test}\n")
        f.write(f"Initial memory usage: {initial_memory:.2f} MB\n\n")
        
        # Dataset Configuration
        f.write("Dataset Configuration:\n")
        f.write("-" * 20 + "\n")
        f.write(f"Training dataset: {TRAIN_DATASET_PATH}\n")
        f.write(f"Evaluation dataset: {EVAL_DATASET_PATH}\n")
        f.write(f"Max sequence length: {args.max_length}\n")
        f.write(f"Train size limit: {args.train_size or 'Full dataset'}\n")
        f.write(f"Eval size limit: {args.eval_size or 'Full dataset'}\n\n")
        
        # Model Configuration
        f.write("Model Configuration:\n")
        f.write("-" * 20 + "\n")
        f.write(f"Int8 quantization: {args.int8}\n")
        f.write(f"Batch size: {BATCH_SIZE}\n")
        f.write(f"Gradient accumulation steps: {GRADIENT_ACCUMULATION_STEPS}\n")
        f.write(f"Effective batch size: {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}\n\n")
        
        # LoRA Configuration
        f.write("LoRA Configuration:\n")
        f.write("-" * 20 + "\n")
        f.write(f"LoRA rank (r): {LORA_RANK}\n")
        f.write(f"LoRA alpha: {LORA_ALPHA}\n")
        f.write(f"LoRA dropout: {LORA_DROPOUT}\n")
        f.write(f"Target modules: {', '.join(LORA_TARGET_MODULES)}\n\n")
        
        # Optimization Configuration
        f.write("Optimization Configuration:\n")
        f.write("-" * 20 + "\n")
        f.write(f"Learning rate: {LEARNING_RATE}\n")
        f.write(f"Warmup ratio: {WARMUP_RATIO}\n")
        f.write(f"Early stopping patience: {EARLY_STOPPING_PATIENCE}\n")
        f.write(f"Early stopping threshold: {EARLY_STOPPING_THRESHOLD}\n")
        
        # System Information
        f.write("\nSystem Information:\n")
        f.write("-" * 20 + "\n")
        f.write(f"CPU count: {os.cpu_count()}\n")
        f.write(f"Torch threads: {torch.get_num_threads()}\n")
        f.write(f"CUDA available: {torch.cuda.is_available()}\n")
        if torch.cuda.is_available():
            f.write(f"GPU: {torch.cuda.get_device_name(0)}\n")

    print(f"Starting fine-tuning with the following settings:")
    print(f"- Model: {args.model}")
    print(f"- Epochs: {args.epochs}")
    print(f"- Max sequence length: {args.max_length}")
    print(f"- Training data limit: {args.train_size or 'full dataset'}")
    print(f"- Int8 quantization: {args.int8}")
    print(f"- Output directory: {output_dir}")
    
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=True,
        padding_side="right"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print_memory_usage()

    print("\nLoading model...")
    model_loading_kwargs = {
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
    }
    
    # Use int8 quantization if requested
    if args.int8:
        print("Using int8 quantization for reduced memory usage")
        model_loading_kwargs["load_in_8bit"] = True
    else:
        model_loading_kwargs["torch_dtype"] = torch.float32
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        **model_loading_kwargs
    )
    
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id
    
    # Free up memory
    gc.collect()
    print_memory_usage()

    print("\nConfiguring LoRA...")
    peft_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        target_modules=LORA_TARGET_MODULES,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    # Prepare for int8 training if using int8
    if args.int8:
        model = prepare_model_for_kbit_training(model)
    
    model = get_peft_model(model, peft_config)
    peft_config.save_pretrained(output_dir)
    print(f"Trainable parameters: {model.print_trainable_parameters()}")
    print_memory_usage()

    # Update paths to datasets
    train_dataset_path = TRAIN_DATASET_PATH
    print("\nPreparing training dataset...")
    train_dataset = prepare_dataset(train_dataset_path, tokenizer, max_length=args.max_length, limit_size=args.train_size)

    eval_dataset_path = EVAL_DATASET_PATH
    print("\nPreparing evaluation dataset...")
    eval_dataset = prepare_dataset(eval_dataset_path, tokenizer, max_length=args.max_length, limit_size=args.eval_size)
    print_memory_usage()

    print("\nCreating data collator...")
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8
    )

    # Calculate the total number of steps
    train_size = len(train_dataset)
    batch_size = BATCH_SIZE
    gradient_accumulation_steps = GRADIENT_ACCUMULATION_STEPS
    effective_batch_size = batch_size * gradient_accumulation_steps
    
    # Calculate total steps
    total_steps = int((train_size // effective_batch_size) * args.epochs)
    
    # Set evaluation steps - evaluate more frequently in quick test mode
    if args.quick_test:
        eval_steps = max(1, total_steps // 2)  # Evaluate twice during quick test
    else:
        eval_steps = max(1, total_steps // EVAL_RATIO)  # Evaluate 4 times during training
    
    save_steps = eval_steps * 2  # Save less frequently than eval
    
    # Configure early stopping
    patience = QUICK_TEST_EARLY_STOPPING_PATIENCE if args.quick_test else EARLY_STOPPING_PATIENCE
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        num_train_epochs=args.epochs,
        learning_rate=LEARNING_RATE,
        logging_steps=LOGGING_STEPS,
        save_steps=save_steps,
        save_total_limit=SAVE_TOTAL_LIMIT,
        optim="adamw_torch", 
        max_grad_norm=0.3,
        warmup_ratio=WARMUP_RATIO,
        lr_scheduler_type="constant_with_warmup",
        no_cuda=True,         # Force CPU
        dataloader_num_workers=0,  # Set to 0 for single-threaded loading
        remove_unused_columns=False,
        evaluation_strategy="steps",
        eval_steps=eval_steps,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        report_to="none",
        fp16=False,           # Disable fp16 on CPU
        logging_first_step=True,
        save_safetensors=True,
        save_strategy="steps",  # Add explicit save strategy
        save_only_model=False,  # Changed to False to save optimizer and scheduler states
    )

    # Initialize metrics logger callback
    metrics_logger = MetricsLoggerCallback(metrics_log_path, args.epochs)
    
    # Add early stopping callback
    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=patience,
        early_stopping_threshold=EARLY_STOPPING_THRESHOLD
    )

    print("\nInitializing trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[metrics_logger, early_stopping_callback]
    )

    print("\n" + "=" * 80)
    print(f"STARTING TRAINING WITH {args.epochs} EPOCHS")
    print(f"Total training examples: {len(train_dataset)}")
    print(f"Total evaluation examples: {len(eval_dataset)}")
    print(f"Total training steps: ~{total_steps}")
    print(f"Evaluation every {eval_steps} steps")
    print(f"Training metrics will be saved to: {metrics_log_path}")
    print("=" * 80 + "\n")
    
    try:
        # Start training with timeout protection
        trainer.train()
        
    except KeyboardInterrupt:
        print("\n\nTraining interrupted! Saving current state...")
    except Exception as e:
        print(f"\n\nError during training: {e}")
    finally:
        # Always try to save the model in the main directory
        try:
            print("Saving model...")
            # Save the model configuration with explicit model type
            config_dict = model.config.to_dict()
            config_dict["model_type"] = "qwen"  # Add model type explicitly
            config_dict["architectures"] = ["QWenLMHeadModel"]  # Add architecture
            
            with open(os.path.join(output_dir, "config.json"), "w") as f:
                json.dump(config_dict, f, indent=2)
            
            # Save base model info
            base_model_info = {
                "base_model_name": args.model,
                "model_type": "qwen",  # Explicitly set for DeepSeek model
                "is_peft_model": True,
                "peft_type": "LORA",
                "lora_config": {
                    "r": LORA_RANK,
                    "alpha": LORA_ALPHA,
                    "dropout": LORA_DROPOUT,
                    "target_modules": LORA_TARGET_MODULES
                }
            }
            
            with open(os.path.join(output_dir, "base_model_info.json"), "w") as f:
                json.dump(base_model_info, f, indent=2)
            
            # Save all tokenizer files
            print("Saving tokenizer files...")
            tokenizer.save_pretrained(output_dir)
            
            # Save the model and all adapter files directly in the output directory
            print("Saving model and adapter files...")
            trainer.save_model(output_dir)  # Save directly to output_dir
            model.save_pretrained(output_dir)  # Save adapter files directly to output_dir
            
            # Save the final checkpoint
            final_checkpoint_dir = os.path.join(output_dir, f"checkpoint-{trainer.state.global_step}")
            os.makedirs(final_checkpoint_dir, exist_ok=True)
            
            # Save adapter files to the checkpoint directory
            model.save_pretrained(final_checkpoint_dir)
            
            # Copy training state files to checkpoint
            for filename in ["optimizer.pt", "scheduler.pt", "trainer_state.json", "training_args.bin"]:
                src_path = os.path.join(output_dir, filename)
                if os.path.exists(src_path):
                    dst_path = os.path.join(final_checkpoint_dir, filename)
                    import shutil
                    shutil.copy2(src_path, dst_path)
            
            # Save RNG state
            rng_states = {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
            }
            torch.save(rng_states, os.path.join(final_checkpoint_dir, "rng_state.pth"))
            
            # Create README.md in the checkpoint directory
            with open(os.path.join(final_checkpoint_dir, "README.md"), "w") as f:
                f.write(f"# Model Checkpoint\n\nThis is a checkpoint of the model saved at step {trainer.state.global_step}.")
            
            print("\n" + "=" * 80)
            print(f"Model saved successfully!")
            print(f"Results saved to: {output_dir}")
            print(f"Metrics log saved to: {metrics_log_path}")
            print("=" * 80)
            
        except Exception as save_error:
            print(f"Error while saving the model: {save_error}")
            # If saving to main directory fails, try saving to error_checkpoint as a last resort
            try:
                error_checkpoint_dir = os.path.join(output_dir, "error_checkpoint")
                os.makedirs(error_checkpoint_dir, exist_ok=True)
                model.save_pretrained(error_checkpoint_dir)
                print(f"Emergency checkpoint saved to: {error_checkpoint_dir}")
            except:
                print("Could not save emergency checkpoint")
        
        # Clean up to free memory
        del model, trainer, train_dataset, eval_dataset
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        print_memory_usage()

if __name__ == "__main__":
    # Install required packages if not available
    try:
        import psutil
    except ImportError:
        print("Installing psutil for memory monitoring...")
        os.system("pip install psutil")
        import psutil
        
    main()