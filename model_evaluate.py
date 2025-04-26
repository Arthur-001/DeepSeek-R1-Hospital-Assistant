import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from evaluate import load as load_metric
from tqdm import tqdm
import numpy as np
import nltk
from nltk.translate.meteor_score import meteor_score
from bert_score import score as bert_score
import warnings
import time
from datetime import datetime
warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('wordnet')
except LookupError:
    nltk.download('wordnet')

# ============= Global Configuration =============
# Quick Test Configuration
QUICK_TEST_CONFIG = {
    "enabled": False,           # Set to True for quick testing
    "test_size": 3,           # Number of examples to test
    "metrics": ["rouge", "exact_match"],  # Only use fast metrics
    "max_length": 128,        # Shorter sequence length
    "temperature": 0.7,       # Lower temperature for more focused outputs
    "results_file": "quick_test_results_{timestamp}.json"  # Different results file with timestamp
}

# Model Paths
MODEL_PATHS = {
    "base_model": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",  # Base model path
    "fine_tuned_model": "./finetuned_model_epochs_0.1_20250425_201530"  # Fine-tuned model path
}

# Model Loading Configuration
MODEL_KWARGS = {
    "trust_remote_code": True,
    "torch_dtype": torch.float32,
    "low_cpu_mem_usage": True,
    "use_flash_attention_2": False
}

TOKENIZER_KWARGS = {
    "trust_remote_code": True,
    "padding_side": "right"
}

# Generation Configuration
GENERATION_CONFIG = {
    "max_length": 256,
    "num_return_sequences": 1,
    "temperature": 0.7,
    "do_sample": True,
    "use_cache": True
}

# Evaluation Configuration
EVAL_CONFIG = {
    "dataset_path": "data-manipulation/dataset.json",
    "num_questions": 2,  # Set to None to use all questions, or specify a number (e.g., 3) to use only first N questions
    "metrics": [
        "rouge",           # ROUGE scores for n-gram overlap
        "sacrebleu",       # BLEU score variant
        "bertscore",       # BERT-based semantic similarity
        "meteor",          # METEOR score for semantic similarity
        "exact_match",     # Exact string matching
        "f1_score",        # Token-level F1 score
        "accuracy"         # Word-level accuracy
    ],
    "batch_size": 4,      # Batch size for evaluation
    "save_results": True, # Whether to save evaluation results
    "results_file": "evaluation_results_{timestamp}.json",  # Path to save results with timestamp
    "bertscore_model": "microsoft/deberta-xlarge-mnli",  # BERTScore model
    "compute_bertscore": True,  # Whether to compute BERTScore (can be slow)
    "compute_meteor": True,     # Whether to compute METEOR score
    "detailed_results": True    # Whether to save detailed per-example results
}

def get_timestamp():
    """Get current timestamp in a formatted string"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def get_results_filename(base_filename, quick_test=False):
    """Get the results filename with timestamp"""
    timestamp = get_timestamp()
    if quick_test:
        return QUICK_TEST_CONFIG["results_file"].format(timestamp=timestamp)
    return EVAL_CONFIG["results_file"].format(timestamp=timestamp)

def load_model_and_tokenizer():
    """Load the fine-tuned model and tokenizer"""
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATHS["base_model"],
        **TOKENIZER_KWARGS
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("\nLoading model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATHS["base_model"],
        **MODEL_KWARGS
    )
    
    print("\nLoading fine-tuned model...")
    model = PeftModel.from_pretrained(
        base_model,
        MODEL_PATHS["fine_tuned_model"],
        adapter_name="default"
    )
    model.eval()
    
    return model, tokenizer

def load_dataset(quick_test=False):
    """Load and prepare the evaluation dataset"""
    print(f"\nLoading dataset from {EVAL_CONFIG['dataset_path']}...")
    with open(EVAL_CONFIG["dataset_path"], "r", encoding="utf-8") as f:
        dataset = json.load(f)
    
    if quick_test:
        print(f"Quick test mode: Using first {QUICK_TEST_CONFIG['test_size']} examples")
        dataset = dataset[:QUICK_TEST_CONFIG["test_size"]]
    elif EVAL_CONFIG["num_questions"] is not None:
        print(f"Full evaluation mode: Using first {EVAL_CONFIG['num_questions']} examples")
        dataset = dataset[:EVAL_CONFIG["num_questions"]]
    else:
        print(f"Full evaluation mode: Using all {len(dataset)} examples")
    
    return dataset

def generate_response(model, tokenizer, instruction, quick_test=False):
    """Generate a response for a given instruction"""
    input_text = f"### Instruction: {instruction}\n### Response:"
    
    # Use shorter max_length for quick test
    max_length = QUICK_TEST_CONFIG["max_length"] if quick_test else GENERATION_CONFIG["max_length"]
    
    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length
    )
    
    generation_kwargs = GENERATION_CONFIG.copy()
    if quick_test:
        generation_kwargs["temperature"] = QUICK_TEST_CONFIG["temperature"]
    
    generation_kwargs.update({
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    })
    
    outputs = model.generate(
        **inputs,
        **generation_kwargs
    )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.split("### Response:")[-1].strip()

def compute_exact_match(prediction, reference):
    """Compute exact match score"""
    return float(prediction.strip().lower() == reference.strip().lower())

def compute_f1_score(prediction, reference):
    """Compute token-level F1 score"""
    pred_tokens = set(prediction.lower().split())
    ref_tokens = set(reference.lower().split())
    
    if not pred_tokens or not ref_tokens:
        return 0.0
    
    # Calculate precision and recall
    common_tokens = pred_tokens.intersection(ref_tokens)
    precision = len(common_tokens) / len(pred_tokens)
    recall = len(common_tokens) / len(ref_tokens)
    
    # Calculate F1 score
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)

def compute_accuracy(prediction, reference):
    """Compute accuracy score by comparing predicted and reference answers"""
    # Convert to lowercase and remove extra whitespace
    pred = prediction.strip().lower()
    ref = reference.strip().lower()
    
    # Split into words
    pred_words = set(pred.split())
    ref_words = set(ref.split())
    
    # Calculate word overlap
    common_words = pred_words.intersection(ref_words)
    total_words = len(ref_words)
    
    if total_words == 0:
        return 0.0
    
    # Calculate accuracy as percentage of matching words
    accuracy = len(common_words) / total_words
    return accuracy

def load_metrics(quick_test=False):
    """Load evaluation metrics with error handling"""
    metrics = {}
    metric_list = QUICK_TEST_CONFIG["metrics"] if quick_test else EVAL_CONFIG["metrics"]
    
    for metric_name in metric_list:
        try:
            if metric_name == "rouge":
                metrics["rouge"] = load_metric("rouge")
            elif metric_name == "sacrebleu":
                metrics["sacrebleu"] = load_metric("sacrebleu")
            elif metric_name == "bertscore":
                # BERTScore doesn't need to be loaded as a metric
                pass
            elif metric_name == "meteor":
                # METEOR doesn't need to be loaded as a metric
                pass
            elif metric_name == "exact_match":
                # Exact match is computed directly
                pass
        except Exception as e:
            print(f"Error loading metric {metric_name}: {str(e)}")
    return metrics

def compute_metrics(metrics, prediction, reference, quick_test=False):
    """Compute all metrics for a prediction-reference pair"""
    scores = {}
    
    # Compute accuracy
    scores["accuracy"] = compute_accuracy(prediction, reference)
    
    # Compute F1 score (always compute this as it's fast)
    scores["f1_score"] = compute_f1_score(prediction, reference)
    
    # Compute ROUGE scores
    if "rouge" in metrics:
        try:
            rouge_scores = metrics["rouge"].compute(
                predictions=[prediction],
                references=[reference]
            )
            scores["rouge1"] = rouge_scores["rouge1"]
            scores["rouge2"] = rouge_scores["rouge2"]
            scores["rougeL"] = rouge_scores["rougeL"]
        except Exception as e:
            print(f"Error computing ROUGE: {str(e)}")
            scores.update({"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0})
    
    # Compute BLEU score
    if "sacrebleu" in metrics:
        try:
            bleu_score = metrics["sacrebleu"].compute(
                predictions=[prediction],
                references=[[reference]]
            )
            scores["bleu"] = bleu_score["score"] / 100
        except Exception as e:
            print(f"Error computing BLEU: {str(e)}")
            scores["bleu"] = 0.0
    
    # Compute BERTScore
    if "bertscore" in EVAL_CONFIG["metrics"]:
        try:
            P, R, F1 = bert_score(
                [prediction],
                [reference],
                lang="en",
                model_type=EVAL_CONFIG["bertscore_model"]
            )
            scores["bertscore_precision"] = P.mean().item()
            scores["bertscore_recall"] = R.mean().item()
            scores["bertscore_f1"] = F1.mean().item()
        except Exception as e:
            print(f"Error computing BERTScore: {str(e)}")
            scores.update({
                "bertscore_precision": 0.0,
                "bertscore_recall": 0.0,
                "bertscore_f1": 0.0
            })
    
    # Compute METEOR score
    if "meteor" in EVAL_CONFIG["metrics"]:
        try:
            meteor = meteor_score([reference], prediction)
            scores["meteor"] = meteor
        except Exception as e:
            print(f"Error computing METEOR: {str(e)}")
            scores["meteor"] = 0.0
    
    # Compute exact match
    if "exact_match" in EVAL_CONFIG["metrics"]:
        scores["exact_match"] = compute_exact_match(prediction, reference)
    
    return scores

def evaluate_model(quick_test=False):
    """Main evaluation function"""
    start_time = time.time()
    
    # Load model, tokenizer and dataset
    model, tokenizer = load_model_and_tokenizer()
    dataset = load_dataset(quick_test)
    
    # Initialize metrics
    print("\nLoading metrics...")
    metrics = load_metrics(quick_test)
    if not metrics:
        raise ValueError("No metrics could be loaded. Please check your metric configuration.")
    
    # Prepare results storage
    results = {
        "predictions": [],
        "metrics": {name: [] for name in (QUICK_TEST_CONFIG["metrics"] if quick_test else EVAL_CONFIG["metrics"])},
        "detailed_metrics": {},
        "evaluation_info": {
            "timestamp": get_timestamp(),
            "quick_test": quick_test,
            "model_path": MODEL_PATHS["fine_tuned_model"],
            "dataset_size": len(dataset),
            "metrics_used": QUICK_TEST_CONFIG["metrics"] if quick_test else EVAL_CONFIG["metrics"]
        }
    }
    
    # Generate predictions and compute metrics
    print("\nGenerating predictions and computing metrics...")
    for item in tqdm(dataset, desc="Evaluating"):
        # Generate prediction
        prediction = generate_response(model, tokenizer, item["question"], quick_test)
        result_item = {
            "question": item["question"],
            "true_answer": item["answer"],
            "predicted_answer": prediction
        }
        
        # Compute metrics
        scores = compute_metrics(metrics, prediction, item["answer"], quick_test)
        result_item["scores"] = scores
        
        # Store results
        results["predictions"].append(result_item)
        for metric_name, score in scores.items():
            if metric_name not in results["metrics"]:
                results["metrics"][metric_name] = []
            results["metrics"][metric_name].append(score)
    
    # Calculate average metrics with proper error handling
    avg_metrics = {}
    for name, scores in results["metrics"].items():
        if len(scores) > 0:  # Only compute if we have scores
            avg_metrics[name] = {
                "mean": float(np.mean(scores)),
                "std": float(np.std(scores)),
                "min": float(np.min(scores)),
                "max": float(np.max(scores))
            }
        else:
            avg_metrics[name] = {
                "mean": 0.0,
                "std": 0.0,
                "min": 0.0,
                "max": 0.0
            }
    
    # Calculate total evaluation time
    total_time = time.time() - start_time
    results["evaluation_info"]["total_time_seconds"] = total_time
    
    # Print results
    print("\n" + "=" * 80)
    print("QUICK TEST RESULTS" if quick_test else "EVALUATION RESULTS")
    print("=" * 80)
    print(f"Evaluation completed in {total_time:.2f} seconds")
    for metric_name, stats in avg_metrics.items():
        print(f"{metric_name.upper():<20} Mean: {stats['mean']:.4f} ± {stats['std']:.4f} (Min: {stats['min']:.4f}, Max: {stats['max']:.4f})")
    
    # Save results if configured
    if EVAL_CONFIG["save_results"]:
        results["statistics"] = avg_metrics
        results_file = get_results_filename(EVAL_CONFIG["results_file"], quick_test)
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nDetailed results saved to {results_file}")

if __name__ == "__main__":
    try:
        # Run quick test if enabled
        if QUICK_TEST_CONFIG["enabled"]:
            print("\n" + "=" * 80)
            print("RUNNING QUICK TEST")
            print("=" * 80)
            evaluate_model(quick_test=True)
        else:
            evaluate_model()
    except Exception as e:
        print(f"\nError during evaluation: {str(e)}")
        raise
