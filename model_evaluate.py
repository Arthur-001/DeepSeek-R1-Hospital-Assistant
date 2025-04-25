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
    "metrics": [
        "rouge",           # ROUGE scores for n-gram overlap
        "sacrebleu",       # BLEU score variant
        "bertscore",       # BERT-based semantic similarity
        "meteor",          # METEOR score for semantic similarity
        "exact_match",     # Exact string matching
        "f1_score"         # Token-level F1 score
    ],
    "batch_size": 4,      # Batch size for evaluation
    "save_results": True, # Whether to save evaluation results
    "results_file": "evaluation_results.json",  # Path to save results
    "bertscore_model": "microsoft/deberta-xlarge-mnli",  # BERTScore model
    "compute_bertscore": True,  # Whether to compute BERTScore (can be slow)
    "compute_meteor": True,     # Whether to compute METEOR score
    "detailed_results": True    # Whether to save detailed per-example results
}

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

def load_dataset():
    """Load and prepare the evaluation dataset"""
    print(f"\nLoading dataset from {EVAL_CONFIG['dataset_path']}...")
    with open(EVAL_CONFIG["dataset_path"], "r", encoding="utf-8") as f:
        dataset = json.load(f)
    return dataset

def generate_response(model, tokenizer, instruction):
    """Generate a response for a given instruction"""
    input_text = f"### Instruction: {instruction}\n### Response:"
    
    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=GENERATION_CONFIG["max_length"]
    )
    
    generation_kwargs = GENERATION_CONFIG.copy()
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

def load_metrics():
    """Load evaluation metrics with error handling"""
    metrics = {}
    for metric_name in EVAL_CONFIG["metrics"]:
        try:
            if metric_name == "rouge":
                metric = load_metric("rouge")
            elif metric_name == "sacrebleu":
                metric = load_metric("sacrebleu")
            else:
                continue
            metrics[metric_name] = metric
        except Exception as e:
            print(f"Error loading metric {metric_name}: {str(e)}")
    return metrics

def compute_metrics(metrics, prediction, reference):
    """Compute all metrics for a prediction-reference pair"""
    scores = {}
    
    # Compute ROUGE scores
    if "rouge" in metrics:
        try:
            rouge_scores = metrics["rouge"].compute(
                predictions=[prediction],
                references=[reference]
            )
            scores["rouge1"] = rouge_scores["rouge1"].mid.fmeasure
            scores["rouge2"] = rouge_scores["rouge2"].mid.fmeasure
            scores["rougeL"] = rouge_scores["rougeL"].mid.fmeasure
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
    
    # Compute BERTScore if enabled
    if "bertscore" in EVAL_CONFIG["metrics"] and EVAL_CONFIG["compute_bertscore"]:
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
    
    # Compute METEOR score if enabled
    if "meteor" in EVAL_CONFIG["metrics"] and EVAL_CONFIG["compute_meteor"]:
        try:
            meteor = meteor_score([reference], prediction)
            scores["meteor"] = meteor
        except Exception as e:
            print(f"Error computing METEOR: {str(e)}")
            scores["meteor"] = 0.0
    
    # Compute exact match
    if "exact_match" in EVAL_CONFIG["metrics"]:
        scores["exact_match"] = compute_exact_match(prediction, reference)
    
    # Compute F1 score
    if "f1_score" in EVAL_CONFIG["metrics"]:
        scores["f1_score"] = compute_f1_score(prediction, reference)
    
    return scores

def evaluate_model():
    """Main evaluation function"""
    # Load model, tokenizer and dataset
    model, tokenizer = load_model_and_tokenizer()
    dataset = load_dataset()
    
    # Initialize metrics
    print("\nLoading metrics...")
    metrics = load_metrics()
    if not metrics:
        raise ValueError("No metrics could be loaded. Please check your metric configuration.")
    
    # Prepare results storage
    results = {
        "predictions": [],
        "metrics": {name: [] for name in EVAL_CONFIG["metrics"]},
        "detailed_metrics": {}
    }
    
    # Generate predictions and compute metrics
    print("\nGenerating predictions and computing metrics...")
    for item in tqdm(dataset, desc="Evaluating"):
        # Generate prediction
        prediction = generate_response(model, tokenizer, item["question"])
        result_item = {
            "question": item["question"],
            "true_answer": item["answer"],
            "predicted_answer": prediction
        }
        
        # Compute metrics
        scores = compute_metrics(metrics, prediction, item["answer"])
        result_item["scores"] = scores
        
        # Store results
        results["predictions"].append(result_item)
        for metric_name, score in scores.items():
            if metric_name not in results["metrics"]:
                results["metrics"][metric_name] = []
            results["metrics"][metric_name].append(score)
    
    # Calculate average metrics
    avg_metrics = {
        name: {
            "mean": np.mean(scores),
            "std": np.std(scores),
            "min": np.min(scores),
            "max": np.max(scores)
        }
        for name, scores in results["metrics"].items()
    }
    
    # Print results
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)
    for metric_name, stats in avg_metrics.items():
        print(f"{metric_name.upper():<20} Mean: {stats['mean']:.4f} ± {stats['std']:.4f} (Min: {stats['min']:.4f}, Max: {stats['max']:.4f})")
    
    # Save results if configured
    if EVAL_CONFIG["save_results"]:
        results["statistics"] = avg_metrics
        with open(EVAL_CONFIG["results_file"], "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nDetailed results saved to {EVAL_CONFIG['results_file']}")

if __name__ == "__main__":
    try:
        evaluate_model()
    except Exception as e:
        print(f"\nError during evaluation: {str(e)}")
        raise
