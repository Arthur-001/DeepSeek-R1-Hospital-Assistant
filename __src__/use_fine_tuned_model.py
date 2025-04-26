import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig
import json

# ============= Global Configuration =============
# Model Paths Configuration
MODEL_PATHS = {
    "base_model": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",  # Base model path
    "fine_tuned_model": os.path.join("..", "Fine-Tuned Models", "finetuned_model0")  # Fine-tuned model path
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

# Test Configuration
TEST_QUESTIONS = [
    "What are the hospital visiting hours?"
    # "How can I book an appointment with a doctor?",
    # "Do you offer emergency services?"
]

def load_fine_tuned_model(model_dir=MODEL_PATHS["fine_tuned_model"]):
    """Load a fine-tuned model and its tokenizer"""
    # Convert relative path to absolute path
    model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), model_dir))
    print(f"Loading model from {model_dir}")
    
    try:
        # First, check if adapter_config.json exists
        adapter_config_path = os.path.join(model_dir, "adapter_config.json")
        if not os.path.exists(adapter_config_path):
            raise ValueError(f"adapter_config.json not found in {model_dir}")
        
        # Load base model info
        base_model_info_path = os.path.join(model_dir, "base_model_info.json")
        if os.path.exists(base_model_info_path):
            with open(base_model_info_path, "r") as f:
                base_model_info = json.load(f)
            base_model_name = base_model_info["base_model_name"]
        else:
            print("Warning: base_model_info.json not found, using default model name")
            base_model_name = MODEL_PATHS["base_model"]
        
        print(f"Loading base model: {base_model_name}")
        
        # Load the base model using global configuration
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            **MODEL_KWARGS
        )
        
        # Load the tokenizer from base model using global configuration
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            base_model_name,
            **TOKENIZER_KWARGS
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load the LoRA configuration
        print("Loading LoRA configuration...")
        peft_config = PeftConfig.from_json_file(adapter_config_path)
        
        # Load the LoRA weights
        print("Loading LoRA weights...")
        model = PeftModel.from_pretrained(
            base_model,
            model_dir,
            adapter_name="default"
        )
        
        model.eval()
        print("Model loaded successfully!")
        return model, tokenizer
        
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise

def generate_response(model, tokenizer, instruction, max_length=GENERATION_CONFIG["max_length"]):
    """Generate a response for a given instruction"""
    input_text = f"### Instruction: {instruction}\n### Response:"
    
    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length
    )
    
    # Use global generation config
    generation_kwargs = GENERATION_CONFIG.copy()
    generation_kwargs.update({
        "max_new_tokens": max_length,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    })
    
    outputs = model.generate(
        **inputs,
        **generation_kwargs
    )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.split("### Response:")[-1].strip()

if __name__ == "__main__":
    # Load model and tokenizer using the global path
    model, tokenizer = load_fine_tuned_model()
    
    print("\nTesting the model with sample questions:")
    print("-" * 50)
    
    for question in TEST_QUESTIONS:
        print(f"\nQuestion: {question}")
        response = generate_response(model, tokenizer, question)
        print(f"Response: {response}")
        print("-" * 50)