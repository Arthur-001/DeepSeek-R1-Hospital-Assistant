import os
import time
from typing import Dict, Any
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from peft import PeftModel, PeftConfig
from threading import Thread

# ============= GLOBAL CONFIGURATION =============
# Model paths
MODEL_PATHS = {
    "base_model": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",  # Base model path
    "fine_tuned_model": os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                                               "Fine-Tuned Models", 
                                               "finetuned_model_epochs_0.1_20250425_201530"))  # Fine-tuned model path
}

# Model configuration
CONFIG = {
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
    "max_length": 200,            # Maximum length of the generated text
    "temperature": 0.7,            # Higher values make output more random
    "top_p": 0.9,                  # Nucleus sampling parameter
    "top_k": 40,                   # Limit vocabulary to top k tokens
    "repetition_penalty": 1.1,     # Penalty for repeating tokens
    "do_sample": True,             # Use sampling (set to False for greedy decoding)
    "use_cache": True,             # Use KV cache for faster generation
    "use_lora": True,              # Whether to use LORA adapter
}

# Chat settings
SYSTEM_PROMPT = "You are a helpful AI assistant fine-tuned with LORA."
USER_PREFIX = "User: "
ASSISTANT_PREFIX = "Assistant: "
EXIT_COMMANDS = ["exit", "quit", "bye", "q"]

# ============= MODEL LOADING =============
def load_model():
    """Load the base model and LORA adapter"""
    if CONFIG["use_lora"]:
        print(f"Loading base model from {MODEL_PATHS['base_model']} and LORA adapter from {MODEL_PATHS['fine_tuned_model']} on {CONFIG['device']}...")
        
        # Load tokenizer from base model
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATHS['base_model'])
        
        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATHS['base_model'],
            device_map=CONFIG["device"],
            torch_dtype=CONFIG["torch_dtype"],
            low_cpu_mem_usage=True,
        )
        
        # Load LORA adapter on top of the base model
        model = PeftModel.from_pretrained(
            base_model,
            MODEL_PATHS['fine_tuned_model'],
            device_map=CONFIG["device"],
        )
    else:
        print(f"Loading model directly from {MODEL_PATHS['fine_tuned_model']} on {CONFIG['device']}...")
        
        # If not using LORA, load the full fine-tuned model
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATHS['fine_tuned_model'])
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATHS['fine_tuned_model'],
            device_map=CONFIG["device"],
            torch_dtype=CONFIG["torch_dtype"],
            low_cpu_mem_usage=True,
        )
    
    model.eval()  # Set model to evaluation mode
    
    print(f"Model loaded successfully on {CONFIG['device']}")
    
    # Return useful model information
    model_info = {
        "model_type": "LORA fine-tuned" if CONFIG["use_lora"] else "Standard fine-tuned",
        "base_model": MODEL_PATHS['base_model'] if CONFIG["use_lora"] else "N/A",
        "adapter_path": MODEL_PATHS['fine_tuned_model'] if CONFIG["use_lora"] else "N/A",
        "device": CONFIG["device"],
        "dtype": str(CONFIG["torch_dtype"]).split(".")[-1],
    }
    
    return model, tokenizer, model_info

# ============= CHAT FUNCTIONS =============
def format_prompt(messages):
    """Format the conversation for the model"""
    prompt = SYSTEM_PROMPT + "\n\n"
    
    for message in messages:
        if message["role"] == "user":
            prompt += USER_PREFIX + message["content"] + "\n"
        elif message["role"] == "assistant":
            prompt += ASSISTANT_PREFIX + message["content"] + "\n"
    
    # Add the assistant prefix for the new response
    prompt += ASSISTANT_PREFIX
    
    return prompt

def generate_response(model, tokenizer, prompt):
    """Generate a response from the model with token speed measurement"""
    inputs = tokenizer(prompt, return_tensors="pt").to(CONFIG["device"])
    
    # Set up streamer for token-by-token generation
    streamer = TextIteratorStreamer(tokenizer, skip_special_tokens=True)
    
    # Set up generation parameters
    gen_kwargs = {
        "input_ids": inputs.input_ids,
        "max_length": CONFIG["max_length"],
        "temperature": CONFIG["temperature"],
        "top_p": CONFIG["top_p"],
        "top_k": CONFIG["top_k"],
        "repetition_penalty": CONFIG["repetition_penalty"],
        "do_sample": CONFIG["do_sample"],
        "streamer": streamer,
        "use_cache": CONFIG["use_cache"],
    }
    
    # Start generation in a separate thread
    thread = Thread(target=model.generate, kwargs=gen_kwargs)
    thread.start()
    
    # Collect the generated tokens and measure speed
    generated_text = ""
    tokens_count = 0
    start_time = time.time()
    
    print("\nAssistant: ", end="", flush=True)
    
    for token in streamer:
        generated_text += token
        tokens_count += 1
        print(token, end="", flush=True)
    
    end_time = time.time()
    generation_time = end_time - start_time
    
    # Calculate and display token generation speed
    if generation_time > 0:
        tokens_per_second = tokens_count / generation_time
        print(f"\n\n[Generated {tokens_count} tokens in {generation_time:.2f}s - {tokens_per_second:.2f} tokens/sec]")
    else:
        print("\n\n[Generation too fast to measure speed]")
    
    return generated_text.strip()

def display_model_info(model_info):
    """Display model information in a formatted way"""
    print("\nModel Information:")
    print(f"  Type: {model_info['model_type']}")
    print(f"  Base model: {model_info['base_model']}")
    print(f"  LORA adapter: {model_info['adapter_path']}")
    print(f"  Running on: {model_info['device']}")
    print(f"  Precision: {model_info['dtype']}")

# ============= MAIN CHATBOT INTERFACE =============
def main():
    """Main chatbot loop"""
    try:
        # Load model and tokenizer
        model, tokenizer, model_info = load_model()
        
        # Welcome message
        print(f"""
{'='*70}
DeepSeek R1 1.5B LORA Fine-Tuned Model Chatbot
{'='*70}""")

        # Display model information
        display_model_info(model_info)
        
        print(f"""
Type your questions and press Enter to get responses.
Type one of {EXIT_COMMANDS} to exit.
{'='*70}
""")
        
        # Initialize conversation history
        conversation = []
        
        # Main interaction loop
        while True:
            # Get user input
            user_input = input("\nUser: ").strip()
            
            # Check for exit command
            if user_input.lower() in EXIT_COMMANDS:
                print("Exiting chatbot. Goodbye!")
                break
            
            # Add user message to conversation
            conversation.append({"role": "user", "content": user_input})
            
            # Format the prompt with the conversation history
            prompt = format_prompt(conversation)
            
            # Generate and measure response
            response = generate_response(model, tokenizer, prompt)
            
            # Add assistant response to conversation history
            conversation.append({"role": "assistant", "content": response})
    
    except KeyboardInterrupt:
        print("\nExiting chatbot. Goodbye!")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()