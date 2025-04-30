import os
import time
import re
from typing import Dict, Any
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from peft import PeftModel, PeftConfig
from threading import Thread

# ============= GLOBAL CONFIGURATION =============
# Model paths
MODEL_PATHS = {
    "base_model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",  # Base model path
    "fine_tuned_model": os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                                               "Fine-Tuned Models", 
                                               "finetuned_model_epochs_5_20250428_164847"))  # Fine-tuned model path
}

# Model configuration
CONFIG = {
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
    "max_length": 2048,            # Maximum length of the generated text
    "temperature": 0.7,            # Higher values make output more random
    "top_p": 0.9,                  # Nucleus sampling parameter
    "top_k": 40,                   # Limit vocabulary to top k tokens
    "repetition_penalty": 1.1,     # Penalty for repeating tokens
    "do_sample": True,             # Use sampling (set to False for greedy decoding)
    "use_cache": True,             # Use KV cache for faster generation
    "use_lora": True,              # Whether to use LORA adapter
    "max_input_length": 1024,      # Maximum allowed input length
}

# Chat settings
SYSTEM_PROMPT = """You are a helpful AI assistant fine-tuned with LORA.

When answering mathematical expressions or equations:
1. NEVER use EXACT 'User:' or EXACT 'Assistant:' in your response
2. For simple calculations, just respond with the result as a single number
3. For complex problems, explain directly without using any dialogue format
4. Always use a concise format without creating mock conversations"""

USER_PREFIX = "User: "
ASSISTANT_PREFIX = "Assistant: "
EXIT_COMMANDS = ["exit", "quit", "bye", "q"]
MAX_RETRY_ATTEMPTS = 3

# Regex pattern to detect if input is a math expression
MATH_EXPRESSION_PATTERN = r'^[\d\s\+\-\*\/\(\)\^\%\=\.x×÷]+$'

# ============= MODEL LOADING =============
def load_model():
    """Load the base model and LORA adapter"""
    try:
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
    
    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()
        raise

# ============= MATH HANDLING =============
def is_math_expression(text):
    """Check if the input text is a simple math expression"""
    # Replace 'x' or '×' with '*' for multiplication
    text = text.replace('x', '*').replace('×', '*').replace('÷', '/')
    return bool(re.match(MATH_EXPRESSION_PATTERN, text.strip()))

def direct_math_eval(expression):
    """
    Safely evaluate a mathematical expression directly.
    This is a fallback for very simple expressions.
    """
    try:
        # Replace 'x' or '×' with '*' for multiplication
        expression = expression.replace('x', '*').replace('×', '*').replace('÷', '/')
        
        # Remove any whitespace
        expression = expression.replace(' ', '')
        
        # Extremely basic security check: only allow digits and basic math operators
        if not re.match(r'^[\d\+\-\*\/\(\)\.\^]+$', expression):
            return None
        
        # Safely evaluate the expression
        # Note: Using eval is generally not recommended but here it's strictly limited
        # For a production system, consider using a proper math expression parser
        result = eval(expression)
        return result
    except:
        return None

# ============= CHAT FUNCTIONS =============
def format_prompt(messages, user_input_is_math=False):
    """Format the conversation for the model"""
    prompt = SYSTEM_PROMPT + "\n\n"
    
    # For math expressions, add special instructions
    if user_input_is_math:
        prompt += "Remember to only respond with the numerical result for simple math expressions.\n\n"
    
    for message in messages:
        if message["role"] == "user":
            prompt += USER_PREFIX + message["content"] + "\n"
        elif message["role"] == "assistant":
            prompt += ASSISTANT_PREFIX + message["content"] + "\n"
    
    # Add the assistant prefix for the new response
    prompt += ASSISTANT_PREFIX
    
    return prompt

def sanitize_input(text):
    """Sanitize input text to prevent errors"""
    # Truncate if too long
    if len(text) > CONFIG["max_input_length"]:
        print(f"\n[Warning: Input too long ({len(text)} chars). Truncated to {CONFIG['max_input_length']} chars.]")
        text = text[:CONFIG["max_input_length"]]
    
    # Replace any problematic control characters but keep math symbols
    # This regex keeps alphanumeric chars, math symbols, punctuation and common special chars
    # but removes control characters and other potentially problematic unicode
    text = re.sub(r'[^\w\s\.\,\;\:\!\?\(\)\[\]\{\}\'\"\+\-\*\/\=\<\>\&\|\^\%\$\#\@\~\`]', '', text)
    
    return text

def post_process_response(response, user_input_is_math=False):
    """Clean up the model's response to remove unwanted patterns"""
    # For math expressions, extract just the answer if possible
    if user_input_is_math:
        # Try to find just a number in the response
        number_match = re.search(r'(\-?\d+\.?\d*)', response)
        if number_match:
            return number_match.group(1)
        
        # Remove any User/Assistant prefixes that might have been generated
        response = re.sub(r'User:.*?Assistant:', '', response)
        response = re.sub(r'User:', '', response)
        response = re.sub(r'Assistant:', '', response)
        
        # Remove any other conversational artifacts
        response = response.replace('The answer is', '').replace('equals', '').strip()
        
    return response.strip()

def generate_response(model, tokenizer, prompt, user_input_is_math=False):
    """Generate a response from the model with token speed measurement and error handling"""
    try:
        # For very simple math expressions, try direct evaluation first
        if user_input_is_math and is_math_expression(prompt.split(USER_PREFIX)[-1].split(ASSISTANT_PREFIX)[0]):
            math_expr = prompt.split(USER_PREFIX)[-1].split(ASSISTANT_PREFIX)[0].strip()
            direct_result = direct_math_eval(math_expr)
            if direct_result is not None:
                # Skip model generation for very simple math
                print(f"\nAssistant: {direct_result}")
                print(f"\n\n[Math expression evaluated directly]")
                return str(direct_result)
        
        inputs = tokenizer(prompt, return_tensors="pt").to(CONFIG["device"])
        
        # Set up streamer for token-by-token generation
        streamer = TextIteratorStreamer(tokenizer, skip_special_tokens=True)
        
        # Adjust generation parameters for math expressions
        gen_kwargs = {
            "input_ids": inputs.input_ids,
            "max_length": CONFIG["max_length"],
            "temperature": 0.1 if user_input_is_math else CONFIG["temperature"],  # Lower temp for math
            "top_p": 0.95 if user_input_is_math else CONFIG["top_p"],
            "top_k": 10 if user_input_is_math else CONFIG["top_k"],  # More deterministic for math
            "repetition_penalty": CONFIG["repetition_penalty"],
            "do_sample": False if user_input_is_math else CONFIG["do_sample"],  # Greedy decoding for math
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
        
        # Post-process the response to clean it up
        clean_response = post_process_response(generated_text, user_input_is_math)
        
        # If the clean response is different from what was printed, print a note
        if clean_response != generated_text and user_input_is_math:
            print(f"\n[Note: Response has been simplified to: {clean_response}]")
        
        # Calculate and display token generation speed
        if generation_time > 0:
            tokens_per_second = tokens_count / generation_time
            print(f"\n\n[Generated {tokens_count} tokens in {generation_time:.2f}s - {tokens_per_second:.2f} tokens/sec]")
        else:
            print("\n\n[Generation too fast to measure speed]")
        
        return clean_response
    
    except torch.cuda.OutOfMemoryError:
        print("\n\n[Error: CUDA out of memory. Try reducing max_length in CONFIG or using a smaller model]")
        return "I encountered a memory error. Please try a shorter question or simpler request."
    
    except Exception as e:
        print(f"\n\n[Error during generation: {e}]")
        return "I encountered an error while generating a response. Let's try again."

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
    """Main chatbot loop with error handling"""
    model = None
    tokenizer = None
    
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
Mathematical operations like "2 + 2" or "5 * 3" are supported.
{'='*70}
""")
        
        # Initialize conversation history
        conversation = []
        
        # Main interaction loop
        while True:
            try:
                # Get user input
                user_input = input("\nUser: ").strip()
                
                # Check for exit command
                if user_input.lower() in EXIT_COMMANDS:
                    print("Exiting chatbot. Goodbye!")
                    break
                
                # Sanitize input
                user_input = sanitize_input(user_input)
                
                # Skip empty inputs
                if not user_input:
                    print("[Empty input. Please try again.]")
                    continue
                
                # Check if input is a math expression
                user_input_is_math = is_math_expression(user_input)
                
                # Add user message to conversation
                conversation.append({"role": "user", "content": user_input})
                
                # Format the prompt with the conversation history
                prompt = format_prompt(conversation, user_input_is_math)
                
                # Generate and measure response with retry logic
                retry_count = 0
                response = None
                
                while response is None and retry_count < MAX_RETRY_ATTEMPTS:
                    try:
                        response = generate_response(model, tokenizer, prompt, user_input_is_math)
                    except Exception as e:
                        retry_count += 1
                        if retry_count < MAX_RETRY_ATTEMPTS:
                            print(f"\n[Error occurred. Retrying ({retry_count}/{MAX_RETRY_ATTEMPTS})...]")
                        else:
                            print(f"\n[Failed after {MAX_RETRY_ATTEMPTS} attempts. Error: {str(e)}]")
                            response = "I'm having trouble generating a response right now. Let's try a different question."
                
                # Add assistant response to conversation history
                conversation.append({"role": "assistant", "content": response})
                
                # Manage conversation length to prevent context overflow
                if len(conversation) > 20:  # Keep last 20 messages
                    conversation = conversation[-20:]
            
            except KeyboardInterrupt:
                # Handle Ctrl+C within the chat loop
                print("\n[Interrupted. Type one of {EXIT_COMMANDS} to exit, or continue chatting.]")
            
            except Exception as e:
                print(f"\n[Error in chat loop: {str(e)}. Continuing...]")
    
    except KeyboardInterrupt:
        print("\nExiting chatbot. Goodbye!")
    except Exception as e:
        print(f"\nCritical error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up resources
        print("Cleaning up resources...")
        if 'model' in locals() and model is not None:
            try:
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                print("Model unloaded and CUDA cache cleared.")
            except Exception as e:
                print(f"Error during cleanup: {e}")

if __name__ == "__main__":
    main()