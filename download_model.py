from transformers import AutoTokenizer, AutoModelForCausalLM

# Specify the model repository on Hugging Face
model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

# Load the tokenizer (using trust_remote_code=True if required by the repo)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
print("Tokenizer downloaded successfully.")

# Load the model
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
print("Model downloaded successfully.")
