from transformers import AutoTokenizer, AutoModelForCausalLM

# Specify the model repository on Hugging Face
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# Load the tokenizer (using trust_remote_code=True if required by the repo)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
print("Tokenizer downloaded successfully.")

# Load the model
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
print("Model downloaded successfully.")
