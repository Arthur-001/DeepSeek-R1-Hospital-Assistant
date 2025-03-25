import time
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Specify the model name
model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)

# Move the model to GPU if available (for faster inference)
# if torch.cuda.is_available():
#     model.to("cuda")

# Set manual seed for reproducibility
torch.manual_seed(42)

# Define the test prompt
prompt = "What is the capital of France?"

# Tokenize the prompt
inputs = tokenizer(prompt, return_tensors="pt")
# if torch.cuda.is_available():
#     inputs = {k: v.to("cuda") for k, v in inputs.items()}

# Start timing the generation process
start_time = time.time()

# Generate the answer (max_new_tokens controls the length of the output)
outputs = model.generate(**inputs, max_new_tokens=50)

# Synchronize GPU operations if applicable
# if torch.cuda.is_available():
#     torch.cuda.synchronize()

# End timing
end_time = time.time()
elapsed_time = end_time - start_time

# Calculate number of tokens generated (subtract input tokens from total)
num_generated = outputs.shape[1] - inputs["input_ids"].shape[1]
tokens_per_second = num_generated / elapsed_time

# Decode the generated tokens into text
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

# Print the results: the generated answer and performance metrics
print("Generated answer:")
print(generated_text)
print("\nPerformance:")
print(f"Generated {num_generated} tokens in {elapsed_time:.2f} seconds.")
print(f"Tokens per second: {tokens_per_second:.2f}")