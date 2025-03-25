from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Directory where your fine-tuned model is saved
model_dir = "./finetuned_model"

# Load the fine-tuned tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_dir, trust_remote_code=True)

# Move the model to GPU if available for faster inference
# if torch.cuda.is_available():
#     model.to("cuda")

# Refined prompt with clear instructions
prompt = "### Instruction: When can I come to the hospital?\n### Response:"


# Tokenize the prompt
inputs = tokenizer(prompt, return_tensors="pt")
# if torch.cuda.is_available():
#     inputs = {k: v.to("cuda") for k, v in inputs.items()}

# Generate a response with adjusted parameters
outputs = model.generate(
    **inputs,
    max_new_tokens=50,     # Increase if necessary
    temperature=0.6,        # Lower temperature for less randomness
    do_sample=True         # Use greedy decoding for more deterministic output
)

# Decode and print the output
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("Generated output:")
print(generated_text)
