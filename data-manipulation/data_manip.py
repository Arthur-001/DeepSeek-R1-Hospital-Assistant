import json
import os

# Get the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
print(script_dir)
# Create the full path to the raw-data.txt file
raw_data_path = os.path.join(script_dir, "raw-data.txt")

# Read the text file
with open(raw_data_path, "r", encoding="utf-8") as file:
    data = file.read()

# Parse Q&A format into JSON structure
qa_pairs = []
current_qa = {}
seen_pairs = set()  # To track unique Q&A pairs

for line in data.strip().split('\n'):
    line = line.strip()
    if not line:  # Skip empty lines
        continue
    if line.startswith('Question: '):
        if current_qa:  # Save previous Q&A pair if exists
            # Create a tuple of question and answer for comparison
            qa_tuple = (current_qa['question'], current_qa['answer'])
            if qa_tuple not in seen_pairs:
                seen_pairs.add(qa_tuple)
                qa_pairs.append(current_qa)
            else:
                print(f"Duplicate Q&A pair found and skipped: {current_qa['question']}")
        current_qa = {'question': line[len('Question: '):]}
    elif line.startswith('Answer: '):
        current_qa['answer'] = line[len('Answer: '):]

# Add the last Q&A pair if it's unique
if current_qa:
    qa_tuple = (current_qa['question'], current_qa['answer'])
    if qa_tuple not in seen_pairs:
        qa_pairs.append(current_qa)
    else:
        print(f"Duplicate Q&A pair found and skipped: {current_qa['question']}")

# Print summary
print(f"\nTotal unique Q&A pairs: {len(qa_pairs)}")
print(f"Duplicates removed: {len(seen_pairs) - len(qa_pairs)}")

# Save as JSON file
output_path = os.path.join(script_dir, "dataset.json")
with open(output_path, "w", encoding="utf-8") as json_file:
    json.dump(qa_pairs, json_file, indent=2, ensure_ascii=False)

print("\nDataset successfully converted to JSON format.")
