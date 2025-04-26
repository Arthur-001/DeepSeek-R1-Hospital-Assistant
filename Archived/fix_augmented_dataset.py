import json

# Load the augmented dataset
with open("augmented_qa_dataset.txt", "r", encoding="utf-8") as f:
    augmented_data = json.load(f)

# Prepare the reformatted dataset
reformatted_dataset = []

# Augmentation types we want to include
aug_types = [
    ("question", "answer"),  # Original pairs
    ("augmented_questions", "augmented_answers"),
    ("swapped_questions", "swapped_answers"),
    ("inserted_questions", "inserted_answers"),
    ("char_aug_questions", "char_aug_answers")
]

# Process each QA pair
for qa_pair in augmented_data:
    for q_key, a_key in aug_types:
        # Handle original pair (string) vs augmented pairs (list)
        if isinstance(qa_pair[q_key], str):
            # Original question-answer pair
            reformatted_dataset.append({
                "instruction": qa_pair[q_key],
                "response": qa_pair[a_key]
            })
        else:
            # Augmented question-answer pairs (lists)
            for i in range(len(qa_pair[q_key])):
                reformatted_dataset.append({
                    "instruction": qa_pair[q_key][i],
                    "response": qa_pair[a_key][i]
                })

# Save the reformatted dataset
with open("deepseek_format_dataset.json", "w", encoding="utf-8") as f:
    json.dump(reformatted_dataset, f, indent=2, ensure_ascii=False)

# Print dataset statistics
print(f"Original dataset: {len(augmented_data)} Q&A pairs")
print(f"Reformatted dataset: {len(reformatted_dataset)} instruction-response pairs")