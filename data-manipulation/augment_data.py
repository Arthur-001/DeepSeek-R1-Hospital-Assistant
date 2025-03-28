import json
import random
import nltk
from nltk.corpus import wordnet
import os
import logging

# Ensure you have the WordNet corpus downloaded
nltk.download('wordnet')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def synonym_replacement(words):
    """Replace a word with a synonym."""
    new_words = words.copy()
    for i, word in enumerate(words):
        synonyms = wordnet.synsets(word)
        if synonyms:
            synonym = random.choice(synonyms).lemmas()[0].name()  # Get the first synonym
            new_words[i] = synonym.replace('_', ' ')  # Replace underscores with spaces
    return new_words

def random_insertion(words):
    """Insert a random word into the sentence."""
    new_words = words.copy()
    random_word = random.choice(words)  # Choose a random word from the sentence
    new_words.insert(random.randint(0, len(new_words)), random_word)  # Insert it at a random position
    return new_words

def augment_data(input_file, output_file):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    datasetJSON = os.path.join(script_dir, input_file)

    # Load dataset
    with open(datasetJSON, "r", encoding="utf-8") as file:
        data = json.load(file)

    # Apply augmentation
    augmented_data = []

    for entry in data:
        question = entry["question"]
        answer = entry["answer"]

        # Skip very short entries
        if len(question.split()) < 3 or len(answer.split()) < 3:
            augmented_data.append(entry)
            continue

        # Tokenize the question and answer
        question_words = question.split()
        answer_words = answer.split()

        # Generate augmented versions
        augmented_data.append(entry)  # Add original entry

        # Synonym Replacement
        aug_question_syn = ' '.join(synonym_replacement(question_words))
        aug_answer_syn = ' '.join(synonym_replacement(answer_words))
        augmented_data.append({"question": aug_question_syn, "answer": aug_answer_syn})

        # Random Insertion
        aug_question_ins = ' '.join(random_insertion(question_words))
        aug_answer_ins = ' '.join(random_insertion(answer_words))
        augmented_data.append({"question": aug_question_ins, "answer": aug_answer_ins})

    # Save augmented dataset in the same directory as the script
    output_path = os.path.join(script_dir, output_file)
    with open(output_path, "w", encoding="utf-8") as outfile:
        json.dump(augmented_data, outfile, indent=2, ensure_ascii=False)

    logger.info(f"Data augmentation completed. Saved to {output_path}")

if __name__ == "__main__":
    augment_data("dataset.json", "augmented_dataset.json")