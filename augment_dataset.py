import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('punkt')  # Added this as it's often needed for tokenization
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.char as nac
import nlpaug.augmenter.sentence as nas
import nlpaug.flow as naf
import random

qa_pairs = [
    {
        "question": "What are the hospital visiting hours?",
        "answer": "The hospital visiting hours are from 9 AM to 7 PM every day."
    },
    {
        "question": "How can I book an appointment with a doctor?",
        "answer": "You can book an appointment with a doctor by calling the hospital's reception or using our online booking system on the hospital website."
    },
    {
        "question": "Do you offer emergency services?",
        "answer": "Yes, the hospital provides 24/7 emergency services. You can visit the emergency room or call the emergency number for assistance."
    },
    {
        "question": "Where is the pharmacy located within the hospital?",
        "answer": "The pharmacy is located on the ground floor near the main entrance."
    },
    {
        "question": "Can I get my medical test results online?",
        "answer": "Yes, you can access your medical test results online through our secure patient portal."
    },
    {
        "question": "How do I get a referral to see a specialist?",
        "answer": "You can get a referral from your primary care doctor or from any of our medical professionals after a consultation."
    },
    {
        "question": "What should I bring to my appointment?",
        "answer": "Please bring your ID, insurance card, medical history, and any relevant test results to your appointment."
    },
    {
        "question": "What are the hospital's COVID-19 protocols?",
        "answer": "All patients and visitors are required to wear masks, maintain social distancing, and use hand sanitizers. Temperature checks will be conducted at the entrance."
    },
    {
        "question": "How can I pay my medical bills?",
        "answer": "You can pay your medical bills in person at the billing department, through our online portal, or via phone using a credit card."
    },
    {
        "question": "Is there parking available at the hospital?",
        "answer": "Yes, there is parking available in the hospital parking lot. We also offer valet services for convenience."
    },
    {
        "question": "What is the process for discharging a patient from the hospital?",
        "answer": "Discharge processes usually start after the doctor has determined the patient is ready. Our staff will provide instructions for home care, medication, and follow-up appointments."
    },
    {
        "question": "Do you have pediatric care available?",
        "answer": "Yes, we have pediatric specialists available to care for children of all ages."
    },
    {
        "question": "How do I contact a specific doctor?",
        "answer": "You can contact a specific doctor through the hospital reception, or by reaching out to their office directly if they have private practice arrangements."
    },
    {
        "question": "Are there any support services for patients recovering from surgery?",
        "answer": "Yes, we offer physical therapy, counseling, and home healthcare services for patients recovering from surgery."
    },
    {
        "question": "What should I do if I need to cancel an appointment?",
        "answer": "If you need to cancel an appointment, please call the hospital at least 24 hours in advance or use the cancellation feature on our online booking system."
    },
    {
        "question": "Do you offer maternity services?",
        "answer": "Yes, we offer maternity services, including prenatal care, delivery, and postnatal care."
    }
]


# Configure augmenters with error handling
def create_augmenter(aug_type):
    try:
        if aug_type == "synonym":
            return naw.SynonymAug(aug_src='wordnet')
        elif aug_type == "swap":
            return naw.RandomWordAug(action="swap")
        elif aug_type == "insert":
            return naw.ContextualWordEmbsAug(model_path='distilbert-base-uncased', action="insert")
        elif aug_type == "char":
            return nac.KeyboardAug()
    except Exception as e:
        print(f"Error creating augmenter {aug_type}: {str(e)}")
        return None

def augment_text(text, augmenter, n=3):
    try:
        if augmenter is None:
            return [text] * n  # Return original text if augmenter failed
        return [augmenter.augment(text)[0] for _ in range(n)]  # Note the [0] to get string instead of list
    except Exception as e:
        print(f"Error during augmentation: {str(e)}")
        return [text] * n

# Apply augmentations with better error handling
augmentation_types = {
    "synonym": ("augmented_questions", "augmented_answers"),
    "swap": ("swapped_questions", "swapped_answers"),
    "insert": ("inserted_questions", "inserted_answers"),
    "char": ("char_aug_questions", "char_aug_answers")
}

for aug_type, (q_key, a_key) in augmentation_types.items():
    print(f"\nApplying {aug_type} augmentation...")
    augmenter = create_augmenter(aug_type)
    
    for qa in qa_pairs:
        qa[q_key] = augment_text(qa["question"], augmenter)
        qa[a_key] = augment_text(qa["answer"], augmenter)
        
        # Print results for verification
        print("\nOriginal Question:", qa["question"])
        print(f"{aug_type.capitalize()} Questions:", qa[q_key])
        print("Original Answer:", qa["answer"])
        print(f"{aug_type.capitalize()} Answers:", qa[a_key])
        print("-" * 50)

# Save the augmented dataset
import json

with open("augmented_qa_dataset.txt", "w", encoding="utf-8") as f:
    json.dump(qa_pairs, f, indent=4, ensure_ascii=False)
