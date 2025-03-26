# DeepSeek-R1-Hospital-Assistant

## Overview
This project focuses on fine-tuning the DeepSeek R1 1.5 Billion parameter model to create a specialized hospital assistant. The goal is to develop an AI model that can effectively support healthcare professionals by providing accurate and contextually relevant information.

## Methods
The project employs advanced machine learning techniques:
- **Supervised Learning**: Training the model with labeled healthcare-related data
- **Transfer Learning**: Utilizing a pretrained DeepSeek R1 model as the base
- **LoRA (Low-Rank Adaptation)**: Efficient fine-tuning technique for model adaptation
- **Data Augmentation**: Expanding the training dataset to improve model robustness

## Prerequisites
- Python 3.8+ (3.13.2 is used)
- Python virtual environment
- Sufficient computational resources for model training

## Installation

### 1. Create Virtual Environment
```bash
python -m venv hospital_assistant_env
source hospital_assistant_env/bin/activate  # On Windows, use `hospital_assistant_env\Scripts\activate`
```

### 2. Clone Repository
```bash
git clone https://github.com/yourusername/DeepSeek-R1-Hospital-Assistant.git
cd DeepSeek-R1-Hospital-Assistant
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

## Usage

### Download Model
```bash
python download_model.py
```

### Optional: Test Downloaded Model
```bash
python test_model.py
```

### Fine-Tuning (Optional)
```bash
python finetune_model.py
```
*Note: The model has already been fine-tuned and saved in the `finedtuned_model` directory.*

### Use Fine-Tuned Model
```bash
python use_fine_tuned_model.py
```

## Model Details
- **Base Model**: DeepSeek R1
- **Parameters**: 1.5 Billion
- **Specialized Domain**: Hospital and Healthcare Assistance

## Limitations
- Model performance depends on the quality and diversity of the training data
- Requires careful validation in real-world healthcare scenarios
- Not a substitute for professional medical advice

## Future Work
- Expand training dataset
- Improve domain-specific performance
- Add more specialized healthcare capabilities

## Contributing
Contributions, issues, and feature requests are welcome. Feel free to check the issues page.

## License
Apache 2.0

## Acknowledgments
- DeepSeek R1 1.5B parameter for the base model
- Hugging Face