import json
import os
import time
from typing import List, Dict

# Global Configuration Variables (for easy modification)
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # Language model to use
EMBEDDINGS_MODEL_NAME = "all-MiniLM-L6-v2"  # Embeddings model

# File and directory paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(BASE_DIR, "data-manipulation", "dataset.json")
VECTORSTORE_PATH = "hospital_assistant_index"  # Path for the vector store

# Model parameters
DEVICE_MAP = "auto"  # Device to use for computation (auto, cpu, cuda:0, etc.)
TORCH_DTYPE = "float16"  # Data type for model (float16, float32, etc.)

# RAG parameters
TOP_K_RESULTS = 3  # Number of documents to retrieve
SIMILARITY_SCORE_THRESHOLD = 0.7  # Minimum similarity score for retrieval

# Prompt settings
PROMPT_TEMPLATE = """
<|system|>You are a helpful hospital assistant. Use the context provided to answer the user's question accurately and concisely. Do not add any information that is not in the context.</s>
<|user|>Context:
{context}
Question: {question}</s>
<|assistant|></s>
"""

# UI settings
UI_WELCOME_MESSAGE = "Welcome to the Hospital Assistant Chatbot!"
UI_INSTRUCTIONS = "Ask me anything about the hospital (type 'exit' to quit)."
UI_METRICS_DISPLAY = True  # Whether to display performance metrics

# Global exit words
EXIT_WORDS = {"exit", "bye", "goodbye"}


# 1. Load the Dataset
def load_dataset(file_path: str) -> List[Dict[str, str]]:
    """
    Loads the dataset from a JSON file.

    Args:
        file_path (str): The path to the JSON file.

    Returns:
        List[Dict[str, str]]: A list of dictionaries, where each dictionary
        contains a "question" and an "answer". Returns an empty list on error.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except FileNotFoundError as e:
        print(f"Error: File not found at {file_path}. Details: {e}")
        return []
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON format in {file_path}. Details: {e}")
        return []
    except Exception as e:
        print(f"An unexpected error occurred while loading the dataset: {e}")
        return []


# 2. Prepare Data for Indexing
def prepare_data(dataset: List[Dict[str, str]]) -> List[str]:
    """
    Prepares the data for indexing by extracting the answers.

    Args:
        dataset (List[Dict[str, str]]): The dataset loaded from the JSON file.

    Returns:
        List[str]: A list of strings, where each string is an answer from the dataset.
    """
    try:
        return [item["answer"] for item in dataset]
    except Exception as e:
        print(f"Error preparing data: {e}")
        return []


# 3. Create Embeddings and Vector Store
def create_vector_store(data: List[str], vectorstore_path: str, embeddings_model_name: str):
    """
    Creates a vector store from the given data using Hugging Face embeddings
    and stores it on disk. Handles the case where the vectorstore_path already
    exists (loads from disk).

    Args:
        data (List[str]): The list of text data to embed.
        vectorstore_path (str): The path to save the vector store.
        embeddings_model_name (str): The name of the Hugging Face model
            to use for embeddings.
    """
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS

    try:
        embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
        if os.path.exists(vectorstore_path):
            # Load existing vector store
            print(f"Loading vector store from {vectorstore_path}")
            vectorstore = FAISS.load_local(vectorstore_path, embeddings,
                                          allow_dangerous_deserialization=True)
        else:
            # Create new vector store
            print("Creating new vector store...")
            vectorstore = FAISS.from_texts(data, embeddings)
            vectorstore.save_local(vectorstore_path)
        return vectorstore
    except Exception as e:
        print(f"Error creating or loading vector store: {e}")
        return None



# 4. Load the Language Model
def load_model(model_name: str, torch_dtype: str = TORCH_DTYPE,
               device_map: str = DEVICE_MAP):
    """
    Loads the language model from Hugging Face Transformers.

    Args:
        model_name (str): The name of the model to load.
        torch_dtype (str): The data type to use (float16, float32)
        device_map (str): The device to use for computation

    Returns:
        pipeline: The loaded pipeline. Returns None on error.
    """
    from transformers import pipeline, AutoTokenizer
    import torch

    try:
        # Convert string dtype to torch dtype
        dtype_map = {
            "float16": torch.float16,
            "float32": torch.float32,
            "bfloat16": torch.bfloat16
        }
        torch_dtype_obj = dtype_map.get(torch_dtype, torch.float16)

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        pipe = pipeline(
            "text-generation",
            model=model_name,
            tokenizer=tokenizer,
            torch_dtype=torch_dtype_obj,
            device_map=device_map,
        )
        return pipe, tokenizer
    except Exception as e:
        print(f"Error loading model {model_name}: {e}")
        return None, None


# 5. Create RAG Pipeline
def create_rag_pipeline(model, vectorstore, top_k: int = TOP_K_RESULTS,
                      score_threshold: float = SIMILARITY_SCORE_THRESHOLD):
    """
    Creates a RAG pipeline that combines the language model and the vector store.

    Args:
        model: The loaded language model.
        vectorstore: The vector store.
        top_k (int): Number of documents to retrieve
        score_threshold (float): Minimum similarity score for retrieval

    Returns:
        function: A function that takes a question and returns an answer using RAG.
                        Returns None on error.
    """
    from langchain.prompts import PromptTemplate
    from langchain.chains import RetrievalQA
    from langchain_community.llms import HuggingFacePipeline
    from langchain_core.runnables import RunnablePassthrough
    from langchain_core.output_parsers import StrOutputParser

    try:
        llm = HuggingFacePipeline(pipeline=model)
        prompt = PromptTemplate.from_template(PROMPT_TEMPLATE)

        # Configure retriever with parameters
        retriever = vectorstore.as_retriever(
            search_kwargs={"k": top_k, "score_threshold": score_threshold}
        )

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        rag_chain = (
            {"context": retriever | format_docs,
             "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        return rag_chain
    except Exception as e:
        print(f"Error creating RAG pipeline: {e}")
        return None


# 6. Chatbot Interaction
def run_chatbot(rag_pipeline, tokenizer, show_metrics: bool = UI_METRICS_DISPLAY):
    """
    Runs the chatbot in a loop, taking user input and providing answers using RAG.

    Args:
        rag_pipeline: The RAG pipeline function.
        tokenizer: The tokenizer for accurate token counting.
        show_metrics (bool): Whether to display performance metrics
    """
    if rag_pipeline is None:
        print("Error: RAG pipeline is not initialized. Exiting chatbot.")
        return

    print("\n" + "=" * 50)
    print(f"{UI_WELCOME_MESSAGE:^50}")
    print(f"{UI_INSTRUCTIONS:^50}")
    print("=" * 50 + "\n")

    while True:
        question = input("\033[1mYou:\033[0m ")
        if question.lower() in EXIT_WORDS:
            print("Goodbye!")
            break

        try:
            print("\033[1mAssistant:\033[0m", end=" ", flush=True)

            # Track generation statistics
            start_time = time.time()
            result = rag_pipeline.invoke(question)
            end_time = time.time()

            # Handle different result types
            if isinstance(result, dict):
                # If result is a dictionary, try to extract text from it
                if "generated_text" in result:
                    text_result = result["generated_text"]
                elif "text" in result:
                    text_result = result["text"]
                else:
                    # Try to convert the entire dict to a string if we can't
                    # find text keys
                    text_result = str(result)
            else:
                # If result is already a string or another format
                text_result = str(result)

            # Print the result
            print(text_result)

            if show_metrics:
                # Calculate performance metrics
                duration = end_time - start_time
                num_tokens = len(tokenizer.encode(text_result))
                tokens_per_second = num_tokens / duration if duration > 0 else 0

                # Print performance metrics in a formatted box
                print("\n" + "-" * 40)
                print(f"| {'Performance Metrics':^36} |")
                print("-" * 40)
                print(f"| {'Tokens generated:':25} {num_tokens:10} |")
                print(f"| {'Time taken:':25} {duration:.2f} sec |")
                print(f"| {'Tokens per second:':25} {tokens_per_second:.2f} |")
                print("-" * 40 + "\n")

        except Exception as e:
            print(f"Error during question answering: {e}")
            print("Sorry, I could not process your request. Please try again.")



def main():
    """
    Main function to orchestrate the loading of data, model, and running
    the chatbot.
    """
    print("\n" + "*" * 50)
    print(f"{'Hospital Assistant Chatbot':^50}")
    print("*" * 50 + "\n")

    print("Initializing system...")

    dataset = load_dataset(DATASET_PATH)
    if not dataset:
        print("Failed to load dataset. Exiting.")
        return

    data = prepare_data(dataset)
    if not data:
        print("Failed to prepare data. Exiting.")
        return

    vectorstore = create_vector_store(data, VECTORSTORE_PATH,
                                     EMBEDDINGS_MODEL_NAME)
    if vectorstore is None:
        print("Failed to create vector store. Exiting.")
        return

    print(f"Loading model: {MODEL_NAME}...")
    model, tokenizer = load_model(MODEL_NAME)
    if model is None or tokenizer is None:
        print("Failed to load the model or tokenizer. Exiting.")
        return

    print("Creating RAG pipeline...")
    rag_pipeline = create_rag_pipeline(model, vectorstore)
    if rag_pipeline is None:
        print("Failed to create the RAG pipeline. Exiting.")
        return

    print("System initialized successfully!")
    run_chatbot(rag_pipeline, tokenizer)



if __name__ == "__main__":
    main()
