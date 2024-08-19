import warnings

# Suppress specific warnings
warnings.filterwarnings("ignore", message=".*parameter name that contains `beta` will be renamed.*")
warnings.filterwarnings("ignore", message=".*parameter name that contains `gamma` will be renamed.*")

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertTokenizer, BertModel
import torch

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')


def get_embeddings(text):
    """
    Get BERT embeddings for a given text.
    """
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    # Use the mean of the last hidden states as the sentence embeddings
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings.numpy()


def compute_similarity(text1, text2):
    """
    Compute cosine similarity between two texts.
    """
    # Get embeddings for both texts
    embedding1 = get_embeddings(text1)
    embedding2 = get_embeddings(text2)

    # Compute cosine similarity
    similarity = cosine_similarity(embedding1, embedding2)
    return similarity[0][0]


# Example ground truth and generated text
ground_truth_text = r'C:/Users/BASVOJU/Desktop/Master_thesis/Transcriptions/ground_truth.txt'
transcribed_file_path = r'C:/Users/BASVOJU/Desktop/Master_thesis/Transcriptions/transcribed_text.txt'


# Compute and print similarity
similarity_score = compute_similarity(ground_truth_text, transcribed_file_path)
print(f"Cosine Similarity: {similarity_score:.4f}")
