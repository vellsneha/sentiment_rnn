"""
Data Preprocessing Module
Handles loading, cleaning, and preparing the IMDb dataset for training.
"""

# Fix SSL certificate verification issue on macOS
import os
import certifi
os.environ['SSL_CERT_FILE'] = certifi.where()

import re
import pickle
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.datasets import imdb

# Set random seeds for reproducibility
import random
import torch
np.random.seed(42)
random.seed(42)
torch.manual_seed(42)


def clean_text(text):
    """
    Clean and preprocess text data.
    
    Args:
        text: Raw text string
        
    Returns:
        Cleaned text string
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation and special characters
    text = re.sub(r'[^a-z0-9\s]', '', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text


def load_and_preprocess_data(vocab_size=10000, save_tokenizer=True):
    """
    Load IMDb dataset and preprocess text.
    
    Args:
        vocab_size: Maximum number of words to keep
        save_tokenizer: Whether to save tokenizer for later use
        
    Returns:
        Tuple of (train_texts, train_labels, test_texts, test_labels, tokenizer)
    """
    print("Loading IMDb dataset...")
    
    # Load the dataset with word indices
    (train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=vocab_size)
    
    # Get word index
    word_index = imdb.get_word_index()
    
    # Reverse word index to decode reviews
    reverse_word_index = {value: key for key, value in word_index.items()}
    
    # Decode reviews back to text
    def decode_review(encoded_review):
        return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])
    
    # Convert encoded reviews to text
    print("Decoding reviews to text...")
    train_texts = [decode_review(review) for review in train_data]
    test_texts = [decode_review(review) for review in test_data]
    
    # Clean all texts
    print("Cleaning text data...")
    train_texts = [clean_text(text) for text in train_texts]
    test_texts = [clean_text(text) for text in test_texts]
    
    # Create tokenizer
    print(f"Creating tokenizer with vocabulary size: {vocab_size}")
    tokenizer = Tokenizer(num_words=vocab_size, oov_token='<OOV>')
    tokenizer.fit_on_texts(train_texts)
    
    # Save tokenizer
    if save_tokenizer:
        with open('data/tokenizer.pkl', 'wb') as f:
            pickle.dump(tokenizer, f)
        print("Tokenizer saved to data/tokenizer.pkl")
    
    print(f"Vocabulary size: {len(tokenizer.word_index)}")
    print(f"Training samples: {len(train_texts)}")
    print(f"Testing samples: {len(test_texts)}")
    
    return train_texts, train_labels, test_texts, test_labels, tokenizer


def prepare_sequences(texts, labels, tokenizer, max_length):
    """
    Convert texts to padded sequences.
    
    Args:
        texts: List of text strings
        labels: Array of labels
        tokenizer: Fitted Tokenizer object
        max_length: Maximum sequence length
        
    Returns:
        Tuple of (padded_sequences, labels)
    """
    # Convert texts to sequences
    sequences = tokenizer.texts_to_sequences(texts)
    
    # Pad sequences to fixed length
    padded = pad_sequences(sequences, maxlen=max_length, padding='post', truncating='post')
    
    return padded, np.array(labels)


def get_dataset_statistics(texts):
    """
    Calculate and print dataset statistics.
    
    Args:
        texts: List of text strings
    """
    lengths = [len(text.split()) for text in texts]
    
    print("\nDataset Statistics:")
    print(f"Average review length: {np.mean(lengths):.2f} words")
    print(f"Median review length: {np.median(lengths):.0f} words")
    print(f"Min review length: {np.min(lengths)} words")
    print(f"Max review length: {np.max(lengths)} words")
    print(f"Std deviation: {np.std(lengths):.2f} words")


if __name__ == "__main__":
    # Create data directory if it doesn't exist
    import os
    os.makedirs('data', exist_ok=True)
    
    # Load and preprocess data
    train_texts, train_labels, test_texts, test_labels, tokenizer = load_and_preprocess_data()
    
    # Get statistics
    get_dataset_statistics(train_texts)
    
    # Prepare sequences for different lengths
    for seq_length in [25, 50, 100]:
        print(f"\nPreparing sequences with length {seq_length}...")
        train_seq, train_y = prepare_sequences(train_texts, train_labels, tokenizer, seq_length)
        test_seq, test_y = prepare_sequences(test_texts, test_labels, tokenizer, seq_length)
        
        # Save preprocessed data
        np.save(f'data/train_sequences_{seq_length}.npy', train_seq)
        np.save(f'data/train_labels_{seq_length}.npy', train_y)
        np.save(f'data/test_sequences_{seq_length}.npy', test_seq)
        np.save(f'data/test_labels_{seq_length}.npy', test_y)
        
        print(f"Saved sequences with shape: {train_seq.shape}")
    
    print("\nPreprocessing complete!")