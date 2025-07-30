from nltk.tokenize import sent_tokenize
from transformers import AutoTokenizer
import warnings

# Initialize tokenizer once globally to avoid reloading
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")

# In utils/chunking.py
def chunk_text(text, max_chunk_size=400):
    """
    Split text into chunks of specified maximum size.
    
    Args:
        text (str): Input text to be chunked
        max_chunk_size (int): Maximum size of each chunk (default: 400)
    
    Returns:
        list: List of text chunks
    """
    chunks = []
    words = text.split()
    current_chunk = []
    current_size = 0
    
    for word in words:
        word_size = len(word) + 1  # +1 for space
        
        if current_size + word_size > max_chunk_size and current_chunk:
            chunks.append(' '.join(current_chunk))
            current_chunk = []
            current_size = 0
            
        current_chunk.append(word)
        current_size += word_size
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
        
    return chunks