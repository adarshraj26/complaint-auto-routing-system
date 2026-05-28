import re
import numpy as np
from sentence_transformers import SentenceTransformer

# Global model cache to avoid reloading
_model = None

def get_transformer_model():
    """
    Lazy loads the multilingual sentence transformer model.
    Using paraphrase-multilingual-MiniLM-L12-v2 which supports 50+ languages,
    essential for multilingual complaint processing.
    """
    global _model
    if _model is None:
        # Downloads model from Hugging Face if not cached, then loads it
        _model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    return _model

def clean_text(text: str) -> str:
    """
    Cleans complaint text by:
    - Converting to lowercase
    - Keeping alphanumeric characters, common punctuation, and international alphabet ranges
      (Hindi, Chinese, Spanish, French, etc.)
    - Collapsing multiple whitespaces
    """
    if not isinstance(text, str):
        return ""
    
    text = text.lower()
    
    # Keep alphanumeric characters, spaces, and basic punctuation, supporting Unicode letters
    # This preserves Hindi (\u0900-\u097F), Chinese (\u4e00-\u9fff), accented characters, etc.
    text = re.sub(
        r'[^a-zA-Z0-9\s\.,!\?\u00C0-\u017F\u0400-\u04FF\u0900-\u097F\u4e00-\u9fff]', 
        '', 
        text
    )
    
    # Replace multiple whitespaces with a single space
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def generate_embeddings(texts):
    """
    Generates text embeddings for a single text or a list of texts.
    Returns:
        numpy.ndarray of shape (num_texts, 384)
    """
    model = get_transformer_model()
    
    # If a single string is passed, wrap it in a list
    if isinstance(texts, str):
        texts = [texts]
        
    embeddings = model.encode(texts, show_progress_bar=False)
    return np.array(embeddings)
