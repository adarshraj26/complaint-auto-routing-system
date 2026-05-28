import os
import faiss
import numpy as np
import pandas as pd
from utils.preprocessing import generate_embeddings

class FAISSSimilarityIndex:
    """
    Manages building, saving, loading, and searching a FAISS index 
    for complaint embeddings, enabling fast semantic similarity search.
    """
    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.index = None
        
    def build_and_add(self, embeddings: np.ndarray):
        """
        Builds a FAISS index using Inner Product (IndexFlatIP).
        By normalizing input embeddings to unit length, Inner Product 
        corresponds exactly to Cosine Similarity.
        """
        # Ensure float32 format for FAISS
        embeddings_f32 = embeddings.astype('float32')
        
        # Normalize vectors in place for cosine similarity
        faiss.normalize_L2(embeddings_f32)
        
        # Initialize IndexFlatIP
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(embeddings_f32)
        
    def save(self, file_path: str):
        """
        Saves the FAISS index to a local file.
        """
        if self.index is None:
            raise ValueError("Cannot save index: Index has not been built.")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        faiss.write_index(self.index, file_path)
        print(f"FAISS index saved successfully to {file_path}")

    def load(self, file_path: str) -> bool:
        """
        Loads a FAISS index from a local file.
        """
        if not os.path.exists(file_path):
            print(f"FAISS index file {file_path} not found.")
            return False
        self.index = faiss.read_index(file_path)
        print(f"FAISS index loaded successfully from {file_path}")
        return True

    def search_similar(
        self, 
        query_text: str, 
        complaints_df: pd.DataFrame, 
        top_k: int = 5
    ) -> list:
        """
        Embeds the query text, searches the FAISS index, and retrieves 
        the top_k most similar complaints along with their metadata.
        """
        if self.index is None:
            print("FAISS index is not initialized.")
            return []
            
        if complaints_df.empty:
            print("Complaints database is empty.")
            return []
            
        # Generate embedding for the query
        query_emb = generate_embeddings(query_text).astype('float32') # (1, 384)
        
        # Normalize query vector for cosine similarity
        faiss.normalize_L2(query_emb)
        
        # Perform FAISS search
        # scores: cosine similarity values, indices: row index in complaints_df
        scores, indices = self.index.search(query_emb, min(top_k, len(complaints_df)))
        
        results = []
        for i in range(len(indices[0])):
            idx = int(indices[0][i])
            score = float(scores[0][i])
            
            # FAISS returns -1 for empty/invalid slots
            if idx == -1 or idx >= len(complaints_df):
                continue
                
            complaint_row = complaints_df.iloc[idx].to_dict()
            complaint_row['similarity_score'] = round(score, 4)
            results.append(complaint_row)
            
        return results
