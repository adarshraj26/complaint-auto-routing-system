import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from utils.preprocessing import generate_embeddings

def build_officer_profile(row) -> str:
    """
    Constructs a rich text profile for an officer to be used for semantic embedding.
    """
    # Handle list of languages stored as string representation or list
    langs = row['languages']
    if isinstance(langs, str):
        if langs.startswith('[') and langs.endswith(']'):
            import ast
            try:
                langs = ", ".join(ast.literal_eval(langs))
            except:
                pass
    elif isinstance(langs, list):
        langs = ", ".join(langs)
        
    profile = (
        f"Department: {row['department']}. "
        f"Specialization: {row['specialization']}. "
        f"Region: {row['region']}. "
        f"Languages: {langs}. "
        f"Experience: {row['experience_years']} years."
    )
    return profile

def get_officer_embeddings(officers_df: pd.DataFrame) -> np.ndarray:
    """
    Generates embeddings for all officers' profiles.
    In a real system, these would be precalculated and cached.
    """
    profiles = officers_df.apply(build_officer_profile, axis=1).tolist()
    return generate_embeddings(profiles)

def route_complaint(
    complaint_text: str,
    complaint_language: str,
    complaint_location: str,
    officers_df: pd.DataFrame,
    officer_embeddings: np.ndarray = None,
    top_k: int = 3
) -> tuple:
    """
    Routes a complaint to the most suitable officers.
    Applies multi-factor scoring:
    1. Semantic Similarity: Cosine similarity between complaint text and officer profiles.
    2. Language Filter: Heavy penalty if the officer does not speak the complaint's language.
    3. Regional Match: Bonus/penalty based on whether the officer operates in the complaint's region.
    4. Workload Balancing: Penalty for high workload scores to prevent officer burnout.
    
    Returns:
        assigned_officer_id: ID of the primary assigned officer (highest score).
        top_officers_list: List of top_k dicts with details of recommended officers.
    """
    if officers_df.empty:
        return None, []

    # Get embeddings
    complaint_embedding = generate_embeddings(complaint_text) # shape (1, 384)
    
    if officer_embeddings is None:
        officer_embeddings = get_officer_embeddings(officers_df) # shape (N, 384)

    # Compute raw cosine similarities
    similarities = cosine_similarity(complaint_embedding, officer_embeddings)[0]

    scored_officers = []
    
    for idx, row in officers_df.iterrows():
        base_similarity = float(similarities[idx])
        
        # 1. Language matching
        langs = row['languages']
        if isinstance(langs, str):
            if langs.startswith('[') and langs.endswith(']'):
                import ast
                try:
                    langs = ast.literal_eval(langs)
                except:
                    langs = [langs]
            else:
                langs = [l.strip() for l in langs.split(',')]
        elif not isinstance(langs, list):
            langs = [str(langs)]
            
        # Check language match (case insensitive)
        lang_match = any(complaint_language.lower() in l.lower() for l in langs)
        lang_multiplier = 1.0 if lang_match else 0.4  # Severe penalty if they can't communicate
        
        # 2. Regional matching
        region_match = str(row['region']).lower() == str(complaint_location).lower()
        region_multiplier = 1.0 if region_match else 0.8  # Moderate penalty for out-of-region
        
        # 3. Workload balancing
        # Workload score is typically 0-100. Let's penalize higher workload scores.
        # multiplier = 1.0 - (workload / 200.0) -> if workload is 100, multiplier is 0.5.
        workload = float(row['workload_score'])
        workload_multiplier = max(0.2, 1.0 - (workload / 150.0))
        
        # Calculate final routing score
        final_score = base_similarity * lang_multiplier * region_multiplier * workload_multiplier
        
        scored_officers.append({
            'officer_id': row['officer_id'],
            'name': row['name'],
            'department': row['department'],
            'specialization': row['specialization'],
            'languages': row['languages'],
            'experience_years': int(row['experience_years']),
            'region': row['region'],
            'workload_score': float(row['workload_score']),
            'semantic_similarity': base_similarity,
            'routing_score': final_score
        })
        
    # Sort by routing score in descending order
    scored_officers_df = pd.DataFrame(scored_officers)
    scored_officers_df = scored_officers_df.sort_values(by='routing_score', ascending=False).reset_index(drop=True)
    
    # Retrieve top K
    top_k_records = scored_officers_df.head(top_k).to_dict(orient='records')
    assigned_officer_id = top_k_records[0]['officer_id'] if top_k_records else None
    
    return assigned_officer_id, top_k_records
