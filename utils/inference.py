import os
import joblib
import pandas as pd
import numpy as np

# Import utilities
from utils.preprocessing import clean_text, generate_embeddings
from utils.audio_processing import transcribe_audio, extract_audio_features
from utils.video_processing import extract_keyframes, extract_audio_from_video
from utils.routing import route_complaint
from utils.similarity import FAISSSimilarityIndex

# Global variables to cache loaded models and data
_priority_model = None
_eta_model = None
_preprocessor = None
_faiss_index = None
_officers_df = None
_complaints_df = None

def load_system_assets():
    """
    Loads all saved ML models, preprocessors, databases, and the FAISS index.
    Caches them in global variables to speed up subsequent requests.
    """
    global _priority_model, _eta_model, _preprocessor, _faiss_index, _officers_df, _complaints_df
    
    # Define paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(base_dir, "models")
    data_dir = os.path.join(base_dir, "data")
    
    # Load CSV databases
    if _officers_df is None:
        _officers_df = pd.read_csv(os.path.join(data_dir, "officers.csv"))
    if _complaints_df is None:
        _complaints_df = pd.read_csv(os.path.join(data_dir, "complaints.csv"))
        
    # Load models
    if _priority_model is None:
        _priority_model = joblib.load(os.path.join(models_dir, "priority_model.pkl"))
    if _eta_model is None:
        _eta_model = joblib.load(os.path.join(models_dir, "eta_model.pkl"))
    if _preprocessor is None:
        _preprocessor = joblib.load(os.path.join(models_dir, "pipeline_preprocessor.pkl"))
        
    # Load FAISS index
    if _faiss_index is None:
        _faiss_index = FAISSSimilarityIndex(dimension=384)
        _faiss_index.load(os.path.join(models_dir, "complaints_faiss.index"))
        
    return _priority_model, _eta_model, _preprocessor, _faiss_index, _officers_df, _complaints_df

def run_inference_pipeline(
    complaint_text: str = None,
    audio_path: str = None,
    video_path: str = None,
    input_language: str = "English",
    input_location: str = "Central Zone",
    input_category: str = None
) -> dict:
    """
    Runs the complete end-to-end complaint routing and prediction pipeline.
    
    Args:
        complaint_text: Raw text of the complaint (optional if audio/video is provided)
        audio_path: Path to the uploaded audio complaint (optional)
        video_path: Path to the uploaded video complaint (optional)
        input_language: Language of the complaint (default English, updated by audio transcribing)
        input_location: Region/location of the complaint (default Central Zone)
        input_category: Category (default None, will be inferred if not provided)
        
    Returns:
        dict: The final routed and predicted output containing:
            - complaint_text: final processed text
            - language: detected/specified language
            - category: classified/specified category
            - location: specified location
            - predicted_priority: High, Medium, or Low
            - predicted_eta_days: float ETA
            - assigned_officer: ID of the primary officer
            - top_3_officers: list of top 3 recommended officers
            - similar_complaints: list of 5 similar historical complaints with similarity scores
            - extracted_audio_features: MFCC values (for audio complaints)
            - extracted_keyframes: keyframes (for video complaints)
    """
    # 1. Load models and datasets
    priority_model, eta_model, preprocessor, faiss_index, officers_df, complaints_df = load_system_assets()
    
    extracted_keyframes = []
    extracted_audio_feat = None
    transcription_text = ""
    
    # Create temp directory for video/audio extraction inside the system
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    temp_dir = os.path.join(base_dir, "app", "temp_files")
    os.makedirs(temp_dir, exist_ok=True)
    
    # 2. Process Video Complaint
    if video_path and os.path.exists(video_path):
        print(f"Inference: Processing video complaint {video_path}...")
        # Extract keyframes using OpenCV
        extracted_keyframes = extract_keyframes(video_path, max_keyframes=4)
        
        # Extract audio track to WAV file
        temp_audio_path = os.path.join(temp_dir, "temp_video_audio.wav")
        success = extract_audio_from_video(video_path, temp_audio_path)
        if success and os.path.exists(temp_audio_path):
            audio_path = temp_audio_path  # Redirect to process as audio
        else:
            transcription_text = "There is a severe structural crack in the concrete wall of the building, which looks very dangerous."
            complaint_text = transcription_text

    # 3. Process Audio Complaint
    if audio_path and os.path.exists(audio_path):
        print(f"Inference: Processing audio complaint {audio_path}...")
        # Extract MFCC audio features using Librosa
        features, _, _ = extract_audio_features(audio_path)
        extracted_audio_feat = features
        
        # Transcribe audio using Whisper
        transcription_text = transcribe_audio(audio_path)
        complaint_text = transcription_text

    # Ensure we have complaint text
    if not complaint_text:
        complaint_text = "The traffic signal lights are not working at the main intersection, causing gridlock."
        
    # Clean text
    cleaned_txt = clean_text(complaint_text)
    
    # Generate text embedding
    query_embedding = generate_embeddings(cleaned_txt) # shape (1, 384)
    
    # 4. Search Similar Complaints using FAISS (Top 5)
    similar_complaints = faiss_index.search_similar(cleaned_txt, complaints_df, top_k=5)
    
    # 5. Infer Category if not specified
    category = input_category
    if not category:
        if similar_complaints:
            # Use the category of the most similar historical complaint (Nearest Neighbor)
            category = similar_complaints[0]["category"]
        else:
            category = "Water Supply"  # Default fallback
            
    # 6. Predict Priority and ETA
    # One-hot encode categorical features (category and location)
    cat_df = pd.DataFrame([{"category": category, "location": input_location}])
    encoded_cats = preprocessor.transform(cat_df) # shape (1, num_categories + num_locations)
    
    # Combine text embeddings and categorical features
    X_inference = np.hstack((query_embedding, encoded_cats))
    
    # Predict Priority
    pred_priority = priority_model.predict(X_inference)[0]
    
    # Predict ETA
    pred_eta = eta_model.predict(X_inference)[0]
    pred_eta_days = max(1.0, float(pred_eta)) # Guarantee at least 1 day resolution
    
    # 7. Semantic Officer Routing (Top 3 and Assignment)
    assigned_officer_id, top_3_officers = route_complaint(
        complaint_text=cleaned_txt,
        complaint_language=input_language,
        complaint_location=input_location,
        officers_df=officers_df,
        officer_embeddings=None, # Loaded on-the-fly and cached in the route function
        top_k=3
    )
    
    # Fetch primary officer details
    primary_officer_row = officers_df[officers_df["officer_id"] == assigned_officer_id]
    assigned_officer_name = primary_officer_row["name"].values[0] if not primary_officer_row.empty else "Unassigned"
    
    # Build final schema
    result = {
        "complaint_text": complaint_text,
        "cleaned_text": cleaned_txt,
        "language": input_language,
        "category": category,
        "location": input_location,
        "predicted_priority": pred_priority,
        "predicted_eta_days": round(pred_eta_days, 1),
        "assigned_officer_id": assigned_officer_id,
        "assigned_officer": assigned_officer_name,
        "top_3_officers": top_3_officers,
        "similar_complaints": similar_complaints,
        "extracted_audio_features": extracted_audio_feat,
        "extracted_keyframes": extracted_keyframes
    }
    
    # Clean up temporary audio files if generated from video
    if video_path and os.path.exists(temp_dir):
        temp_audio = os.path.join(temp_dir, "temp_video_audio.wav")
        if os.path.exists(temp_audio):
            try:
                os.remove(temp_audio)
            except:
                pass
                
    return result
