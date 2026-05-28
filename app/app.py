import os
import wave
import struct
import math
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import cv2

# Set page config for wide layout and premium title
st.set_page_config(
    page_title="AI/ML Complaint Auto-Routing System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Set matplotlib style to a clean, modern aesthetic
plt.rcParams['figure.facecolor'] = '#0e1117'
plt.rcParams['axes.facecolor'] = '#161b22'
plt.rcParams['text.color'] = '#c9d1d9'
plt.rcParams['axes.labelcolor'] = '#8b949e'
plt.rcParams['xtick.color'] = '#8b949e'
plt.rcParams['ytick.color'] = '#8b949e'

# ---------------------------------------------------------
# SAMPLE FILES GENERATOR FOR EASY TESTING
# ---------------------------------------------------------

def ensure_sample_media():
    """
    Creates dummy audio (.wav) and video (.mp4) files inside the project
    so users can instantly test the media processing pipelines.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    samples_dir = os.path.join(base_dir, "app", "sample_files")
    os.makedirs(samples_dir, exist_ok=True)
    
    sample_wav = os.path.join(samples_dir, "sample_complaint.wav")
    sample_mp4 = os.path.join(samples_dir, "sample_complaint.mp4")
    
    # 1. Generate 3-second WAV audio file (sine wave simulating audio data)
    if not os.path.exists(sample_wav):
        sample_rate = 16000
        duration = 3.0
        num_samples = int(sample_rate * duration)
        with wave.open(sample_wav, 'wb') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(sample_rate)
            for i in range(num_samples):
                # Sine wave at 440Hz
                val = int(25000.0 * math.sin(2.0 * math.pi * 440.0 * i / sample_rate))
                data = struct.pack('<h', val)
                wav_file.writeframesraw(data)
                
    # 2. Generate 3-second MP4 video file with moving text and shapes
    if not os.path.exists(sample_mp4):
        fps = 10
        duration = 3
        num_frames = fps * duration
        width, height = 640, 480
        
        # Use MP4v codec which is standard and supported across platforms
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(sample_mp4, fourcc, fps, (width, height))
        
        for frame_idx in range(num_frames):
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            frame[:, :] = [30, 20, 15]  # Deep dark blue background (BGR)
            
            # Draw moving circle to simulate visual activity
            x_pos = int(120 + frame_idx * 14)
            y_pos = int(240 + 60 * math.sin(frame_idx * 0.6))
            cv2.circle(frame, (x_pos, y_pos), 40, (230, 110, 85), -1)  # Light blue circle
            
            # Overlay texts
            cv2.putText(frame, "COMPLAINT: WATER MAIN LEAKAGE", (40, 80), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, "Maple Avenue - Flooding sidewalk", (40, 120), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)
            cv2.putText(frame, f"Simulated Frame: {frame_idx + 1}/{num_frames}", (40, 420), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 120, 120), 1)
            
            out.write(frame)
        out.release()
        
    return sample_wav, sample_mp4

# ---------------------------------------------------------
# SYSTEM IMPORTS & PIPELINE INITIALIZATION
# ---------------------------------------------------------

# Try importing the inference pipeline.
# If it fails, raise exception.
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.inference import run_inference_pipeline, load_system_assets

# ---------------------------------------------------------
# CUSTOM PREMIUM STYLING INJECTION (CSS)
# ---------------------------------------------------------

st.markdown("""
<style>
    /* Dark themes and glassmorphic cards */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    .gradient-header {
        background: linear-gradient(135deg, #00f2fe 0%, #4facfe 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
        font-size: 2.8rem;
        margin-bottom: 0.2rem;
    }
    
    .subtitle-header {
        color: #8b949e;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    .card {
        background-color: #1f2937;
        border-radius: 12px;
        padding: 1.5rem;
        border: 1px solid #374151;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 1.5rem;
        transition: transform 0.2s, border-color 0.2s;
    }
    
    .card:hover {
        transform: translateY(-2px);
        border-color: #4facfe;
    }
    
    .priority-high {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.2) 0%, rgba(220, 38, 38, 0.05) 100%);
        border: 1px solid #ef4444;
        border-radius: 8px;
        padding: 0.3rem 0.8rem;
        color: #ef4444;
        font-weight: bold;
        display: inline-block;
    }
    
    .priority-medium {
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.2) 0%, rgba(217, 119, 6, 0.05) 100%);
        border: 1px solid #f59e0b;
        border-radius: 8px;
        padding: 0.3rem 0.8rem;
        color: #f59e0b;
        font-weight: bold;
        display: inline-block;
    }
    
    .priority-low {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.2) 0%, rgba(5, 150, 105, 0.05) 100%);
        border: 1px solid #10b981;
        border-radius: 8px;
        padding: 0.3rem 0.8rem;
        color: #10b981;
        font-weight: bold;
        display: inline-block;
    }
    
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #f3f4f6;
        margin-top: 0.5rem;
    }
    
    .metric-label {
        font-size: 0.85rem;
        color: #9ca3af;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .officer-badge {
        background-color: #1e1b4b;
        border: 1px solid #4f46e5;
        border-radius: 6px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        color: #e0e7ff;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# APP LAYOUT
# ---------------------------------------------------------

# Title banner
st.markdown('<div class="gradient-header">AI/ML Complaint Auto-Routing System</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle-header">Local Intelligent Operations Command — Offline ML & Speech-to-Text</div>', unsafe_allow_html=True)

# Ensure sample files exist
sample_wav_path, sample_mp4_path = ensure_sample_media()

# Load system assets (read-only for displaying stats in sidebar)
_, _, _, _, officers_df, complaints_df = load_system_assets()

# Sidebar: System Status & Meta-Inputs
with st.sidebar:
    st.image("https://img.icons8.com/nolan/96/artificial-intelligence.png", width=70)
    st.markdown("### System Configuration")
    
    # Region & Language dropdowns
    complaint_location = st.selectbox("Submitting Location/Region", ["Central Zone", "North Zone", "South Zone", "East Zone", "West Zone"])
    complaint_language = st.selectbox("Submitting Language", ["English", "Spanish", "Hindi", "French", "Chinese"])
    
    st.markdown("---")
    st.markdown("### Historical Database Stats")
    
    # Simple Metrics inside sidebar
    col_side1, col_side2 = st.columns(2)
    with col_side1:
        st.metric("Total Officers", len(officers_df))
    with col_side2:
        st.metric("Past Cases", len(complaints_df))
        
    # Department workload chart in sidebar
    st.markdown("#### Workload by Department")
    workload_dept = officers_df.groupby("department")["workload_score"].mean().round(1)
    st.bar_chart(workload_dept)

# Tab structure for input modes
tab_text, tab_audio, tab_video = st.tabs(["✍️ Text Complaint", "🎙️ Audio Upload", "📹 Video Upload"])

complaint_submitted = False
active_text = None
active_audio = None
active_video = None

# Tab 1: Text Complaint
with tab_text:
    st.markdown("#### Describe the issue:")
    text_input = st.text_area(
        "Enter complaint details (Supports multilingual submissions):",
        placeholder="E.g., There is a massive water leak flooding the main road...",
        height=150
    )
    
    # Quick fill templates for testing
    st.markdown("💡 *Quick Fill Templates (Click to copy/paste):*")
    col_t1, col_t2, col_t3 = st.columns(3)
    with col_t1:
        st.info("**English (High Priority - Water):**\n\n'Water is gushing out of the street. The entire sidewalk is submerged, please send repair teams immediately.'")
    with col_t2:
        st.warning("**Spanish (Medium Priority - Trash):**\n\n'Pilas de basura maloliente en la acera de Linking Road. Está atrayendo ratas y moscas, por favor envíen camiones de basura.'")
    with col_t3:
        st.success("**Hindi (Low Priority - Light):**\n\n'MG Road पर स्ट्रीट लाइटों का एक पूरा ब्लॉक एक सप्ताह से बंद है। सड़क पर घने अंधेरे से निवासी असुरक्षित महसूस करते हैं।'")

    if st.button("Route Text Complaint", type="primary"):
        if text_input.strip() == "":
            st.error("Please enter complaint text.")
        else:
            active_text = text_input
            complaint_submitted = True

# Tab 2: Audio Complaint
with tab_audio:
    st.markdown("#### Upload spoken complaint audio:")
    audio_file = st.file_uploader("Upload Audio File (.wav)", type=["wav"])
    
    # Use sample wav file toggle
    st.markdown("💡 *Don't have a test audio file?*")
    use_sample_audio = st.checkbox("Use pre-generated system test WAV (16kHz sine-wave representing water leak description)")
    
    if st.button("Route Audio Complaint", type="primary"):
        if use_sample_audio:
            active_audio = sample_wav_path
            complaint_submitted = True
        elif audio_file is not None:
            # Save uploaded audio file to temp location
            temp_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp_files", audio_file.name)
            os.makedirs(os.path.dirname(temp_path), exist_ok=True)
            with open(temp_path, "wb") as f:
                f.write(audio_file.getbuffer())
            active_audio = temp_path
            complaint_submitted = True
        else:
            st.error("Please upload an audio file or select the pre-generated sample.")

# Tab 3: Video Complaint
with tab_video:
    st.markdown("#### Upload complaint video:")
    video_file = st.file_uploader("Upload Video File (.mp4)", type=["mp4"])
    
    # Use sample mp4 file toggle
    st.markdown("💡 *Don't have a test video file?*")
    use_sample_video = st.checkbox("Use pre-generated system test MP4 (Moving blue block overlay with 'WATER MAIN LEAKAGE' text)")
    
    if st.button("Route Video Complaint", type="primary"):
        if use_sample_video:
            active_video = sample_mp4_path
            complaint_submitted = True
        elif video_file is not None:
            # Save uploaded video file to temp location
            temp_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp_files", video_file.name)
            os.makedirs(os.path.dirname(temp_path), exist_ok=True)
            with open(temp_path, "wb") as f:
                f.write(video_file.getbuffer())
            active_video = temp_path
            complaint_submitted = True
        else:
            st.error("Please upload a video file or select the pre-generated sample.")

# ---------------------------------------------------------
# ROUTING & PREDICTION RESULTS DASHBOARD
# ---------------------------------------------------------

if complaint_submitted:
    with st.spinner("🧠 Local AI Pipelines executing: transcribing audio/video, running features extraction, running neural predictions, executing FAISS indices..."):
        # Run the unified inference pipeline
        results = run_inference_pipeline(
            complaint_text=active_text,
            audio_path=active_audio,
            video_path=active_video,
            input_language=complaint_language,
            input_location=complaint_location
        )
        
    st.markdown("## 📊 Complaint Intelligence Dashboard")
    
    # Display processing path details if audio or video
    if active_audio or active_video:
        st.success(f"**Speech-to-Text Transcription Result:** \"{results['complaint_text']}\"")
        
    # --- ROW 1: CORE PREDICTIONS (3 COLUMNS) ---
    col1, col2, col3 = st.columns(3)
    
    with col1:
        priority = results["predicted_priority"]
        priority_class = "priority-high" if priority == "High" else ("priority-medium" if priority == "Medium" else "priority-low")
        
        st.markdown(f"""
        <div class="card">
            <div class="metric-label">Predicted Priority</div>
            <div style="margin-top: 0.8rem;">
                <span class="{priority_class}">{priority} Severity</span>
            </div>
            <p style="margin-top: 1rem; color: #8b949e; font-size: 0.9rem;">
                Determined by ML Random Forest model based on complaint text semantics and department type.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        eta = results["predicted_eta_days"]
        st.markdown(f"""
        <div class="card">
            <div class="metric-label">Estimated Resolution ETA</div>
            <div class="metric-value">⏱️ {eta} Days</div>
            <p style="margin-top: 1.1rem; color: #8b949e; font-size: 0.9rem;">
                Predicted by ML Random Forest Regressor. Average timeline based on category and region factors.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
    with col3:
        officer_name = results["assigned_officer"]
        dept = results["category"]
        st.markdown(f"""
        <div class="card">
            <div class="metric-label">Assigned Primary Officer</div>
            <div class="metric-value" style="font-size: 1.8rem; color: #818cf8;">👤 {officer_name}</div>
            <div style="margin-top: 0.5rem; font-size: 0.95rem; color: #a5b4fc;">
                <strong>Department:</strong> {dept}
            </div>
        </div>
        """, unsafe_allow_html=True)

    # --- ROW 2: DETAILED MATCHES & SIMILAR CASES ---
    col_left, col_right = st.columns([1, 1.2])
    
    with col_left:
        st.markdown("### 👮 Top-3 Recommended Officers")
        st.markdown("Based on expertise matching, languages, region, and workload score:")
        
        for i, officer in enumerate(results["top_3_officers"]):
            rank_emoji = "🥇" if i == 0 else ("🥈" if i == 1 else "🥉")
            st.markdown(f"""
            <div class="card" style="padding: 1rem; margin-bottom: 0.8rem;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div style="font-size: 1.1rem; font-weight: bold; color: #60a5fa;">
                        {rank_emoji} {officer['name']} (ID: {officer['officer_id']})
                    </div>
                    <div style="font-size: 0.85rem; color: #34d399; font-weight: bold;">
                        Match Score: {officer['routing_score']:.3f}
                    </div>
                </div>
                <div style="margin-top: 0.5rem; font-size: 0.9rem; color: #e5e7eb;">
                    <strong>Specialization:</strong> {officer['specialization']} in <strong>{officer['department']}</strong>
                </div>
                <div style="display: flex; gap: 1.5rem; margin-top: 0.5rem; font-size: 0.85rem; color: #9ca3af;">
                    <div>📍 Region: {officer['region']}</div>
                    <div>💼 Current Workload: {officer['workload_score']:.0f}</div>
                    <div>🗣️ Languages: {officer['languages']}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
    with col_right:
        st.markdown("### 🔍 Top-5 Similar Past Complaints (FAISS)")
        st.markdown("Retrieved in sub-milliseconds from database embeddings index:")
        
        for k, comp in enumerate(results["similar_complaints"]):
            score = comp["similarity_score"]
            # Color indicator for similarity
            sim_pct = int(score * 100)
            st.markdown(f"""
            <div class="card" style="padding: 1rem; margin-bottom: 0.8rem;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.3rem;">
                    <span style="font-size: 0.9rem; font-weight: bold; color: #f472b6;">
                        #{k+1} Case ID: {comp['complaint_id']} (Category: {comp['category']})
                    </span>
                    <span style="font-size: 0.85rem; color: #f472b6; font-weight: bold;">
                        Similarity Match: {sim_pct}%
                    </span>
                </div>
                <div style="font-style: italic; font-size: 0.9rem; color: #d1d5db; margin-bottom: 0.5rem;">
                    "{comp['complaint_text']}"
                </div>
                <div style="display: flex; justify-content: space-between; font-size: 0.8rem; color: #9ca3af;">
                    <div>⚠️ Priority: {comp['priority']}</div>
                    <div>⏱️ Resolution: {comp['resolution_days']} days</div>
                    <div>👤 Assigned: {comp['assigned_officer']}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # --- ROW 3: FEATURES VISUALIZATIONS (IF AUDIO OR VIDEO) ---
    if results["extracted_audio_features"] is not None or len(results["extracted_keyframes"]) > 0:
        st.markdown("---")
        st.markdown("### 🔬 Media Features Processing & Insights")
        
        col_vis1, col_vis2 = st.columns(2)
        
        with col_vis1:
            if results["extracted_audio_features"] is not None:
                st.markdown("#### Mel-Frequency Cepstral Coefficients (MFCC)")
                st.markdown("Acoustic features computed from Librosa used for audio signature matching:")
                
                # Visualize MFCC (We plot a visual graph of MFCC average coefficients)
                mfcc_feat = results["extracted_audio_features"]
                n_mfcc = len(mfcc_feat) // 2
                means = mfcc_feat[:n_mfcc]
                stds = mfcc_feat[n_mfcc:]
                
                fig, ax = plt.subplots(figsize=(6, 3))
                x_ticks = np.arange(1, n_mfcc + 1)
                ax.bar(x_ticks - 0.2, means, width=0.4, label='Mean value', color='#4facfe')
                ax.bar(x_ticks + 0.2, stds, width=0.4, label='Std Dev', color='#00f2fe')
                ax.set_xlabel('MFCC Coefficient Index')
                ax.set_ylabel('Amplitude')
                ax.set_title('Acoustic Spectrum MFCC Signature')
                ax.set_xticks(x_ticks)
                ax.legend(facecolor='#161b22', edgecolor='#374151')
                ax.grid(True, color='#374151', linestyle='--', alpha=0.5)
                
                st.pyplot(fig)
                
        with col_vis2:
            if len(results["extracted_keyframes"]) > 0:
                st.markdown("#### OpenCV Keyframe Extraction Gallery")
                st.markdown("Frames extracted at uniform intervals representing visual events:")
                
                kf_list = results["extracted_keyframes"]
                
                # Render keyframes in a horizontal grid
                col_kf_list = st.columns(len(kf_list))
                for idx, kf in enumerate(kf_list):
                    with col_kf_list[idx]:
                        st.image(
                            kf["image_rgb"],
                            caption=f"Frame: {kf['frame_idx']} ({kf['timestamp_sec']}s)",
                            use_container_width=True
                        )
