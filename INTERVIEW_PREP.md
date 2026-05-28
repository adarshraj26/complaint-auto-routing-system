# Interview Preparation Guide: AI/ML Complaint Auto-Routing System

This guide is designed to help you explain your project confidently in an interview and ace any technical or conceptual questions.

---

## 1. The 30-Second Elevator Pitch (How to introduce your project)
> *"I built a local, offline AI-powered municipal dashboard that automates how city complaints are handled. When a citizen submits a complaint (in text, audio, or video, and in multiple languages like Hindi, English, Spanish, etc.), the system automatically detects the problem category, predicts its urgency priority, estimates the resolution time in days, retrieves similar historical complaints, and assigns it to the most suitable municipal officer based on their language, zone, expertise, and current workload."*

---

## 2. The Core Problem & Solution (In Simple Words)
* **The Old Way (Problem)**: Municipal complaints are routed manually. If a citizen reports a broken pipe, a human clerk has to read it, find who is free, make sure they speak the language, and assign it. This is slow, prone to errors, and fails if the complaint is in another language (like Hindi or Spanish) or submitted as an audio/video clip.
* **The New Way (Solution)**: AI does the heavy lifting. It listens to audio, watches video, translates meaning, matches it to the right department, makes predictions, and assigns it to the best officer in under a second.

---

## 3. Technology Stack: The "What & Why" (Analogy Guide)

| Technology | What it is | Why we used it (Simple Analogy) |
| :--- | :--- | :--- |
| **Streamlit** | Python Web Framework | **The Face**: Allows us to build a beautiful, interactive web interface for the dashboard in pure Python. |
| **Sentence Transformers** (`paraphrase-multilingual-MiniLM-L12-v2`) | NLP Text Embedding Model | **The Translator**: Converts words into a list of 384 numbers (embeddings) that represent **meaning**. It allows the system to understand that *"water leak"* and *"पानी बह रहा है"* mean the same thing. |
| **Random Forest** | Machine Learning Classifier & Regressor | **The Brain**: An ensemble of "decision trees" that vote on the output. It predicts **Priority** (High/Medium/Low) and **ETA** (number of days to resolve). |
| **FAISS** (by Meta) | Semantic Vector Database | **The Memory**: An ultra-fast index that searches through thousands of past complaints in microseconds to find the top-5 most similar cases. |
| **Whisper-Tiny** (by OpenAI) | Speech-to-Text Model | **The Ears**: A tiny, offline model that transcribes voice complaints (audio files) into clean text. |
| **Librosa** | Audio Analysis Library | **The Voice Fingerprint**: Extracts acoustic features (MFCCs) from voice clips to show the frequency spectrum/signature. |
| **OpenCV** (Headless) | Computer Vision Library | **The Eyes**: Extracts screenshot frames (keyframes) from video complaints to show visual evidence of the issue. |

---

## 4. Likely Interview Questions & Answers

### 💬 General / Project Overview Questions

#### Q1. Walk me through the lifetime of a complaint in your system.
**Answer**: 
1. **Ingestion**: The user submits a text, audio, or video complaint on the Streamlit dashboard, selecting their location (e.g., *South Zone*) and language.
2. **Processing**: If it's audio or video, Whisper transcribes it to text. OpenCV extracts video frames, and Librosa plots the audio frequencies.
3. **Semantic Matching**: The text is converted into a vector embedding.
4. **History Search**: The embedding is sent to FAISS, which finds the 5 most similar past cases.
5. **AI Predictions**: The embedding is passed to our Random Forest models to predict the **Priority** (e.g., *High*) and **ETA** (e.g., *3 days*).
6. **Smart Routing**: The routing engine calculates a score for every officer and assigns the complaint to the officer with the highest score.
7. **UI Render**: The dashboard displays the assignment, officer rank, similar cases, and media visualization.

#### Q2. Why is your system designed to run entirely offline and local?
**Answer**: 
* **Privacy & Security**: Citizen complaints might contain private details, addresses, or faces. Processing locally ensures no data leaves the city's servers.
* **Cost Efficiency**: There are no expensive API calls to OpenAI or Google Cloud.
* **Network Independence**: The system can function in municipal control rooms during network outages or emergency situations.

---

### 🧠 NLP & Machine Learning Questions

#### Q3. What is an "embedding" and why did you use a multilingual one?
**Answer**: 
An embedding is a vector (a list of numbers) representing the semantic meaning of a text. We used `paraphrase-multilingual-MiniLM-L12-v2` because it maps 50+ languages to the same vector space. This means a complaint in Hindi (*"पाइप फट गया"*) is located right next to the English translation (*"pipe burst"*) in the vector space, allowing our AI models and FAISS search to understand both without needing a separate translation step.

#### Q4. Why did you use FAISS? Why not just do a standard database search (like SQL `LIKE` or regex)?
**Answer**: 
Standard database searches look for exact keyword matches. If a user types *"water is leaking"* and the database has *"pipe burst"*, a keyword search will fail. FAISS performs a **semantic search**—it compares the vector coordinates (meanings) using cosine similarity. It is also designed to scale to millions of records, returning results in microseconds.

#### Q5. Why did you choose Random Forest over simple models like Logistic Regression or Linear Regression?
**Answer**: 
During training, we compared both:
* For **Priority Classification**, Random Forest and Logistic Regression both performed very well (~92-93% accuracy), capturing department terms and priority rules.
* For **ETA Regression**, Linear Regression failed completely (negative $R^2$ score) because resolution timelines are highly non-linear (e.g., high priority takes 1-3 days, low priority takes 8-15 days, regardless of street name). The Random Forest Regressor handled these non-linear decision boundaries perfectly, achieving a **Mean Absolute Error (MAE) of only 2.2 days**.

---

### 👮 Routing Engine Questions

#### Q6. How does your routing algorithm work? What factors does it consider?
**Answer**: 
It routes using a **Multi-Factor Score**:
$$\text{Final Score} = \text{Semantic Cosine Similarity} \times M_{\text{language}} \times M_{\text{region}} \times M_{\text{workload}}$$
1. **Semantic Similarity**: How close is the complaint description to the officer's specialization (e.g., *pothole repair*)?
2. **Language Multiplier ($M_{\text{language}}$)**: If the officer doesn't speak the complaint language, they get a heavy penalty (0.4 multiplier).
3. **Region Multiplier ($M_{\text{region}}$)**: If the officer doesn't work in the complaint's zone (e.g., *North Zone*), they get a moderate penalty (0.8 multiplier) to prefer local officers.
4. **Workload Multiplier ($M_{\text{workload}}$)**: As an officer's current active cases increase, this multiplier decays (down to 0.2) to divert new cases to less busy officers and prevent bottlenecks.

#### Q7. How does the workload balancing work?
**Answer**: 
The workload multiplier is calculated as: `max(0.2, 1.0 - (workload / 150.0))`. 
If an officer is heavily overloaded (e.g., workload score of 90), their workload multiplier drops to `0.4`. Even if they are the perfect technical match, their final score drops, allowing a slightly less specialized but completely free officer to take the case.

---

### 📹 Media & Audio Processing Questions

#### Q8. Explain what Librosa and MFCC mean in simple terms.
**Answer**: 
**Librosa** is a Python library for audio analysis. **MFCC** stands for *Mel-Frequency Cepstral Coefficients*. In simple words, it is an **acoustic fingerprint**. It breaks down the sound wave into frequency bands that mimic how the human ear perceives sound. By plotting the MFCC coefficients on the dashboard, we show the unique spectral signature of the voice complaint.

#### Q9. Why did you use `opencv-python-headless` instead of standard `opencv-python`?
**Answer**: 
Standard `opencv-python` tries to pull in GUI dependencies (like GTK or Qt) to open video window popups (like `cv2.imshow`). In headless environments (like Streamlit Cloud or remote Linux servers), there is no screen display, which causes OpenCV to crash. The `headless` version includes the full code logic but strips out the GUI requirements, making it perfect for cloud deployments.

---

### 🇮🇳 Localization Questions

#### Q10. How is this project customized for the Indian context?
**Answer**: 
* **Officer Database**: Updated with 55 Indian municipal officers across 7 departments (e.g., *Rajesh Kumar, Sanjay Sharma, Arjun Reddy, Swati Kapoor*).
* **Zones**: Set up with standard municipal zones used in Indian corporations: `North Zone`, `South Zone`, `East Zone`, `West Zone`, and `Central Zone`.
* **Streets**: Integrated with 20 major Indian city streets (e.g., `MG Road`, `Linking Road`, `Brigade Road`, `Chandni Chowk`, `Janpath`).
* **Multilingual Input**: Native support for Hindi speech and text inputs, which are routed successfully to Hindi-speaking officers.
