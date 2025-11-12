# 🎧 VoxaSense – AI-Powered Early Detection of Parkinson’s Disease from Voice

**VoxaSense** is an AI-powered Streamlit web application designed to detect **early signs of Parkinson’s disease** from voice recordings.  
It analyzes short speech samples, extracts acoustic features, and predicts the likelihood of Parkinson’s using a trained machine learning model.

> ⚠️ *Note: VoxaSense is an early detection tool for research and awareness. It is **not a medical diagnostic system**. Please consult a healthcare professional for medical advice.*

---

## 🧠 Overview

Parkinson’s disease is a **neurodegenerative disorder** affecting movement and speech due to the loss of dopamine-producing neurons in the brain.  
VoxaSense focuses on detecting **early voice-related symptoms**, such as reduced vocal tone, hoarseness, and instability, by analyzing subtle vocal changes.

---

## 🚀 Features

- 🎙️ **Voice Input Options**
  - Record voice directly using the in-browser recorder.
  - Upload existing audio files in `.wav`, `.mp3`, or `.ogg` formats.

- ⚙️ **Feature Extraction**
  - Extracts 22 essential acoustic features using **Librosa**, including:
    - MFCCs, Pitch variation, Spectral flatness, and Energy metrics.
  
- 🧮 **Machine Learning Model**
  - Trained **Support Vector Machine (SVM)** model.
  - Standardized using `StandardScaler` for consistent predictions.
  - Outputs a **risk percentage** and **health status**.

- 📊 **Interactive Visual Insights**
  - Displays **voice behavior breakdown** (strength, pitch stability, clarity).
  - Provides **stage-based suggestions** (Normal, Early Onset, or Developing).

- 🧬 **Educational Guidance**
  - Explains **possible causes** and **early-stage symptoms**.
  - Highlights that the system is for **early detection only**.

---

## 🩺 Parkinson’s Disease Information

Parkinson’s disease is caused by the **gradual loss of dopamine neurons** in the brain, affecting speech, movement, and facial expressions.

### 🧩 Common Causes
- Genetic mutations or family history  
- Environmental factors (like pesticide exposure)  
- Age-related dopamine neuron loss  
- Long-term exposure to toxins  

### ⚠️ Early Symptoms
- Tremors in voice or jaw  
- Softer, monotone, or shaky speech  
- Difficulty pronouncing words clearly  
- Reduced vocal strength  

---

## 💡 How to Use VoxaSense

1. **Launch the Application**
   - Run the Streamlit app on your local machine.

2. **Get Started**
   - Click the **“Get Started”** button (available on top-right and bottom-center of the home page).

3. **Record or Upload**
   - Record your voice (3–8 seconds) using the built-in recorder, or upload an audio file.

4. **Predict**
   - Click **🔮 Predict** to analyze your voice and view the prediction.

5. **View Results**
   - Check your **Parkinson’s risk percentage**.
   - View **voice analysis pie chart** and **personalized suggestions**.

---

## ⚙️ Environment Setup

Below is the **complete environment setup guide** for running VoxaSense locally.  
You can follow these steps in your terminal or VS Code command prompt 👇  

1️⃣ Clone the Repository

git clone https://github.com/Sri-dathu/Early-Detection-of-Parkinson-s-Disease-from-Voice.git

cd voxasense

2️⃣ Create a Virtual Environment

python -m venv .venv

3️⃣ Activate the Environment
On Windows:

.venv\Scripts\activate

On macOS/Linux:

source .venv/bin/activate

Website:
https://early-detection-of-parkinson-s-disease-from-voice-mq5tryozimrt.streamlit.app
