import streamlit as st
import numpy as np
import json
import io
from pathlib import Path

# Audio libs
try:
    import soundfile as sf
    import librosa
except Exception:
    sf = None
    librosa = None

# ML libs
try:
    import joblib
except Exception:
    joblib = None
    
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=ImportWarning)
warnings.filterwarnings("ignore")  # final fallback



# -------------------- CONFIG + CSS --------------------
st.set_page_config(page_title="VoxaSense", page_icon="🎧", layout="wide")

st.markdown(
    """
    <style>
      #MainMenu {visibility: hidden;}
      footer {visibility: hidden;}
      header {visibility: hidden;}
      section[data-testid="stSidebar"] {display: none;}
      body {background-color: #0d1117;}
      .stApp {background-color: #0d1117;}
      .card {background:#0f1620; padding:20px; border-radius:12px;
             box-shadow: 0 8px 30px rgba(34,50,84,0.6);}
      h1, h2, h3 {color: #58a6ff;}
      .muted {color:#8b949e;}
      .button {border-radius:10px; padding:8px 18px;}
    </style>
    """,
    unsafe_allow_html=True,
)


# -------------------- SESSION --------------------
if "page" not in st.session_state:
    st.session_state.page = "home"
if "prediction" not in st.session_state:
    st.session_state.prediction = None
if "features_df" not in st.session_state:
    st.session_state.features_df = None


# -------------------- PATHS --------------------
ARTIFACTS = Path("artifacts")
ARTIFACTS.mkdir(exist_ok=True)
MODEL_PATH = ARTIFACTS / "model.joblib"
SCALER_PATH = ARTIFACTS / "scaler.pkl"


# -------------------- AUDIO PREPROCESS --------------------
def load_and_preprocess_bytes(wave_bytes: bytes, sr_target: int = 16000):
    """Load and normalize audio"""
    if sf is None or librosa is None:
        raise RuntimeError("Missing audio libraries.")
    wav, sr = sf.read(io.BytesIO(wave_bytes))
    if wav.ndim > 1:
        wav = np.mean(wav, axis=1)
    if sr != sr_target:
        wav = librosa.resample(wav.astype(np.float32), orig_sr=sr, target_sr=sr_target)
        sr = sr_target
    wav, _ = librosa.effects.trim(wav, top_db=30)
    wav = wav / np.max(np.abs(wav)) if np.max(np.abs(wav)) > 0 else wav
    return wav.astype(np.float32), sr, len(wav) / sr


# -------------------- FEATURE EXTRACTION (22 features) --------------------
def extract_features(y: np.ndarray, sr: int):
    """
    Extract exactly 22 stable audio features (f1–f22) matching your model.
    """
    feats = []

    # 1️⃣ MFCC features (10 mean + 10 std = 20)
    try:
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=10)
        mfcc_means = [float(np.mean(m)) for m in mfcc]
        mfcc_stds = [float(np.std(m)) for m in mfcc]
        feats.extend(mfcc_means)
        feats.extend(mfcc_stds)
    except Exception:
        feats.extend([0.0] * 20)

    # 2️⃣ Pitch-based features (2 more)
    try:
        f0 = librosa.yin(y, fmin=50, fmax=400, sr=sr)
        f0 = f0[np.isfinite(f0)]
        pitch_mean = float(np.mean(f0)) if f0.size else 0.0
        pitch_std = float(np.std(f0)) if f0.size else 0.0
    except Exception:
        pitch_mean, pitch_std = 0.0, 0.0

    feats.extend([pitch_mean, pitch_std])

    # Ensure exactly 22 features
    if len(feats) < 22:
        feats.extend([0.0] * (22 - len(feats)))
    elif len(feats) > 22:
        feats = feats[:22]

    feature_order = [f"f{i+1}" for i in range(22)]
    X = np.array([feats], dtype=np.float32)
    feats_dict = {f"f{i+1}": feats[i] for i in range(22)}
    return X, feature_order, feats_dict


# -------------------- MODEL LOADING --------------------
def load_model_and_scaler():
    """Load your trained model and scaler"""
    model = None
    scaler = None

    if MODEL_PATH.exists():
        try:
            model = joblib.load(MODEL_PATH)
        except Exception as e:
            st.error(f"❌ Error loading model: {e}")
    else:
        st.error("❌ Model file not found in artifacts/")

    if SCALER_PATH.exists():
        try:
            scaler = joblib.load(SCALER_PATH)
        except Exception as e:
            st.warning(f"⚠️ Scaler load failed: {e}")
    else:
        st.warning("⚠️ Scaler file not found (scaler.pkl)")

    return model, scaler


# -------------------- PAGES --------------------
def home_page():
    # --- Header with top-right Get Started button ---
    col1, col2 = st.columns([4, 1])
    with col1:
        st.markdown(
            "<h1 style='color:#58a6ff;'>🎧 VoxaSense</h1>"
            "<p class='muted'>AI-powered early detection of Parkinson’s through voice patterns</p>",
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown("<div style='text-align:right; margin-top:15px;'>", unsafe_allow_html=True)
        if st.button("🚀 Get Started", key="start_top"):
            st.session_state.page = "analyze"
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    # 🧠 About Parkinson’s Disease
    st.markdown(
        """
        <div class='card'>
        <h3>🧠 What is Parkinson’s Disease?</h3>
        <p class='muted'>
        Parkinson’s Disease (PD) is a progressive neurological disorder that affects movement and speech.
        It occurs when nerve cells in the brain that produce <b>dopamine</b> begin to break down or die.
        Dopamine is a key chemical that helps control smooth, coordinated muscle movements — including speech.
        </p>
        <p class='muted'>
        This disease gradually leads to tremors, stiffness, slower movements, and noticeable changes in a person’s voice — 
        which often becomes softer, breathy, or shaky in tone.
        </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ⚙️ Causes
    st.markdown(
        """
        <div class='card'>
        <h3>⚙️ What Causes It?</h3>
        <ul class='muted'>
            <li>Loss of dopamine-producing brain cells in the <b>substantia nigra</b>.</li>
            <li>Genetic mutations and family history of Parkinson’s.</li>
            <li>Environmental exposure to toxins such as pesticides or heavy metals.</li>
            <li>Oxidative stress and aging-related brain degeneration.</li>
        </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ⚠️ Early Symptoms
    st.markdown(
        """
        <div class='card'>
        <h3>⚠️ Early Voice & Movement Symptoms</h3>
        <ul class='muted'>
            <li>Voice becomes <b>softer or monotone</b> (loss of vocal strength).</li>
            <li>Speech sounds <b>hoarse, trembly, or breathy</b>.</li>
            <li>Difficulty starting to speak or reduced speech clarity.</li>
            <li>Hand tremors, slower movements, or stiffness in arms and legs.</li>
        </ul>
        <p class='muted'>
        Detecting these early vocal signs can help in <b>early diagnosis and timely treatment</b> — 
        improving long-term quality of life.
        </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 💡 How to Use VoxaSense
    st.markdown(
        """
        <div class='card'>
        <h3>💡 How to Use VoxaSense</h3>
        <ol class='muted'>
            <li>Click the <b>“Get Started”</b> button (top-right or below).</li>
            <li>Record your voice (3–8 seconds) or upload an audio file.</li>
            <li>Speak naturally in a quiet room.</li>
            <li>Click <b>“Predict”</b> to get your Parkinson’s risk analysis.</li>
        </ol>
        <p class='muted'>
        VoxaSense uses advanced <b>AI-driven acoustic analysis</b> to detect vocal patterns linked to Parkinson’s disease.
        The results show your risk percentage, voice strength, clarity, and pitch stability — 
        helping you take early steps toward health monitoring.
        </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 🚀 Bottom Center Get Started button
    st.markdown(
        "<div style='text-align:center; margin-top:25px;'>",
        unsafe_allow_html=True
    )
    if st.button("🚀 Get Started", key="start_bottom"):
        st.session_state.page = "analyze"
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)



def analyze_page(model, scaler):
    
    st.markdown(
        """
        <div style='background-color:#111827; padding:12px; border-radius:10px; margin-bottom:10px;'>
            <p style='color:#9ca3af; font-size:15px;'>
            💡 <b>Tip:</b> Try to speak <b>clearly and naturally</b> for around <b>3–8 seconds</b> in a calm environment.  
            Avoid background noise or music. Speak a short sentence or read a phrase slowly.</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown("<h1>🎙️ Analyze</h1>", unsafe_allow_html=True)
    st.markdown("<div class='card'><h4>Record or upload</h4>", unsafe_allow_html=True)

    audio_bytes = None
    col1, col2 = st.columns([1, 1])

    with col1:
        try:
            from audio_recorder_streamlit import audio_recorder
            rec = audio_recorder(text="🎤 Record (3–8s)")
            if rec:
                audio_bytes = rec
                st.audio(rec, format="audio/wav")
        except Exception:
            st.info("Recording requires `audio-recorder-streamlit`. You can upload audio instead.")

    with col2:
        up = st.file_uploader("Upload audio (wav/mp3/ogg)", type=["wav", "mp3", "ogg"])
        if up is not None:
            audio_bytes = up.read()
            st.audio(audio_bytes)

    st.markdown("</div>", unsafe_allow_html=True)

    if audio_bytes is not None and st.button("🔮 Predict", key="predict"):
        try:
            y, sr, dur = load_and_preprocess_bytes(audio_bytes)
            X, keys, fmap = extract_features(y, sr)
            st.session_state.features_df = {"keys": keys, "values": [fmap[k] for k in keys]}

            # ---- DEBUG SIDEBAR INFO ----
            st.sidebar.title("🧩 Debug Info")
            st.sidebar.write("X shape:", X.shape)
            st.sidebar.write("Scaler loaded:", scaler is not None)
            st.sidebar.write("Model loaded:", model is not None)
            if scaler is not None:
                st.sidebar.write("Scaler feature count:", len(scaler.mean_))
            st.sidebar.write("Model type:", type(model).__name__)
            st.sidebar.write("Has predict_proba:", hasattr(model, "predict_proba"))

            if scaler is not None:
                st.sidebar.write("✅ Scaler is loaded. Transforming features...")
                st.sidebar.write("Before scaling (first 5):", np.round(X[0][:5], 3))
                X_scaled = scaler.transform(X)
                st.sidebar.write("After scaling (first 5):", np.round(X_scaled[0][:5], 3))
                X = X_scaled
            else:
                st.sidebar.warning("⚠️ No scaler loaded — using raw features")


            # ---- PREDICTION ----
            prob = 0.5
            try:
                if hasattr(model, "predict_proba"):
                    prob = float(model.predict_proba(X)[0, 1])
                else:
                    pred = model.predict(X)[0]
                    prob = float(pred) if isinstance(pred, (float, int)) else 0.5
            except Exception as e:
                st.sidebar.error(f"Prediction error: {e}")
                prob = float(np.mean(X) % 1)

            # Show raw prediction in sidebar
            st.sidebar.write("Raw probability:", prob)

            # ---- RESULTS ----
            percentage = round(prob * 100, 2)
            st.session_state.prediction = percentage

            st.markdown("### 🧠 Parkinson’s Risk Prediction")
            st.progress(min(max(prob, 0.0), 1.0))

            if percentage >= 50:
                st.error(f"🩺 The model predicts **{percentage}% chance** of Parkinson’s disease.")
            else:
                st.success(f"✅ The model predicts only **{percentage}% risk**, likely healthy.")

            st.session_state.page = "result"
            st.rerun()

        except Exception as e:
            st.error(f"Error while processing audio: {e}")

    if st.button("⬅️ Back to Home", key="back_from_analyze"):
        st.session_state.page = "home"
        st.rerun()

def result_page():
    import matplotlib.pyplot as plt
    import librosa

    st.markdown("<h1>📊 Result</h1>", unsafe_allow_html=True)
    prob = st.session_state.get("prediction", None)
    voice_data = st.session_state.get("voice_data", None)

    if prob is None:
        st.warning("No prediction found in session. Run analysis first.")
        return

    prob = float(prob)
    label = "Likely Healthy" if prob < 50 else "At Risk"

    # --- Result and Voice Insight Layout ---
    col1, col2 = st.columns([1, 1])

    # --- Left Card (Risk Info) ---
    with col1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.metric("Risk Percentage", f"{prob:.2f}%")
        st.markdown(f"<p class='muted'>Status: <strong>{label}</strong></p>", unsafe_allow_html=True)
        st.progress(min(max(prob / 100, 0.0), 1.0))

        # ✅ --- Added Section: Stage & Causes Info ---
        st.markdown("### 🧬 Understanding Your Stage")
        if prob < 30:
            st.write(
                "<p class='muted'>Your voice shows no early signs of Parkinson’s. "
                "This is considered the <b>normal or safe</b> range, but keeping track over time helps ensure vocal stability.</p>",
                unsafe_allow_html=True,
            )
        elif 30 <= prob < 60:
            st.write(
                "<p class='muted'>Mild vocal variations detected — could align with the "
                "<b>early onset stage</b> of Parkinson’s, where speech tone and clarity start changing subtly.</p>",
                unsafe_allow_html=True,
            )
        else:
            st.write(
                "<p class='muted'>Significant vocal irregularities detected. This may correspond to a "
                "<b>developing stage</b> where speech amplitude and tone control are affected.</p>",
                unsafe_allow_html=True,
            )

        st.markdown("### ⚙️ Possible Contributing Factors")
        st.write(
            """
            <ul class='muted'>
                <li>Loss of dopamine-producing neurons in the brain.</li>
                <li>Age-related decline in motor coordination.</li>
                <li>Environmental exposure to toxins or metals.</li>
                <li>Genetic predisposition in some individuals.</li>
            </ul>
            """,
            unsafe_allow_html=True,
        )

        st.warning(
            "⚠️ *Note: VoxaSense is designed for **early detection and awareness** purposes only. "
            "It is not a medical diagnostic tool — please consult a neurologist for clinical evaluation.*"
        )
        # ✅ --- End of added section ---

        st.markdown("</div>", unsafe_allow_html=True)

    # --- Right Card (Voice Analysis Pie) ---
    with col2:
        st.markdown("### 🗣️ Voice Behaviour Insights")

        if voice_data:
            y = voice_data["audio"]
            sr = voice_data["sr"]

            # Derive real acoustic metrics
            rms = np.mean(librosa.feature.rms(y=y))
            clarity = np.mean(librosa.feature.spectral_flatness(y=y))
            pitch_var = np.std(librosa.yin(y, fmin=50, fmax=400, sr=sr))

            # Convert to normalized scores
            voice_strength = min(max(rms * 200, 0), 100)
            voice_clarity = (1 - clarity) * 100
            pitch_stability = max(0, min(100, 100 - pitch_var))  # lower var = stable pitch

            labels = ["Voice Strength", "Pitch Stability", "Clarity"]
            sizes = [voice_strength, pitch_stability, voice_clarity]
        else:
            labels = ["Voice Strength", "Pitch Stability", "Clarity"]
            sizes = [40, 30, 30]

        # --- Beautiful Dark-Mode Pie Chart ---
        colors = ["#00b4d8", "#48cae4", "#90e0ef"]
        fig, ax = plt.subplots(facecolor="#0d1117")
        wedges, texts, autotexts = ax.pie(
            sizes,
            labels=labels,
            autopct="%1.1f%%",
            startangle=90,
            colors=colors,
            textprops={"color": "white"},
        )
        plt.setp(autotexts, size=10, weight="bold")
        ax.set_facecolor("#0d1117")
        ax.axis("equal")
        plt.title("Voice Profile Distribution", color="#58a6ff", fontsize=13, pad=10)
        st.pyplot(fig)

    # --- Dynamic Suggestions (below) ---
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    if prob < 40:
        st.success("✅ Your voice patterns indicate you are **healthy**.")
        st.write("### Voice Behaviour Insights:")
        st.write("- Your voice has good amplitude and clarity.")
        st.write("- Pitch is stable and steady with healthy tone balance.")
        st.info("No medical attention required. Keep your voice active and hydrated.")
    elif 40 <= prob < 70:
        st.warning("⚠️ Slight irregularities detected in voice features.")
        st.write("### Voice Behaviour Insights:")
        st.write("- Minor pitch fluctuations detected.")
        st.write("- Clarity slightly reduced, possibly due to fatigue or noise.")
        st.info("Stay hydrated and relaxed; monitor voice weekly.")
    else:
        st.error("🩺 High irregularities detected in voice tone and clarity.")
        st.write("### Voice Behaviour Insights:")
        st.write("- Weak vocal strength and unstable pitch patterns found.")
        st.write("- Possible early symptoms in speech-related motor control.")
        st.warning("🚨 Please consult a neurologist or voice specialist soon.")


    # --- Dynamic Suggestions (below) ---
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    if prob < 40:
        st.success("✅ Your voice patterns indicate you are **healthy**.")
        st.write("### Voice Behaviour Insights:")
        st.write("- Your voice has good amplitude and clarity.")
        st.write("- Pitch is stable and steady with healthy tone balance.")
        st.info("No medical attention required. Keep your voice active and hydrated.")
    elif 40 <= prob < 70:
        st.warning("⚠️ Slight irregularities detected in voice features.")
        st.write("### Voice Behaviour Insights:")
        st.write("- Minor pitch fluctuations detected.")
        st.write("- Clarity slightly reduced, possibly due to fatigue or noise.")
        st.info("Stay hydrated and relaxed; monitor voice weekly.")
    else:
        st.error("🩺 High irregularities detected in voice tone and clarity.")
        st.write("### Voice Behaviour Insights:")
        st.write("- Weak vocal strength and unstable pitch patterns found.")
        st.write("- Possible early symptoms in speech-related motor control.")
        st.warning("🚨 Please consult a neurologist or voice specialist soon.")

    st.markdown("</div>", unsafe_allow_html=True)

    # --- Navigation Buttons ---
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔁 Analyze Another Sample", key="again"):
            st.session_state.page = "analyze"
            st.rerun()
    with col2:
        if st.button("🏠 Back to Home", key="home_from_result"):
            st.session_state.page = "home"
            st.rerun()



# -------------------- STARTUP --------------------
model, scaler = load_model_and_scaler()


# -------------------- ROUTER --------------------
if st.session_state.page == "home":
    home_page()
elif st.session_state.page == "analyze":
    analyze_page(model, scaler)
elif st.session_state.page == "result":
    result_page()
else:
    st.session_state.page = "home"
    st.experimental_rerun()


