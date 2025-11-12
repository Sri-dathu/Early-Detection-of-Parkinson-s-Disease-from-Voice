# results.py — Clean futuristic results screen for VoxaSense

import streamlit as st

# ------------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------------
st.set_page_config(page_title="VoxaSense Results", page_icon="📊", layout="wide")

# ------------------------------------------------------------
# CUSTOM STYLING
# ------------------------------------------------------------
st.markdown("""
    <style>
        body {
            background-color: #0d1117;
            color: #e6edf3;
        }
        .stApp {
            background-color: #0d1117;
        }
        h1, h2, h3 {
            color: #58a6ff;
            text-align: center;
        }
        .title {
            text-align: center;
            font-size: 2.2rem;
            font-weight: 700;
            color: #58a6ff;
            margin-bottom: 10px;
        }
        .subtitle {
            text-align: center;
            font-size: 1.1rem;
            color: #8b949e;
            margin-bottom: 40px;
        }
        .card {
            background: #161b22;
            padding: 2rem;
            border-radius: 15px;
            box-shadow: 0 0 10px rgba(88,166,255,0.2);
        }
        .stButton>button {
            background-color: #238636;
            color: white;
            border: none;
            border-radius: 8px;
            padding: 0.6em 1.2em;
            font-size: 16px;
        }
        .stButton>button:hover {
            background-color: #2ea043;
            color: white;
        }
    </style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------
# HEADER
# ------------------------------------------------------------
st.markdown("<div class='title'>📊 VoxaSense Prediction Results</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>AI-based Parkinson’s Voice Screening</div>", unsafe_allow_html=True)

# ------------------------------------------------------------
# GET PREDICTION
# ------------------------------------------------------------
prob = st.session_state.get("prediction", 0.5)

try:
    prob = float(prob)
except Exception:
    prob = 0.5

label = "🟢 Likely Healthy" if prob < 0.5 else "🔴 At Risk"

# ------------------------------------------------------------
# DISPLAY RESULTS
# ------------------------------------------------------------
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.metric("Predicted Risk Score", f"{prob:.3f}")
st.metric("Status", label)
st.progress(min(max(prob, 0.0), 1.0))
st.markdown("</div>", unsafe_allow_html=True)

# ------------------------------------------------------------
# MESSAGE BOX
# ------------------------------------------------------------
st.markdown("<br>", unsafe_allow_html=True)
if label == "🔴 At Risk":
    st.warning("⚠️ Your voice patterns may indicate early Parkinson’s risk. Please consult a neurologist for further examination.")
else:
    st.success("✅ Your voice appears within a healthy range. Keep up regular check-ups and maintain voice health!")

# ------------------------------------------------------------
# NAVIGATION BUTTONS
# ------------------------------------------------------------
st.markdown("<br>", unsafe_allow_html=True)
col1, col2 = st.columns(2)

with col1:
    if st.button("🔁 Analyze Another Sample", use_container_width=True):
        st.session_state.page = "analyze"
        st.switch_page("app.py")

with col2:
    if st.button("🏠 Back to Home", use_container_width=True):
        st.session_state.page = "home"
        st.switch_page("app.py")

# ------------------------------------------------------------
# FOOTER
# ------------------------------------------------------------
st.markdown("<br><hr>", unsafe_allow_html=True)
st.caption("🧠 VoxaSense — Hackathon Prototype | Not a medical diagnosis tool.")
