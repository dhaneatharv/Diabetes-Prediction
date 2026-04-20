import streamlit as st
import numpy as np
from predict import predict

st.set_page_config(
    page_title="Diabetes Risk Predictor",
    page_icon="🩺",
    layout="centered"
)

# CSS (same as yours)
st.markdown("""
<style>
    .main { padding-top: 1rem; }
    .stButton>button {
        width: 100%;
        background-color: #1D9E75;
        color: white;
        font-size: 16px;
        font-weight: 600;
        padding: 0.6rem;
        border: none;
        border-radius: 8px;
    }
    .stButton>button:hover { background-color: #0F6E56; color: white; }
    .result-box {
        padding: 1.2rem;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Risk conversion
MIN_SCORE = 25
MAX_SCORE = 346

def score_to_risk_pct(score):
    pct = (score - MIN_SCORE) / (MAX_SCORE - MIN_SCORE) * 100
    return round(float(np.clip(pct, 0, 100)), 1)

# Header
st.title("🩺 Diabetes Risk Predictor")
st.markdown("Fill in the patient details below.")
st.divider()

# Inputs
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", 1, 100, 45)
    bmi = st.number_input("BMI", 10.0, 60.0, 28.5)
    bp  = st.number_input("Blood Pressure", 40, 200, 80)

with col2:
    glucose = st.number_input("Blood Sugar", 50, 300, 110)
    insulin = st.number_input("Insulin", 0, 900, 80)
    dpf     = st.number_input("DPF", 0.0, 3.0, 0.47)

st.divider()

# Button
if st.button("🔍 Predict Diabetes Risk"):

    input_features = np.array([[
        (age - 50) / 10,
        0,
        (bmi - 26) / 5,
        (bp - 83) / 10,
        (glucose - 110) / 30,
        (insulin - 80) / 50,
        (dpf - 0.45) / 0.3,
        0, 0, 0
    ]])

    rf, lr, dt = predict(input_features[0])
    avg = (rf + lr + dt) / 3

    rf_pct  = score_to_risk_pct(rf)
    lr_pct  = score_to_risk_pct(lr)
    dt_pct  = score_to_risk_pct(dt)
    avg_pct = score_to_risk_pct(avg)

    # SAVE FOR NEXT PAGE 🔥
    st.session_state["rf"] = rf_pct
    st.session_state["lr"] = lr_pct
    st.session_state["dt"] = dt_pct
    st.session_state["avg"] = avg_pct

    # Risk label
    if avg_pct < 35:
        risk = "🟢 Low Risk"
        color = "#DCB2B2"
    elif avg_pct < 65:
        risk = "🟡 Moderate Risk"
        color = "#D9C458"
    else:
        risk = "🔴 High Risk"
        color = "#A74C4C"

    # RESULT CARD
    st.subheader("Prediction Result")
    st.markdown(f"""
    <div class="result-box" style="background:{color};">
        <h2>{avg_pct}% Diabetes Risk</h2>
        <p>{risk}</p>
    </div>
    """, unsafe_allow_html=True)

    