# ============================================================
#  DIABETES RISK PREDICTOR — Advanced Version
#  - Dataset   : Pima Indians (768 rows) + synthetic augment → 1200 rows
#  - Models    : Random Forest, Logistic Regression, Decision Tree
#  - Tuning    : GridSearchCV hyperparameter tuning on all 3
#  - API       : Claude AI explains prediction in plain English
# ============================================================

import streamlit as st
import numpy as np
import pandas as pd
import requests
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings("ignore")

# ──────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────
st.set_page_config(
    page_title="Diabetes Risk Predictor",
    page_icon="🩺",
    layout="centered"
)

st.markdown("""
<style>
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
    .model-card {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.6rem;
    }
    .result-box {
        padding: 1.4rem;
        border-radius: 12px;
        text-align: center;
        margin-bottom: 1rem;
    }
    .ai-box {
        background: #f0f7ff;
        border-left: 4px solid #378ADD;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────
# LOAD & AUGMENT DATASET (min 1000 rows)
# ──────────────────────────────────────────
@st.cache_data
def load_dataset():
    """
    Load Pima Indians diabetes dataset from URL.
    If unavailable, generate a realistic synthetic dataset.
    Augment to reach 1200+ rows total.
    """
    columns = ['pregnancies','glucose','blood_pressure','skin_thickness',
               'insulin','bmi','dpf','age','outcome']
    try:
        url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
        df = pd.read_csv(url, header=None, names=columns)
        st.sidebar.success(f"✅ Pima dataset loaded ({len(df)} rows)")
    except Exception:
        st.sidebar.warning("⚠️ Using synthetic dataset (no internet)")
        df = None

    if df is None or len(df) < 100:
        np.random.seed(42)
        n = 768
        df = pd.DataFrame({
            'pregnancies':    np.random.poisson(3.8, n),
            'glucose':        np.random.normal(120, 32, n).clip(50, 200),
            'blood_pressure': np.random.normal(69, 19, n).clip(30, 130),
            'skin_thickness': np.random.normal(20, 16, n).clip(0, 70),
            'insulin':        np.random.exponential(80, n).clip(0, 600),
            'bmi':            np.random.normal(32, 7, n).clip(15, 60),
            'dpf':            np.random.exponential(0.47, n).clip(0.07, 2.5),
            'age':            np.random.normal(33, 12, n).clip(18, 80).astype(int),
        })
        # Realistic outcome based on risk factors
        risk = (
            (df['glucose'] > 140).astype(int) * 2 +
            (df['bmi'] > 30).astype(int) +
            (df['age'] > 45).astype(int) +
            (df['dpf'] > 0.8).astype(int)
        )
        df['outcome'] = (risk >= 2).astype(int)

    # ── Augment to 1200+ rows ──
    original_len = len(df)
    needed = max(0, 1200 - original_len)
    if needed > 0:
        np.random.seed(99)
        augmented = df.sample(needed, replace=True).copy()
        noise_cols = ['glucose','blood_pressure','bmi','insulin','skin_thickness']
        for col in noise_cols:
            augmented[col] = (augmented[col] + np.random.normal(0, df[col].std() * 0.05, needed)).clip(df[col].min(), df[col].max())
        df = pd.concat([df, augmented], ignore_index=True)

    df = df.replace(0, np.nan)
    for col in ['glucose','blood_pressure','skin_thickness','insulin','bmi']:
        df[col].fillna(df[col].median(), inplace=True)
    df = df.fillna(0)

    return df

# ──────────────────────────────────────────
# TRAIN MODELS WITH HYPERPARAMETER TUNING
# ──────────────────────────────────────────
@st.cache_resource
def train_tuned_models():
    df = load_dataset()
    feature_cols = ['pregnancies','glucose','blood_pressure','skin_thickness',
                    'insulin','bmi','dpf','age']
    X = df[feature_cols].values
    y = df['outcome'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    results = {}

    # ── 1. Random Forest ──
    rf_params = {
        'n_estimators': [100, 200],
        'max_depth':    [None, 10, 20],
        'min_samples_split': [2, 5],
    }
    rf_grid = GridSearchCV(
        RandomForestClassifier(random_state=42),
        rf_params, cv=5, scoring='accuracy', n_jobs=-1
    )
    rf_grid.fit(X_train_s, y_train)
    rf_best = rf_grid.best_estimator_
    rf_acc  = accuracy_score(y_test, rf_best.predict(X_test_s))
    rf_cv   = cross_val_score(rf_best, X_train_s, y_train, cv=5).mean()
    results['Random Forest'] = {
        'model':       rf_best,
        'accuracy':    round(rf_acc * 100, 2),
        'cv_score':    round(rf_cv * 100, 2),
        'best_params': rf_grid.best_params_,
        'color':       '#378ADD'
    }

    # ── 2. Logistic Regression (hypertuned) ──
    lr_params = {
        'C':        [0.01, 0.1, 1, 10, 100],
        'solver':   ['lbfgs', 'liblinear'],
        'max_iter': [200, 500, 1000],
        'penalty':  ['l2']
    }
    lr_grid = GridSearchCV(
        LogisticRegression(random_state=42),
        lr_params, cv=5, scoring='accuracy', n_jobs=-1
    )
    lr_grid.fit(X_train_s, y_train)
    lr_best = lr_grid.best_estimator_
    lr_acc  = accuracy_score(y_test, lr_best.predict(X_test_s))
    lr_cv   = cross_val_score(lr_best, X_train_s, y_train, cv=5).mean()
    results['Logistic Regression'] = {
        'model':       lr_best,
        'accuracy':    round(lr_acc * 100, 2),
        'cv_score':    round(lr_cv * 100, 2),
        'best_params': lr_grid.best_params_,
        'color':       '#1D9E75'
    }

    # ── 3. Decision Tree ──
    dt_params = {
        'max_depth':        [3, 5, 7, 10, None],
        'min_samples_split':[2, 5, 10],
        'criterion':        ['gini', 'entropy']
    }
    dt_grid = GridSearchCV(
        DecisionTreeClassifier(random_state=42),
        dt_params, cv=5, scoring='accuracy', n_jobs=-1
    )
    dt_grid.fit(X_train_s, y_train)
    dt_best = dt_grid.best_estimator_
    dt_acc  = accuracy_score(y_test, dt_best.predict(X_test_s))
    dt_cv   = cross_val_score(dt_best, X_train_s, y_train, cv=5).mean()
    results['Decision Tree'] = {
        'model':       dt_best,
        'accuracy':    round(dt_acc * 100, 2),
        'cv_score':    round(dt_cv * 100, 2),
        'best_params': dt_grid.best_params_,
        'color':       '#D85A30'
    }

    return results, scaler, X_test_s, y_test

# ──────────────────────────────────────────
# CLAUDE API — AI EXPLANATION
# ──────────────────────────────────────────
def get_ai_explanation(patient_data: dict, predictions: dict, avg_risk: float, api_key: str) -> str:
    """Call Claude API to explain the prediction in plain English."""

    prompt = f"""
You are a medical AI assistant helping explain diabetes risk predictions.

Patient details:
- Age: {patient_data['age']} years
- BMI: {patient_data['bmi']}
- Blood Glucose: {patient_data['glucose']} mg/dL
- Blood Pressure: {patient_data['blood_pressure']} mmHg
- Insulin Level: {patient_data['insulin']} μU/mL
- Diabetes Pedigree Score: {patient_data['dpf']}
- Pregnancies: {patient_data['pregnancies']}

Model predictions:
- Random Forest:       {predictions['Random Forest']}% diabetes risk
- Logistic Regression: {predictions['Logistic Regression']}% diabetes risk
- Decision Tree:       {predictions['Decision Tree']}% diabetes risk
- Average Risk:        {avg_risk}%

Please give a short, clear explanation (3-4 sentences) of:
1. What the key risk factors are for this patient
2. What the prediction means in simple terms
3. One practical health tip

Keep it simple, friendly, and avoid medical jargon. End with a reminder to consult a doctor.
"""

    headers = {
        "Content-Type": "application/json",
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01"
    }
    body = {
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 300,
        "messages": [{"role": "user", "content": prompt}]
    }

    try:
        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers=headers,
            json=body,
            timeout=15
        )
        data = response.json()
        if response.status_code == 200:
            return data["content"][0]["text"]
        else:
            return f"API Error {response.status_code}: {data.get('error', {}).get('message', 'Unknown error')}"
    except Exception as e:
        return f"Could not connect to Claude API: {str(e)}"

# ──────────────────────────────────────────
# SIDEBAR — API KEY + MODEL INFO
# ──────────────────────────────────────────
st.sidebar.title("⚙️ Settings")
api_key = st.sidebar.text_input(
    "Claude API Key",
    type="password",
    placeholder="sk-ant-...",
    help="Get your key from console.anthropic.com"
)
use_ai = st.sidebar.toggle("Enable AI Explanation", value=True)

st.sidebar.divider()
st.sidebar.markdown("### 📊 Dataset Info")

df_info = load_dataset()
st.sidebar.markdown(f"""
- **Rows:** {len(df_info)}
- **Features:** 8
- **Target:** Diabetic (1) / Not (0)
- **Source:** Pima Indians + augmented
""")

# ──────────────────────────────────────────
# LOAD MODELS (with spinner)
# ──────────────────────────────────────────
with st.spinner("🔧 Training & hypertuning models... (first run only)"):
    model_results, scaler, X_test_s, y_test = train_tuned_models()

# Show model accuracy in sidebar
st.sidebar.divider()
st.sidebar.markdown("### 🎯 Model Accuracies (after tuning)")
for name, info in model_results.items():
    st.sidebar.markdown(f"**{name}**")
    st.sidebar.progress(info['accuracy'] / 100, text=f"{info['accuracy']}% accuracy")

# ──────────────────────────────────────────
# MAIN UI
# ──────────────────────────────────────────
st.title("🩺 Diabetes Risk Predictor")
st.markdown("Uses **3 hypertuned ML models** + **Claude AI** to predict and explain diabetes risk.")
st.divider()

st.subheader("Patient Details")
col1, col2 = st.columns(2)

with col1:
    pregnancies = st.number_input("Pregnancies",            min_value=0,  max_value=20,  value=2)
    glucose     = st.number_input("Blood Glucose (mg/dL)",  min_value=50, max_value=300, value=110,
                                   help="Normal fasting: 70–100 mg/dL")
    bp          = st.number_input("Blood Pressure (mmHg)",  min_value=40, max_value=200, value=72,
                                   help="Normal: 60–90 mmHg")
    skin        = st.number_input("Skin Thickness (mm)",    min_value=0,  max_value=100, value=20,
                                   help="Triceps skinfold thickness")

with col2:
    insulin = st.number_input("Insulin (μU/mL)",         min_value=0,   max_value=900, value=80,
                               help="Normal: 2–25 μU/mL")
    bmi     = st.number_input("BMI",                     min_value=10.0, max_value=70.0, value=28.5, step=0.1,
                               help="Normal: 18.5–24.9")
    dpf     = st.number_input("Diabetes Pedigree Score", min_value=0.0, max_value=3.0,  value=0.47, step=0.01,
                               help="Family history score (0 = no history)")
    age     = st.number_input("Age (years)",             min_value=1,   max_value=100,  value=33)

st.divider()

# ──────────────────────────────────────────
# PREDICT
# ──────────────────────────────────────────
if st.button("🔍 Predict Diabetes Risk"):

    input_arr = np.array([[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]])
    input_scaled = scaler.transform(input_arr)

    predictions = {}
    confidences = {}

    for name, info in model_results.items():
        pred_proba = info['model'].predict_proba(input_scaled)[0]
        risk_pct   = round(pred_proba[1] * 100, 1)
        conf_pct   = round(max(pred_proba) * 100, 1)
        predictions[name]  = risk_pct
        confidences[name]  = conf_pct

    avg_risk = round(sum(predictions.values()) / 3, 1)
    avg_conf = round(sum(confidences.values()) / 3, 1)

    # ── Risk verdict ──
    if avg_risk < 35:
        risk_label, box_color, text_color = "🟢 Low Risk", "#d4edda", "#155724"
    elif avg_risk < 60:
        risk_label, box_color, text_color = "🟡 Moderate Risk", "#fff3cd", "#856404"
    else:
        risk_label, box_color, text_color = "🔴 High Risk", "#f8d7da", "#721c24"

    st.subheader("Prediction Results")

    # ── Overall result box ──
    st.markdown(f"""
    <div class="result-box" style="background:{box_color}; color:{text_color};">
        <h2 style="margin:0; color:{text_color};">{avg_risk}% Diabetes Risk</h2>
        <p style="margin:4px 0 0; font-size:18px;">{risk_label}</p>
        <p style="margin:4px 0 0; font-size:13px; opacity:0.8;">
            Average confidence: {avg_conf}% &nbsp;|&nbsp; Across all 3 models
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Per-model breakdown ──
    st.markdown("#### Model breakdown")
    for name, info in model_results.items():
        risk = predictions[name]
        conf = confidences[name]
        color = info['color']
        acc   = info['accuracy']
        st.markdown(f"""
        <div class="model-card">
            <div style="display:flex; justify-content:space-between; margin-bottom:4px;">
                <span style="font-weight:600;">{name}</span>
                <span style="font-weight:600; color:{color};">{risk}% risk &nbsp;|&nbsp; {conf}% confidence</span>
            </div>
            <div style="background:#e0e0e0; border-radius:6px; height:10px; margin-bottom:6px;">
                <div style="width:{risk}%; background:{color}; height:10px; border-radius:6px;"></div>
            </div>
            <span style="font-size:12px; color:#666;">Model accuracy after tuning: {acc}%</span>
        </div>
        """, unsafe_allow_html=True)

    # ── Metric summary ──
    st.markdown("#### Score summary")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Random Forest",       f"{predictions['Random Forest']}%",       f"conf {confidences['Random Forest']}%")
    c2.metric("Logistic Regression", f"{predictions['Logistic Regression']}%", f"conf {confidences['Logistic Regression']}%")
    c3.metric("Decision Tree",       f"{predictions['Decision Tree']}%",        f"conf {confidences['Decision Tree']}%")
    c4.metric("Avg Risk",            f"{avg_risk}%",                            f"conf {avg_conf}%")

    # ── Best params expander ──
    with st.expander("🔧 View best hyperparameters (GridSearchCV)"):
        for name, info in model_results.items():
            st.markdown(f"**{name}:** `{info['best_params']}`")
            st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;CV Score: **{info['cv_score']}%** &nbsp;|&nbsp; Test Accuracy: **{info['accuracy']}%**")

    # ── Claude AI Explanation ──
    if use_ai:
        st.markdown("#### 🤖 AI Explanation (Claude)")
        if not api_key:
            st.warning("Add your Claude API key in the sidebar to get an AI explanation.")
        else:
            with st.spinner("Claude is analyzing..."):
                patient_data = {
                    'age': age, 'bmi': bmi, 'glucose': glucose,
                    'blood_pressure': bp, 'insulin': insulin,
                    'dpf': dpf, 'pregnancies': pregnancies
                }
                explanation = get_ai_explanation(patient_data, predictions, avg_risk, api_key)
            st.markdown(f"""
            <div class="ai-box">
                <p style="margin:0; font-size:14px; line-height:1.7;">{explanation}</p>
            </div>
            """, unsafe_allow_html=True)

    st.divider()
    st.caption("⚠️ For educational purposes only. Not a substitute for professional medical diagnosis.")
