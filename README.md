# 🩺 Diabetes Risk Predictor (ML + Streamlit)

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![Status](https://img.shields.io/badge/Status-Active-success)

---

## 📌 Overview

This project builds a **Diabetes Risk Prediction system** using multiple machine learning models and serves it via a **Streamlit web app**. The app collects patient inputs and predicts diabetes risk using an ensemble of models.

---

## 🎥 Demo

![Demo GIF](demo.gif)
![Demo2 GIF](demo2.gif)

> Record your app screen and save it as `demo.gif` to showcase this section.

---

## 🎯 Key Features

* 🔍 Predict diabetes risk from patient inputs
* 🌲 Uses 3 ML models:

  * Random Forest
  * Linear Regression
  * Decision Tree
* 📊 Displays individual model predictions + average risk
* ⚡ Fast predictions using pre-trained models
* 🌐 Interactive UI with Streamlit

---

## 🧠 Models Used

* Random Forest Regressor
* Linear Regression
* Decision Tree Regressor

👉 Final prediction = **Average of all 3 models** fileciteturn1file7

---

## 📂 Project Structure

```
diabetes-predictor/
│
├── models/                # Saved models (pickle)
├── preprocess.py          # Data preprocessing
├── train.py               # Train & save models
├── predict.py             # Load models + predict
├── evaluate.py            # Model evaluation (R2 score)
├── app.py                 # Simple Streamlit UI
├── Diabetes_app_v2.py     # Advanced UI (better UX)
└── README.md
```

---

## ⚙️ Workflow

### 1. Data Preprocessing

* Uses sklearn diabetes dataset
* Splits into train/test
* Applies StandardScaler fileciteturn1file6

---

### 2. Model Training

* Trains 3 models
* Saves them using pickle
* Saves scaler for inference fileciteturn1file8

---

### 3. Prediction

* Loads saved models
* Scales input
* Returns predictions from all models fileciteturn1file5

---

### 4. Evaluation

* Uses R² score for performance comparison fileciteturn1file4

---

### 5. Web App

* Takes user input (age, BMI, etc.)
* Shows prediction instantly
* Displays model-wise output fileciteturn1file1

---

## 🚀 How to Run

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Train Models

```bash
python train.py
```

### Step 3: Run App

```bash
streamlit run app.py
```

---

## 📊 Results

* Multiple models improve reliability
* Ensemble average gives stable prediction
* Random Forest usually performs best

---

## 🛠 Tech Stack

* Python
* Scikit-learn
* NumPy
* Streamlit
* Pickle

---

## ⚠️ Notes

* Models are saved locally in `/models`
* Ensure models are trained before running app
* This is for **educational use only**

---

## 💡 Future Improvements

* Convert to classification (diabetic / non-diabetic)
* Add probability-based risk score
* Deploy on cloud (Streamlit Cloud)
* Add explainability (SHAP)

---

## 🧾 Conclusion

This project demonstrates a **complete ML pipeline** from preprocessing → training → evaluation → deployment. It highlights how multiple models can be combined to improve prediction reliability.

