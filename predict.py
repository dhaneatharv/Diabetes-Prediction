import pickle
import numpy as np

rf = pickle.load(open("models/rf.pkl", "rb"))
lr = pickle.load(open("models/lr.pkl", "rb"))
dt = pickle.load(open("models/dt.pkl", "rb"))
scaler = pickle.load(open("models/scaler.pkl", "rb"))

def predict(input_features):
    input_scaled = scaler.transform([input_features])

    rf_pred = rf.predict(input_scaled)[0]
    lr_pred = lr.predict(input_scaled)[0]
    dt_pred = dt.predict(input_scaled)[0]

    return rf_pred, lr_pred, dt_pred