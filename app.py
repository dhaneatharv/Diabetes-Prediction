from predict import predict
import streamlit as st
from predict import predict

st.title("Diabetes Predictor")

age = st.number_input("Age")
bmi = st.number_input("BMI")

if st.button("Predict"):
    features = [age, 0, bmi, 0, 0, 0, 0, 0, 0, 0]

    rf, lr, dt = predict(features)

    st.write("Random Forest:", rf)
    st.write("Linear Regression:", lr)
    st.write("Decision Tree:", dt)