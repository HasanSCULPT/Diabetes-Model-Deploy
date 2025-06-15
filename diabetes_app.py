# diabetes_app.py
import streamlit as st
import numpy as np
import joblib

# Load the model
model = joblib.load("diabetes_model.pkl")

st.title("🩺 Diabetes Prediction")
st.write("## By DSA 2025")

st.markdown("""
Enter your medical information below to predict whether you're likely to have diabetes.
""")

# Input fields
glucose = st.number_input("Glucose Level", min_value=0.0, step=1.0)
blood_pressure = st.number_input("Blood Pressure", min_value=0.0, step=1.0)
skin_thickness = st.number_input("Skin Thickness", min_value=0.0, step=1.0)
insulin = st.number_input("Insulin", min_value=0.0, step=1.0)
bmi = st.number_input("BMI", min_value=0.0, step=0.1)
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, step=0.01)
age = st.number_input("Age", min_value=1.0, step=1.0)

# Predict
if st.button("Predict"):
    input_data = np.array([[glucose, blood_pressure, skin_thickness,
                            insulin, bmi, dpf, age]])
    prediction = model.predict(input_data)[0]
    result = "🟢 Non-Diabetic" if prediction == 0 else "🔴 Diabetic"
    
    st.subheader("Prediction Result:")
    st.success(result if prediction == 0 else result)

