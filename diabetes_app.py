import streamlit as st
import joblib
import numpy as np
import shap
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# Load model
model = joblib.load("best_rf_model_optuna.pkl")

# Streamlit UI
st.set_page_config(page_title="Diabetes Predictor", layout="centered")
st.title("🩺 Diabetes Prediction Centre App")
st.write("## By HasanSCULPT | DSA 2025")
st.markdown("""
Enter your medical information below to predict whether you're likely to have diabetes.
""")


# Input fields
pregnancies = st.number_input("Pregnancies", min_value=0, step=1)
glucose = st.number_input("Glucose Level", min_value=0)
bloodpressure = st.number_input("Blood Pressure", min_value=0)
skinthickness = st.number_input("Skin Thickness", min_value=0)
insulin = st.number_input("Insulin Level", min_value=0)
bmi = st.number_input("BMI", min_value=0.0, format="%.2f")
dpf = st.number_input("Diabetes Pedigree Function (DPF)", min_value=0.0, format="%.3f")
age = st.number_input("Age", min_value=0, step=1)

# Predict button
if st.button("Predict"):
    input_data = np.array([[pregnancies, glucose, bloodpressure, skinthickness, insulin, bmi, dpf, age]])
    prediction = model.predict(input_data)[0]
    prediction_proba = model.predict_proba(input_data)[0]

    if prediction == 1:
        st.error(f"🛑 Prediction: **Diabetic** (Confidence: {prediction_proba[1]:.2%})")
    else:
        st.success(f"✅ Prediction: **Not Diabetic** (Confidence: {prediction_proba[0]:.2%})")

    # Show probability bar chart
    st.subheader("📊 Prediction Confidence")
    st.bar_chart({"Probability": {"Not Diabetic": prediction_proba[0], "Diabetic": prediction_proba[1]}})

    # SHAP Explanation
    explainer = shap.Explainer(model)
shap_values = explainer(input_data)

shap.plots.waterfall(shap_values[0], show=False)
st.pyplot(bbox_inches='tight')

