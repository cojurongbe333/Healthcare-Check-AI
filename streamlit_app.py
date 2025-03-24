
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model and scaler
model = joblib.load("models/risk_classifier.pkl")
scaler = joblib.load("models/scaler.pkl")

st.title("🏥 HealthCheckAI: Patient Risk Predictor")

st.write("""Enter patient data below to predict risk of ICU transfer.""")

# Input fields
age = st.slider("Age", 18, 100, 65)
gender = st.selectbox("Gender", ["Male", "Female"])
comorbidities = st.slider("Comorbidities", 0, 5, 2)
admission_type = st.selectbox("Admission Type", ["Emergency", "Elective", "Urgent"])
heart_rate = st.number_input("Heart Rate (bpm)", 30.0, 180.0, 85.0)
resp_rate = st.number_input("Respiratory Rate", 10.0, 40.0, 18.0)
systolic_bp = st.number_input("Systolic BP (mmHg)", 60.0, 200.0, 120.0)
temperature = st.number_input("Temperature (°C)", 34.0, 41.0, 37.0)
hemoglobin = st.number_input("Hemoglobin (g/dL)", 5.0, 20.0, 13.5)
creatinine = st.number_input("Creatinine (mg/dL)", 0.2, 5.0, 1.0)
lactate = st.number_input("Lactate (mmol/L)", 0.5, 10.0, 1.5)
wbc = st.number_input("White Cell Count (x10^9/L)", 1.0, 25.0, 7.0)

# Process input
input_df = pd.DataFrame([{
    "age": age,
    "comorbidities": comorbidities,
    "heart_rate": heart_rate,
    "respiratory_rate": resp_rate,
    "systolic_bp": systolic_bp,
    "temperature": temperature,
    "hemoglobin": hemoglobin,
    "creatinine": creatinine,
    "lactate": lactate,
    "white_cell_count": wbc,
    "gender_Male": int(gender == "Male"),
    "admission_type_Emergency": int(admission_type == "Emergency"),
    "admission_type_Urgent": int(admission_type == "Urgent")
}])

# Predict
scaled_input = scaler.transform(input_df)
pred = model.predict(scaled_input)[0]
prob = model.predict_proba(scaled_input)[0][pred]

# Output
st.subheader("Prediction")
if pred == 1:
    st.error(f"⚠️ High Risk of ICU Transfer ({prob:.2%} confidence)")
else:
    st.success(f"✅ Stable Condition ({prob:.2%} confidence)")
