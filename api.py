
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib

# Load model and scaler
model = joblib.load("models/risk_classifier.pkl")
scaler = joblib.load("models/scaler.pkl")

app = FastAPI(title="HealthCheckAI API")

class PatientInput(BaseModel):
    age: int
    comorbidities: int
    heart_rate: float
    respiratory_rate: float
    systolic_bp: float
    temperature: float
    hemoglobin: float
    creatinine: float
    lactate: float
    white_cell_count: float
    gender: str
    admission_type: str

@app.post("/predict")
def predict(input: PatientInput):
    # One-hot encoding
    gender_male = 1 if input.gender == "Male" else 0
    admission_emergency = 1 if input.admission_type == "Emergency" else 0
    admission_urgent = 1 if input.admission_type == "Urgent" else 0

    features = [[
        input.age,
        input.comorbidities,
        input.heart_rate,
        input.respiratory_rate,
        input.systolic_bp,
        input.temperature,
        input.hemoglobin,
        input.creatinine,
        input.lactate,
        input.white_cell_count,
        gender_male,
        admission_emergency,
        admission_urgent
    ]]

    scaled = scaler.transform(features)
    prediction = model.predict(scaled)[0]
    probability = model.predict_proba(scaled)[0][prediction]

    return {
        "risk_label": int(prediction),
        "confidence": round(float(probability), 4)
    }
