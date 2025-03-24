
# 🏥 HealthCheckAI — Predictive Patient Risk Monitoring

HealthCheckAI is an end-to-end machine learning project that predicts the risk of ICU transfer for hospital patients using vital signs and lab results.

---

## 📦 Features

- Predicts patient deterioration risk using a Random Forest Classifier
- Trained on synthetic EHR data (age, vitals, labs, etc.)
- SHAP explainability for feature attribution
- Streamlit frontend for live clinical risk prediction
- REST API via FastAPI for system integration
- Docker & Docker Compose setup for deployment

---

## 🚀 Quickstart

### 1. Clone and Build
```bash
git clone https://github.com/yourusername/healthcheckai.git
cd healthcheckai
docker-compose up --build
```

- Streamlit: [http://localhost:8501](http://localhost:8501)
- FastAPI Docs: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 📊 Dataset

The project uses synthetic data with features like:
- `age`, `gender`, `admission_type`, `comorbidities`
- `heart_rate`, `respiratory_rate`, `systolic_bp`, `temperature`
- `hemoglobin`, `creatinine`, `lactate`, `white_cell_count`

Label: `risk_label` = 1 indicates ICU transfer risk.

---

## 🛠 Tech Stack

- Python, Pandas, scikit-learn, SHAP
- Streamlit, FastAPI, Docker
- Synthetic data generated for reproducibility

---

## 📂 File Structure

```
├── data/                  # Raw and processed datasets
├── models/                # Trained model, scaler, SHAP plots
├── app/
│   ├── streamlit_app.py   # Streamlit UI
│   └── api.py             # FastAPI backend
├── notebooks/             # EDA and feature engineering
├── Dockerfile             # Container build
├── docker-compose.yml     # Multi-service orchestration
├── requirements.txt       # Python dependencies
└── README.md              # Project overview
```

---

## 👩‍⚕️ Example Prediction

Send a POST request to `/predict`:

```json
{
  "age": 70,
  "comorbidities": 3,
  "heart_rate": 105,
  "respiratory_rate": 25,
  "systolic_bp": 85,
  "temperature": 38.1,
  "hemoglobin": 11.5,
  "creatinine": 1.2,
  "lactate": 3.0,
  "white_cell_count": 9.5,
  "gender": "Male",
  "admission_type": "Emergency"
}
```

---

## 📜 License

MIT
