
import requests

url = "http://localhost:8000/predict"

sample_payload = {
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

response = requests.post(url, json=sample_payload)

print("Status Code:", response.status_code)
print("Response JSON:", response.json())
