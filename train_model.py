
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import shap
import matplotlib.pyplot as plt

# Load processed data
df = pd.read_csv("data/processed_ehr.csv")

# Features and label
X = df.drop("risk_label", axis=1)
y = df["risk_label"]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train model
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))

# Save model and scaler
joblib.dump(clf, "models/risk_classifier.pkl")
joblib.dump(scaler, "models/scaler.pkl")

# SHAP Explainability
explainer = shap.Explainer(clf, X)
shap_values = explainer(X)

# Plot summary
shap.summary_plot(shap_values, X, show=False)
plt.tight_layout()
plt.savefig("models/shap_summary_plot.png")
