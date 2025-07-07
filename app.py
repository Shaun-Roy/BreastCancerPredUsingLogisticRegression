import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load model and scaler
model = joblib.load("logistic_regression_model.pkl")
scaler = joblib.load("scaler.pkl")

# Define feature list
feature_names = [
    'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean',
    'compactness_mean', 'concavity_mean', 'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
    'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
    'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se', 'fractal_dimension_se',
    'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst', 'smoothness_worst',
    'compactness_worst', 'concavity_worst', 'concave points_worst', 'symmetry_worst', 'fractal_dimension_worst'
]

# Load original dataset just to get realistic slider ranges
data = pd.read_csv("data.csv")
data = data.drop(["id", "Unnamed: 32"], axis=1)
data["diagnosis"] = data["diagnosis"].map({"M": 1, "B": 0})

st.set_page_config(page_title="Breast Cancer Predictor", layout="wide")
st.title("🔬 Breast Cancer Diagnosis Predictor")
st.write("Adjust the sliders below based on test results to predict whether a tumor is **benign** or **malignant**.")

input_data = []



# Create sliders in three columns
cols = st.columns(3)

for i, feature in enumerate(feature_names):
    min_val = float(data[feature].min())
    max_val = float(data[feature].max())
    mean_val = float(data[feature].mean())
    
    col = cols[i % 3]
    val = col.slider(f"{feature}", min_val, max_val, mean_val)
    input_data.append(val)

# Convert input to DataFrame
input_df = pd.DataFrame([input_data], columns=feature_names)

# Scale and predict
input_scaled = scaler.transform(input_df)
prediction = model.predict(input_scaled)[0]
proba = model.predict_proba(input_scaled)[0]

# Display results


# Confidence block - centered and styled
benign_pct = proba[0] * 100
malignant_pct = proba[1] * 100

st.markdown(
    f"""
    <div style="text-align: center; padding-top: 1rem;">
        <h3>📊 Prediction Confidence</h3>
        <p style="font-size: 22px;">🟢 Benign: <strong>{benign_pct:.2f}%</strong></p>
        <p style="font-size: 22px;">🔴 Malignant: <strong>{malignant_pct:.2f}%</strong></p>
    </div>
    """,
    unsafe_allow_html=True
)
