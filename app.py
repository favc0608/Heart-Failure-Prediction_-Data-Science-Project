import streamlit as st
import joblib
import pandas as pd
import numpy as np
import logging
import json
from pathlib import Path

# Page configuration (English)
st.set_page_config(page_title="Cardiac Risk Prediction Assistant", layout="centered")

# Keep model-loading unchanged per request
@st.cache_resource
def load_model():
    return joblib.load("model/modelo_heart_disease_stacking.pkl")

model = load_model()

# Logging setup
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
logging.basicConfig(
    filename=LOG_DIR / "app.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)

st.title("Cardiac Risk Prediction Assistant Model - DEMO")
st.write("Enter patient clinical data below to obtain a model-based risk estimate for coronary heart disease.")

# Sidebar and presets
st.sidebar.header("Patient data")

def set_preset(preset: str):
    """Populate `st.session_state` with example values for quick demo."""
    if preset == "low":
        st.session_state.update({
            "age": 45,
            "sex": "M",
            "cp": "ATA",
            "trestbps": 118,
            "chol": 190,
            "fbs": 0,
            "restecg": "Normal",
            "thalach": 160,
            "exang": "N",
            "oldpeak": 0.2,
            "slope": "Up",
        })
    else:
        st.session_state.update({
            "age": 68,
            "sex": "M",
            "cp": "ASY",
            "trestbps": 150,
            "chol": 280,
            "fbs": 1,
            "restecg": "ST",
            "thalach": 110,
            "exang": "Y",
            "oldpeak": 3.2,
            "slope": "Flat",
        })

st.sidebar.write("Quick presets:")
colp1, colp2 = st.sidebar.columns(2)
if colp1.button("Load low-risk example"):
    set_preset("low")
if colp2.button("Load high-risk example"):
    set_preset("high")

# Input widgets (use session_state defaults when available)
col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 20, 90, value=st.session_state.get("age", 50), help="Patient age in years", key="age")
    sex = st.selectbox("Sex", ["M", "F"], index=0 if st.session_state.get("sex", "M") == "M" else 1, key="sex", help="Biological sex: M or F")
    cp = st.selectbox("Chest pain type", ["ATA", "NAP", "ASY", "TA"], index=0, key="cp", help="Typical/atypical/other chest pain encoding")
    trestbps = st.slider("Resting blood pressure (mm Hg)", 80, 200, value=st.session_state.get("trestbps", 120), key="trestbps", help="Resting systolic blood pressure")
    chol = st.slider("Serum cholesterol (mg/dl)", 100, 600, value=st.session_state.get("chol", 200), key="chol", help="Total serum cholesterol")

with col2:
    fbs = st.radio("Fasting blood sugar > 120 mg/dl?", [1, 0], index=0 if st.session_state.get("fbs", 0) == 1 else 1, key="fbs", help="1 = true, 0 = false")
    restecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"], index=0, key="restecg", help="Resting electrocardiographic results")
    thalach = st.slider("Maximum heart rate achieved", 60, 210, value=st.session_state.get("thalach", 150), key="thalach", help="Maximum heart rate recorded during exercise test")
    exang = st.radio("Exercise-induced angina?", ["Y", "N"], index=0 if st.session_state.get("exang", "N") == "Y" else 1, key="exang", help="Y = yes, N = no")
    oldpeak = st.slider("ST depression (Oldpeak)", 0.0, 6.0, value=st.session_state.get("oldpeak", 1.0), step=0.1, key="oldpeak", help="ST depression induced by exercise relative to rest")
    slope = st.selectbox("ST slope", ["Up", "Flat", "Down"], index=0, key="slope", help="Slope of the peak exercise ST segment")

# Build input dictionary and validation
input_data = {
    "Age": age,
    "Sex": sex,
    "ChestPainType": cp,
    "RestingBP": trestbps,
    "Cholesterol": chol,
    "FastingBS": fbs,
    "RestingECG": restecg,
    "MaxHR": thalach,
    "ExerciseAngina": exang,
    "Oldpeak": oldpeak,
    "ST_Slope": slope,
}

def validate_input(d: dict) -> (bool, str):
    """Validate types and ranges; return (valid, message)."""
    if d["Age"] < 18 or d["Age"] > 120:
        return False, "Age outside plausible range (18-120)."
    if d["RestingBP"] < 50 or d["RestingBP"] > 300:
        return False, "Resting blood pressure outside plausible range."
    if d["Cholesterol"] <= 0 or d["Cholesterol"] > 2000:
        return False, "Cholesterol value appears invalid."
    if d["Oldpeak"] < 0:
        return False, "Oldpeak cannot be negative."
    return True, ""

def format_inputs_for_model(d: dict) -> pd.DataFrame:
    df = pd.DataFrame([d])
    # enforce dtypes
    df["Age"] = df["Age"].astype(int)
    df["RestingBP"] = df["RestingBP"].astype(int)
    df["Cholesterol"] = df["Cholesterol"].astype(int)
    df["FastingBS"] = df["FastingBS"].astype(int)
    df["MaxHR"] = df["MaxHR"].astype(int)
    df["Oldpeak"] = df["Oldpeak"].astype(float)
    # Ensure column order expected by pipeline
    expected_cols = ["Age","Sex","ChestPainType","RestingBP","Cholesterol","FastingBS","RestingECG","MaxHR","ExerciseAngina","Oldpeak","ST_Slope"]
    for c in expected_cols:
        if c not in df.columns:
            df[c] = np.nan
    return df[expected_cols]

st.divider()

if st.button("Analyze risk"):
    valid, msg = validate_input(input_data)
    if not valid:
        st.error(f"Invalid input: {msg}")
        logging.warning("Input validation failed: %s", msg)
    else:
        df_input = format_inputs_for_model(input_data)
        try:
            # Safe prediction logic: prefer predict_proba, then decision_function, then predict
            if hasattr(model, "predict_proba"):
                prob = float(model.predict_proba(df_input)[0, 1])
            elif hasattr(model, "decision_function"):
                score = model.decision_function(df_input)[0]
                prob = float(1.0 / (1.0 + np.exp(-score)))
            else:
                pred = model.predict(df_input)[0]
                prob = float(pred)
        except Exception as e:
            st.error("An error occurred during prediction. Please check input types and model compatibility.")
            logging.exception("Prediction error:")
        else:
            # Display polished results
            st.subheader("Prediction result")
            pct = prob * 100.0
            st.metric(label="Estimated probability of heart disease", value=f"{pct:.2f}%")
            if prob > 0.5:
                st.warning("HIGH RISK — consult a clinician for further evaluation.")
            else:
                st.success("LOW RISK — consider routine follow-up and clinical correlation.")

            # Persist a small log (no PHI) and provide downloadable result
            out = {"probability": prob, "risk": ("High" if prob > 0.5 else "Low")}
            logging.info("Prediction made: %s", json.dumps({"inputs": {
                "Age": int(df_input.loc[0, "Age"]),
                "Sex": df_input.loc[0, "Sex"],
                "ChestPainType": df_input.loc[0, "ChestPainType"]
            }, "result": out}))

            st.download_button("Download result (JSON)", data=json.dumps({"inputs": input_data, "result": out}, indent=2), file_name="prediction_result.json", mime="application/json")

            st.caption("This application is for demonstration purposes only ")
