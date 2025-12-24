import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Configuración de la página
st.set_page_config(page_title="Asistente de Diagnóstico Cardíaco", layout="centered")

# 1. Cargar el modelo guardado
@st.cache_resource
def load_model():
    return joblib.load('modelo_heart_disease_stacking.pkl')

model = load_model()

st.title("🏥 Sistema de Predicción de Riesgo Cardíaco")
st.write("Ingrese los datos clínicos del paciente para evaluar el riesgo de enfermedad coronaria.")

# 2. Crear la interfaz de usuario (Inputs)
st.sidebar.header("Datos del Paciente")

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Edad", 20, 90, 50)
    sex = st.selectbox("Sexo", ["M", "F"])
    cp = st.selectbox("Tipo de Dolor de Pecho", ["ATA", "NAP", "ASY", "TA"])
    trestbps = st.slider("Presión Arterial en Reposo (mm Hg)", 80, 200, 120)
    chol = st.slider("Colesterol Sérico (mm/dl)", 100, 600, 200)

with col2:
    fbs = st.radio("Azúcar en sangre en ayunas > 120 mg/dl", [1, 0])
    restecg = st.selectbox("Electrocardiograma en Reposo", ["Normal", "ST", "LVH"])
    thalach = st.slider("Frecuencia Cardíaca Máxima", 60, 210, 150)
    exang = st.radio("Angina inducida por ejercicio", ["Y", "N"])
    oldpeak = st.slider("Depresión del ST (Oldpeak)", 0.0, 6.0, 1.0, 0.1)
    slope = st.selectbox("Pendiente del Segmento ST", ["Up", "Flat", "Down"])

# 3. Procesar los datos ingresados
input_data = {
    'Age': age, 'Sex': sex, 'ChestPainType': cp, 'RestingBP': trestbps,
    'Cholesterol': chol, 'FastingBS': fbs, 'RestingECG': restecg,
    'MaxHR': thalach, 'ExerciseAngina': exang, 'Oldpeak': oldpeak, 'ST_Slope': slope
}

# 4. Botón de Predicción
if st.button("Analizar Riesgo"):
    df_input = pd.DataFrame([input_data])
    
    # Obtener probabilidad del modelo
    prob = model.predict_proba(df_input)[0][1]
    
    st.divider()
    
    # Mostrar resultados
    if prob > 0.5:
        st.error(f"### RIESGO ALTO DETECTADO")
        st.write(f"La probabilidad estimada de enfermedad cardíaca es del **{prob*100:.2f}%**.")
    else:
        st.success(f"### RIESGO BAJO")
        st.write(f"La probabilidad estimada de enfermedad cardíaca es del **{prob*100:.2f}%**.")

    st.info("Nota: Este sistema es una herramienta de apoyo basada en datos y no sustituye un diagnóstico médico profesional.")
