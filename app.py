import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load model, scaler, and features
@st.cache_resource
def load_model():
    model = joblib.load('lr_model.pkl')
    scaler = joblib.load('scaler.pkl')
    feature_names = joblib.load('feature_names.pkl')
    return model, scaler, feature_names

model, scaler, feature_names = load_model()

st.title("🫀 Heart Disease Predictor")
st.markdown("Enter patient details to get disease risk prediction")

# Create input fields for key features (adjust based on your feature_names)
col1, col2 = st.columns(2)
with col1:
    age = st.slider("Age", 20, 80, 50)
    sex = st.selectbox("Sex", ["Male", "Female"])
    cp = st.selectbox("Chest Pain Type", ["typical angina", "atypical angina", "non-anginal", "asymptomatic"])
    trestbps = st.slider("Resting Blood Pressure", 90, 200, 120)
    
with col2:
    chol = st.slider("Cholesterol", 100, 600, 250)
    fbs = st.selectbox("Fasting Blood Sugar >120 mg/dl", ["True", "False"])
    restecg = st.selectbox("Resting ECG", ["normal", "ST-T wave", "lv hypertrophy"])
    thalch = st.slider("Max Heart Rate", 70, 220, 150)

if st.button("🔮 Predict Risk", type="primary"):
    # Create input array (must match exact feature order!)
    input_data = np.zeros(len(feature_names))
    
    # Map inputs to correct positions (simplified - update with your exact feature_names)
    input_data[feature_names.index('age')] = (age - 55) / 10  # rough scaling
    input_data[feature_names.index('sex')] = 1 if sex == "Male" else 0
    # Add other mappings...
    
    input_scaled = scaler.transform(input_data.reshape(1, -1))
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]
    
    st.success(f"**Prediction:** {'🛑 HEART DISEASE' if prediction == 1 else '✅ LOW RISK'}")
    st.info(f"**Disease Probability:** {probability:.1%}")




