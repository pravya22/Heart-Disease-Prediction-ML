import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load model files
@st.cache_resource
def load_model():
    model = joblib.load('lr_model.pkl')
    scaler = joblib.load('scaler.pkl')
    feature_names = joblib.load('feature_names.pkl')
    return model, scaler, feature_names

model, scaler, feature_names = load_model()

st.title("🫀 Heart Disease Predictor")
st.markdown("Powered by Logistic Regression (83.3% accuracy)")

# Sidebar for inputs
st.sidebar.header("Patient Information")
age = st.sidebar.slider("Age", 29, 77, 50)
sex = st.sidebar.selectbox("Sex", ["Male", "Female"])
chest_pain = st.sidebar.selectbox("Chest Pain Type", 
    ["typical angina", "atypical angina", "non-anginal pain", "asymptomatic"])
trestbps = st.sidebar.slider("Resting Blood Pressure (mm Hg)", 94, 200, 130)
chol = st.sidebar.slider("Cholesterol (mg/dl)", 126, 564, 246)
fbs = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dl", ["False", "True"])
restecg = st.sidebar.selectbox("Resting ECG Results", 
    ["normal", "ST-T wave abnormality", "left ventricular hypertrophy"])
thalach = st.sidebar.slider("Maximum Heart Rate", 71, 202, 150)
exang = st.sidebar.selectbox("Exercise Induced Angina", ["No", "Yes"])
oldpeak = st.sidebar.slider("ST Depression", 0.0, 6.2, 1.0)
slope = st.sidebar.selectbox("Slope of Peak Exercise ST", ["upsloping", "flat", "downsloping"])
ca = st.sidebar.slider("Major Vessels (0-3)", 0, 4, 0)
thal = st.sidebar.selectbox("Thalassemia", ["normal", "fixed defect", "reversable defect"])

# Create EXACT feature vector matching training data
if st.button("🔮 Predict Disease Risk", type="primary"):
    # Initialize with zeros (exact length as training)
    input_data = np.zeros(len(feature_names))
    
    # Map inputs to correct positions (handles ANY feature order)
    feature_mapping = {
        'age': age,
        'sex': 1 if sex == "Male" else 0,
        'cp': {"typical angina": 0, "atypical angina": 1, "non-anginal pain": 2, "asymptomatic": 3}[chest_pain],
        'trestbps': trestbps,
        'chol': chol,
        'fbs': 1 if fbs == "True" else 0,
        'restecg': {"normal": 0, "ST-T wave abnormality": 1, "left ventricular hypertrophy": 2}[restecg],
        'thalch': thalach,
        'exang': 1 if exang == "Yes" else 0,
        'oldpeak': oldpeak,
        'slope': {"upsloping": 2, "flat": 1, "downsloping": 0}[slope],
        'ca': ca,
        'thal': {"normal": 0, "fixed defect": 1, "reversable defect": 2}[thal]
    }
    
    # Fill only features that exist in training data
    for feature, value in feature_mapping.items():
        if feature in feature_names:
            idx = feature_names.index(feature)
            input_data[idx] = value
    
    # Scale and predict
    input_scaled = scaler.transform(input_data.reshape(1, -1))
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]
    
    # Results
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Prediction", "🛑 DISEASE" if prediction == 1 else "✅ LOW RISK", 
                 "HIGH RISK" if probability > 0.7 else "LOW RISK")
    with col2:
        st.metric("Disease Probability", f"{probability:.1%}")
    
    # Risk interpretation
    if prediction == 1:
        st.error("⚠️ **HIGH RISK** - Patient likely has heart disease")
        if probability > 0.8:
            st.warning("🚨 **CRITICAL** - Immediate medical attention recommended")
    else:
        st.success("✅ **LOW RISK** - Patient appears healthy")
    
    st.info(f"**Confidence:** {max(model.predict_proba(input_scaled)[0])*100:.1f}%")

st.markdown("---")
st.markdown("*Built with your Logistic Regression model (83.3% test accuracy)*")





