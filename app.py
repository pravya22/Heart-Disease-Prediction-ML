import streamlit as st
import joblib
import pandas as pd

@st.cache_resource
def load_all():
    model = joblib.load('lr_model.pkl')
    scaler = joblib.load('scaler.pkl')
    sample_input = joblib.load('sample_input.pkl')
    feature_names = joblib.load('feature_names.pkl')
    return model, scaler, sample_input, feature_names

model, scaler, sample_input, feature_names = load_all()

st.title("🫀 Heart Disease Predictor")
st.success(f"✅ Loaded | {len(feature_names)} features")

with st.form("predict"):
    st.header("Patient Data")
    
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", 29, 77, 50)
        trestbps = st.number_input("Resting BP", 94, 200, 130)
        chol = st.number_input("Cholesterol", 126, 564, 246)
    
    with col2:
        thalch = st.number_input("Max Heart Rate", 71, 202, 150)
        oldpeak = st.number_input("Oldpeak", 0.0, 6.2, 1.0)
        ca = st.number_input("CA Vessels", 0, 4, 0)
    
    sex = st.selectbox("Sex", [0, 1])
    fbs = st.selectbox("FBS>120?", [0, 1])
    cp = st.selectbox("Chest Pain", [0, 1, 2, 3])
    exang = st.selectbox("Exercise Angina", [0, 1])
    slope = st.selectbox("Slope", [0, 1, 2])
    thal = st.selectbox("Thal", [0, 1, 2])
    restecg = st.selectbox("Rest ECG", [0, 1, 2])
    
    if st.form_submit_button("🔮 Predict"):
        # Copy EXACT structure from training
        input_df = sample_input.copy()
        
        # Update values safely
        updates = {
            'age': age, 'trestbps': trestbps, 'chol': chol, 'thalch': thalch,
            'oldpeak': oldpeak, 'ca': ca, 'sex': sex, 'fbs': fbs,
            'cp': cp, 'exang': exang, 'slope': slope, 'thal': thal, 'restecg': restecg
        }
        
        for col, val in updates.items():
            if col in input_df.columns:
                input_df.at[0, col] = val
        
        # Scale & predict (EXACTLY like training)
        input_scaled = scaler.transform(input_df)
        pred = model.predict(input_scaled)[0]
        prob = model.predict_proba(input_scaled)[0][1]
        
        # Results
        if pred == 1:
            st.error("🛑 **HEART DISEASE**")
        else:
            st.success("✅ **LOW RISK**")
        st.metric("Risk", f"{prob:.1%}")
        st.balloons()






