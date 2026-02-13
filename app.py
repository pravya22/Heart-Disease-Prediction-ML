import streamlit as st
import joblib
import numpy as np
scaler = joblib.load('scaler.pkl')  # Save scaler too
model = joblib.load('lr_model.pkl')
st.title('Heart Disease Predictor')
# Add inputs for 15 features (id, age_scaled, sex_encoded, etc.)
input_data = np.array([...]).reshape(1, -1)
input_scaled = scaler.transform(input_data)
pred = model.predict(input_scaled)
st.write('Prediction:', 'Disease' if pred[0]==1 else 'No Disease')



