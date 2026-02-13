import streamlit as st
import pickle
import numpy as np

st.title("Heart Disease Prediction")

# Load saved model
model = pickle.load(open("heart_model.pkl", "rb"))

age = st.number_input("Age")
sex = st.selectbox("Sex (0 = Female, 1 = Male)", [0,1])
chol = st.number_input("Cholesterol")

if st.button("Predict"):
    data = np.array([[age, sex, chol]])
    result = model.predict(data)

    if result[0] == 1:
        st.error("High Risk of Heart Disease")
    else:
        st.success("Low Risk of Heart Disease")

