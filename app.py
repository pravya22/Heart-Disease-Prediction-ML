import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

st.title("Heart Disease Prediction App ❤️")

# Load dataset
df = pd.read_csv("https://raw.githubusercontent.com/plotly/datasets/master/heart.csv")

X = df.drop("target", axis=1)
y = df["target"]

# Train model
model = RandomForestClassifier()
model.fit(X, y)

st.subheader("Enter Patient Details")

age = st.number_input("Age", 1, 120)
chol = st.number_input("Cholesterol")
thalach = st.number_input("Max Heart Rate")

if st.button("Predict"):
    input_data = [[age, 1, 2, 120, chol, 0, 0, thalach, 0, 0, 1, 0, 2]]
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.error("⚠️ High Risk of Heart Disease")
    else:
        st.success("✅ Low Risk of Heart Disease")

