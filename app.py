import joblib
import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
from streamlit_extras.stylable_container import stylable_container

# Load saved model and scaler
model = joblib.load("best_model.pkl")
scaler = joblib.load("heart-disease-scaler.pkl")  # Make sure this scaler file exists

# Define heart disease feature names
feature_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
                 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']

# Load and resize the image (optional heart-related image)
  # Make sure this image exists


# App title and instructions
st.title("Heart Disease Prediction")
st.markdown("Enter the clinical values below to predict the likelihood of heart disease.")

# Collect user input
def get_user_input():
    user_data = {}
    with st.expander("Enter Feature Values"):
        for feature in feature_names:
            user_data[feature] = st.text_input(f"Enter value for {feature}", "0.0")
    df = pd.DataFrame([user_data])
    return df

input_df = get_user_input()

# Show input data
st.subheader("Your Data :")
st.data_editor(input_df, num_rows="fixed", use_container_width=True)

# Make prediction
if st.button("Predict", use_container_width=True):
    try:
        input_array = input_df.astype(float).to_numpy().reshape(1, -1)
        scaled_input = scaler.transform(input_array)

        prediction = model.predict(scaled_input)
        prediction_proba = model.predict_proba(scaled_input)

        with stylable_container("prediction-box", css_styles="background-color: #f0f0f5; padding: 50px; border-radius: 10px; text-align: center;"):
            st.write("## **Heart Disease Detected**" if prediction[0] == 1 else "## **No Heart Disease**")

            st.subheader("Prediction Probability:")
            st.write(f"**No Heart Disease:** {prediction_proba[0][0] * 100:.2f}%")
            st.write(f"**Heart Disease:** {prediction_proba[0][1] * 100:.2f}%")

    except Exception as e:
        st.error(f"Error: {e}")
