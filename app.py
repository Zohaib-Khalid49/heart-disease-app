import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Heart Disease Prediction", page_icon="❤️")

# Same label encoders as in the notebook (Step 5.1)
def make_encoder(values):
    enc = LabelEncoder()
    enc.fit(np.ravel(values))
    return enc

SEX = ["Female", "Male"]
CHEST_PAIN = ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"]
EXERCISE_ANGINA = ["No", "Yes"]
SLOPE_ST = ["Upsloping", "Flat", "Downsloping"]

sex_enc = make_encoder(SEX)
cp_enc = make_encoder(CHEST_PAIN)
exang_enc = make_encoder(EXERCISE_ANGINA)
slope_enc = make_encoder(SLOPE_ST)

@st.cache_resource
def load_model():
    path = Path(__file__).parent / "heart_svc_trained_model.pkl"
    with open(path, "rb") as f:
        return pickle.load(f)

model = load_model()

st.title("Heart Disease Prediction System")
st.write("Machine Learning Fundamentals (AIC354) - COMSATS University Islamabad, Lahore Campus")
st.caption("Learning project only. Not for real medical decisions.")

sex = st.selectbox("Sex", SEX)
chest_pain = st.selectbox("Chest Pain Type", CHEST_PAIN)
exercise_angina = st.selectbox("Exercise Induced Angina", EXERCISE_ANGINA)
slope_st = st.selectbox("Slope of the Peak Exercise ST Segment", SLOPE_ST)

if st.button("Predict"):
    features = pd.DataFrame({
        "Sex": [sex_enc.transform([sex])[0]],
        "ChestPain": [cp_enc.transform([chest_pain])[0]],
        "ExerciseAngina": [exang_enc.transform([exercise_angina])[0]],
        "SlopeST": [slope_enc.transform([slope_st])[0]],
    })
    prediction = model.predict(features)[0]
    if prediction == 1:
        st.error("Prediction: HEART DISEASE")
    else:
        st.success("Prediction: NO HEART DISEASE")
