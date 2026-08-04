import streamlit as st
import pandas as pd
import joblib
import os

from predict import predict_student

# ----------------------------------------
# Page Configuration
# ----------------------------------------

st.set_page_config(
    page_title="Student Dropout Prediction",
    page_icon="🎓",
    layout="wide"
)

# ----------------------------------------
# Load Saved Files
# ----------------------------------------

DATA_PATH = "data/student_dropout.csv"

MODEL_PATH = "model/student_dropout_model.pkl"
DEFAULT_VALUES_PATH = "model/default_values.pkl"

try:
    model = joblib.load(MODEL_PATH)
    default_values = joblib.load(DEFAULT_VALUES_PATH)

except FileNotFoundError:

    st.error("❌ Model files not found.")

    st.info("Please run train_model.py first.")

    st.stop()

# ----------------------------------------
# Page Title
# ----------------------------------------

st.title("🎓 Student Dropout Prediction System")

st.markdown(
"""
This application predicts whether a student is likely to **Graduate**
or **Drop Out** using Machine Learning.
"""
)

st.divider()

# ----------------------------------------
# Sidebar
# ----------------------------------------

st.sidebar.title("Navigation")

page = st.sidebar.radio(

    "Select Page",

    [

        "🏠 Home",
        "📊 Dataset",
        "🎯 Prediction",
        "ℹ️ About"

    ]

)
