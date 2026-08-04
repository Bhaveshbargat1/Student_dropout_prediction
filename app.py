"""
app.py

Student Dropout Prediction System
Developed using Streamlit
"""

import streamlit as st
import pandas as pd
import joblib
import os

from predict import predict_student

from utils.preprocessing import (
    load_dataset,
    get_dataset_information
)

from utils.helper import (
    display_dataset,
    display_dataset_info,
    plot_target_distribution,
    plot_feature_importance,
    show_prediction
)

# =====================================================
# Streamlit Configuration
# =====================================================

st.set_page_config(

    page_title="Student Dropout Prediction System",

    page_icon="🎓",

    layout="wide",

    initial_sidebar_state="expanded"

)

# =====================================================
# Constants
# =====================================================

DATA_PATH = "data/student_dropout.csv"

MODEL_FOLDER = "model"

FEATURE_IMPORTANCE = os.path.join(

    MODEL_FOLDER,

    "feature_importance.csv"

)

DEFAULT_VALUES = os.path.join(

    MODEL_FOLDER,

    "default_values.pkl"

)

# =====================================================
# Load Dataset
# =====================================================

try:

    df = load_dataset(DATA_PATH)

except Exception as e:

    st.error(f"Dataset Error : {e}")

    st.stop()

# =====================================================
# Load Default Values
# =====================================================

try:

    default_values = joblib.load(DEFAULT_VALUES)

except Exception:

    st.error(
        "Model files not found.\nRun train_model.py first."
    )

    st.stop()

# =====================================================
# Sidebar
# =====================================================

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

st.sidebar.markdown("---")

st.sidebar.info(

"""
Student Dropout Prediction

B.Tech Mini Project

Artificial Intelligence & Data Science

"""

)

# =====================================================
# HOME PAGE
# =====================================================

if page == "🏠 Home":

    st.title("🎓 Student Dropout Prediction System")

    st.markdown("---")

    st.subheader("Project Overview")

    st.write(

"""
This application predicts whether a student is likely to

- Graduate
- Dropout

using Machine Learning Classification algorithms.

The objective is to identify students who are at risk
of dropping out so that educational institutions can
take preventive actions.
"""

    )

    st.markdown("---")

    info = get_dataset_information(df)

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(

        "Rows",

        info["rows"]

    )

    c2.metric(

        "Columns",

        info["columns"]

    )

    c3.metric(

        "Missing Values",

        info["missing_values"]

    )

    c4.metric(

        "Duplicate Rows",

        info["duplicate_rows"]

    )

    st.markdown("---")

    st.subheader("Algorithms Used")

    st.write("""

✔ Logistic Regression

✔ Decision Tree

✔ Random Forest

✔ Gradient Boosting

The application automatically selects the best-performing model during training.

""")
