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

# ----------------------------------------
# PREDICTION PAGE
# ----------------------------------------

elif page == "🎯 Prediction":

    st.header("Student Dropout Prediction")

    st.write("Enter the student's academic information below.")

    st.divider()

    col1, col2 = st.columns(2)

    with col1:

        course = st.number_input(
            "Course",
            min_value=1,
            max_value=9999,
            value=9500
        )

        admission_grade = st.slider(
            "Admission Grade",
            0.0,
            200.0,
            130.0
        )

        previous_grade = st.slider(
            "Previous Qualification Grade",
            0.0,
            200.0,
            130.0
        )

        age = st.number_input(
            "Age at Enrollment",
            17,
            70,
            20
        )

        gender = st.selectbox(
            "Gender",
            ["Male", "Female"]
        )

        debtor = st.selectbox(
            "Debtor",
            ["No", "Yes"]
        )

    with col2:

        tuition = st.selectbox(
            "Tuition Fees Up To Date",
            ["Yes", "No"]
        )

        scholarship = st.selectbox(
            "Scholarship Holder",
            ["No", "Yes"]
        )

        sem1_approved = st.number_input(
            "1st Semester Approved Subjects",
            0,
            20,
            6
        )

        sem1_grade = st.slider(
            "1st Semester Average Grade",
            0.0,
            20.0,
            13.0
        )

        sem2_approved = st.number_input(
            "2nd Semester Approved Subjects",
            0,
            20,
            6
        )

        sem2_grade = st.slider(
            "2nd Semester Average Grade",
            0.0,
            20.0,
            13.0
        )

    predict_button = st.button(
        "Predict Student Status",
        use_container_width=True
    )

    if predict_button:

        student = default_values.copy()

        student["Course"] = course
        student["Admission grade"] = admission_grade
        student["Previous qualification (grade)"] = previous_grade
        student["Age at enrollment"] = age

        student["Gender"] = 1 if gender == "Male" else 0

        student["Debtor"] = 1 if debtor == "Yes" else 0

        student["Tuition fees up to date"] = (
            1 if tuition == "Yes" else 0
        )

        student["Scholarship holder"] = (
            1 if scholarship == "Yes" else 0
        )

        student["Curricular units 1st sem (approved)"] = sem1_approved

        student["Curricular units 1st sem (grade)"] = sem1_grade

        student["Curricular units 2nd sem (approved)"] = sem2_approved

        student["Curricular units 2nd sem (grade)"] = sem2_grade

        prediction, confidence = predict_student(student)

        st.divider()

        if prediction == "Graduate":

            st.success(
                f"Prediction : {prediction}"
            )

        else:

            st.error(
                f"Prediction : {prediction}"
            )

        st.metric(
            "Confidence",
            f"{confidence}%"
        )

        if confidence >= 90:

            st.success("High Confidence Prediction")

        elif confidence >= 70:

            st.warning("Moderate Confidence")

        else:

            st.info("Low Confidence")
