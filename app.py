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

# =====================================================
# DATASET PAGE
# =====================================================

elif page == "📊 Dataset":

    st.title("📊 Dataset Overview")

    st.markdown("---")

    display_dataset_info(df)

    st.markdown("---")

    st.subheader("Dataset Preview")

    display_dataset(df)

    st.markdown("---")

    st.subheader("Target Distribution")

    plot_target_distribution(df)

    st.markdown("---")

    st.subheader("Dataset Columns")

    columns_df = pd.DataFrame({

        "Column Number": range(1, len(df.columns) + 1),

        "Column Name": df.columns

    })

    st.dataframe(

        columns_df,

        use_container_width=True,

        hide_index=True

    )

    st.markdown("---")

    st.subheader("Summary Statistics")

    st.dataframe(

        df.describe(),

        use_container_width=True

    )

    # ===============================================

    # Feature Importance

    # ===============================================

    if os.path.exists(FEATURE_IMPORTANCE):

        st.markdown("---")

        st.subheader("Top Important Features")

        importance_df = pd.read_csv(

            FEATURE_IMPORTANCE

        )

        plot_feature_importance(

            importance_df

        )

        st.dataframe(

            importance_df,

            use_container_width=True

        )

    else:

        st.info(

            "Run train_model.py to generate feature importance."

        )


# =====================================================
# ABOUT PAGE
# =====================================================

elif page == "ℹ️ About":

    st.title("ℹ️ About")

    st.markdown("---")

    st.subheader("Project")

    st.write("""

Student Dropout Prediction System

This project predicts whether a student is likely to

Graduate

or

Dropout

using Machine Learning Classification.

""")

    st.markdown("---")

    st.subheader("Technology Stack")

    tech = pd.DataFrame({

        "Technology": [

            "Python",

            "Streamlit",

            "Pandas",

            "NumPy",

            "Scikit-Learn",

            "Plotly",

            "Joblib"

        ]

    })

    st.table(tech)

    st.markdown("---")

    st.subheader("Machine Learning Algorithms")

    algo = pd.DataFrame({

        "Algorithms": [

            "Logistic Regression",

            "Decision Tree",

            "Random Forest",

            "Gradient Boosting"

        ]

    })

    st.table(algo)

    st.markdown("---")

    st.subheader("Dataset")

    st.write("""

Source:

Predict Students' Dropout and Academic Success Dataset

University student academic records

Binary Classification

Graduate vs Dropout

""")

# =====================================================
# PREDICTION PAGE
# =====================================================

elif page == "🎯 Prediction":

    st.title("🎯 Student Dropout Prediction")

    st.markdown("---")

    st.write(
        "Fill in the student's academic details and click **Predict**."
    )

    student = default_values.copy()

    col1, col2 = st.columns(2)

    # ============================================
    # LEFT COLUMN
    # ============================================

    with col1:

        student["Course"] = st.number_input(
            "Course Code",
            min_value=0,
            value=int(default_values["Course"])
        )

        student["Admission grade"] = st.number_input(
            "Admission Grade",
            min_value=0.0,
            max_value=200.0,
            value=float(default_values["Admission grade"])
        )

        student["Previous qualification (grade)"] = st.number_input(
            "Previous Qualification Grade",
            min_value=0.0,
            max_value=200.0,
            value=float(default_values["Previous qualification (grade)"])
        )

        student["Age at enrollment"] = st.number_input(
            "Age at Enrollment",
            min_value=15,
            max_value=80,
            value=int(default_values["Age at enrollment"])
        )

        student["Gender"] = st.selectbox(

            "Gender",

            [0, 1],

            index=int(default_values["Gender"])

        )

        student["Debtor"] = st.selectbox(

            "Debtor",

            [0, 1],

            index=int(default_values["Debtor"])

        )

    # ============================================
    # RIGHT COLUMN
    # ============================================

    with col2:

        student["Tuition fees up to date"] = st.selectbox(

            "Tuition Fees Up To Date",

            [0, 1],

            index=int(default_values["Tuition fees up to date"])

        )

        student["Scholarship holder"] = st.selectbox(

            "Scholarship Holder",

            [0, 1],

            index=int(default_values["Scholarship holder"])

        )

        student["Curricular units 1st sem (approved)"] = st.number_input(

            "1st Semester Approved Units",

            min_value=0,

            value=int(
                default_values[
                    "Curricular units 1st sem (approved)"
                ]
            )

        )

        student["Curricular units 1st sem (grade)"] = st.number_input(

            "1st Semester Grade",

            min_value=0.0,

            max_value=20.0,

            value=float(
                default_values[
                    "Curricular units 1st sem (grade)"
                ]
            )

        )

        student["Curricular units 2nd sem (approved)"] = st.number_input(

            "2nd Semester Approved Units",

            min_value=0,

            value=int(
                default_values[
                    "Curricular units 2nd sem (approved)"
                ]
            )

        )

        student["Curricular units 2nd sem (grade)"] = st.number_input(

            "2nd Semester Grade",

            min_value=0.0,

            max_value=20.0,

            value=float(
                default_values[
                    "Curricular units 2nd sem (grade)"
                ]
            )

        )

    st.markdown("---")

    predict = st.button(

        "Predict Student Status",

        use_container_width=True

    )

    # ============================================
    # Prediction
    # ============================================

    if predict:

        prediction, confidence = predict_student(student)

        st.markdown("---")

        show_prediction(

            prediction,

            confidence

        )

        st.markdown("---")

        st.subheader("Recommendation")

        if prediction == "Dropout":

            st.error("""

The student appears to be at risk.

Suggested interventions:

• Academic counselling

• Financial assistance

• Attendance monitoring

• Faculty mentoring

• Psychological support

""")

        else:

            st.success("""

The student is likely to graduate successfully.

Suggested actions:

• Continue academic support

• Encourage internships

• Promote skill development

• Career guidance

""")

# =====================================================
# FOOTER
# =====================================================

st.markdown("---")

st.caption(
    "Student Dropout Prediction System | "
    "Developed using Python, Streamlit and Scikit-Learn"
)

# =====================================================
# SIDEBAR MODEL INFORMATION
# =====================================================

st.sidebar.markdown("---")

st.sidebar.subheader("Project Information")

st.sidebar.write("**Dataset:**")
st.sidebar.write("Predict Students' Dropout and Academic Success")

st.sidebar.write("**Classification Type:**")
st.sidebar.write("Binary Classification")

st.sidebar.write("**Algorithms:**")
st.sidebar.write("""
• Logistic Regression

• Decision Tree

• Random Forest

• Gradient Boosting
""")

# =====================================================
# MODEL DETAILS
# =====================================================

BEST_MODEL_FILE = os.path.join(
    MODEL_FOLDER,
    "best_model.txt"
)

MODEL_ACCURACY_FILE = os.path.join(
    MODEL_FOLDER,
    "model_accuracy.csv"
)

if os.path.exists(BEST_MODEL_FILE):

    with open(BEST_MODEL_FILE, "r") as f:

        best_model = f.read()

    st.sidebar.success(
        f"Best Model\n\n{best_model}"
    )

if os.path.exists(MODEL_ACCURACY_FILE):

    accuracy_df = pd.read_csv(
        MODEL_ACCURACY_FILE
    )

    st.sidebar.dataframe(
        accuracy_df,
        use_container_width=True,
        hide_index=True
    )

# =====================================================
# END OF FILE
# =====================================================
