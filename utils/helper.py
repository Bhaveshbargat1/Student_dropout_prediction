"""
helper.py

Reusable helper functions for the Student Dropout Prediction
Streamlit application.
"""

import streamlit as st
import pandas as pd
import plotly.express as px


# ======================================================
# Message Functions
# ======================================================

def show_success(message):
    """Display success message."""
    st.success(message)


def show_error(message):
    """Display error message."""
    st.error(message)


def show_warning(message):
    """Display warning message."""
    st.warning(message)


def show_info(message):
    """Display information message."""
    st.info(message)


# ======================================================
# Dataset Functions
# ======================================================

def display_dataset(df, rows=10):
    """
    Display dataset preview.
    """
    st.dataframe(df.head(rows), use_container_width=True)


def display_dataset_info(df):
    """
    Display dataset statistics.
    """

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Rows", df.shape[0])
    col2.metric("Columns", df.shape[1])
    col3.metric("Missing Values", df.isnull().sum().sum())
    col4.metric("Duplicate Rows", df.duplicated().sum())


# ======================================================
# Charts
# ======================================================

def plot_target_distribution(df):
    """
    Plot target distribution.
    """

    counts = df["Target"].value_counts().reset_index()

    counts.columns = ["Target", "Count"]

    fig = px.bar(

        counts,

        x="Target",

        y="Count",

        color="Target",

        text="Count",

        title="Target Class Distribution"

    )

    fig.update_layout(

        xaxis_title="Student Status",

        yaxis_title="Number of Students"

    )

    st.plotly_chart(

        fig,

        use_container_width=True

    )


def plot_feature_importance(feature_df):
    """
    Display feature importance chart.
    """

    fig = px.bar(

        feature_df.head(10),

        x="Importance",

        y="Feature",

        orientation="h",

        title="Top 10 Important Features"

    )

    fig.update_layout(

        yaxis={"categoryorder": "total ascending"}

    )

    st.plotly_chart(

        fig,

        use_container_width=True

    )


# ======================================================
# Prediction Result
# ======================================================

def show_prediction(prediction, confidence):
    """
    Display prediction result.
    """

    st.subheader("Prediction Result")

    if prediction == "Graduate":

        st.success("🎉 Student is likely to Graduate")

    else:

        st.error("⚠ Student is likely to Dropout")

    st.metric(

        "Confidence Score",

        f"{confidence:.2f}%"

    )

    if confidence >= 90:

        st.success("Risk Assessment: High Confidence")

    elif confidence >= 70:

        st.warning("Risk Assessment: Medium Confidence")

    else:

        st.info("Risk Assessment: Low Confidence")
