"""
helper.py

Reusable Streamlit helper functions for the
Student Dropout Prediction System.
"""

import streamlit as st
import pandas as pd
import plotly.express as px


# ==========================================================
# Message Functions
# ==========================================================

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
    """Display info message."""
    st.info(message)


# ==========================================================
# Dataset Preview
# ==========================================================

def display_dataset(df, rows=10):
    """
    Display dataset preview.
    """
    st.dataframe(
        df.head(rows),
        use_container_width=True
    )


# ==========================================================
# Dataset Statistics
# ==========================================================

def display_dataset_info(info):
    """
    Display dataset statistics.
    """

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Rows",
        info["rows"]
    )

    col2.metric(
        "Columns",
        info["columns"]
    )

    col3.metric(
        "Missing",
        info["missing_values"]
    )

    col4.metric(
        "Duplicates",
        info["duplicate_rows"]
    )


# ==========================================================
# Target Distribution
# ==========================================================

def plot_target_distribution(target_counts):
    """
    Plot class distribution.
    """

    chart_df = pd.DataFrame({

        "Class": target_counts.index,

        "Count": target_counts.values

    })

    fig = px.bar(

        chart_df,

        x="Class",

        y="Count",

        color="Class",

        text="Count",

        title="Student Target Distribution"

    )

    fig.update_traces(
        textposition="outside"
    )

    fig.update_layout(

        xaxis_title="Target",

        yaxis_title="Students",

        height=500

    )

    st.plotly_chart(

        fig,

        use_container_width=True

    )


# ==========================================================
# Feature Importance
# ==========================================================

def plot_feature_importance(feature_df):
    """
    Display top feature importance.
    """

    feature_df = feature_df.sort_values(

        by="Importance",

        ascending=False

    ).head(15)

    fig = px.bar(

        feature_df,

        x="Importance",

        y="Feature",

        orientation="h",

        text="Importance",

        title="Top 15 Important Features"

    )

    fig.update_layout(

        yaxis=dict(

            categoryorder="total ascending"

        ),

        height=650

    )

    st.plotly_chart(

        fig,

        use_container_width=True

    )


# ==========================================================
# Model Comparison
# ==========================================================

def show_model_accuracy(df):
    """
    Display model comparison table.
    """

    st.dataframe(

        df,

        use_container_width=True,

        hide_index=True

    )


# ==========================================================
# Prediction Result
# ==========================================================

def show_prediction(prediction, confidence):
    """
    Display prediction result.
    """

    st.subheader("Prediction")

    if prediction == "Graduate":

        st.success(
            "🎉 Student is likely to Graduate"
        )

    else:

        st.error(
            "⚠️ Student is likely to Dropout"
        )

    st.metric(

        "Confidence",

        f"{confidence:.2f}%"

    )


# ==========================================================
# Footer
# ==========================================================

def show_footer():

    st.markdown("---")

    st.caption(

        "Student Dropout Prediction System | "
        "Developed using Streamlit & Scikit-Learn"

    )
