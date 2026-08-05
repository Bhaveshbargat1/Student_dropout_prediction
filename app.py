import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from predict import predict_from_raw
from utils.preprocessing import get_categorical_columns
import os

# Page configuration
st.set_page_config(
    page_title="Student Dropout Prediction",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling (optional)
st.markdown("""
<style>
    .main-header { font-size: 2.5rem; font-weight: 700; color: #1f77b4; }
    .sub-header { font-size: 1.2rem; color: #666; }
    .result-box { background-color: #d4edda; padding: 1.5rem; border-radius: 10px; border-left: 5px solid #28a745; }
    .result-box-dropout { background-color: #f8d7da; padding: 1.5rem; border-radius: 10px; border-left: 5px solid #dc3545; }
    .result-box-enrolled { background-color: #fff3cd; padding: 1.5rem; border-radius: 10px; border-left: 5px solid #ffc107; }
</style>
""", unsafe_allow_html=True)

# Title and description
st.markdown('<p class="main-header">🎓 Student Dropout Prediction</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Enter student details to predict their academic outcome</p>', unsafe_allow_html=True)

# Load sample data to get column names and possible values
@st.cache_data
def load_sample_data():
    return pd.read_csv('data/data.csv', sep=';')

df_sample = load_sample_data()
feature_names = [col for col in df_sample.columns if col != 'Target']
cat_cols = get_categorical_columns()
cat_cols = [col for col in cat_cols if col in df_sample.columns]
num_cols = [col for col in feature_names if col not in cat_cols]

# Sidebar for user input
st.sidebar.header("Student Features")
st.sidebar.markdown("Fill in all fields below. For categorical fields, select from the dropdown; for numerical fields, use the number inputs.")

user_input = {}

# Organize inputs into two columns for better layout (optional)
# We'll keep in sidebar for simplicity

# Create input widgets
for col in feature_names:
    if col in cat_cols:
        unique_vals = df_sample[col].dropna().unique().tolist()
        unique_vals = sorted([str(v) for v in unique_vals])
        if len(unique_vals) > 20:
            user_input[col] = st.sidebar.text_input(f"{col}", value=unique_vals[0] if unique_vals else "")
        else:
            user_input[col] = st.sidebar.selectbox(f"{col}", options=unique_vals)
    else:
        min_val = float(df_sample[col].min())
        max_val = float(df_sample[col].max())
        default_val = float(df_sample[col].median())
        user_input[col] = st.sidebar.number_input(
            f"{col}",
            min_value=min_val,
            max_value=max_val,
            value=default_val,
            step=0.1 if col != 'Age at enrollment' else 1.0
        )

# Prediction button
predict_btn = st.sidebar.button("Predict Outcome", type="primary", use_container_width=True)

# Show dataset preview in main area (optional)
with st.expander("📊 Dataset Overview (first 5 rows)"):
    st.dataframe(df_sample.head(), use_container_width=True)
    st.caption(f"Total records: {len(df_sample)}")

# When prediction is triggered
if predict_btn:
    try:
        # Convert all values to appropriate types
        for col in feature_names:
            if col in cat_cols:
                user_input[col] = str(user_input[col])
            else:
                user_input[col] = float(user_input[col])
        
        # Predict
        label, probs = predict_from_raw(user_input)
        
        # Display result
        st.markdown("## Prediction Result")
        
        # Choose color based on outcome
        if label == 'Dropout':
            box_class = "result-box-dropout"
            emoji = "❌"
        elif label == 'Graduate':
            box_class = "result-box"
            emoji = "✅"
        else:  # Enrolled
            box_class = "result-box-enrolled"
            emoji = "🔄"
        
        st.markdown(f"""
        <div class="{box_class}">
            <h3>{emoji} Predicted Outcome: <strong>{label}</strong></h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Show probabilities
        if probs:
            st.markdown("### Probability Distribution")
            prob_df = pd.DataFrame({
                'Outcome': list(probs.keys()),
                'Probability': list(probs.values())
            })
            
            # Bar chart
            st.bar_chart(prob_df.set_index('Outcome'))
            
            # Table
            st.dataframe(prob_df.style.format({'Probability': '{:.2%}'}), use_container_width=True)
        
        # Additional insights (optional)
        st.info("💡 This prediction is based on a machine learning model trained on historical student data.")
        
    except Exception as e:
        st.error(f"❌ Error during prediction: {e}")
        st.error("Please check that all inputs are filled correctly and try again.")

# Footer
st.sidebar.markdown("---")
st.sidebar.caption("Built with ❤️ using Streamlit")
