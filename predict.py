import pandas as pd
import numpy as np
from utils.helper import load_artifacts, predict_single

# Load artifacts once at module load
(model, scaler, encoders, label_encoder,
 num_cols, cat_cols, default_values, feature_names) = load_artifacts()

def predict_from_raw(raw_dict):
    """
    Predict student outcome from a dictionary of raw input values.
    Returns: predicted_label, probabilities (dict)
    """
    pred_label, probs = predict_single(
        raw_dict, model, scaler, encoders, label_encoder,
        num_cols, cat_cols, default_values
    )
    # Convert probs to dict
    if probs is not None:
        prob_dict = {label: prob for label, prob in zip(label_encoder.classes_, probs)}
    else:
        prob_dict = None
    return pred_label, prob_dict

# Example usage
if __name__ == '__main__':
    # Example input (must contain all features)
    sample = {
        'Marital status': '1',
        'Application mode': '17',
        'Application order': '5',
        'Course': '171',
        'Daytime/evening attendance': '1',
        'Previous qualification': '1',
        'Previous qualification (grade)': 122.0,
        'Nacionality': '1',
        "Mother's qualification": '19',
        "Father's qualification": '12',
        "Mother's occupation": '5',
        "Father's occupation": '9',
        'Admission grade': 127.3,
        'Displaced': '0',
        'Educational special needs': '0',
        'Debtor': '0',
        'Tuition fees up to date': '1',
        'Gender': '1',
        'Scholarship holder': '0',
        'Age at enrollment': 20,
        'International': '0',
        'Curricular units 1st sem (credited)': 0,
        'Curricular units 1st sem (enrolled)': 0,
        'Curricular units 1st sem (evaluations)': 0,
        'Curricular units 1st sem (approved)': 0,
        'Curricular units 1st sem (grade)': 0.0,
        'Curricular units 1st sem (without evaluations)': 0,
        'Curricular units 2nd sem (credited)': 0,
        'Curricular units 2nd sem (enrolled)': 0,
        'Curricular units 2nd sem (evaluations)': 0,
        'Curricular units 2nd sem (approved)': 0,
        'Curricular units 2nd sem (grade)': 0.0,
        'Curricular units 2nd sem (without evaluations)': 0,
        'Unemployment rate': 10.8,
        'Inflation rate': 1.4,
        'GDP': 1.74
    }
    label, probs = predict_from_raw(sample)
    print(f"Predicted: {label}")
    print(f"Probabilities: {probs}")
