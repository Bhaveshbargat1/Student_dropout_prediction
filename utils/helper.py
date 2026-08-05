import joblib
import pandas as pd
import numpy as np

def load_artifacts():
    """Load all saved artifacts from model/ folder."""
    model = joblib.load('model/student_dropout_model.pkl')
    scaler = joblib.load('model/scaler.pkl')
    encoders = joblib.load('model/encoders.pkl')
    label_encoder = joblib.load('model/target_encoder.pkl')
    num_cols = joblib.load('model/num_cols.pkl')
    cat_cols = joblib.load('model/cat_cols.pkl')
    default_values = joblib.load('model/default_values.pkl')
    feature_names = joblib.load('model/feature_names.pkl')
    return model, scaler, encoders, label_encoder, num_cols, cat_cols, default_values, feature_names

def safe_transform(encoder, values):
    """Transform values, mapping unseen categories to the mode (0) of training."""
    classes = list(encoder.classes_)
    def map_value(x):
        if str(x) in classes:
            return classes.index(str(x))
        else:
            return 0  # default for unseen
    return [map_value(v) for v in values]

def preprocess_raw_input(raw_dict, encoders, scaler, num_cols, cat_cols, default_values):
    """
    Convert raw input dictionary to a preprocessed DataFrame.
    Handles missing values by filling with defaults.
    """
    # Create a copy to avoid modifying input
    data = raw_dict.copy()
    
    # Fill missing with defaults for categorical
    for col in cat_cols:
        if col not in data or pd.isna(data[col]):
            data[col] = default_values['cat'][col]
        else:
            data[col] = str(data[col])
    
    # Fill missing with defaults for numerical
    for col in num_cols:
        if col not in data or pd.isna(data[col]):
            data[col] = default_values['num'][col]
        else:
            data[col] = float(data[col])
    
    # Create DataFrame
    input_df = pd.DataFrame([data])
    
    # Encode categorical columns
    for col in cat_cols:
        input_df[col] = safe_transform(encoders[col], input_df[col].tolist())
    
    # Scale numerical columns
    input_df[num_cols] = scaler.transform(input_df[num_cols])
    
    return input_df

def predict_single(raw_dict, model, scaler, encoders, label_encoder, num_cols, cat_cols, default_values):
    """Make a single prediction from raw input dictionary."""
    input_df = preprocess_raw_input(raw_dict, encoders, scaler, num_cols, cat_cols, default_values)
    pred_encoded = model.predict(input_df)[0]
    pred_label = label_encoder.inverse_transform([pred_encoded])[0]
    probs = model.predict_proba(input_df)[0] if hasattr(model, 'predict_proba') else None
    return pred_label, probs
