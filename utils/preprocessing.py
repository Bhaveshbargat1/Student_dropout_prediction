import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

def load_data(filepath):
    """Load dataset with semicolon separator."""
    return pd.read_csv(filepath, sep=';')

def get_categorical_columns():
    """Return list of categorical column names based on dataset description."""
    return [
        'Marital status', 'Application mode', 'Application order', 'Course',
        'Daytime/evening attendance', 'Previous qualification', 'Nacionality',
        "Mother's qualification", "Father's qualification", "Mother's occupation",
        "Father's occupation", 'Displaced', 'Educational special needs', 'Debtor',
        'Tuition fees up to date', 'Gender', 'Scholarship holder', 'International'
    ]

def preprocess_data(df, target_col='Target', test_size=0.2, random_state=42):
    """
    Preprocess data: impute missing values, encode categorical, scale numerical,
    split into train/test.
    Returns:
        X_train, X_test, y_train, y_test,
        encoders (dict), scaler, label_encoder,
        num_cols, cat_cols,
        default_values (dict for imputation)
    """
    data = df.copy()
    
    # Separate target
    y = data[target_col]
    X = data.drop(columns=[target_col])
    
    # Encode target
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Define categorical and numerical columns
    cat_cols = get_categorical_columns()
    # Ensure all cat_cols exist in X
    cat_cols = [col for col in cat_cols if col in X.columns]
    num_cols = [col for col in X.columns if col not in cat_cols]
    
    # Impute missing values: use median for numerical, mode for categorical
    # Create default values for later use in prediction
    default_values = {}
    
    # Numerical imputer
    num_imputer = SimpleImputer(strategy='median')
    X_num = X[num_cols].copy()
    X_num_imputed = num_imputer.fit_transform(X_num)
    # Store default values (median)
    default_values['num'] = {col: num_imputer.statistics_[i] for i, col in enumerate(num_cols)}
    
    # Categorical imputer
    cat_imputer = SimpleImputer(strategy='most_frequent')
    X_cat = X[cat_cols].astype(str)
    X_cat_imputed = cat_imputer.fit_transform(X_cat)
    # Store default values (mode)
    default_values['cat'] = {col: cat_imputer.statistics_[0][i] for i, col in enumerate(cat_cols)}
    
    # Reconstruct DataFrame with imputed values
    X_imputed = pd.DataFrame(
        np.concatenate([X_num_imputed, X_cat_imputed], axis=1),
        columns=num_cols + cat_cols
    )
    
    # Encode categorical features
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        X_imputed[col] = le.fit_transform(X_imputed[col])
        encoders[col] = le
    
    # Scale numerical features
    scaler = StandardScaler()
    X_imputed[num_cols] = scaler.fit_transform(X_imputed[num_cols])
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_imputed, y_encoded, test_size=test_size,
        random_state=random_state, stratify=y_encoded
    )
    
    return (X_train, X_test, y_train, y_test,
            encoders, scaler, label_encoder,
            num_cols, cat_cols, default_values)
