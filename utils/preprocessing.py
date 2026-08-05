"""
preprocessing.py

Utility functions for loading and preprocessing the
Student Dropout dataset.
"""

import pandas as pd


# ==========================================================
# Load Dataset
# ==========================================================

def load_dataset(file_path):
    """
    Load the dataset from CSV.

    Parameters
    ----------
    file_path : str

    Returns
    -------
    pandas.DataFrame
    """

    try:

        df = pd.read_csv(file_path, sep=";")

    except Exception as e:

        raise Exception(f"Unable to load dataset : {e}")

    # Remove hidden spaces and tabs from column names
    df.columns = df.columns.str.strip()

    return df


# ==========================================================
# Dataset Validation
# ==========================================================

def validate_dataset(df):
    """
    Validate dataset structure.
    """

    if df.empty:
        raise ValueError("Dataset is empty.")

    if "Target" not in df.columns:
        raise ValueError("Target column not found.")

    return True


# ==========================================================
# Remove Missing Values
# ==========================================================

def remove_missing_values(df):
    """
    Remove rows containing missing values.
    """

    return df.dropna()


# ==========================================================
# Remove Enrolled Students
# ==========================================================

def remove_enrolled_students(df):
    """
    Keep only Graduate and Dropout classes.
    """

    return df[df["Target"] != "Enrolled"]


# ==========================================================
# Split Features and Target
# ==========================================================

def split_features_target(df):
    """
    Split dataset into X and y.
    """

    X = df.drop("Target", axis=1)

    y = df["Target"]

    return X, y


# ==========================================================
# Dataset Information
# ==========================================================

def get_dataset_information(df):
    """
    Return dataset statistics.
    """

    return {

        "rows": int(df.shape[0]),

        "columns": int(df.shape[1]),

        "missing_values": int(df.isnull().sum().sum()),

        "duplicate_rows": int(df.duplicated().sum())

    }


# ==========================================================
# Target Distribution
# ==========================================================

def get_target_distribution(df):
    """
    Return class distribution.
    """

    return df["Target"].value_counts()


# ==========================================================
# Feature Names
# ==========================================================

def get_feature_names(df):
    """
    Return feature names.
    """

    return list(df.drop("Target", axis=1).columns)
