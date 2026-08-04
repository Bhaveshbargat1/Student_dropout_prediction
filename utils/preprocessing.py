"""
preprocessing.py

This module contains helper functions for loading and preprocessing
the Student Dropout dataset.
"""

import pandas as pd


def load_dataset(file_path):
    """
    Load the student dropout dataset.

    Parameters
    ----------
    file_path : str
        Path to CSV file.

    Returns
    -------
    pandas.DataFrame
    """
    df = pd.read_csv(file_path, sep=";")
    return df


def remove_missing_values(df):
    """
    Remove missing values from the dataset.
    """

    return df.dropna()


def remove_enrolled_students(df):
    """
    Remove students whose target is 'Enrolled'.

    The project performs binary classification:
        Graduate
        Dropout
    """

    return df[df["Target"] != "Enrolled"]


def split_features_target(df):
    """
    Split dataframe into features and target.

    Returns
    -------
    X : DataFrame
    y : Series
    """

    X = df.drop("Target", axis=1)
    y = df["Target"]

    return X, y


def get_dataset_information(df):
    """
    Return useful dataset information.
    """

    info = {

        "rows": df.shape[0],

        "columns": df.shape[1],

        "missing_values": int(df.isnull().sum().sum()),

        "duplicate_rows": int(df.duplicated().sum())

    }

    return info


def target_distribution(df):
    """
    Return target class distribution.
    """

    return df["Target"].value_counts()
