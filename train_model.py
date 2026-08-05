"""
train_model.py

Train and save the best model for the
Student Dropout Prediction System.
"""

import os
import warnings
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier
)

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from utils.preprocessing import (
    load_dataset,
    validate_dataset,
    remove_missing_values,
    remove_enrolled_students,
    split_features_target
)

warnings.filterwarnings("ignore")

# ==========================================================
# Paths
# ==========================================================

DATA_PATH = "data/data.csv"
MODEL_DIR = "model"

os.makedirs(MODEL_DIR, exist_ok=True)

# ==========================================================
# Load and Prepare Data
# ==========================================================

def load_and_prepare_data():

    df = load_dataset(DATA_PATH)

    validate_dataset(df)

    df = remove_missing_values(df)

    df = remove_enrolled_students(df)

    target_encoder = LabelEncoder()

    df["Target"] = target_encoder.fit_transform(df["Target"])

    X, y = split_features_target(df)

    feature_names = list(X.columns)

    default_values = {}

    for column in X.columns:

        if pd.api.types.is_integer_dtype(X[column]):

            default_values[column] = int(X[column].median())

        else:

            default_values[column] = float(X[column].median())

    X_train, X_test, y_train, y_test = train_test_split(

        X,

        y,

        test_size=0.20,

        random_state=42,

        stratify=y

    )

    return (

        X_train,

        X_test,

        y_train,

        y_test,

        feature_names,

        default_values,

        target_encoder

    )

# ==========================================================
# Train Models
# ==========================================================

def train_models(X_train, y_train):

    models = {

        "Logistic Regression": LogisticRegression(

            max_iter=2000,

            random_state=42

        ),

        "Decision Tree": DecisionTreeClassifier(

            random_state=42

        ),

        "Random Forest": RandomForestClassifier(

            n_estimators=300,

            random_state=42

        ),

        "Gradient Boosting": GradientBoostingClassifier(

            random_state=42

        )

    }

    return models
