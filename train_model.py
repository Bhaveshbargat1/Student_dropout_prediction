"""
train_model.py

Student Dropout Prediction System
---------------------------------
This script:
1. Loads the dataset
2. Preprocesses the data
3. Trains multiple ML models
4. Selects the best model
5. Saves the trained model and metadata
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
    remove_missing_values,
    remove_enrolled_students,
    split_features_target
)

warnings.filterwarnings("ignore")

# ==========================================================
# Create Model Directory
# ==========================================================

os.makedirs("model", exist_ok=True)

# ==========================================================
# Load Dataset
# ==========================================================

print("=" * 60)
print("Loading Dataset...")
print("=" * 60)

df = load_dataset("data/student_dropout.csv")

print(f"Dataset Shape : {df.shape}")

# ==========================================================
# Data Cleaning
# ==========================================================

df = remove_missing_values(df)
df = remove_enrolled_students(df)

print(f"Shape After Cleaning : {df.shape}")

# ==========================================================
# Encode Target Variable
# ==========================================================

target_encoder = LabelEncoder()

df["Target"] = target_encoder.fit_transform(df["Target"])

joblib.dump(
    target_encoder,
    "model/target_encoder.pkl"
)


# ==========================================================
# Input features are already numeric in this dataset.
# No feature encoding is required.
# ==========================================================

joblib.dump(
    {},
    "model/label_encoders.pkl"
)

# ==========================================================
# Split Features & Target
# ==========================================================

X, y = split_features_target(df)

feature_names = list(X.columns)

joblib.dump(
    feature_names,
    "model/feature_names.pkl"
)

# ==========================================================
# Save Default Values
# ==========================================================

default_values = {}

for column in X.columns:

    if pd.api.types.is_integer_dtype(X[column]):

        default_values[column] = int(X[column].median())

    elif pd.api.types.is_float_dtype(X[column]):

        default_values[column] = float(X[column].median())

    else:

        default_values[column] = X[column].mode()[0]

joblib.dump(
    default_values,
    "model/default_values.pkl"
)

# ==========================================================
# Train-Test Split
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,

    y,

    test_size=0.20,

    random_state=42,

    stratify=y

)

print("Training Samples :", len(X_train))
print("Testing Samples  :", len(X_test))

# ==========================================================
# Machine Learning Models
# ==========================================================

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

best_model = None
best_model_name = None
best_accuracy = 0

results = []

print("\n" + "=" * 60)
print("Training Models")
print("=" * 60)

for name, model in models.items():

    print(f"\nTraining {name}...")

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    results.append({

        "Model": name,

        "Accuracy (%)": round(
            accuracy * 100,
            2
        )

    })

    print(f"Accuracy : {accuracy * 100:.2f}%")

    if accuracy > best_accuracy:

        best_accuracy = accuracy

        best_model = model

        best_model_name = name

# ==========================================================
# Model Comparison
# ==========================================================

results_df = pd.DataFrame(results)

results_df.to_csv(

    "model/model_accuracy.csv",

    index=False

)

print("\n")
print(results_df)

print("\nBest Model :", best_model_name)
print(
    "Best Accuracy :",
    round(best_accuracy * 100, 2),
    "%"
)

# ==========================================================
# Save Best Model
# ==========================================================

joblib.dump(

    best_model,

    "model/student_dropout_model.pkl"

)

with open(

    "model/best_model.txt",

    "w"

) as file:

    file.write(best_model_name)

# ==========================================================
# Final Prediction
# ==========================================================

y_pred = best_model.predict(X_test)

# ==========================================================
# Classification Report
# ==========================================================

report = classification_report(

    y_test,

    y_pred,

    target_names=target_encoder.classes_

)

print("\nClassification Report\n")

print(report)

with open(

    "model/classification_report.txt",

    "w"

) as file:

    file.write(report)

# ==========================================================
# Confusion Matrix
# ==========================================================

cm = confusion_matrix(

    y_test,

    y_pred

)

cm_df = pd.DataFrame(cm)

cm_df.to_csv(

    "model/confusion_matrix.csv",

    index=False

)

# ==========================================================
# Feature Importance
# ==========================================================

if hasattr(best_model, "feature_importances_"):

    importance = pd.DataFrame({

        "Feature": X.columns,

        "Importance": best_model.feature_importances_

    })

    importance = importance.sort_values(

        by="Importance",

        ascending=False

    )

    importance.to_csv(

        "model/feature_importance.csv",

        index=False

    )

    print("\nTop 10 Important Features\n")

    print(importance.head(10))

else:

    print(
        "\nFeature Importance not available for this model."
    )

# ==========================================================
# Training Completed
# ==========================================================

print("\n" + "=" * 60)

print("TRAINING COMPLETED SUCCESSFULLY")

print("=" * 60)

print(f"Best Model : {best_model_name}")

print(f"Accuracy   : {best_accuracy * 100:.2f}%")

print("\nSaved Files")

print("--------------------------------------")

print("student_dropout_model.pkl")

print("feature_names.pkl")

print("default_values.pkl")

print("target_encoder.pkl")

print("label_encoders.pkl")

print("classification_report.txt")

print("confusion_matrix.csv")

print("feature_importance.csv")

print("model_accuracy.csv")

print("best_model.txt")

print("=" * 60)
