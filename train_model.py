# ============================================
# Student Dropout Prediction System
# Train Model
# ============================================

import os
import warnings
import joblib
import pandas as pd
import numpy as np

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

warnings.filterwarnings("ignore")

# ============================================
# Create Required Directories
# ============================================

os.makedirs("model", exist_ok=True)

# ============================================
# Load Dataset
# ============================================

DATA_PATH = "data/student_dropout.csv"

print("=" * 60)
print("Loading Dataset...")
print("=" * 60)

df = pd.read_csv(DATA_PATH, sep=";")

print("\nDataset Loaded Successfully\n")

print(df.head())

print("\nShape :", df.shape)

# ============================================
# Remove Missing Values
# ============================================

df.dropna(inplace=True)

print("\nMissing Values Removed")

# ============================================
# Remove Enrolled Students
# ============================================

df = df[df["Target"] != "Enrolled"]

print("Enrolled Students Removed")

# ============================================
# Encode Target
# Graduate -> 0
# Dropout -> 1
# ============================================

target_encoder = LabelEncoder()

df["Target"] = target_encoder.fit_transform(df["Target"])

# ============================================
# Save Target Encoder
# ============================================

joblib.dump(
    target_encoder,
    "model/target_encoder.pkl"
)

print("Target Encoder Saved")

# ============================================
# Encode Categorical Columns
# ============================================

label_encoders = {}

categorical_columns = df.select_dtypes(
    include=["object"]
).columns

for column in categorical_columns:

    if column != "Target":

        encoder = LabelEncoder()

        df[column] = encoder.fit_transform(df[column])

        label_encoders[column] = encoder

print("Categorical Columns Encoded")

# ============================================
# Save Label Encoders
# ============================================

joblib.dump(
    label_encoders,
    "model/label_encoders.pkl"
)

print("Label Encoders Saved")

# ============================================
# Split Features and Target
# ============================================

X = df.drop("Target", axis=1)

y = df["Target"]

# ============================================
# Save Feature Names
# ============================================

feature_names = list(X.columns)

joblib.dump(
    feature_names,
    "model/feature_names.pkl"
)

print("Feature Names Saved")

# ============================================
# Save Default Values
# ============================================

default_values = {}

for column in X.columns:

    if X[column].dtype == object:

        default_values[column] = X[column].mode()[0]

    else:

        default_values[column] = float(
            X[column].median()
        )

joblib.dump(
    default_values,
    "model/default_values.pkl"
)

print("Default Values Saved")

# ============================================
# Train Test Split
# ============================================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.20,

    random_state=42,

    stratify=y

)

print("\nTraining Samples :", len(X_train))
print("Testing Samples  :", len(X_test))

# ============================================
# Machine Learning Models
# ============================================

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
best_model_name = ""
best_accuracy = 0

results = []

print("\n" + "=" * 60)
print("Training Machine Learning Models")
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
        "Accuracy": round(accuracy * 100, 2)

    })

    print(f"Accuracy : {accuracy * 100:.2f}%")

    if accuracy > best_accuracy:

        best_accuracy = accuracy

        best_model = model

        best_model_name = name

print("\n" + "=" * 60)
print("Model Comparison")
print("=" * 60)

results_df = pd.DataFrame(results)

print(results_df)

print("\nBest Model :", best_model_name)

print("Best Accuracy :", round(best_accuracy * 100, 2), "%")

# ============================================
# Final Prediction
# ============================================

y_pred = best_model.predict(X_test)

# ============================================
# Classification Report
# ============================================

print("\n" + "=" * 60)
print("Classification Report")
print("=" * 60)

report = classification_report(
    y_test,
    y_pred
)

print(report)

with open(
    "model/classification_report.txt",
    "w"
) as file:

    file.write(report)

# ============================================
# Confusion Matrix
# ============================================

cm = confusion_matrix(
    y_test,
    y_pred
)

cm_df = pd.DataFrame(cm)

cm_df.to_csv(
    "model/confusion_matrix.csv",
    index=False
)

print("\nConfusion Matrix Saved")

# ============================================
# Save Best Model
# ============================================

joblib.dump(

    best_model,

    "model/student_dropout_model.pkl"

)

print("\nBest Model Saved Successfully")

# ============================================
# Save Feature Importance
# ============================================

print("\n" + "=" * 60)
print("Saving Feature Importance")
print("=" * 60)

if best_model_name == "Random Forest":

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
        "Feature Importance is only available for Random Forest."
    )

# ============================================
# Save Model Accuracy
# ============================================

accuracy_df = pd.DataFrame({

    "Best Model": [best_model_name],

    "Accuracy (%)": [

        round(best_accuracy * 100, 2)

    ]

})

accuracy_df.to_csv(

    "model/model_accuracy.csv",

    index=False

)

print("\nModel Accuracy Saved")

# ============================================
# Training Summary
# ============================================

print("\n" + "=" * 60)
print("TRAINING COMPLETED SUCCESSFULLY")
print("=" * 60)

print(f"""
Best Model        : {best_model_name}

Accuracy          : {best_accuracy * 100:.2f} %

Training Samples  : {len(X_train)}

Testing Samples   : {len(X_test)}

Number of Features: {len(feature_names)}

""")

print("=" * 60)

print("Files Saved Successfully")

print("""
student_dropout_model.pkl

feature_names.pkl

label_encoders.pkl

target_encoder.pkl

default_values.pkl

classification_report.txt

confusion_matrix.csv

feature_importance.csv

model_accuracy.csv
""")

print("=" * 60)
print("Project Ready For Streamlit")
print("=" * 60)
