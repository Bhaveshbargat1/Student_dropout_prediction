import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# -----------------------------------
# Create model directory
# -----------------------------------
os.makedirs("model", exist_ok=True)

# -----------------------------------
# Load Dataset
# -----------------------------------
df = pd.read_csv("data/student_dropout.csv", sep=";")

print("=" * 60)
print("Dataset Loaded Successfully")
print("=" * 60)
print(df.head())

# -----------------------------------
# Remove missing values
# -----------------------------------
df.dropna(inplace=True)

# -----------------------------------
# Remove Enrolled students
# Keep only Graduate and Dropout
# -----------------------------------
df = df[df["Target"] != "Enrolled"]

# -----------------------------------
# Encode Target
# Graduate = 0
# Dropout = 1
# -----------------------------------
target_encoder = LabelEncoder()

df["Target"] = target_encoder.fit_transform(df["Target"])

# -----------------------------------
# Encode categorical columns
# -----------------------------------
label_encoders = {}

categorical_columns = df.select_dtypes(include=["object"]).columns

for column in categorical_columns:

    if column != "Target":

        encoder = LabelEncoder()

        df[column] = encoder.fit_transform(df[column])

        label_encoders[column] = encoder

# -----------------------------------
# Features & Target
# -----------------------------------
X = df.drop("Target", axis=1)

y = df["Target"]

# -----------------------------------
# Train Test Split
# -----------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# -----------------------------------
# Models
# -----------------------------------
models = {

    "Logistic Regression": LogisticRegression(max_iter=2000),

    "Decision Tree": DecisionTreeClassifier(random_state=42),

    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        random_state=42
    ),

    "Gradient Boosting": GradientBoostingClassifier(
        random_state=42
    )

}

best_model = None
best_accuracy = 0
best_model_name = ""

print("\nTraining Models...\n")

for name, model in models.items():

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    print(f"{name} Accuracy : {accuracy:.4f}")

    if accuracy > best_accuracy:

        best_accuracy = accuracy
        best_model = model
        best_model_name = name

print("\n" + "=" * 60)
print("Best Model :", best_model_name)
print("Accuracy   :", round(best_accuracy * 100, 2), "%")
print("=" * 60)

# -----------------------------------
# Classification Report
# -----------------------------------
predictions = best_model.predict(X_test)

print("\nClassification Report\n")

print(classification_report(y_test, predictions))

# -----------------------------------
# Save Model
# -----------------------------------
joblib.dump(best_model, "model/student_dropout_model.pkl")
joblib.dump(target_encoder, "model/target_encoder.pkl")
joblib.dump(label_encoders, "model/label_encoders.pkl")
joblib.dump(list(X.columns), "model/feature_names.pkl")

print("\nModel Saved Successfully!")
