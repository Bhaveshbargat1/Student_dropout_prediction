import joblib
import pandas as pd

# ---------------------------------------
# Load Saved Objects
# ---------------------------------------

model = joblib.load("model/student_dropout_model.pkl")
feature_names = joblib.load("model/feature_names.pkl")
label_encoders = joblib.load("model/label_encoders.pkl")
target_encoder = joblib.load("model/target_encoder.pkl")


def preprocess_input(input_dict):
    """
    Convert user input into a DataFrame that matches the
    training data format.
    """

    df = pd.DataFrame([input_dict])

    # Encode categorical columns if needed
    for column, encoder in label_encoders.items():

        if column in df.columns:

            df[column] = encoder.transform(df[column])

    # Arrange columns exactly like training
    df = df[feature_names]

    return df


def predict_student(input_dict):
    """
    Predict Dropout / Graduate
    """

    processed_data = preprocess_input(input_dict)

    prediction = model.predict(processed_data)[0]

    probability = model.predict_proba(processed_data)[0]

    confidence = round(max(probability) * 100, 2)

    predicted_label = target_encoder.inverse_transform([prediction])[0]

    return predicted_label, confidence
