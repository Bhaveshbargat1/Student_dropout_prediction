import joblib
import pandas as pd

# -------------------------
# Load Saved Files
# -------------------------

model = joblib.load("model/student_dropout_model.pkl")
feature_names = joblib.load("model/feature_names.pkl")
label_encoders = joblib.load("model/label_encoders.pkl")


def preprocess_input(input_data):
    """
    Converts categorical values using saved LabelEncoders
    """

    data = input_data.copy()

    for column, encoder in label_encoders.items():

        if column in data.columns:

            data[column] = encoder.transform(data[column])

    return data


def predict_student(student_data):
    """
    Predict student dropout
    """

    df = pd.DataFrame([student_data])

    df = preprocess_input(df)

    # Arrange columns in correct order
    df = df[feature_names]

    prediction = model.predict(df)[0]

    probability = model.predict_proba(df)[0]

    confidence = round(max(probability) * 100, 2)

    if prediction == 1:

        result = "Dropout"

    else:

        result = "Graduate"

    return result, confidence
