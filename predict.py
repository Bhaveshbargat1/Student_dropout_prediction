"""
predict.py

Prediction module for the Student Dropout Prediction System.
"""

import joblib
import pandas as pd


MODEL_PATH = "model/student_dropout_model.pkl"
FEATURE_PATH = "model/feature_names.pkl"
TARGET_ENCODER_PATH = "model/target_encoder.pkl"


class StudentDropoutPredictor:
    """
    Loads the trained model and predicts
    student academic outcome.
    """

    def __init__(self):

        self.model = joblib.load(MODEL_PATH)

        self.feature_names = joblib.load(FEATURE_PATH)

        self.target_encoder = joblib.load(TARGET_ENCODER_PATH)

    def predict(self, student_data):
        """
        Parameters
        ----------
        student_data : dict

        Returns
        -------
        prediction : str
        confidence : float
        """

        df = pd.DataFrame([student_data])

        # Arrange columns exactly as training
        df = df[self.feature_names]

        prediction = self.model.predict(df)[0]

        probabilities = self.model.predict_proba(df)[0]

        confidence = max(probabilities) * 100

        prediction = self.target_encoder.inverse_transform(
            [prediction]
        )[0]

        return prediction, round(confidence, 2)


predictor = StudentDropoutPredictor()


def predict_student(student_data):
    """
    Wrapper function for Streamlit.
    """

    return predictor.predict(student_data)
