# Student Dropout Prediction

This project predicts whether a student will **Dropout**, **Graduate**, or remain **Enrolled** based on academic and demographic features. It uses machine learning models trained on a dataset of student records.

## Features

- **Data Preprocessing**: Handles missing values, encodes categorical variables, scales numerical features.
- **Model Training**: Trains Logistic Regression, Random Forest, and XGBoost, selects the best based on accuracy.
- **Web Interface**: Interactive Streamlit app for easy prediction.
- **Deployment Ready**: Can be deployed on Streamlit Cloud, Heroku, or any Python web server.

## Project Structure
Student-Dropout-Prediction/
│
├── app.py # Streamlit frontend
├── train_model.py # Train ML models
├── predict.py # Prediction engine
├── requirements.txt
├── README.md
├── LICENSE
├── .gitignore
│
├── data/
│ └── data.csv # Dataset
│
├── model/ # Saved models and reports
│ ├── student_dropout_model.pkl
│ ├── feature_names.pkl
│ ├── target_encoder.pkl
│ ├── default_values.pkl
│ ├── feature_importance.csv
│ ├── model_accuracy.csv
│ ├── classification_report.csv
│ ├── classification_report.txt
│ ├── confusion_matrix.csv
│ └── best_model.txt
│
├── utils/
│ ├── init.py
│ ├── preprocessing.py
│ └── helper.py
│
├── images/ # Screenshots for README
│ ├── home.png
│ ├── dataset.png
│ ├── prediction.png
│ └── result.png
│
└── .streamlit/
└── config.toml
