import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from utils.preprocessing import load_data, preprocess_data

def save_model_artifacts(model, scaler, encoders, label_encoder,
                         num_cols, cat_cols, default_values,
                         feature_names, model_name, accuracy,
                         classification_report_dict, confusion_matrix,
                         feature_importance=None):
    """Save all model artifacts and evaluation results."""
    os.makedirs('model', exist_ok=True)
    
    joblib.dump(model, 'model/student_dropout_model.pkl')
    joblib.dump(scaler, 'model/scaler.pkl')
    joblib.dump(encoders, 'model/encoders.pkl')
    joblib.dump(label_encoder, 'model/target_encoder.pkl')
    joblib.dump(num_cols, 'model/num_cols.pkl')
    joblib.dump(cat_cols, 'model/cat_cols.pkl')
    joblib.dump(default_values, 'model/default_values.pkl')
    joblib.dump(feature_names, 'model/feature_names.pkl')
    
    # Save best model info
    with open('model/best_model.txt', 'w') as f:
        f.write(f"Best Model: {model_name}\n")
        f.write(f"Test Accuracy: {accuracy:.4f}\n")
    
    # Save accuracy
    pd.DataFrame({'Model': [model_name], 'Accuracy': [accuracy]}).to_csv('model/model_accuracy.csv', index=False)
    
    # Save classification report as CSV and TXT
    if classification_report_dict:
        df_report = pd.DataFrame(classification_report_dict).transpose()
        df_report.to_csv('model/classification_report.csv')
        with open('model/classification_report.txt', 'w') as f:
            f.write(str(classification_report_dict))
    
    # Save confusion matrix
    if confusion_matrix is not None:
        pd.DataFrame(confusion_matrix).to_csv('model/confusion_matrix.csv', index=False)
    
    # Save feature importance if available
    if feature_importance is not None:
        fi_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importance})
        fi_df = fi_df.sort_values('Importance', ascending=False)
        fi_df.to_csv('model/feature_importance.csv', index=False)

def train_and_save():
    # Load data
    df = load_data('data/data.csv')
    
    # Preprocess
    (X_train, X_test, y_train, y_test,
     encoders, scaler, label_encoder,
     num_cols, cat_cols, default_values) = preprocess_data(df)
    
    # Get feature names
    feature_names = X_train.columns.tolist()
    
    # Define models
    models = {
        'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
        'XGBoost': XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='mlogloss')
    }
    
    best_model = None
    best_acc = 0.0
    best_name = ''
    best_report = None
    best_cm = None
    best_fi = None
    
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True, target_names=label_encoder.classes_)
        cm = confusion_matrix(y_test, y_pred)
        print(f"{name} Accuracy: {acc:.4f}")
        if acc > best_acc:
            best_acc = acc
            best_model = model
            best_name = name
            best_report = report
            best_cm = cm
            if hasattr(model, 'feature_importances_'):
                best_fi = model.feature_importances_
            else:
                best_fi = None
    
    print(f"\nBest model: {best_name} with accuracy {best_acc:.4f}")
    
    # Save all artifacts
    save_model_artifacts(
        model=best_model,
        scaler=scaler,
        encoders=encoders,
        label_encoder=label_encoder,
        num_cols=num_cols,
        cat_cols=cat_cols,
        default_values=default_values,
        feature_names=feature_names,
        model_name=best_name,
        accuracy=best_acc,
        classification_report_dict=best_report,
        confusion_matrix=best_cm,
        feature_importance=best_fi
    )
    
    print("All artifacts saved successfully in the 'model/' folder.")

if __name__ == '__main__':
    train_and_save()
