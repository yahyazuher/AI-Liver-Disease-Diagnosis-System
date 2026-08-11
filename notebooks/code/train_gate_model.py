"""
AI Liver Disease Diagnosis System - Gate Model Training Pipeline
----------------------------------------------------------------
Target: Gate Model Binary Classification (1 = Sick/Patient, 0 = Healthy)
Author: Yahya Zuher
Project: AI-Liver-Diseases-Diagnosis-System
"""

import os
import sys
import joblib
import pandas as pd
import xgboost as xgb
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Configuration
DATASET_FILENAME = 'Liver_Patient_Dataset_Cleaned_19k.csv'
MODEL_FILENAME = 'gate_model.pkl'
CONFUSION_MATRIX_FILENAME = 'confusion_matrix_gate.png'

# Direct link to the raw CSV file on GitHub
GITHUB_RAW_URL = 'https://raw.githubusercontent.com/yahyazuher/AI-Liver-Diseases-Diagnosis-System/main/data/processed/Liver_Patient_Dataset_Cleaned_19k.csv'

def download_dataset_if_missing():
    if not os.path.exists(DATASET_FILENAME):
        try:
            print("Dataset not found locally. Downloading from https://raw.githubusercontent.com/yahyazuher/AI-Liver-Diseases-Diagnosis-System/main/data/processed/Liver_Patient_Dataset_Cleaned_19k.csv...")
            df = pd.read_csv(GITHUB_RAW_URL)
            df.to_csv(DATASET_FILENAME, index=False)
            print("Download successful. Dataset saved locally.")
        except Exception as e:
            print(f"Error downloading dataset: {e}")
            sys.exit(1)

def train_liver_prediction_model():
    print("Starting Liver Disease Prediction Pipeline...")

    # 1. Data Acquisition
    download_dataset_if_missing()

    print("Loading dataset...")
    df = pd.read_csv(DATASET_FILENAME).dropna()

    # 2. Clean Feature Names & Enforce Standard Column Mapping
    doc_features = [
        'Age', 'Gender', 'Total_Bilirubin', 'Direct_Bilirubin', 
        'ALP', 'ALT', 'AST', 'Total_Protiens', 
        'Albumin', 'Albumin_and_Globulin_Ratio'
    ]
    
    # Strip hidden non-breaking spaces (\xa0) and align feature names
    df.columns = [col.replace('\xa0', '').strip() for col in df.columns]
    df.columns = doc_features + ['Target']

    # 3. Explicit Data Encoding
    # Gender Encoding: Male = 1, Female = 0
    if df['Gender'].dtype == object or str(df['Gender'].dtype) == 'category':
        df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0, 'M': 1, 'F': 0})

    # Target Mapping: Medical Standard (1 = Sick/Patient, 0 = Healthy)
    if set(df['Target'].unique()) == {1, 2}:
        df['Target'] = df['Target'].map({1: 1, 2: 0})

    X = df[doc_features]
    y = df['Target']

    # 4. Stratified Train-Test Split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 5. Model Initialization
    print("Training XGBoost Classifier...")
    model = xgb.XGBClassifier(
        n_estimators=200,
        learning_rate=0.04,
        max_depth=5,
        subsample=0.85,
        colsample_bytree=0.85,
        eval_metric='logloss',
        random_state=42
    )

    # 6. Model Training
    model.fit(X_train, y_train)

    # 7. Evaluation
    print("Evaluating Model Performance...")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print("-" * 50)
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print("-" * 50)
    print(classification_report(y_test, y_pred))

    # 8. Confusion Matrix Visualization
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                xticklabels=['Class 0', 'Class 1'],
                yticklabels=['Class 0', 'Class 1'])

    plt.title('Confusion Matrix: Gate Model (Liver Disease)', fontsize=14, pad=20)
    plt.ylabel('Actual Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)

    print(f"Saving confusion matrix to {CONFUSION_MATRIX_FILENAME}...")
    plt.savefig(CONFUSION_MATRIX_FILENAME, dpi=300, bbox_inches='tight')
    plt.show()

    # 9. Serialization (Saving only gate_model.pkl)
    joblib.dump(model, MODEL_FILENAME)
    print(f"Model saved successfully: '{MODEL_FILENAME}'")

if __name__ == "__main__":
    train_liver_prediction_model()
