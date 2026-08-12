"""
AI Liver Disease Diagnosis System - Gate Model Training Pipeline
----------------------------------------------------------------
Target: Gate Model Binary Classification (1 = Sick/Patient, 0 = Healthy)
Author: Yahya Zuher
Project: AI-Liver-Diseases-Diagnosis-System
"""
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os
import seaborn as sns
import matplotlib.pyplot as plt

DATASET_FILENAME = 'Liver_Patient_Dataset_Cleaned_19k.csv'
MODEL_FILENAME = 'gate_model.pkl'
CONFUSION_MATRIX_FILENAME = 'confusion_matrix_gate.png'
GITHUB_RAW_URL = 'https://raw.githubusercontent.com/yahyazuher/AI-Liver-Diseases-Diagnosis-System/main/data/processed/Liver_Patient_Dataset_Cleaned_19k.csv'

def train_balanced_gate_model():
    if not os.path.exists(DATASET_FILENAME):
        df = pd.read_csv(GITHUB_RAW_URL)
        df.to_csv(DATASET_FILENAME, index=False)
    else:
        df = pd.read_csv(DATASET_FILENAME)

    df = df.dropna()

    doc_features = [
        'Age', 'Gender', 'Total_Bilirubin', 'Direct_Bilirubin',
        'ALP', 'ALT', 'AST', 'Total_Protiens',
        'Albumin', 'Albumin_and_Globulin_Ratio'
    ]
    df.columns = doc_features + ['Target']

    if df['Gender'].dtype == object or str(df['Gender'].dtype) == 'category':
        df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0, 'M': 1, 'F': 0})

    if set(df['Target'].unique()) == {1, 2}:
        df['Target'] = df['Target'].map({1: 0, 2: 1})

    X = df[doc_features]
    y = df['Target']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 1. التدريب الأولي على 80% لغرض التقييم والاختبار
    model = xgb.XGBClassifier(
        n_estimators=200,
        learning_rate=0.04,
        max_depth=5,
        subsample=0.85,
        colsample_bytree=0.85,
        scale_pos_weight=1.5,
        eval_metric='logloss',
        random_state=42
    )

    model.fit(X_train, y_train)

    # التقييم على 20%
    y_pred = model.predict(X_test)
    print(f"Optimal Model Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%\n")
    print(classification_report(y_test, y_pred, target_names=['Sick (0)', 'Healthy (1)']))

    # ================================================================
    # CONFUSION MATRIX VISUALIZATION
    # ================================================================
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Sick (0)', 'Healthy (1)'],
                yticklabels=['Sick (0)', 'Healthy (1)'])
    plt.title('Confusion Matrix - Gate Model Evaluation', fontsize=12, pad=15)
    plt.xlabel('Predicted Diagnosis', fontsize=10)
    plt.ylabel('Actual Diagnosis', fontsize=10)

    print(f"Saving confusion matrix to '{CONFUSION_MATRIX_FILENAME}'...")
    plt.savefig(CONFUSION_MATRIX_FILENAME, dpi=300, bbox_inches='tight')
    plt.show()

    # اختبار الحالة السليمة
    test_healthy_case = pd.DataFrame([[25, 1, 0.6, 0.1, 110, 15, 18, 7.8, 4.5, 1.30]], columns=X.columns)
    prob = model.predict_proba(test_healthy_case)[0]
    pred = model.predict(test_healthy_case)[0]

    print("\n--- HEALTHY CASE VALIDATION ---")
    print(f"Prediction Output  : {pred} ({'Healthy' if pred == 1 else 'Sick'})")
    print(f"Healthy Probability: {prob[1]*100:.2f}%")
    print(f"Sick Probability   : {prob[0]*100:.2f}%")

    # 2. إعادة التدريب على 100% من البيانات قبل الحفظ النهائي
    print("\n" + "="*50)
    print("Retraining gate model on 100% of dataset for deployment...")
    model.fit(X, y)
    print("✔ Retraining completed on all samples.")

    # الحفظ النهائي بالصيغتين (.pkl و .json)
    joblib.dump(model, MODEL_FILENAME)
    model.save_model("gate_model.json")
    print(f"\nModel saved successfully as '{MODEL_FILENAME}' and 'gate_model.json'")
    print("="*50)

if __name__ == "__main__":
    train_balanced_gate_model()
