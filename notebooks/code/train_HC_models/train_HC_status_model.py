"""
[IMPORTANT NOTE / ملاحظة هامة]
--------------------------------------------------
English: This script is specifically designed and optimized to run in the GOOGLE COLAB environment.
- It is configured to automatically download models and training files directly from GitHub.
- Copy-pasting this code to other environments (local IDEs) may require adjustments 
  to file paths and library configurations.

Arabic: Google Colab هذا الكود مخصص ومجهز للعمل مباشرة داخل بيئة 
- GitHub لضمان التشغيل الفوري تم إعداد الكود ليقوم بتحميل النماذج وملفات التدريب تلقائياً من 
- نسخ هذا الكود وتشغيله في تطبيقات أو بيئات أخرى قد يتطلب تعديلات في مسارات الملفات وإعدادات المكتبات.
--------------------------------------------------
Created by: Yahya Zuher
Project: AI-Liver-Diseases-Diagnosis-System

        - Target Output: 0.0 - 1.0 Presentage
"""


import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os
import sys
import seaborn as sns
import matplotlib.pyplot as plt

# --- Configuration (GitHub Integration) ---
RAW_DATA_URL = 'https://raw.githubusercontent.com/yahyazuher/AI-Liver-Diseases-Diagnosis-System/main/data/processed/hepatitisC_status.csv'
MODEL_FILENAME = 'hepatitisC_status_model.pkl'
CONFUSION_MATRIX_FILENAME = 'confusion_matrix_status.png'

def get_live_dataset():
    """Downloads the dataset from GitHub."""
    try:
        print(f"⬇Downloading dataset from GitHub...")
        df = pd.read_csv(RAW_DATA_URL)
        print(f"Successfully loaded {len(df)} records.")
        return df
    except Exception as e:
        sys.exit(f"Critical Error: Could not fetch data. {e}")

def add_medical_features(df):
    """Adds ALBI and APRI features for high-precision mortality prediction."""
    df_eng = df.copy()

    # 1. APRI Score
    df_eng['APRI'] = ((df_eng['SGOT'] / 40.0) / (df_eng['Platelets'] + 0.1)) * 100

    # 2. ALBI Score (Log-based liver function assessment)
    df_eng['ALBI_Score'] = (np.log10(df_eng['Bilirubin'].clip(lower=0.1) * 17.1) * 0.66) + \
                            (df_eng['Albumin'] * 10 * -0.085)

    # 3. Bilirubin to Albumin Ratio
    df_eng['Bili_Alb_Ratio'] = df_eng['Bilirubin'] / (df_eng['Albumin'] + 0.1)

    return df_eng

def run_pipeline():
    print("Starting Automated Training Pipeline...")

    # 1. Get Data from GitHub
    df = get_live_dataset()

    # 2. Add Engineered Features
    df = add_medical_features(df)

    # 3. Define Features & Target
    # We exclude 'Status' (Target) and 'Stage' (due to its low accuracy)
    X = df.drop(columns=['Status', 'Stage'], errors='ignore')
    y = df['Status'].astype(int)

    # 4. Preprocessing Layers
    numeric_features = X.select_dtypes(include=['number']).columns
    categorical_features = X.select_dtypes(include=['object']).columns

    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

    # 5. XGBoost Model Configuration (Tuned for 125/187 ratio)
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', xgb.XGBClassifier(
            n_estimators=250,
            learning_rate=0.03,
            max_depth=4,
            scale_pos_weight=1.5, # Balance for your specific dataset
            subsample=0.8,
            colsample_bytree=0.8,
            objective='binary:logistic',
            eval_metric='logloss',
            random_state=42
        ))
    ])

    # 6. Stratified Train-Test Split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 7. Model Training
    print("Training model on remote data...")
    model.fit(X_train, y_train)

    # 8. Evaluation
    y_pred = model.predict(X_test)
    print("\nFinal Validation Results:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
    print("-" * 30)
    print(classification_report(y_test, y_pred))

    # ================================================================
    # SECTION: CONFUSION MATRIX VISUALIZATION (BINARY)
    # ================================================================
    # This matrix evaluates the model's ability to predict patient mortality risk.
    # It specifically highlights:
    # - True Positives: Correctly identifying high-risk patients.
    # - False Negatives: Critical misses where a high-risk patient is classified as stable.
    # ================================================================
    
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                xticklabels=['Status 0 (Stable)', 'Status 1 (Risk)'],
                yticklabels=['Status 0 (Stable)', 'Status 1 (Risk)'])
    
    plt.title('Confusion Matrix: Mortality Risk Prediction', fontsize=14, pad=20)
    plt.ylabel('Actual Patient Status', fontsize=12)
    plt.xlabel('Model Prediction', fontsize=12)
    
    print(f" Saving confusion matrix to {CONFUSION_MATRIX_FILENAME}...")
    plt.savefig(CONFUSION_MATRIX_FILENAME, dpi=300, bbox_inches='tight')
    plt.show()
    
    # ================================================================
    # END VISUALIZATION
    # ================================================================

    # 9. Global Save
    print(f"Saving model locally: {MODEL_FILENAME}")
    model.fit(X, y) # Retrain on full dataset
    joblib.dump(model, MODEL_FILENAME)

if __name__ == "__main__":
    run_pipeline()
