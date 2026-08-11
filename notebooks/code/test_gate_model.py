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
"""

import io
import joblib
import pandas as pd
import requests

# GitHub Raw URL pointing explicitly to models/gate_model.pkl
MODEL_URL = 'https://raw.githubusercontent.com/yahyazuher/AI-Liver-Diseases-Diagnosis-System/main/models/gate_model.pkl'

print("Fetching model from GitHub: models/gate_model.pkl...")
try:
    response = requests.get(MODEL_URL)
    response.raise_for_status()
    model = joblib.load(io.BytesIO(response.content))
    print("Model loaded successfully from GitHub!\n")
except Exception as e:
    print(f"Error fetching from GitHub ({e}). Loading locally...")
    try:
        model = joblib.load('models/gate_model.pkl')
    except FileNotFoundError:
        model = joblib.load('gate_model.pkl')
    print("Model loaded from local storage!\n")

# Benchmark Test Data (10 Patients)
benchmark_cases = [
    [45, 1, 8.5, 4.5, 400, 200, 180, 6.8, 3.0, 0.80],
    [60, 1, 2.5, 1.2, 200, 45,  60,  5.0, 1.8, 0.50],
    [30, 0, 15.0, 8.0, 550, 120, 110, 7.0, 3.2, 0.80],
    [40, 1, 1.5, 0.6, 190, 150, 140, 7.2, 3.8, 1.10],
    [25, 1, 0.7, 0.1, 110, 18,  20,  7.6, 4.2, 1.20],
    [32, 0, 0.6, 0.1, 125, 15,  18,  7.4, 4.1, 1.15],
    [18, 0, 0.6, 0.1, 140, 15,  18,  7.8, 4.2, 1.20],
    [22, 1, 3.2, 0.3, 85,  22,  20,  7.2, 4.2, 1.20],
    [35, 0, 0.9, 0.2, 180, 25,  19,  6.8, 3.5, 1.00],
    [42, 0, 0.8, 0.2, 130, 52,  45,  7.6, 4.0, 1.05]
]

feature_columns = [
    'Age', 'Gender', 'Total_Bilirubin', 'Direct_Bilirubin', 
    'ALP', 'ALT', 'AST', 'Total_Protiens', 'Albumin', 'Albumin_and_Globulin_Ratio'
]

df_benchmark = pd.DataFrame(benchmark_cases, columns=feature_columns)

# Inference
predictions = model.predict(df_benchmark)
probabilities = model.predict_proba(df_benchmark)

# Output Display
print("=" * 62)
print("       AI LIVER DIAGNOSIS SYSTEM - GATE MODEL BENCHMARK       ")
print("=" * 62)
print(f"{'Case':<8} | {'Classification':<12} | {'Sick Prob (%)':<14} | {'Healthy Prob (%)':<14}")
print("-" * 62)

for i in range(len(df_benchmark)):
    is_sick = (predictions[i] == 1)
    label = "Sick (1)" if is_sick else "Healthy (0)"
    
    sick_prob = probabilities[i][1] * 100
    healthy_prob = probabilities[i][0] * 100
    
    print(f"Case {i+1:<3}   | {label:<12} | {sick_prob:<14.2f} | {healthy_prob:<14.2f}")

print("=" * 62)
