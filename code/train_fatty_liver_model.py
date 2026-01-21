import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
from google.colab import files

print("--- Fatty Liver (NAFLD) Diagnostic System Training ---")

# Step 1: Upload Dataset
print("\nStep 1: Please upload the dataset file: 'FattyLiver.csv'")
uploaded = files.upload()

try:
    filename = "FattyLiver.csv"
    df = pd.read_csv(filename)
    print(f" Successfully loaded {len(df)} patient records.")
except FileNotFoundError:
    print(f" Error: File '{filename}' not found.")
    raise SystemExit

# 2. Data Cleaning
df.columns = df.columns.str.strip()
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')
df = df.dropna()

# 3. Clinical Target Engineering
def create_clinical_target(row):
    # Diagnosis logic: Triglycerides > 150 AND (ALT or GGT > 40)
    trig_high = row['Triglycerides'] > 150
    alt_high = row['ALT'] > 40
    ggt_high = row['GGT'] > 40
    if (trig_high and (alt_high or ggt_high)) or (alt_high and ggt_high):
        return 1
    return 0

df['Diagnosis'] = df.apply(create_clinical_target, axis=1)

# 4. Feature Selection (13 Features)
X = df.drop(['Diagnosis'], axis=1)
if 'SEQN' in X.columns:
    X = X.drop(['SEQN'], axis=1)
y = df['Diagnosis']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Model Training (Optimized for NAFLD)
model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=4, subsample=0.8, eval_metric='logloss')
model.fit(X_train, y_train)

# 6. Evaluation
y_pred = model.predict(X_test)
print(f"\n Model Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")

# ---------------------------------------------------------
# 7. Saving the Final Model (Updated Name)
# ---------------------------------------------------------
model_filename = "fatty_liver_model.pkl"
pickle.dump(model, open(model_filename, "wb"))

print(f"\n Optimized model saved as: {model_filename}")
