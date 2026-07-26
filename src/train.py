# ==========================================
# Diabetes Prediction using Machine Learning
# train.py
# ==========================================

# Import Libraries
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# ==========================================
# Load Dataset
# ==========================================

df = pd.read_csv("../data/diabetes.csv")

# ==========================================
# Data Cleaning
# ==========================================

columns = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI"
]

# Replace 0 with NaN
df[columns] = df[columns].replace(0, np.nan)

# Fill Missing Values with Median
for col in columns:
    df[col] = df[col].fillna(df[col].median())

# ==========================================
# Feature Selection
# ==========================================

X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# ==========================================
# Train-Test Split
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# ==========================================
# Feature Scaling
# ==========================================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ==========================================
# Model Training
# ==========================================

model = LogisticRegression(random_state=42)

model.fit(X_train, y_train)

# ==========================================
# Prediction
# ==========================================

y_pred = model.predict(X_test)

# ==========================================
# Evaluation
# ==========================================

print("="*40)
print("Accuracy Score")
print("="*40)

accuracy = accuracy_score(y_test, y_pred)
print(accuracy)

print("\n")

print("="*40)
print("Confusion Matrix")
print("="*40)

print(confusion_matrix(y_test, y_pred))

print("\n")

print("="*40)
print("Classification Report")
print("="*40)

print(classification_report(y_test, y_pred))

# ==========================================
# Save Model
# ==========================================

joblib.dump(model, "../models/Diabetes_Model.pkl")
joblib.dump(scaler, "../models/Scaler.pkl")

print("\nModel Saved Successfully!")
