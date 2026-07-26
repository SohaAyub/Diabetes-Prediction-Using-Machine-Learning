# ==========================================
# Diabetes Prediction
# predict.py
# ==========================================

import pandas as pd
import joblib

# ==========================================
# Load Saved Model
# ==========================================

model = joblib.load("../models/Diabetes_Model.pkl")
scaler = joblib.load("../models/Scaler.pkl")

print("="*40)
print("Diabetes Prediction System")
print("="*40)

# ==========================================
# User Input
# ==========================================

pregnancies = int(input("Pregnancies: "))
glucose = float(input("Glucose: "))
blood_pressure = float(input("Blood Pressure: "))
skin_thickness = float(input("Skin Thickness: "))
insulin = float(input("Insulin: "))
bmi = float(input("BMI: "))
dpf = float(input("Diabetes Pedigree Function: "))
age = int(input("Age: "))

# ==========================================
# Create DataFrame
# ==========================================

new_data = pd.DataFrame({
    "Pregnancies":[pregnancies],
    "Glucose":[glucose],
    "BloodPressure":[blood_pressure],
    "SkinThickness":[skin_thickness],
    "Insulin":[insulin],
    "BMI":[bmi],
    "DiabetesPedigreeFunction":[dpf],
    "Age":[age]
})

# ==========================================
# Feature Scaling
# ==========================================

new_data = scaler.transform(new_data)

# ==========================================
# Prediction
# ==========================================

prediction = model.predict(new_data)

print("\n")

print("="*40)
print("Prediction")
print("="*40)

if prediction[0] == 1:
    print("Patient is likely to have Diabetes.")
else:
    print("Patient is NOT likely to have Diabetes.")
