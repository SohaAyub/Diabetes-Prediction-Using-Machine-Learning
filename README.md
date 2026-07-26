# Diabetes Prediction using Machine Learning

## Project Overview

This project predicts whether a patient has diabetes based on medical information such as glucose level, blood pressure, BMI, insulin, age, and other health-related features.

The project demonstrates a complete Machine Learning workflow including data preprocessing, exploratory data analysis (EDA), feature scaling, model training, evaluation, and prediction.

---

## Dataset

Dataset: diabetes.csv

The dataset contains the following features:

- Pregnancies
- Glucose
- BloodPressure
- SkinThickness
- Insulin
- BMI
- DiabetesPedigreeFunction
- Age

Target Variable:

- Outcome
  - 0 = No Diabetes
  - 1 = Diabetes

---

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib
- Google Colab

---

## Machine Learning Algorithm

- Logistic Regression

---

## Data Preprocessing

- Replaced invalid zero values with missing values
- Filled missing values using median
- Feature Scaling using StandardScaler
- Train-Test Split

---

## Exploratory Data Analysis (EDA)

- Diabetes Distribution
- Age Distribution
- Glucose Distribution
- BMI Distribution
- Glucose vs Outcome
- Correlation Heatmap

---

## Model Evaluation

- Accuracy Score
- Confusion Matrix
- Classification Report

---

## Project Structure

```text
Diabetes-Prediction-Using-Machine-Learning/
│
├── data/
├── images/
├── models/
├── notebook/
├── src/
├── README.md
├── requirements.txt
├── LICENSE
└── .gitignore
