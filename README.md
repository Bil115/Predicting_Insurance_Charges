# 🏥 Insurance Charges Prediction

A machine learning project that accurately predicts medical insurance charges using demographic and lifestyle data. This project explores multiple supervised learning models, feature engineering, and model tuning techniques to develop a robust regression model. Final deployment-ready model is built using **XGBoost**, achieving **85% R² score**.

---

## 📌 Problem Statement

Can we build a reliable machine learning model that predicts a customer's insurance charges based on features such as:

- Age
- BMI (Body Mass Index)
- Smoking status
- Number of children
- Gender
- Residential region

---

## 🎯 Objective

- Build and evaluate machine learning models to predict insurance charges.
- Identify key features that drive insurance costs.
- Deploy the best-performing model for real-time predictions.

---

## 📊 Dataset Overview

- **Source**: [Medical Cost Personal Dataset](https://www.kaggle.com/mirichoi0218/insurance)
- **Size**: 1,300+ records
- **Features**:
  - `age`: Age of the policyholder
  - `sex`: Gender of the policyholder
  - `bmi`: Body mass index
  - `children`: Number of dependents
  - `smoker`: Smoking status
  - `region`: Residential region
  - `charges`: Annual medical insurance charges (target)

---

## 💡 Key Hypotheses

1. Smoking status significantly increases insurance costs.
2. Older individuals are likely to incur higher charges.
3. Higher BMI correlates with increased medical expenses.

---

## 🔧 Tools & Technologies

- **Languages**: Python
- **Libraries**: Pandas, NumPy, Scikit-learn, XGBoost, Seaborn, Matplotlib, SHAP
- **Modeling Techniques**:
  - Linear Regression
  - Ridge & Lasso Regression (Regularization)
  - Random Forest Regressor
  - **XGBRegressor (Best performing)**
- **Validation**: K-Fold Cross Validation, Hyperparameter Tuning (GridSearchCV)

---

## 🧼 Data Preprocessing

- Handled missing/null values (if any)
- Label encoded categorical features (`smoker`, `sex`, `region`)
- Feature scaling using StandardScaler
- Performed correlation analysis and outlier detection

---

## 🧪 Model Evaluation

| Model               | R² Score | MAE      | RMSE     |
|--------------------|----------|----------|----------|
| Linear Regression   | 0.78     | ~4100    | ~6000    |
| **XGBRegressor**    | **0.86** | **~3100**| **~5100**|

✅ Performance improved by **~9% R²** using `XGBRegressor` with tuned hyperparameters.

---

## 📈 Feature Importance

Key drivers of cost identified using model-based feature importance and SHAP:

- Smoking status 🚬
- Age 👴
- BMI ⚖️

---

## 🚀 Deployment (Coming Soon)

The trained XGBoost model will be deployed as a **REST API using Flask**, allowing users to input customer data and receive real-time charge predictions.

---

## 🧠 What You’ll Learn

- How to perform regression analysis in healthcare
- Use of regularization techniques (Ridge, Lasso)
- Handling skewed data and outliers
- Feature importance and interpretability (SHAP)
- Model optimization and hyperparameter tuning

---

## 📎 Project Structure

```bash
InsuranceChargesPrediction/
│
├── data/                # Raw and cleaned datasets
├── notebooks/           # EDA and model building Jupyter Notebooks
├── models/              # Trained models (Pickle files)
├── app/                 # Deployment (Flask API - coming soon)
├── images/              # Visualizations (correlation, importance, etc.)
└── README.md            # Project overview

