# Diabetes Prediction — Data Visualization & Linear Regression

**Course:** MSDS 550-01 — Computational Machine Learning (Summer 2026)
**Author:** Antonio Brown
**Instructor:** Dr. Mohammad Mahmudur Rahman Khan
**Assignment 1:** Data Visualization and Linear Regression Model Implementation

---

## Overview

This project performs exploratory data analysis and builds a **multiple linear regression model** to predict **HbA1c level** from demographic and health-related features in a diabetes dataset. HbA1c reflects average blood glucose control over the previous two to three months, making it a clinically meaningful continuous target.

The workflow covers data inspection, visualization, categorical encoding, an 80/20 train–test split, OLS model fitting with `statsmodels`, and evaluation via MSE, MAE, R², and residual analysis.

---

## Dataset

- **Source:** `diabetes_prediction_dataset.csv` (Kaggle)
- **Observations:** 100,000
- **Variables:** 9
- **Missing values:** None

| Variable | Type | Description |
|---|---|---|
| `gender` | Categorical | Male / Female / Other |
| `age` | Numerical | Age in years |
| `hypertension` | Binary | 0 = no, 1 = yes |
| `heart_disease` | Binary | 0 = no, 1 = yes |
| `smoking_history` | Categorical | never, No Info, current, former, ever, not current |
| `bmi` | Numerical | Body Mass Index |
| `HbA1c_level` | Numerical | **Target variable** |
| `blood_glucose_level` | Numerical | Blood glucose (mg/dL) |
| `diabetes` | Binary | 0 = no, 1 = yes |

**Target:** `HbA1c_level`
**Predictors:** age, hypertension, heart_disease, bmi, blood_glucose_level, diabetes, gender, smoking_history

---

## Methodology

1. **Data inspection** — verified shape, dtypes, and confirmed zero missing values with `df.info()` and `df.isnull().sum()`.
2. **Visualization** — histograms (age, BMI), box plots (HbA1c and blood glucose by diabetes status), a correlation heatmap, and a count plot of smoking categories.
3. **Encoding** — categorical features (`gender`, `smoking_history`) converted to dummy variables with `pd.get_dummies(..., drop_first=True)`, yielding 13 predictor columns.
4. **Split** — 80% train / 20% test via `train_test_split(random_state=42)`; intercept added with `sm.add_constant()`.
5. **Modeling** — Ordinary Least Squares (OLS) regression using `statsmodels`.
6. **Evaluation** — MSE, MAE, R², plus residual scatter and residual distribution plots.

---

## Results

### Model performance (test set)

| Metric | Value | Interpretation |
|---|---|---|
| **MSE** | 0.9667 | Average squared prediction error below one HbA1c unit |
| **MAE** | 0.8664 | Predictions off by ~0.87 HbA1c units on average |
| **R²** | 0.1561 | Model explains ~15.6% of variation in HbA1c |

### Key coefficient findings

| Predictor | Coefficient | p-value | Significance |
|---|---|---|---|
| **diabetes** | 1.5505 | < 0.001 | Strong, significant |
| gender (Male) | 0.0093 | 0.193 | Not significant |
| age | −0.0002 | 0.331 | Not significant |
| hypertension | 0.0217 | 0.113 | Not significant |
| heart_disease | −0.0042 | 0.819 | Not significant |
| bmi | −0.0002 | 0.703 | Not significant |
| blood_glucose_level | −0.0001 | 0.269 | Not significant |
| smoking categories | various | > 0.05 | Not significant |
| intercept (const) | 5.416 | < 0.001 | Baseline HbA1c |

**Takeaway:** Diabetes status is by far the dominant predictor — individuals with diabetes are predicted to have HbA1c levels roughly **1.55 units higher**, holding all else constant. After accounting for diabetes status, the remaining predictors add little independent explanatory value.

---

## Residual Analysis

- Residuals are centered around zero (no major systematic bias).
- No strong nonlinear trend, so the linearity assumption is not severely violated.
- The distribution is not perfectly normal and shows multiple peaks/clustering.
- Consistent with the modest R² (0.1561): a substantial portion of HbA1c variability remains unexplained.

---

## Conclusion & Future Work

The model captures a real but limited relationship between the predictors and HbA1c level, with diabetes status carrying most of the signal. Roughly 84% of the variation stays unexplained, suggesting the linear form is too simple for this data.

Potential improvements:

- Add clinical/lifestyle variables (diet, physical activity, medication use, genetics).
- Introduce interaction terms or polynomial features.
- Try nonlinear models — Decision Trees, Random Forests, or Gradient Boosting.

---

## Tech Stack

- **Python**
- **pandas** — data handling
- **NumPy** — numerical operations
- **matplotlib** / **seaborn** — visualization
- **statsmodels** — OLS regression
- **scikit-learn** — train/test split and evaluation metrics

---

## How to Run

```bash
# Install dependencies
pip install pandas numpy matplotlib seaborn statsmodels scikit-learn

# Launch the notebook
jupyter notebook
```

1. Place `diabetes_prediction_dataset.csv` in the project directory.
2. Update the file path in the data-loading cell if needed.
3. Run all cells top to bottom to reproduce the visualizations, model, and evaluation.

---

## Project Structure

```
.
├── diabetes_prediction_dataset.csv    # Dataset (from Kaggle)
├── diabetes_linear_regression.ipynb   # Analysis notebook
└── README.md                          # This file
```
