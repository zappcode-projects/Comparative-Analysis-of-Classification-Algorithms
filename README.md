# Stroke Prediction — Comparative Classification Analysis

**Course:** MSDS 550 — Computational Machine Learning (Summer 2026)
**Author:** Antonio Brown
**Assignment 2:** Comparative Analysis of Classification Algorithms

Predicting stroke risk from patient health records by training and evaluating **eight classification algorithms** on a severely imbalanced clinical dataset.

---

## Overview

This project designs, trains, and evaluates eight classifiers on the Healthcare Stroke Prediction dataset to identify patients at risk of stroke. The core challenge is **severe class imbalance** — only 4.87% of patients experienced a stroke — which makes accuracy a misleading metric. Left unaddressed, every classifier collapses to predicting "no stroke" for everyone, scoring ~95% accuracy while detecting essentially no real cases.

The analysis therefore prioritizes **recall (sensitivity)** and **ROC-AUC**, and applies **SMOTE** oversampling to the training data so each algorithm can learn the minority class.

**Headline result:** Logistic Regression is the best model for this clinical task — highest ROC-AUC (0.845) and highest recall (0.840), correctly identifying 42 of 50 strokes in the held-out test set while remaining fully interpretable.

---

## Dataset

- **Source:** Healthcare Stroke Prediction dataset (public)
- **Records:** 5,110 patients
- **Attributes:** 12 (demographics, health conditions, lifestyle)
- **Target:** `stroke` — 249 stroke cases vs. 4,861 non-stroke (**4.87% positive rate**)

| Feature | Type | Notes |
|---|---|---|
| `gender` | Categorical | Male / Female / Other (single "Other" record removed) |
| `age` | Numeric | Strongest single predictor |
| `hypertension` | Binary | 1 = diagnosed |
| `heart_disease` | Binary | 1 = diagnosed |
| `ever_married` | Categorical | Yes / No |
| `work_type` | Categorical | Private, Self-employed, Govt_job, children, Never_worked |
| `Residence_type` | Categorical | Urban / Rural |
| `avg_glucose_level` | Numeric | Average blood-glucose level |
| `bmi` | Numeric | 201 missing values, median-imputed |
| `smoking_status` | Categorical | formerly / never / smokes / Unknown |
| `stroke` | Binary | **Target** |

---

## Methodology & Preprocessing

All preprocessing was fit on the **training partition only** and applied to the test partition, to prevent leakage.

1. **Drop identifier** — `id` removed (non-predictive).
2. **Remove separating categories** — `gender = "Other"` (1 record) and `work_type = "Never_worked"` (22 records, all non-stroke) removed; each perfectly separates the outcome and destabilizes the logistic MLE. Total: 23 rows (0.45%).
3. **Impute missing BMI** — 201 values filled with training-set median (28.1), computed after the split.
4. **Encode categoricals** — one-hot with first level dropped, yielding 14 model features.
5. **Stratified 80/20 split** — preserves the 4.87% positive rate (test set: 1,018 patients, 50 strokes).
6. **Scaling** — `StandardScaler` for distance/gradient-sensitive models (Logistic Regression, KNN, Naïve Bayes); tree models use raw features.
7. **SMOTE** — applied to the **training set only** (balanced to 3,870 per class). The test set is never resampled, so all metrics reflect the true distribution.

**Inference vs. prediction:** For the coefficient/p-value table, logistic regression is fit on the original (non-SMOTE) data, since synthetic points would inflate significance. The predictive logistic model used for scoring is trained on SMOTE-balanced data. This keeps statistical inference valid while giving the predictive model a balanced signal.

---

## Models Evaluated

Logistic Regression · Decision Tree · Random Forest · AdaBoost · Gradient Boosting · XGBoost · K-Nearest Neighbors · Naïve Bayes

Each model was trained on SMOTE-balanced data and evaluated on the untouched test set. A shared routine produced a confusion matrix, ROC curve, and six metrics per model.

---

## Results

### Model comparison (test set, sorted by ROC-AUC)

| Model | Accuracy | Precision | Recall | F1 | Specificity | ROC-AUC |
|---|---|---|---|---|---|---|
| **Logistic Regression** | 0.706 | 0.126 | **0.840** | 0.219 | 0.699 | **0.845** |
| Naïve Bayes | 0.601 | 0.092 | 0.800 | 0.165 | 0.591 | 0.793 |
| Random Forest | 0.886 | 0.107 | 0.180 | 0.134 | **0.923** | 0.773 |
| Decision Tree (d=4) | 0.815 | **0.152** | 0.600 | **0.242** | 0.826 | 0.773 |
| AdaBoost | 0.833 | 0.147 | 0.500 | 0.227 | 0.850 | 0.764 |
| XGBoost | **0.887** | 0.151 | 0.280 | 0.196 | 0.918 | 0.763 |
| Gradient Boosting | 0.881 | 0.149 | 0.300 | 0.199 | 0.911 | 0.732 |
| KNN (k=5) | 0.806 | 0.084 | 0.300 | 0.132 | 0.832 | 0.669 |

**Best per metric:** Accuracy — XGBoost (0.887) · Precision — Decision Tree (0.152) · Recall — Logistic Regression (0.840) · F1 — Decision Tree (0.242) · Specificity — Random Forest (0.923) · ROC-AUC — Logistic Regression (0.845).

### How to read this

Accuracy is the wrong lens here — the three highest-accuracy models (XGBoost, Random Forest, Gradient Boosting) are precisely the ones that **miss the most strokes**. The clinically meaningful metrics are recall/sensitivity (real strokes caught) and ROC-AUC (ranking quality), and Logistic Regression leads on both. Low precision across all models is an unavoidable consequence of a 4.87% positive rate combined with SMOTE — expected, not a sign of a broken model.

---

## Key Predictors

Across **every** algorithm, the signal is consistent:

1. **age** — dominant driver (a one-SD increase multiplies stroke odds by ~5.5)
2. **avg_glucose_level** — +19% odds per SD
3. **hypertension** — +13% odds per SD

These three are the only statistically significant predictors in the logistic model (p < 0.05). Demographic and lifestyle variables — BMI, gender, marital status, residence, work type, smoking — add little once age is accounted for, and their odds-ratio confidence intervals all span 1.0. All VIFs are well below 5, so multicollinearity is not a concern.

---

## Threshold Tuning & Deployment

A default 0.50 cut-off is arbitrary for an imbalanced clinical problem. Because logistic regression ranks cases well (AUC 0.845), the threshold can slide the model along the precision–recall curve:

| Strategy | Threshold | Accuracy | Recall | F1 | Specificity | Strokes caught |
|---|---|---|---|---|---|---|
| Default cut-off | 0.500 | 0.706 | 0.840 | 0.219 | 0.699 | 42 / 50 |
| **Screening (recall ≥ 90%)** | 0.252 | 0.546 | 0.900 | 0.163 | 0.528 | **45 / 50** |
| Balanced (max F1) | 0.810 | 0.882 | 0.600 | 0.333 | 0.897 | 30 / 50 |

**Recommendation:** Deploy **Logistic Regression as a high-recall screening stage** with a tuned threshold of ~0.25, raising stroke detection to 45 of 50 cases (90% recall) and flagging ~49% of patients for clinical follow-up. Where missing a stroke is far costlier than a false alarm, this trade-off is appropriate. If a sharper balance is later needed, raise the threshold toward the F1-optimal point (~0.81) — **no retraining required**. Threshold tuning, not model replacement, is the right lever for matching the classifier to the clinical cost structure.

---

## Tech Stack

- **Python 3**
- **scikit-learn** — models, splitting, metrics
- **statsmodels** — logistic inference (coefficients, p-values, VIF)
- **xgboost** — gradient-boosted trees
- **imbalanced-learn** — SMOTE oversampling
- **pandas** / **numpy** — data handling
- **matplotlib** / **seaborn** — visualization

---

## Reproducibility

All results reproduce from the accompanying notebook (`CMLS-Assignment_2.ipynb`). A fixed seed (42) governs the split, SMOTE, and all stochastic models.

**Pipeline order:**

```
load → drop id → remove separating categories → encode →
stratified split → impute (train median) → scale →
SMOTE (train only) → fit → evaluate on untouched test set
```

---

## How to Run

```bash
# Install dependencies
pip install scikit-learn statsmodels xgboost imbalanced-learn pandas numpy matplotlib seaborn

# Launch the notebook
jupyter notebook CMLS-Assignment_2.ipynb
```

1. Place the stroke dataset CSV in the project directory.
2. Update the data path in the loading cell if needed.
3. Run all cells top to bottom to reproduce preprocessing, all eight models, the comparison, and threshold analysis.

---

## Project Structure

```
.
├── healthcare-dataset-stroke-data.csv   # Dataset
├── CMLS-Assignment_2.ipynb              # Analysis notebook
└── README.md                            # This file
```
