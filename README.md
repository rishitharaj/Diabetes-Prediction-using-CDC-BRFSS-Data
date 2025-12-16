# 🩺 Diabetes Risk Prediction using CDC BRFSS Data

> End-to-end Machine Learning project for predicting diabetes risk using public health survey data, with a focus on **real-world evaluation, interpretability, and deployment readiness**.

---

## 📌 Project Overview

Diabetes is a major public health challenge where **early risk identification** can significantly improve outcomes.  
This project builds a **screening-oriented machine learning model** using the **CDC Behavioral Risk Factor Surveillance System (BRFSS) 2015 dataset** to predict the likelihood of diabetes based on demographic, lifestyle, and health indicators.

The emphasis is not just on model accuracy, but on:
- Handling **class imbalance correctly**
- Choosing **clinically meaningful evaluation metrics**
- Making **transparent, defensible modeling decisions**
- Deploying the model via a **Flask web application**

---

## 📊 Dataset Description

- **Source**: CDC BRFSS 2015
- **Records**: ~253,000 survey responses
- **Target variable**: `Diabetes_binary`
  - `1` → Diabetic
  - `0` → Non-diabetic
- **Original class distribution**:
  - ~14% diabetic
  - ~86% non-diabetic (highly imbalanced)

### Feature Categories
- Health conditions (HighBP, HighChol, Stroke, etc.)
- Lifestyle indicators (Smoking, Alcohol, Physical activity)
- Self-reported health status (General, Physical, Mental health)
- Demographics (Age group, Sex, Education, Income)

---

## 🧠 Key Design Decisions

### 1️⃣ Proper Handling of Class Imbalance
- **Training set** was deliberately **balanced (50/50)** to help models learn minority-class patterns.
- **Test set** was kept **unbalanced (~86/14)** to reflect real-world prevalence.

This prevents data leakage and avoids overly optimistic evaluation.

---

### 2️⃣ Meaningful Evaluation Metrics
Accuracy was intentionally **not** used as the primary metric.

Instead, evaluation focused on:
- **ROC AUC** → Overall ranking ability
- **PR AUC** → Minority-class performance
- **Recall (Diabetes)** → Critical for screening use cases

---

## 🔬 Modeling Approach

### Baseline Model
- **Logistic Regression**
- Served as a strong and interpretable benchmark

### Models Compared
- Random Forest
- Gradient Boosting
- XGBoost
- LightGBM
- **CatBoost (final model)**

All models were trained on the balanced training set and evaluated on the unbalanced test set.

---

## 🏆 Model Performance Summary

| Model | ROC AUC | PR AUC | Recall (Diabetes) | Notes |
|------|--------|--------|------------------|------|
| Logistic Regression | 0.819 | 0.392 | 0.763 | Strong linear baseline |
| Random Forest | 0.808 | 0.380 | 0.775 | Underperformed |
| Gradient Boosting | 0.826 | 0.419 | 0.789 | Competitive |
| LightGBM | 0.826 | 0.419 | 0.794 | Fast & stable |
| XGBoost | 0.827 | 0.421 | 0.792 | Strong ensemble |
| **CatBoost (Final)** | **0.827** | **0.422** | **0.795** | **Best overall** |

---

## 🎯 Threshold Optimization

Rather than using the default threshold of `0.5`, the decision threshold was explicitly optimized.

### Final Operating Point
- **Threshold**: `0.495`
- **Recall (Diabetes)**: **80%**
- **Precision**: 30%

This operating point is suitable for **screening**, where missing diabetic cases is more costly than generating false positives.

---

## 🔍 Key Insights

- Logistic regression already performed strongly, indicating **strong linear separability** in the data.
- Boosting models provided **incremental improvements**, primarily in ranking quality rather than thresholded classification.
- Feature importance analysis showed that **Age, BMI, General Health, Blood Pressure, and Cholesterol** were dominant predictors, aligning with medical understanding.

---

## 🌐 Deployment (Flask Application)

The final model was:
- Trained **offline** in a notebook
- Serialized as a **`.pkl` artifact**
- Loaded into a **Flask web application** for inference only

### Deployment Best Practices
- No retraining inside the application
- Probability-based predictions
- Explicit thresholding
- Clear separation between training and inference

---

## 📁 Project Structure

Diabetes Prediction_Flask/
│
├─ app.py
├─ model.py
├─ diabetes_catBoost_model.pkl   👈 final model
├─ feature_labels.json
├─ README.md
├─ requirements.txt
├─ Dockerfile
│
├─ templates/
│   └── form.html & result.html
│
└─ notebooks/
    └── Diabetes_EDA_Notebook.ipynb



---

## ✅ Key Takeaways

- Proper data splitting and evaluation are more important than model complexity.
- Balanced training with realistic testing leads to trustworthy metrics.
- Threshold tuning transforms a model from academic to practical.
- Honest evaluation and interpretability are critical in healthcare ML.

## 🧩 Results

<img width="3600" height="2598" alt="Diabetics Prediction Positive" src="https://github.com/user-attachments/assets/751ea894-d3ae-4614-bcc7-c5ec5dcd67a8" />

---

## 👩‍💻 Author

**Rishitha Raj**  
Aspiring Data Scientist  
Background in Engineering, Project Management & UX  
Focused on interpretable, real-world machine learning systems
