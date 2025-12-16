# Diabetes Prediction using CDC BRFSS Data

## Overview
This project builds a machine learning pipeline to predict diabetes risk using the CDC BRFSS 2015 dataset.  
The goal is to develop a screening-oriented model that maximizes recall while maintaining reasonable precision.

## Dataset
- Source: CDC Behavioral Risk Factor Surveillance System (BRFSS)
- Target: `Diabetes_binary`
- Class distribution: Imbalanced (~14% diabetic)

## Methodology
- Exploratory Data Analysis (EDA)
- Feature preprocessing and engineering
- Train–test split with:
  - Balanced training set
  - Real-world (unbalanced) test set
- Model comparison across:
  - Logistic Regression
  - Random Forest
  - Gradient Boosting
  - XGBoost
  - LightGBM
  - CatBoost
- Evaluation using ROC AUC, PR AUC, and Recall
- Threshold optimization for screening use case

## Results
- Best model: **CatBoost**
- ROC AUC: 0.827
- PR AUC: 0.422
- Recall (Diabetes): 0.80 at threshold 0.495

## Files
- `diabetes_project.ipynb` — full analysis notebook
- `diabetes_catboost_model.pkl` — trained model + threshold
- `README.md` — project summary
- Flask App

## Key Takeaway
Balancing only the training data while preserving real-world test distributions enables realistic evaluation.  
Boosting models offer incremental improvements over strong linear baselines in epidemiological datasets.

## Author
Rishitha Raj
