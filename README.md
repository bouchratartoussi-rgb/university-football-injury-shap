# Interpretable Machine Learning for Injury Risk Prediction in University Football Players: A SHAP-Based Analysis

Anonymous code repository accompanying the manuscript submitted to *Scientific Reports* (Nature Portfolio).

## Contents
- `university-football-injury-shap.py` — full analysis pipeline (data preprocessing, VIF correction, 70/15/15 split, model training and hyperparameter optimisation, statistical significance testing (McNemar, bootstrap), probability calibration, SHAP interpretability).
- `outputs/resultats_final.xlsx` — per-player predicted injury probability and risk category (all 800 players), as produced by the SVM-optimised pipeline.
- `outputs/plan_intervention_sportif.xlsx` — per-player intervention recommendations derived from the SHAP-based risk stratification described in the manuscript.

## Dataset
This study uses the publicly available **University Football Injury Prediction Dataset**
(Hong, Y., 2024) hosted on Kaggle:
https://www.kaggle.com/datasets/yuanchunhong/university-football-injury-prediction-dataset

The raw `data.csv` file is **not redistributed in this repository**; please download it
directly from the Kaggle link above and place it at the path expected by the script
(`/content/drive/MyDrive/data.csv` if running in Google Colab, or adjust the path locally).

## Pipeline overview
1. One-hot encoding of `Position` (`drop='first'`)
2. Variance Inflation Factor (VIF) correction — removes `Height_cm`, `Weight_kg`
3. Stratified 70/15/15 train / validation / test split
4. Stratified 5-fold cross-validation (stability check)
5. Hyperparameter optimisation via `RandomizedSearchCV` (F1 objective)
6. Youden-index decision threshold optimisation
7. Seven classifiers benchmarked: Logistic Regression, SVM, Random Forest, XGBoost,
   Naïve Bayes, KNN, Decision Tree
8. McNemar exact test + 1,000-iteration bootstrap confidence intervals
9. Probability calibration (Brier score, reliability curves)
10. SHAP interpretability (`KernelExplainer`, background=200, `nsamples=500`) —
    global feature importance, beeswarm plot, individual waterfall explanations,
    Pearson–SHAP coherence check

## Requirements
```
numpy
pandas
matplotlib
seaborn
scikit-learn
xgboost
shap
```

## Reproducibility
All random operations use `random_state=42`. Running the script end-to-end (after
placing `data.csv` at the expected path) reproduces every figure, table, and metric
reported in the manuscript.
