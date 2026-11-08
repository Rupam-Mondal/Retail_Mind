# Retail Mind — Day 1

Today I built the first baseline model for predicting customer churn using Logistic Regression.

The Telco Customer Churn dataset was cleaned by removing duplicate records, dropping the customer ID column, and converting `TotalCharges` to a numeric value. I used a preprocessing pipeline to fill missing values, scale numerical columns, and one-hot encode categorical columns. The data was split into 80% training and 20% testing sets.

## Baseline results

| Metric | Score |
| --- | ---: |
| Accuracy | 73.81% |
| Precision | 50.43% |
| Recall | 78.34% |
| F1 score | 61.36% |
| ROC-AUC | 84.13% |

The model was trained with balanced class weights, which helps it identify customers likely to churn. The trained pipeline is saved at `models/churn_pipeline.joblib`.

## Requirements

Install the libraries listed in `requirement.txt`:

```bash
pip install -r requirement.txt
```

Training steps are available in `notebooks/train.ipynb`.
