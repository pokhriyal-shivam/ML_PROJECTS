# Credit Card Fraud Detection using Machine Learning

## Problem Statement

Credit card fraud causes significant financial losses to banks and customers. The goal of this project is to build a machine learning model that can identify fraudulent transactions.

## Dataset

* Total Transactions: 284,807
* Fraud Transactions: 492
* Normal Transactions: 284,315
* Fraud Percentage: 0.172%

Dataset:
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Imbalanced-learn (SMOTE)

## Machine Learning Workflow

1. Data Loading and Exploration
2. Missing Value Analysis
3. Class Imbalance Analysis
4. Exploratory Data Analysis
5. Train-Test Split
6. Baseline Logistic Regression
7. Model Evaluation
8. SMOTE Oversampling
9. Threshold Tuning
10. ROC-AUC Analysis

## Results

### Baseline Logistic Regression

* Accuracy: 99.92%
* Recall: 69%

### Logistic Regression + SMOTE

* Accuracy: 98.84%
* Recall: 90%
