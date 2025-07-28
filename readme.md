# Financial Fraud Detection ML Project

## Overview

Machine learning system for detecting fraudulent financial transactions with exceptional accuracy. Built using ensemble methods and advanced feature engineering on 6.3M transaction dataset.

## Key Results

**Champion Model: Random Forest**
- **99.72% F1-Score** 
- **99.76% Precision**
- **99.67% Recall**
- **99.98% AUC**

**Business Impact:**
- $12+ billion fraud detected
- <0.001% false positive rate  
- 99.67% fraud prevention accuracy
- Only 6 false alarms out of 1.9M legitimate transactions

## Model Comparison

| Model | F1-Score | Precision | Recall | AUC |
|-------|----------|-----------|--------|-----|
| **Random Forest** | **99.72%** | **99.76%** | **99.67%** | **99.98%** |
| XGBoost | 98.06% | 96.46% | 99.72% | 99.96% |
| Logistic Regression | 98.00% | 96.38% | 99.67% | 99.92% |

## Quick Start

**Install Dependencies:**
pip install pandas numpy scikit-learn xgboost imbalanced-learn matplotlib seaborn

**Run Analysis:**
jupyter notebook Fraud_Detection_ML_Project.ipynb


## Project Structure

├── data/ # Transaction dataset (6.3M records)
├── models/ # Trained ML models
├── src/ # Source code modules
├── Fraud_Detection_ML_Project.ipynb # Main analysis notebook
├── main.py # Complete pipeline
└── README.md # This file


## Technical Approach

**Data Processing:**
- 6,362,620 financial transactions
- 24 engineered features (temporal, balance, behavioral patterns)
- SMOTE balancing for 99.87% class imbalance

**Models:**
- Random Forest (100 trees)
- XGBoost (gradient boosting) 
- Logistic Regression (baseline)

**Key Features:**
- Transaction amount patterns
- Balance change analysis
- Temporal fraud indicators
- Account behavior profiling

## Business Value

**Financial Impact:**
- Average fraud: $1.47M per case
- Monthly savings: $50K-$75K projected
- Annual ROI: 400-600%

**Operational Benefits:**
- Automated 99.67% fraud detection
- Minimal customer friction
- Production-ready deployment

## Dataset Features

- **step**: Time intervals (30-day period)
- **type**: Transaction categories (TRANSFER, CASH-OUT, etc.)
- **amount**: Transaction values
- **balances**: Origin/destination account balances
- **isFraud**: Target variable

## Results Summary

The Random Forest model successfully identifies fraudulent transactions with near-perfect accuracy while maintaining minimal false positives. The system is ready for production deployment in financial institutions.

**Confusion Matrix (Test Set):**
            Predicted
Actual Legitimate Fraud
Legit 1,906,317 6
Fraud 8 2,451

## Technologies Used

Python • scikit-learn • XGBoost • SMOTE • Pandas • Jupyter

*Professional-grade fraud detection system achieving world-class performance.*
