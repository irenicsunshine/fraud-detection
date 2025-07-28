import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
import joblib
import os

class ModelTrainer:
    def __init__(self):
        self.models = {}
        self.smote = SMOTE(random_state=42)
        
    def train_models(self, df):
        """Train multiple models and save them"""
        # Prepare features and target
        X = df.drop(['isFraud'], axis=1)
        y = df['isFraud']
        
        print(f"Training data shape: {X.shape}")
        print(f"Class distribution: {y.value_counts()}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        # Apply SMOTE for balanced training
        print("Applying SMOTE for class balancing...")
        X_train_balanced, y_train_balanced = self.smote.fit_resample(X_train, y_train)
        print(f"Balanced training data: {X_train_balanced.shape}")
        
        # Train XGBoost (Primary model based on research)
        print("Training XGBoost...")
        xgb_model = XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            eval_metric='logloss'
        )
        xgb_model.fit(X_train_balanced, y_train_balanced)
        self.models['xgboost'] = xgb_model
        
        # Train Random Forest
        print("Training Random Forest...")
        rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        rf_model.fit(X_train_balanced, y_train_balanced)
        self.models['random_forest'] = rf_model
        
        # Train Logistic Regression
        print("Training Logistic Regression...")
        lr_model = LogisticRegression(random_state=42, max_iter=1000)
        lr_model.fit(X_train_balanced, y_train_balanced)
        self.models['logistic_regression'] = lr_model
        
        # Save models
        os.makedirs('models', exist_ok=True)
        joblib.dump(self.models, 'models/trained_models.pkl')
        joblib.dump(X.columns.tolist(), 'models/feature_names.pkl')
        
        print("Models trained and saved successfully!")
        return self.models, X_test, y_test
