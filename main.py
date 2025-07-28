import pandas as pd
import numpy as np
from src.data_preprocessing import DataPreprocessor
from src.feature_engineering import FeatureEngineer
from src.model_training import ModelTrainer
from src.model_evaluation import ModelEvaluator
from src.business_insights import BusinessAnalyzer

def main():
    print("Starting Fraud Detection Model Development...")
    
    # Step 1: Load and preprocess data
    print("\n1. Loading and preprocessing data...")
    preprocessor = DataPreprocessor()
    df = preprocessor.load_data('data/PS_20174392719_1491204439457_log.csv')
    df_clean = preprocessor.clean_data(df)
    
    # Step 2: Feature engineering
    print("\n2. Engineering features...")
    feature_engineer = FeatureEngineer()
    df_features = feature_engineer.create_features(df_clean)
    
    # Step 3: Train models
    print("\n3. Training models...")
    trainer = ModelTrainer()
    models, X_test, y_test = trainer.train_models(df_features)
    
    # Step 4: Evaluate models
    print("\n4. Evaluating models...")
    evaluator = ModelEvaluator()
    results = evaluator.evaluate_models(models, X_test, y_test)
    
    # Step 5: Business insights
    print("\n5. Generating business insights...")
    analyzer = BusinessAnalyzer()
    insights = analyzer.generate_insights(results, df_features)
    
    print("\nFraud Detection Model Development Complete!")
    return models, results, insights

if __name__ == "__main__":
    models, results, insights = main()
