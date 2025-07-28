import pandas as pd
import numpy as np

class DataPreprocessor:
    def __init__(self):
        self.categorical_columns = ['type']
        self.numerical_columns = ['step', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 
                                 'oldbalanceDest', 'newbalanceDest']
    
    def load_data(self, file_path):
        """Load dataset from CSV file"""
        try:
            df = pd.read_csv(file_path)
            print(f"Dataset loaded successfully: {df.shape}")
            return df
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
    
    def clean_data(self, df):
        """Clean and validate the dataset"""
        # Handle missing values
        print(f"Missing values before cleaning: {df.isnull().sum().sum()}")
        df = df.dropna()
        
        # Remove invalid transactions
        df = df[df['amount'] > 0]
        
        # Handle merchant accounts (accounts starting with 'M')
        df['isMerchantDest'] = df['nameDest'].str.startswith('M').astype(int)
        df['isMerchantOrig'] = df['nameOrig'].str.startswith('M').astype(int)
        
        # Basic data validation
        assert 'isFraud' in df.columns, "Target variable 'isFraud' not found"
        assert 'isFlaggedFraud' in df.columns, "Flag variable 'isFlaggedFraud' not found"
        
        print(f"Data cleaned successfully: {df.shape}")
        print(f"Fraud rate: {df['isFraud'].mean():.4f}")
        
        return df
