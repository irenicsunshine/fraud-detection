import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

class FeatureEngineer:
    def __init__(self):
        self.label_encoder = LabelEncoder()
        
    def create_features(self, df):
        """Create engineered features for fraud detection"""
        df_features = df.copy()
        
        # 1. Transaction amount features
        df_features['amount_log'] = np.log1p(df['amount'])
        df_features['amount_zscore'] = (df['amount'] - df['amount'].mean()) / df['amount'].std()
        
        # 2. Balance change features
        df_features['balance_change_orig'] = df['newbalanceOrig'] - df['oldbalanceOrg']
        df_features['balance_change_dest'] = df['newbalanceDest'] - df['oldbalanceDest']
        
        # 3. Balance ratio features
        df_features['balance_ratio_orig'] = np.where(
            df['oldbalanceOrg'] > 0, 
            df['newbalanceOrig'] / df['oldbalanceOrg'], 
            0
        )
        
        # 4. Transaction timing features
        df_features['hour_of_day'] = df['step'] % 24
        df_features['day_of_month'] = df['step'] // 24
        
        # 5. Suspicious transaction indicators
        df_features['zero_balance_orig'] = (df['oldbalanceOrg'] == 0).astype(int)
        df_features['zero_balance_dest'] = (df['oldbalanceDest'] == 0).astype(int)
        df_features['amount_equals_old_balance'] = (df['amount'] == df['oldbalanceOrg']).astype(int)
        
        # 6. Transaction type encoding
        df_features = pd.get_dummies(df_features, columns=['type'], prefix='type')
        
        # 7. Remove non-predictive columns
        cols_to_drop = ['nameOrig', 'nameDest', 'isFlaggedFraud']
        df_features = df_features.drop(columns=cols_to_drop)
        
        print(f"Feature engineering complete: {df_features.shape}")
        print(f"Features created: {list(df_features.columns)}")
        
        return df_features
