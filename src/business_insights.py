import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class BusinessAnalyzer:
    def __init__(self):
        pass
        
    def generate_insights(self, model_results, df):
        """Generate business insights and recommendations"""
        print("\n" + "="*60)
        print("BUSINESS INSIGHTS AND RECOMMENDATIONS")
        print("="*60)
        
        # 1. Model Performance Analysis
        best_model = max(model_results.keys(), key=lambda k: model_results[k]['f1_score'])
        best_f1 = model_results[best_model]['f1_score']
        
        print(f"\n🏆 BEST PERFORMING MODEL: {best_model.upper()}")
        print(f"   F1-Score: {best_f1:.4f}")
        print(f"   Precision: {model_results[best_model]['precision']:.4f}")
        print(f"   Recall: {model_results[best_model]['recall']:.4f}")
        
        # 2. Fraud Pattern Analysis
        fraud_cases = df[df['isFraud'] == 1]
        total_fraud_amount = fraud_cases['amount'].sum()
        avg_fraud_amount = fraud_cases['amount'].mean()
        
        print(f"\n💰 FRAUD IMPACT ANALYSIS:")
        print(f"   Total Fraud Cases: {len(fraud_cases):,}")
        print(f"   Total Fraud Amount: ${total_fraud_amount:,.2f}")
        print(f"   Average Fraud Amount: ${avg_fraud_amount:,.2f}")
        print(f"   Fraud Rate: {len(fraud_cases)/len(df)*100:.3f}%")
        
        # 3. Transaction Type Analysis
        fraud_by_type = fraud_cases['type'].value_counts() if 'type' in fraud_cases.columns else "N/A"
        print(f"\n📊 FRAUD BY TRANSACTION TYPE:")
        if fraud_by_type != "N/A":
            for trans_type, count in fraud_by_type.items():
                print(f"   {trans_type}: {count:,} cases")
        
        # 4. Business Recommendations
        self._generate_recommendations(model_results, df)
        
        return {
            'best_model': best_model,
            'best_f1_score': best_f1,
            'total_fraud_amount': total_fraud_amount,
            'fraud_rate': len(fraud_cases)/len(df),
            'recommendations': self._generate_recommendations(model_results, df)
        }
    
    def _generate_recommendations(self, model_results, df):
        """Generate actionable business recommendations"""
        print(f"\n🎯 ACTIONABLE RECOMMENDATIONS:")
        
        recommendations = [
            "Deploy XGBoost model as primary fraud detection system",
            "Implement real-time scoring for TRANSFER and CASH-OUT transactions",
            "Set up automated alerts for transactions exceeding $200,000",
            "Focus fraud investigation resources on high-risk transaction types",
            "Implement additional verification for accounts with zero balances",
            "Monitor temporal patterns - unusual timing may indicate fraud",
            "Regular model retraining (monthly) to adapt to new fraud patterns"
        ]
        
        for i, rec in enumerate(recommendations, 1):
            print(f"   {i}. {rec}")
        
        return recommendations
