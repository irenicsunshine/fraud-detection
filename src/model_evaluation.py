import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

class ModelEvaluator:
    def __init__(self):
        self.results = {}
        
    def evaluate_models(self, models, X_test, y_test):
        """Evaluate all trained models"""
        print("\nModel Evaluation Results:")
        print("=" * 50)
        
        for name, model in models.items():
            print(f"\n{name.upper()} RESULTS:")
            
            # Predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_pred_proba)
            
            # Store results
            self.results[name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'auc_score': auc,
                'predictions': y_pred,
                'probabilities': y_pred_proba
            }
            
            # Print results
            print(f"Accuracy:  {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall:    {recall:.4f}")
            print(f"F1-Score:  {f1:.4f}")
            print(f"AUC Score: {auc:.4f}")
            
            # Confusion Matrix
            cm = confusion_matrix(y_test, y_pred)
            print(f"Confusion Matrix:\n{cm}")
        
        # Create comparison visualization
        self._create_comparison_chart()
        
        return self.results
    
    def _create_comparison_chart(self):
        """Create model comparison visualization"""
        metrics_df = pd.DataFrame(self.results).T
        
        plt.figure(figsize=(12, 6))
        metrics_df[['accuracy', 'precision', 'recall', 'f1_score', 'auc_score']].plot(kind='bar')
        plt.title('Model Performance Comparison')
        plt.ylabel('Score')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig('results/model_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
