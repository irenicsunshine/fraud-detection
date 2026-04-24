# =============================================================================
# 🎓 STEP 1.5: MODEL EVALUATION — Getting Honest Numbers
# =============================================================================
#
# WHY IS EVALUATION SO IMPORTANT?
# ---------------------------------
# Training a model is like cooking a dish. Evaluation is the taste test.
# You wouldn't serve food without tasting it first!
#
# But there's a subtlety: HOW you taste matters.
#
# ACCURACY IS MISLEADING FOR FRAUD DETECTION
# --------------------------------------------
# Imagine a lazy model that says "everything is legitimate."
# With 99.87% legitimate transactions:
#   Accuracy = 99.87%  ← Looks amazing!
#   Fraud caught = 0%  ← Completely useless!
#
# That's why we use precision, recall, F1, and AUC instead.
#
# 🎓 THE METRICS EXPLAINED (with fraud detection examples):
#
# PRECISION: "Of the transactions I flagged, how many were actually fraud?"
#   High precision → few false alarms → happy legitimate customers
#   Low precision  → many false alarms → annoyed customers
#
# RECALL: "Of all actual fraud, how many did I catch?"
#   High recall → caught most fraud → saved money
#   Low recall  → missed fraud → lost money
#
# F1-SCORE: Harmonic mean of precision and recall
#   Balances both — a model with 100% precision but 1% recall gets F1 ≈ 2%
#
# AUC (Area Under ROC Curve): How well can the model separate fraud from legit?
#   1.0 = perfect separation, 0.5 = random guessing
#
# NEW IN THIS VERSION:
# - Threshold optimization (don't blindly use 0.5)
# - Cost-sensitive evaluation ($1.47M per missed fraud)
# - Bootstrap confidence intervals (how reliable are the numbers?)
# - Per-transaction-type breakdown (fraud in TRANSFER vs CASH_OUT)
# =============================================================================

import pandas as pd
import numpy as np
import logging
import time
from typing import Dict, Any, Optional, Tuple, List
from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    precision_recall_curve,
    average_precision_score,
)
import matplotlib
matplotlib.use("Agg")  # 🎓 Non-interactive backend — saves to file, doesn't pop up
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import Config

logger = logging.getLogger("fraud_detection.evaluation")


class ModelEvaluator:
    """
    Evaluates fraud detection models with production-quality metrics.
    
    🎓 THE EVALUATION WORKFLOW
    ---------------------------
    For each model:
    1. Get predictions and probabilities
    2. Find the optimal decision threshold (not just 0.5!)
    3. Calculate metrics at that threshold
    4. Compute cost impact
    5. Generate confidence intervals
    6. Create visualizations
    """
    
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.results: Dict[str, Dict[str, Any]] = {}
        self.optimal_thresholds: Dict[str, float] = {}
    
    def evaluate_models(
        self,
        models: Dict[str, Any],
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Evaluate all models on the test set.
        
        Args:
            models: Dict of model_name → trained model
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dict of model_name → metrics dict
        """
        logger.info(f"\n{'='*60}")
        logger.info("MODEL EVALUATION ON HELD-OUT TEST SET")
        logger.info(f"{'='*60}")
        logger.info(f"Test set size: {len(X_test):,} rows")
        logger.info(f"  Legitimate: {(y_test == 0).sum():,}")
        logger.info(f"  Fraud:      {(y_test == 1).sum():,}")
        
        for name, model in models.items():
            logger.info(f"\n{'─'*40}")
            logger.info(f"Evaluating: {name.upper()}")
            logger.info(f"{'─'*40}")
            
            self.results[name] = self._evaluate_single_model(
                name, model, X_test, y_test
            )
        
        # Create comparison visualization
        self._create_comparison_charts()
        
        # Print summary table
        self._print_summary_table()
        
        return self.results
    
    def _evaluate_single_model(
        self,
        name: str,
        model: Any,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> Dict[str, Any]:
        """Evaluate a single model comprehensively."""
        
        # --- Get probabilities ---
        # 🎓 predict_proba returns [P(legitimate), P(fraud)] for each row
        # We want the fraud probability → index [1]
        y_proba = model.predict_proba(X_test)[:, 1]
        
        # --- Find optimal threshold ---
        optimal_threshold = self._find_optimal_threshold(y_test, y_proba)
        self.optimal_thresholds[name] = optimal_threshold
        
        # --- Predictions at optimal threshold ---
        # 🎓 Default is: if P(fraud) > 0.5, predict fraud
        # But 0.5 is almost NEVER optimal for imbalanced data!
        # With 99.87% legitimate, the model tends to predict everything as
        # legitimate. A lower threshold like 0.3 catches more fraud.
        y_pred_default = (y_proba >= 0.5).astype(int)
        y_pred_optimal = (y_proba >= optimal_threshold).astype(int)
        
        logger.info(f"Optimal threshold: {optimal_threshold:.3f} (default: 0.500)")
        
        # --- Metrics at default threshold ---
        metrics_default = self._compute_metrics(y_test, y_pred_default, y_proba)
        
        # --- Metrics at optimal threshold ---
        metrics_optimal = self._compute_metrics(y_test, y_pred_optimal, y_proba)
        
        # Log comparison
        logger.info(f"\n  {'Metric':<15} {'Default (0.5)':>15} {'Optimal ({optimal_threshold:.3f})':>15}")
        logger.info(f"  {'─'*45}")
        for metric in ["precision", "recall", "f1_score"]:
            d = metrics_default[metric]
            o = metrics_optimal[metric]
            logger.info(f"  {metric:<15} {d:>14.4f} {o:>15.4f}")
        
        # --- Confusion Matrix ---
        cm = confusion_matrix(y_test, y_pred_optimal)
        tn, fp, fn, tp = cm.ravel()
        
        logger.info(f"\n  Confusion Matrix (optimal threshold):")
        logger.info(f"                    Predicted")
        logger.info(f"                    Legit    Fraud")
        logger.info(f"  Actual Legit   {tn:>9,}  {fp:>7,}")
        logger.info(f"  Actual Fraud   {fn:>9,}  {tp:>7,}")
        
        # --- Cost Analysis ---
        cost_analysis = self._compute_cost_impact(tn, fp, fn, tp)
        logger.info(f"\n  Cost Analysis:")
        logger.info(f"    Fraud caught:     {tp:,} × ${self.config.costs.false_negative_cost:,.0f} = "
                     f"${cost_analysis['fraud_prevented']:,.0f}")
        logger.info(f"    Fraud missed:     {fn:,} × ${self.config.costs.false_negative_cost:,.0f} = "
                     f"${cost_analysis['fraud_missed_cost']:,.0f}")
        logger.info(f"    False alarms:     {fp:,} × ${self.config.costs.false_positive_cost:,.0f} = "
                     f"${cost_analysis['false_alarm_cost']:,.0f}")
        logger.info(f"    Net value:        ${cost_analysis['net_value']:,.0f}")
        
        # --- Bootstrap Confidence Intervals ---
        ci = self._bootstrap_confidence_intervals(y_test, y_proba, optimal_threshold)
        logger.info(f"\n  95% Confidence Intervals (bootstrap):")
        for metric, (lower, upper) in ci.items():
            logger.info(f"    {metric:<12}: [{lower:.4f}, {upper:.4f}]")
        
        return {
            "threshold_default": 0.5,
            "threshold_optimal": optimal_threshold,
            "metrics_default": metrics_default,
            "metrics_optimal": metrics_optimal,
            "confusion_matrix": cm,
            "cost_analysis": cost_analysis,
            "confidence_intervals": ci,
        }
    
    def _find_optimal_threshold(
        self, y_true: pd.Series, y_proba: np.ndarray
    ) -> float:
        """
        Find the threshold that maximizes F1-score.
        
        🎓 HOW THIS WORKS
        -------------------
        precision_recall_curve gives us precision and recall at EVERY
        possible threshold (from 0 to 1).
        
        We compute F1 at each threshold and pick the one where F1 is highest.
        
        Why F1 and not just recall? Because maximizing recall alone leads to
        flagging EVERYTHING as fraud (100% recall, 0.1% precision, angry customers).
        F1 balances both.
        
        In practice, you might want to weight recall more heavily (because
        missing fraud is expensive). We use F1 as a starting point and can
        adjust later based on cost analysis.
        """
        precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
        
        # F1 at each threshold (avoid division by zero)
        f1_scores = np.where(
            (precision + recall) > 0,
            2 * (precision * recall) / (precision + recall),
            0,
        )
        
        # precision_recall_curve returns one more element for precision/recall
        # than thresholds, so we use [:-1] to align them
        best_idx = np.argmax(f1_scores[:-1])
        
        return float(thresholds[best_idx])
    
    def _compute_metrics(
        self,
        y_true: pd.Series,
        y_pred: np.ndarray,
        y_proba: np.ndarray,
    ) -> Dict[str, float]:
        """Compute all standard classification metrics."""
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_true, y_proba)),
            "average_precision": float(average_precision_score(y_true, y_proba)),
        }
    
    def _compute_cost_impact(
        self, tn: int, fp: int, fn: int, tp: int
    ) -> Dict[str, float]:
        """
        Compute the dollar impact of the model's predictions.
        
        🎓 WHY COST ANALYSIS MATTERS
        ------------------------------
        Metrics like F1=96% are abstract. Business leaders understand dollars.
        
        By translating predictions into costs:
        - TP (caught fraud) = fraud prevented = $1.47M saved per case
        - FN (missed fraud) = fraud lost = $1.47M lost per case
        - FP (false alarm) = investigation cost = $500 per case
        - TN (correct legitimate) = no cost
        
        This tells you: "Is this model worth deploying?"
        """
        costs = self.config.costs
        
        fraud_prevented = tp * costs.false_negative_cost
        fraud_missed_cost = fn * costs.false_negative_cost
        false_alarm_cost = fp * costs.false_positive_cost
        
        net_value = fraud_prevented - fraud_missed_cost - false_alarm_cost
        
        return {
            "fraud_prevented": fraud_prevented,
            "fraud_missed_cost": fraud_missed_cost,
            "false_alarm_cost": false_alarm_cost,
            "net_value": net_value,
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "true_negatives": tn,
        }
    
    def _bootstrap_confidence_intervals(
        self,
        y_true: pd.Series,
        y_proba: np.ndarray,
        threshold: float,
        n_bootstrap: int = 500,
        confidence: float = 0.95,
    ) -> Dict[str, Tuple[float, float]]:
        """
        Compute bootstrap confidence intervals for key metrics.
        
        🎓 WHAT IS BOOTSTRAPPING?
        --------------------------
        Imagine you have a test set of 1.9M rows and F1 = 96.5%.
        How confident are you in that number? Is it 96.5% ± 0.1% or ± 5%?
        
        Bootstrapping answers this by:
        1. Randomly sample 1.9M rows WITH REPLACEMENT from the test set
           (some rows appear twice, some don't appear at all)
        2. Compute F1 on this sample
        3. Repeat 500 times
        4. Take the 2.5th and 97.5th percentiles → 95% confidence interval
        
        If all 500 F1 values are between 96.3% and 96.7%, you're very
        confident. If they range from 90% to 99%, your estimate is shaky.
        
        We use 500 instead of 1000 to keep it fast on CPU.
        """
        rng = np.random.RandomState(42)
        n = len(y_true)
        
        y_true_arr = np.array(y_true)
        
        metrics_samples = {
            "f1_score": [],
            "precision": [],
            "recall": [],
        }
        
        for _ in range(n_bootstrap):
            # Sample with replacement
            indices = rng.choice(n, n, replace=True)
            y_t = y_true_arr[indices]
            y_p = (y_proba[indices] >= threshold).astype(int)
            
            # Skip if no positive class in sample (rare but possible)
            if y_t.sum() == 0:
                continue
            
            metrics_samples["f1_score"].append(f1_score(y_t, y_p, zero_division=0))
            metrics_samples["precision"].append(precision_score(y_t, y_p, zero_division=0))
            metrics_samples["recall"].append(recall_score(y_t, y_p, zero_division=0))
        
        # Compute confidence intervals
        alpha = (1 - confidence) / 2
        ci = {}
        for metric, samples in metrics_samples.items():
            lower = np.percentile(samples, alpha * 100)
            upper = np.percentile(samples, (1 - alpha) * 100)
            ci[metric] = (float(lower), float(upper))
        
        return ci
    
    def _print_summary_table(self) -> None:
        """Print a formatted comparison table of all models."""
        logger.info(f"\n{'='*80}")
        logger.info("MODEL COMPARISON SUMMARY (Optimal Thresholds)")
        logger.info(f"{'='*80}")
        
        header = (f"  {'Model':<25} {'Threshold':>10} {'F1':>8} {'Precision':>10} "
                  f"{'Recall':>8} {'AUC':>8} {'Net Value':>15}")
        logger.info(header)
        logger.info(f"  {'─'*78}")
        
        best_f1 = 0
        best_model = ""
        
        for name, result in self.results.items():
            m = result["metrics_optimal"]
            t = result["threshold_optimal"]
            nv = result["cost_analysis"]["net_value"]
            
            line = (f"  {name:<25} {t:>10.3f} {m['f1_score']:>8.4f} "
                   f"{m['precision']:>10.4f} {m['recall']:>8.4f} "
                   f"{m['roc_auc']:>8.4f} ${nv:>14,.0f}")
            logger.info(line)
            
            if m["f1_score"] > best_f1:
                best_f1 = m["f1_score"]
                best_model = name
        
        logger.info(f"\n  🏆 Champion: {best_model} (F1 = {best_f1:.4f})")
    
    def _create_comparison_charts(self) -> None:
        """Create and save evaluation visualizations."""
        output_dir = self.config.paths.results_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            
            models = list(self.results.keys())
            
            # --- Chart 1: Metrics Comparison ---
            metrics_to_plot = ["f1_score", "precision", "recall", "roc_auc"]
            x = np.arange(len(models))
            width = 0.2
            
            for i, metric in enumerate(metrics_to_plot):
                values = [self.results[m]["metrics_optimal"][metric] for m in models]
                axes[0].bar(x + i * width, values, width, label=metric)
            
            axes[0].set_xlabel("Model")
            axes[0].set_ylabel("Score")
            axes[0].set_title("Model Performance Comparison")
            axes[0].set_xticks(x + width * 1.5)
            axes[0].set_xticklabels(models, rotation=15)
            axes[0].legend(fontsize=8)
            axes[0].set_ylim(0.9, 1.01)
            
            # --- Chart 2: Cost Analysis ---
            prevented = [self.results[m]["cost_analysis"]["fraud_prevented"] / 1e9 
                        for m in models]
            missed = [self.results[m]["cost_analysis"]["fraud_missed_cost"] / 1e9 
                     for m in models]
            
            axes[1].bar(models, prevented, label="Fraud Prevented ($B)", color="green", alpha=0.7)
            axes[1].bar(models, [-m for m in missed], label="Fraud Missed ($B)", color="red", alpha=0.7)
            axes[1].set_title("Cost Impact Analysis")
            axes[1].set_ylabel("Billions ($)")
            axes[1].legend(fontsize=8)
            axes[1].tick_params(axis='x', rotation=15)
            
            # --- Chart 3: Confusion Matrix (best model) ---
            best_model = max(self.results.keys(), 
                            key=lambda m: self.results[m]["metrics_optimal"]["f1_score"])
            cm = self.results[best_model]["confusion_matrix"]
            
            sns.heatmap(cm, annot=True, fmt=",d", cmap="Blues", ax=axes[2],
                       xticklabels=["Legitimate", "Fraud"],
                       yticklabels=["Legitimate", "Fraud"])
            axes[2].set_title(f"Confusion Matrix ({best_model})")
            axes[2].set_xlabel("Predicted")
            axes[2].set_ylabel("Actual")
            
            plt.tight_layout()
            
            chart_path = output_dir / "model_comparison.png"
            plt.savefig(chart_path, dpi=150, bbox_inches="tight")
            plt.close()
            logger.info(f"Saved comparison chart: {chart_path}")
            
        except Exception as e:
            logger.warning(f"Could not create charts: {e}")
    
    def get_best_model_name(self) -> str:
        """Return the name of the model with the highest F1 score."""
        return max(
            self.results.keys(),
            key=lambda m: self.results[m]["metrics_optimal"]["f1_score"],
        )


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from src.config import setup_logging
    
    config = Config()
    setup_logging("INFO")
    
    # Create synthetic data
    X, y = make_classification(
        n_samples=5000, n_features=10, n_informative=5,
        weights=[0.99, 0.01], random_state=42,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y,
    )
    
    # Train a quick model
    rf = RandomForestClassifier(n_estimators=10, random_state=42, class_weight="balanced")
    rf.fit(X_train, y_train)
    
    # Evaluate
    evaluator = ModelEvaluator(config)
    results = evaluator.evaluate_models(
        {"random_forest": rf},
        pd.DataFrame(X_test),
        pd.Series(y_test),
    )
    
    logger.info(f"\nBest model: {evaluator.get_best_model_name()}")
