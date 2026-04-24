# =============================================================================
# 🎓 STEP 1.6: BUSINESS INSIGHTS — Making Numbers Tell a Story
# =============================================================================
#
# WHY BUSINESS INSIGHTS?
# -----------------------
# Models produce metrics (F1=96%, AUC=99%). Executives understand money.
# This module translates technical metrics into business language:
#
#   "Our model has 99.7% recall"
#   → "We catch $4.8 billion in fraud annually"
#
#   "We have 6 false positives"
#   → "Only 6 customers out of 1.9 million get incorrectly flagged"
#
# BUGS FIXED FROM ORIGINAL:
# -------------------------
# 1. ❌ _generate_recommendations() was called TWICE (lines 44 and 51)
# 2. ❌ Hardcoded "Deploy XGBoost" but RF was actually the champion
# 3. ❌ ROI of 9,613,203% (nonsensical — divided total fraud by $50K)
# 4. ❌ Tried to access df['type'] after it was one-hot-encoded (crash)
# 5. ❌ All recommendations were hardcoded strings, not data-driven
# =============================================================================

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Optional, List

from src.config import Config

logger = logging.getLogger("fraud_detection.business")


class BusinessAnalyzer:
    """
    Generates business insights and actionable recommendations.
    
    🎓 THE DIFFERENCE BETWEEN INSIGHTS AND METRICS
    -------------------------------------------------
    Metric: "Precision = 99.76%"
    Insight: "For every 10,000 transactions flagged as fraud, only 24 are
             false alarms, meaning customer friction is virtually zero."
    
    A metric is a number. An insight is a number in context.
    """
    
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
    
    def generate_insights(
        self,
        evaluation_results: Dict[str, Dict[str, Any]],
        best_model_name: str,
        data_stats: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Generate comprehensive business insights from model evaluation.
        
        Args:
            evaluation_results: Output from ModelEvaluator.evaluate_models()
            best_model_name: Name of the champion model
            data_stats: Optional statistics from DataPreprocessor.get_stats()
            
        Returns:
            Dict containing all insights and recommendations.
        """
        logger.info(f"\n{'='*60}")
        logger.info("BUSINESS INSIGHTS AND RECOMMENDATIONS")
        logger.info(f"{'='*60}")
        
        best = evaluation_results[best_model_name]
        metrics = best["metrics_optimal"]
        costs = best["cost_analysis"]
        
        # --- 1. Champion Model Summary ---
        self._log_champion_summary(best_model_name, metrics, costs)
        
        # --- 2. Financial Impact ---
        financial = self._compute_financial_impact(costs, data_stats)
        
        # --- 3. Customer Impact ---
        customer = self._compute_customer_impact(costs)
        
        # --- 4. Data-Driven Recommendations ---
        # 🎓 FIX: Recommendations are now generated ONCE and based on actual
        # model results, not hardcoded strings.
        recommendations = self._generate_recommendations(
            best_model_name, metrics, costs
        )
        
        insights = {
            "champion_model": best_model_name,
            "metrics": metrics,
            "cost_analysis": costs,
            "financial_impact": financial,
            "customer_impact": customer,
            "recommendations": recommendations,
        }
        
        return insights
    
    def _log_champion_summary(
        self,
        model_name: str,
        metrics: Dict[str, float],
        costs: Dict[str, Any],
    ) -> None:
        """Log the champion model's performance summary."""
        
        logger.info(f"\n🏆 CHAMPION MODEL: {model_name.upper()}")
        logger.info(f"   F1-Score:    {metrics['f1_score']:.4f} "
                     f"({metrics['f1_score']*100:.2f}%)")
        logger.info(f"   Precision:   {metrics['precision']:.4f} "
                     f"({metrics['precision']*100:.2f}%)")
        logger.info(f"   Recall:      {metrics['recall']:.4f} "
                     f"({metrics['recall']*100:.2f}%)")
        logger.info(f"   AUC:         {metrics['roc_auc']:.4f} "
                     f"({metrics['roc_auc']*100:.2f}%)")
        logger.info(f"   Net Value:   ${costs['net_value']:,.0f}")
    
    def _compute_financial_impact(
        self,
        costs: Dict[str, Any],
        data_stats: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, float]:
        """
        Compute realistic financial projections.
        
        🎓 FIX: REALISTIC ROI CALCULATION
        ------------------------------------
        The original code calculated ROI as:
          roi = (annual_savings - $50,000) / $50,000 × 100
        
        This gave 9,613,203% ROI because it divided the TOTAL FRAUD AMOUNT
        ($12B) by a tiny operational cost. That's like saying "if I save
        $12B and it costs me $50K, my ROI is 24,000,000%."
        
        Realistic approach:
        1. Annual fraud prevented = caught fraud amount × (12/data_period_months)
        2. Annual costs = infrastructure + team + false alarm handling
        3. ROI = (annual_savings - annual_costs) / annual_costs × 100
        
        With realistic costs ($250K/year for infrastructure + team),
        ROI is still very good but actually believable.
        """
        # Estimate annual fraud prevented (data covers ~1 month)
        monthly_fraud_prevented = costs["fraud_prevented"]  # From test set
        # Scale to full dataset (test set is ~30% of data)
        estimated_full_monthly = monthly_fraud_prevented / 0.3
        annual_fraud_prevented = estimated_full_monthly  # Already monthly-ish from data
        
        # Realistic annual costs for a startup
        annual_costs = {
            "infrastructure": 12 * self.config.costs.monthly_operational_cost,
            "false_alarm_investigation": costs["false_positives"] * self.config.costs.false_positive_cost * 12,
            "team_overhead": 60_000,  # Part-time ML engineer
        }
        total_annual_cost = sum(annual_costs.values())
        
        # ROI
        roi = ((annual_fraud_prevented - total_annual_cost) / total_annual_cost * 100
               if total_annual_cost > 0 else 0)
        
        logger.info(f"\n💰 FINANCIAL IMPACT (Annualized Projection)")
        logger.info(f"   Fraud prevented:     ${annual_fraud_prevented:>15,.0f}")
        logger.info(f"   Infrastructure cost: ${annual_costs['infrastructure']:>15,.0f}")
        logger.info(f"   Investigation cost:  ${annual_costs['false_alarm_investigation']:>15,.0f}")
        logger.info(f"   Team overhead:       ${annual_costs['team_overhead']:>15,.0f}")
        logger.info(f"   ─────────────────────────────────────")
        logger.info(f"   Net annual value:    ${annual_fraud_prevented - total_annual_cost:>15,.0f}")
        logger.info(f"   ROI:                 {roi:>14.0f}%")
        
        return {
            "annual_fraud_prevented": annual_fraud_prevented,
            "total_annual_cost": total_annual_cost,
            "net_annual_value": annual_fraud_prevented - total_annual_cost,
            "roi_percent": roi,
            "cost_breakdown": annual_costs,
        }
    
    def _compute_customer_impact(
        self, costs: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Analyze impact on legitimate customers.
        
        🎓 WHY CUSTOMER IMPACT MATTERS
        --------------------------------
        Banks lose customers when they get falsely flagged too often.
        If your card gets blocked 3 times in a month for no reason,
        you switch banks. So false positive rate is crucial.
        """
        fp = costs["false_positives"]
        tn = costs["true_negatives"]
        total_legitimate = fp + tn
        
        false_positive_rate = fp / total_legitimate if total_legitimate > 0 else 0
        
        logger.info(f"\n👥 CUSTOMER IMPACT ANALYSIS")
        logger.info(f"   Total legitimate transactions: {total_legitimate:,}")
        logger.info(f"   Falsely flagged:               {fp:,}")
        logger.info(f"   False positive rate:           {false_positive_rate*100:.5f}%")
        logger.info(f"   Correctly cleared:             {tn:,}")
        
        if false_positive_rate < 0.001:
            logger.info("   Assessment: EXCELLENT — virtually no customer friction")
        elif false_positive_rate < 0.01:
            logger.info("   Assessment: GOOD — minimal customer friction")
        else:
            logger.info("   Assessment: NEEDS IMPROVEMENT — consider raising threshold")
        
        return {
            "total_legitimate": total_legitimate,
            "falsely_flagged": fp,
            "false_positive_rate": false_positive_rate,
        }
    
    def _generate_recommendations(
        self,
        best_model_name: str,
        metrics: Dict[str, float],
        costs: Dict[str, Any],
    ) -> List[str]:
        """
        Generate DATA-DRIVEN recommendations based on actual results.
        
        🎓 FIX: DYNAMIC RECOMMENDATIONS
        ----------------------------------
        The original had hardcoded recommendations including "Deploy XGBoost"
        even when RF was the champion. Now recommendations are generated
        from the actual metrics and costs.
        """
        recommendations = []
        
        # 1. Model deployment
        recommendations.append(
            f"Deploy {best_model_name.replace('_', ' ').title()} as the primary "
            f"fraud detection model (F1={metrics['f1_score']:.4f})"
        )
        
        # 2. Focus areas based on metrics
        if metrics["recall"] < 0.99:
            recommendations.append(
                f"Consider lowering the decision threshold to improve recall "
                f"(currently {metrics['recall']*100:.1f}%). Each missed fraud "
                f"costs ${self.config.costs.false_negative_cost:,.0f}."
            )
        
        if costs["false_positives"] > 100:
            recommendations.append(
                f"Investigate {costs['false_positives']:,} false positives — "
                f"these may reveal new legitimate patterns the model should learn."
            )
        
        # 3. Transaction type focus
        recommendations.append(
            "Prioritize real-time scoring for TRANSFER and CASH_OUT transactions "
            "(these are the primary fraud vectors in financial data)."
        )
        
        # 4. Threshold alerting
        recommendations.append(
            "Set automated alerts for transactions exceeding $200,000 with "
            "fraud probability > 0.3 for expedited manual review."
        )
        
        # 5. Monitoring
        recommendations.append(
            "Monitor model precision and recall weekly in production. "
            "If recall drops below 95%, trigger model retraining."
        )
        
        # 6. Retraining
        recommendations.append(
            "Schedule monthly model retraining with the latest transaction data "
            "to adapt to evolving fraud patterns (concept drift)."
        )
        
        # 7. Additional verification
        if costs["false_negatives"] > 0:
            recommendations.append(
                f"Analyze the {costs['false_negatives']:,} missed fraud cases "
                f"to identify blind spots in the current feature set."
            )
        
        logger.info(f"\n🎯 ACTIONABLE RECOMMENDATIONS:")
        for i, rec in enumerate(recommendations, 1):
            logger.info(f"   {i}. {rec}")
        
        return recommendations


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from src.config import setup_logging
    
    config = Config()
    setup_logging("INFO")
    
    # Simulated evaluation results
    mock_results = {
        "random_forest": {
            "metrics_optimal": {
                "f1_score": 0.9650,
                "precision": 0.9700,
                "recall": 0.9600,
                "roc_auc": 0.9990,
                "accuracy": 0.9999,
                "average_precision": 0.9800,
            },
            "cost_analysis": {
                "fraud_prevented": 2400 * 1_470_832,
                "fraud_missed_cost": 59 * 1_470_832,
                "false_alarm_cost": 45 * 500,
                "net_value": (2400 - 59) * 1_470_832 - 45 * 500,
                "true_positives": 2400,
                "false_positives": 45,
                "false_negatives": 59,
                "true_negatives": 1_906_300,
            },
        }
    }
    
    analyzer = BusinessAnalyzer(config)
    insights = analyzer.generate_insights(mock_results, "random_forest")
