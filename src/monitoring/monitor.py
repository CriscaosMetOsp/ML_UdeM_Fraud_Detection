"""
Model monitoring module.
Detects data drift and performance degradation over time.
"""

import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FraudModelMonitor:
    """
    Lightweight monitor for the fraud detection model.

    Tracks:
    - Prediction score distribution drift (KS test)
    - Feature drift on key numeric columns
    - Fraud rate drift
    - Business metrics ($ at risk detected vs missed)
    """

    def __init__(self, reference_scores: np.ndarray, reference_features: pd.DataFrame):
        self.reference_scores = reference_scores
        self.reference_features = reference_features
        self.alerts: list[dict] = []
        self.report: dict = {}

    def check_score_drift(
        self, current_scores: np.ndarray, threshold: float = 0.05
    ) -> dict:
        """KS test on predicted probability distributions."""
        stat, p_value = stats.ks_2samp(self.reference_scores, current_scores)
        drift_detected = p_value < threshold
        result = {
            "test": "KS (score distribution)",
            "ks_statistic": round(float(stat), 4),
            "p_value": round(float(p_value), 4),
            "drift_detected": drift_detected,
            "threshold": threshold,
        }
        if drift_detected:
            self.alerts.append({"type": "SCORE_DRIFT", **result})
            logger.warning(f"⚠️ Score drift detected! KS={stat:.4f} p={p_value:.4f}")
        return result

    def check_feature_drift(
        self,
        current_features: pd.DataFrame,
        numerical_cols: list,
        threshold: float = 0.05,
    ) -> list[dict]:
        """KS test per numerical feature."""
        results = []
        for col in numerical_cols:
            if (
                col not in self.reference_features.columns
                or col not in current_features.columns
            ):
                continue
            stat, p_value = stats.ks_2samp(
                self.reference_features[col].dropna(),
                current_features[col].dropna(),
            )
            drift = p_value < threshold
            r = {
                "feature": col,
                "ks_statistic": round(float(stat), 4),
                "p_value": round(float(p_value), 4),
                "drift_detected": drift,
            }
            results.append(r)
            if drift:
                self.alerts.append({"type": "FEATURE_DRIFT", **r})
                logger.warning(
                    f"⚠️ Feature drift on '{col}': KS={stat:.4f} p={p_value:.4f}"
                )
        return results

    def check_fraud_rate(
        self,
        current_fraud_rate: float,
        expected_rate: float = 0.0052,
        tolerance: float = 0.005,
    ) -> dict:
        """Flag if fraud rate deviates too much from training baseline."""
        deviation = abs(current_fraud_rate - expected_rate)
        alert = deviation > tolerance
        result = {
            "current_fraud_rate": round(current_fraud_rate, 6),
            "expected_fraud_rate": expected_rate,
            "deviation": round(deviation, 6),
            "alert": alert,
        }
        if alert:
            self.alerts.append({"type": "FRAUD_RATE_ANOMALY", **result})
            logger.warning(
                f"⚠️ Unusual fraud rate: {current_fraud_rate:.4%} (expected ~{expected_rate:.4%})"
            )
        return result

    def check_business_impact(
        self, y_true: np.ndarray, y_pred: np.ndarray, amounts: np.ndarray
    ) -> dict:
        """Compute $ detected vs missed fraud."""
        fraud_mask = y_true == 1
        detected = amounts[fraud_mask & (y_pred == 1)].sum()
        missed = amounts[fraud_mask & (y_pred == 0)].sum()
        false_positives = int(((y_pred == 1) & (y_true == 0)).sum())
        result = {
            "fraud_usd_detected": round(float(detected), 2),
            "fraud_usd_missed": round(float(missed), 2),
            "detection_rate_usd": round(
                float(detected / (detected + missed + 1e-9)), 4
            ),
            "false_positives": false_positives,
        }
        logger.info(
            f"$ Detected: ${detected:,.2f} | Missed: ${missed:,.2f} | FP: {false_positives}"
        )
        return result

    def generate_report(
        self, score_result, feature_results, fraud_rate_result, business_result
    ) -> dict:
        self.report = {
            "timestamp": datetime.utcnow().isoformat(),
            "score_drift": score_result,
            "feature_drift": feature_results,
            "fraud_rate": fraud_rate_result,
            "business_impact": business_result,
            "total_alerts": len(self.alerts),
            "alerts": self.alerts,
            "status": "🔴 ACTION REQUIRED" if self.alerts else "🟢 HEALTHY",
        }
        return self.report

    def save_report(self, path: str = "logs/monitoring_report.json"):
        Path(path).parent.mkdir(exist_ok=True)

        def _default(obj):
            if isinstance(obj, (bool, int, float, str)):
                return obj
            if hasattr(obj, "item"):  # numpy scalar
                return obj.item()
            raise TypeError(f"Not serializable: {type(obj)}")

        with open(path, "w") as f:
            json.dump(self.report, f, indent=2, default=_default)
        logger.info(f"Monitoring report saved to {path}")


def run_monitoring_check(
    model,
    scaler,
    X_train: pd.DataFrame,
    y_train,
    X_test: pd.DataFrame,
    y_test,
    amounts: pd.Series,
    numerical_cols: list,
):
    """Full monitoring check: drift + business impact."""
    # Reference scores from training data (sample for speed)
    ref_sample = X_train.sample(min(10000, len(X_train)), random_state=42)
    ref_scores = model.predict_proba(ref_sample)[:, 1]

    monitor = FraudModelMonitor(ref_scores, X_train)

    current_scores = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    score_result = monitor.check_score_drift(current_scores)
    feature_results = monitor.check_feature_drift(X_test, numerical_cols)
    fraud_rate_result = monitor.check_fraud_rate(float(y_test.mean()))
    business_result = monitor.check_business_impact(
        np.array(y_test), y_pred, np.array(amounts)
    )

    report = monitor.generate_report(
        score_result, feature_results, fraud_rate_result, business_result
    )
    monitor.save_report()

    print(f"\n=== Monitoring Report ===")
    print(f"Status: {report['status']}")
    print(f"Alerts: {report['total_alerts']}")
    print(f"Business Impact: ${business_result['fraud_usd_detected']:,.2f} detected")
    return report
