"""Unit tests para el módulo de monitoreo con Evidently + Prometheus."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.monitoring.evidently_monitor import (
    _extract_drift_metrics,
    _extract_classification_metrics,
    compute_business_metrics,
    update_prometheus_gauges,
    build_reference_dataframe,
    build_current_dataframe,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def sample_predictions():
    rng = np.random.default_rng(42)
    n = 500
    y_true = np.array([0] * 475 + [1] * 25)
    y_pred = y_true.copy()
    y_pred[:10] = 1 - y_pred[:10]  # introduce some errors
    amounts = rng.exponential(70, n)
    return y_true, y_pred, amounts


@pytest.fixture
def mock_drift_metrics():
    return [
        {
            "metric_name": "DriftedColumnsCount(drift_share=0.5)",
            "value": {"count": 2.0, "share": 0.25},
        },
        {
            "metric_name": "ValueDrift(column=fraud_proba,method=K-S p_value,threshold=0.05)",
            "value": 0.03,
        },
        {
            "metric_name": "ValueDrift(column=amt,method=K-S p_value,threshold=0.05)",
            "value": 0.02,
        },
    ]


@pytest.fixture
def mock_clf_metrics():
    return [
        {"metric_name": "Precision", "value": 0.72},
        {"metric_name": "Recall", "value": 0.85},
        {"metric_name": "F1", "value": 0.78},
    ]


# ── Business metrics ──────────────────────────────────────────────────────────


def test_compute_business_metrics_shapes(sample_predictions):
    y_true, y_pred, amounts = sample_predictions
    result = compute_business_metrics(y_true, y_pred, amounts)
    for key in [
        "fraud_usd_detected",
        "fraud_usd_missed",
        "false_positives",
        "detection_rate_usd",
        "current_fraud_rate",
        "prediction_count",
        "high_risk_count",
    ]:
        assert key in result, f"Missing key: {key}"


def test_detection_rate_between_0_and_1(sample_predictions):
    y_true, y_pred, amounts = sample_predictions
    result = compute_business_metrics(y_true, y_pred, amounts)
    assert 0.0 <= result["detection_rate_usd"] <= 1.0


def test_usd_detected_plus_missed_equals_total_fraud(sample_predictions):
    y_true, y_pred, amounts = sample_predictions
    result = compute_business_metrics(y_true, y_pred, amounts)
    total = result["fraud_usd_detected"] + result["fraud_usd_missed"]
    expected = float(amounts[y_true == 1].sum())
    assert abs(total - expected) < 0.01


def test_perfect_model_zero_missed():
    y_true = np.array([0, 0, 0, 1, 1])
    y_pred = np.array([0, 0, 0, 1, 1])
    amounts = np.array([50.0, 60.0, 70.0, 100.0, 200.0])
    result = compute_business_metrics(y_true, y_pred, amounts)
    assert result["fraud_usd_missed"] == 0.0
    assert result["false_positives"] == 0


def test_fraud_rate_correct(sample_predictions):
    y_true, y_pred, amounts = sample_predictions
    result = compute_business_metrics(y_true, y_pred, amounts)
    assert abs(result["current_fraud_rate"] - y_true.mean()) < 1e-6


# ── Drift metric extraction ───────────────────────────────────────────────────


def test_extract_drift_metrics_share(mock_drift_metrics):
    result = _extract_drift_metrics(mock_drift_metrics)
    assert result["drift_share"] == pytest.approx(0.25)


def test_extract_drift_metrics_pvalue(mock_drift_metrics):
    result = _extract_drift_metrics(mock_drift_metrics)
    assert result["score_drift_pvalue"] == pytest.approx(0.03)


def test_extract_drift_metrics_drifted_features(mock_drift_metrics):
    """Features con p-value < 0.05 deben aparecer en drifted_features."""
    result = _extract_drift_metrics(mock_drift_metrics)
    # amt tiene p=0.02 < 0.05
    assert "amt" in result["drifted_features"]


def test_extract_drift_empty_input():
    result = _extract_drift_metrics([])
    assert result["drift_share"] == 0.0
    assert result["score_drift_pvalue"] == 1.0
    assert result["drifted_features"] == []


# ── Classification metric extraction ─────────────────────────────────────────


def test_extract_classification_metrics(mock_clf_metrics):
    result = _extract_classification_metrics(mock_clf_metrics)
    assert result["precision"] == pytest.approx(0.72)
    assert result["recall"] == pytest.approx(0.85)
    assert result["f1"] == pytest.approx(0.78)


def test_extract_classification_empty():
    result = _extract_classification_metrics([])
    assert result == {"precision": 0.0, "recall": 0.0, "f1": 0.0, "pr_auc": 0.0}


# ── Prometheus gauges ─────────────────────────────────────────────────────────


def test_update_prometheus_gauges_no_exception(sample_predictions):
    """Verificar que actualizar los gauges no lanza excepciones."""
    y_true, y_pred, amounts = sample_predictions
    business = compute_business_metrics(y_true, y_pred, amounts)
    drift = {
        "drift_share": 0.1,
        "score_drift_pvalue": 0.3,
        "precision": 0.7,
        "recall": 0.85,
        "f1": 0.77,
        "pr_auc": 0.88,
    }
    # No debe lanzar excepción
    update_prometheus_gauges(drift, business)


# ── DataFrame builders ────────────────────────────────────────────────────────


def test_build_reference_dataframe_columns():
    """El DataFrame de referencia debe incluir is_fraud y fraud_proba."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import make_classification

    X_arr, y_arr = make_classification(n_samples=300, n_features=5, random_state=42)
    X = pd.DataFrame(X_arr, columns=[f"f{i}" for i in range(5)])
    y = pd.Series(y_arr)
    model = RandomForestClassifier(n_estimators=5, random_state=42).fit(X, y)

    ref = build_reference_dataframe(model, X, y, [f"f{i}" for i in range(5)])
    assert "is_fraud" in ref.columns
    assert "fraud_proba" in ref.columns
    assert ref["fraud_proba"].between(0, 1).all()


def test_build_current_dataframe_columns():
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import make_classification

    X_arr, y_arr = make_classification(n_samples=200, n_features=5, random_state=0)
    X = pd.DataFrame(X_arr, columns=[f"f{i}" for i in range(5)])
    y = pd.Series(y_arr)
    model = RandomForestClassifier(n_estimators=5, random_state=0).fit(X, y)

    cur = build_current_dataframe(model, X, y, [f"f{i}" for i in range(5)])
    assert "is_fraud" in cur.columns
    assert "fraud_proba" in cur.columns
    assert len(cur) == len(X)
