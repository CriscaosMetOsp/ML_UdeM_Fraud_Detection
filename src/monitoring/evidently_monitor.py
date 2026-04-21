"""
Monitoring con Evidently AI + Prometheus.

Flujo:
1. Evidently genera reportes de drift (HTML) y extrae métricas clave.
2. Las métricas se exponen como Prometheus Gauges en /metrics (puerto 8001).
3. Prometheus scrapea ese endpoint cada 30s.
4. Grafana visualiza los datos de Prometheus en dashboards.

Uso:
    python src/monitoring/evidently_monitor.py
    # Luego: docker compose up -d  (levanta Prometheus + Grafana)
"""

import json
import logging
import time
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from prometheus_client import Gauge, start_http_server

# Evidently v0.7 API
from evidently import BinaryClassification, DataDefinition, Dataset
from evidently.core.report import Report
from evidently.presets import ClassificationPreset, DataDriftPreset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ── Prometheus metrics ─────────────────────────────────────────────────────
# Data drift
DRIFT_SHARE = Gauge(
    "fraud_model_drift_share",
    "Fracción de features con drift detectado (Evidently)",
)
SCORE_DRIFT = Gauge(
    "fraud_model_score_drift_pvalue",
    "P-value KS test en la distribución de scores de fraude",
)

# Model performance (cuando hay etiquetas disponibles)
PRECISION = Gauge("fraud_model_precision", "Precision en ventana de evaluación")
RECALL = Gauge("fraud_model_recall", "Recall en ventana de evaluación")
F1 = Gauge("fraud_model_f1", "F1-score en ventana de evaluación")
PR_AUC = Gauge("fraud_model_pr_auc", "PR-AUC en ventana de evaluación")

# Business metrics
FRAUD_USD_DETECTED = Gauge(
    "fraud_model_usd_detected", "USD en fraude detectado en la ventana"
)
FRAUD_USD_MISSED = Gauge(
    "fraud_model_usd_missed", "USD en fraude no detectado en la ventana"
)
FALSE_POSITIVES = Gauge("fraud_model_false_positives", "Falsos positivos en la ventana")
FRAUD_RATE_CURRENT = Gauge(
    "fraud_model_current_fraud_rate", "Tasa de fraude actual en la ventana"
)

# Inference
PREDICTION_COUNT = Gauge(
    "fraud_model_prediction_count", "Número de predicciones en la ventana"
)
HIGH_RISK_COUNT = Gauge(
    "fraud_model_high_risk_count", "Predicciones con riesgo ALTO (proba >= 0.6)"
)


# ── Evidently report builder ───────────────────────────────────────────────


def _build_data_definition(feature_cols: list[str]) -> DataDefinition:
    """Construye el DataDefinition de Evidently para clasificación binaria."""
    return DataDefinition(
        numerical_columns=feature_cols,
        classification=[
            BinaryClassification(
                target="is_fraud",
                prediction_probas="fraud_proba",
            )
        ],
    )


def run_evidently_drift_report(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    feature_cols: list[str],
    output_dir: str = "monitoring_reports",
) -> dict:
    """
    Genera un reporte de drift con Evidently y devuelve métricas clave.

    Parameters
    ----------
    reference_df : DataFrame con features + is_fraud + fraud_proba (datos de entrenamiento)
    current_df   : DataFrame con las mismas columnas (ventana de producción)
    feature_cols : Lista de columnas numéricas a monitorear
    output_dir   : Directorio donde guardar el HTML

    Returns
    -------
    dict con métricas extraídas del reporte
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    dd = _build_data_definition(feature_cols)
    ref_ds = Dataset.from_pandas(reference_df, data_definition=dd)
    cur_ds = Dataset.from_pandas(current_df, data_definition=dd)

    # Reporte de drift
    drift_report = Report([DataDriftPreset()])
    drift_snap = drift_report.run(reference_data=ref_ds, current_data=cur_ds)

    # Reporte de clasificación (cuando hay ground truth)
    clf_report = Report([ClassificationPreset()])
    clf_snap = clf_report.run(reference_data=ref_ds, current_data=cur_ds)

    # Guardar HTMLs
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    drift_snap.save_html(f"{output_dir}/drift_{ts}.html")
    clf_snap.save_html(f"{output_dir}/classification_{ts}.html")
    logger.info(f"Reportes Evidently guardados en {output_dir}/")

    # Extraer métricas del reporte de drift
    drift_metrics = _extract_drift_metrics(drift_snap.dict()["metrics"])

    # Extraer métricas de clasificación
    clf_metrics = _extract_classification_metrics(clf_snap.dict()["metrics"])

    result = {
        **drift_metrics,
        **clf_metrics,
        "timestamp": datetime.utcnow().isoformat(),
    }
    _save_metrics_json(result, output_dir, ts)
    return result


def _extract_drift_metrics(metrics: list) -> dict:
    """Extrae drift_share y p-values del dict de métricas de Evidently."""
    result = {"drift_share": 0.0, "score_drift_pvalue": 1.0, "drifted_features": []}

    for m in metrics:
        name = m.get("metric_name", "")
        val = m.get("value")

        # Fracción de columnas con drift
        if "DriftedColumnsCount" in name and isinstance(val, dict):
            result["drift_share"] = float(val.get("share", 0.0))

        # Drift del score de fraude (p-value)
        if "ValueDrift" in name and "fraud_proba" in name:
            result["score_drift_pvalue"] = float(val) if val is not None else 1.0

        # Features con drift
        if "ValueDrift" in name and isinstance(val, (int, float)):
            if float(val) < 0.05:
                col = (
                    name.split("column=")[1].split(",")[0] if "column=" in name else ""
                )
                if col:
                    result["drifted_features"].append(col)

    return result


def _extract_classification_metrics(metrics: list) -> dict:
    """Extrae precision, recall, f1, roc_auc del dict de métricas de clasificación."""
    result = {"precision": 0.0, "recall": 0.0, "f1": 0.0, "pr_auc": 0.0}

    for m in metrics:
        name = m.get("metric_name", "")
        val = m.get("value")
        if val is None:
            continue
        if isinstance(val, dict):
            # Algunos presets anidan las métricas por clase
            val_current = val.get("current", val)
            if "Precision" in name:
                result["precision"] = (
                    float(val_current) if not isinstance(val_current, dict) else 0.0
                )
            elif "Recall" in name:
                result["recall"] = (
                    float(val_current) if not isinstance(val_current, dict) else 0.0
                )
            elif "F1" in name:
                result["f1"] = (
                    float(val_current) if not isinstance(val_current, dict) else 0.0
                )
            elif "PRAuc" in name or "PR AUC" in name or "AveragePrecision" in name:
                result["pr_auc"] = (
                    float(val_current) if not isinstance(val_current, dict) else 0.0
                )
        elif isinstance(val, (int, float)):
            if "Precision" in name:
                result["precision"] = float(val)
            elif "Recall" in name:
                result["recall"] = float(val)
            elif "F1" in name:
                result["f1"] = float(val)

    return result


def _save_metrics_json(metrics: dict, output_dir: str, ts: str):
    """Serializa métricas a JSON ignorando tipos numpy."""

    def _default(o):
        if hasattr(o, "item"):
            return o.item()
        return str(o)

    path = Path(output_dir) / f"metrics_{ts}.json"
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2, default=_default)

    # Último reporte siempre disponible como latest
    latest = Path(output_dir) / "latest_metrics.json"
    with open(latest, "w") as f:
        json.dump(metrics, f, indent=2, default=_default)

    logger.info(f"Métricas guardadas en {path}")


# ── Business impact ─────────────────────────────────────────────────────────


def compute_business_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, amounts: np.ndarray
) -> dict:
    fraud_mask = y_true == 1
    detected = float(amounts[fraud_mask & (y_pred == 1)].sum())
    missed = float(amounts[fraud_mask & (y_pred == 0)].sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    detection_rate = detected / (detected + missed + 1e-9)

    return {
        "fraud_usd_detected": round(detected, 2),
        "fraud_usd_missed": round(missed, 2),
        "false_positives": fp,
        "detection_rate_usd": round(detection_rate, 4),
        "current_fraud_rate": round(float(y_true.mean()), 6),
        "prediction_count": len(y_true),
        "high_risk_count": int((y_pred == 1).sum()),
    }


# ── Prometheus exporter ──────────────────────────────────────────────────────


def update_prometheus_gauges(drift_metrics: dict, business_metrics: dict):
    """Actualiza todos los Prometheus Gauges con las métricas calculadas."""
    DRIFT_SHARE.set(drift_metrics.get("drift_share", 0.0))
    SCORE_DRIFT.set(drift_metrics.get("score_drift_pvalue", 1.0))
    PRECISION.set(drift_metrics.get("precision", 0.0))
    RECALL.set(drift_metrics.get("recall", 0.0))
    F1.set(drift_metrics.get("f1", 0.0))
    PR_AUC.set(drift_metrics.get("pr_auc", 0.0))

    FRAUD_USD_DETECTED.set(business_metrics.get("fraud_usd_detected", 0.0))
    FRAUD_USD_MISSED.set(business_metrics.get("fraud_usd_missed", 0.0))
    FALSE_POSITIVES.set(business_metrics.get("false_positives", 0))
    FRAUD_RATE_CURRENT.set(business_metrics.get("current_fraud_rate", 0.0))
    PREDICTION_COUNT.set(business_metrics.get("prediction_count", 0))
    HIGH_RISK_COUNT.set(business_metrics.get("high_risk_count", 0))

    logger.info(
        f"Prometheus actualizado | drift_share={drift_metrics.get('drift_share', 0):.2%} "
        f"| recall={drift_metrics.get('recall', 0):.4f} "
        f"| USD detected=${business_metrics.get('fraud_usd_detected', 0):,.0f}"
    )


# ── Main monitoring loop ─────────────────────────────────────────────────────


def build_reference_dataframe(
    model, X_train: pd.DataFrame, y_train: pd.Series, feature_cols: list[str]
) -> pd.DataFrame:
    """Construye el DataFrame de referencia con scores del modelo."""
    sample = X_train.sample(min(10_000, len(X_train)), random_state=42)
    proba = model.predict_proba(sample)[:, 1]
    ref = sample[feature_cols].copy()
    ref["is_fraud"] = y_train.loc[sample.index].values
    ref["fraud_proba"] = proba
    return ref


def build_current_dataframe(
    model, X_test: pd.DataFrame, y_test: pd.Series, feature_cols: list[str]
) -> pd.DataFrame:
    """Construye el DataFrame de producción con scores del modelo."""
    proba = model.predict_proba(X_test)[:, 1]
    cur = X_test[feature_cols].copy()
    cur["is_fraud"] = y_test.values
    cur["fraud_proba"] = proba
    return cur


def run_monitoring_loop(
    model,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    amounts: pd.Series,
    feature_cols: list[str],
    prometheus_port: int = 8001,
    interval_seconds: int = 60,
    output_dir: str = "monitoring_reports",
):
    """
    Loop principal de monitoreo.

    1. Arranca servidor Prometheus en `prometheus_port`.
    2. Cada `interval_seconds` genera reporte Evidently y actualiza métricas.
    """
    logger.info(f"Iniciando servidor Prometheus en puerto {prometheus_port}...")
    start_http_server(prometheus_port)
    logger.info(f"Métricas disponibles en http://localhost:{prometheus_port}/metrics")

    ref_df = build_reference_dataframe(model, X_train, y_train, feature_cols)
    y_pred = model.predict(X_test)

    iteration = 0
    while True:
        iteration += 1
        logger.info(f"=== Iteración de monitoreo #{iteration} ===")

        try:
            cur_df = build_current_dataframe(model, X_test, y_test, feature_cols)

            drift_metrics = run_evidently_drift_report(
                ref_df, cur_df, feature_cols, output_dir
            )

            business_metrics = compute_business_metrics(
                np.array(y_test), y_pred, np.array(amounts)
            )

            update_prometheus_gauges(drift_metrics, business_metrics)

            # Log drifted features
            if drift_metrics.get("drifted_features"):
                logger.warning(
                    f"⚠️  Features con drift: {drift_metrics['drifted_features']}"
                )
            else:
                logger.info("✅ Sin drift detectado en features")

        except Exception as e:
            logger.error(f"Error en iteración de monitoreo: {e}", exc_info=True)

        logger.info(f"Próxima revisión en {interval_seconds}s...")
        time.sleep(interval_seconds)


# ── Standalone entry point ───────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

    from src.data.preprocessing import (
        build_feature_matrix,
        load_data,
        scale_features,
        split_data,
    )

    cfg = yaml.safe_load(open("configs/config.yaml"))
    feature_cols = cfg["features"]["numerical_cols"]

    df = load_data(cfg["data"]["raw_path"])
    X, y = build_feature_matrix(df, cfg["features"])
    X_train, X_test, y_train, y_test = split_data(
        X, y, cfg["data"]["test_size"], cfg["data"]["random_state"]
    )
    X_train_sc, X_test_sc, scaler = scale_features(
        X_train, X_test, cfg["features"]["numerical_cols"]
    )

    model = joblib.load(cfg["api"]["model_path"])
    amounts = df.loc[X_test.index, "amt"]

    run_monitoring_loop(
        model=model,
        X_train=X_train_sc,
        y_train=y_train,
        X_test=X_test_sc,
        y_test=y_test,
        amounts=amounts,
        feature_cols=feature_cols,
        prometheus_port=8001,
        interval_seconds=60,
        output_dir="monitoring_reports",
    )
