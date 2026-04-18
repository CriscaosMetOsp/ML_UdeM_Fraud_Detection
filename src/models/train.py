"""
Model training with MLflow experiment tracking.
Trains Random Forest and XGBoost, registers the best model.
"""
import logging
import os
import sys
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import numpy as np
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data.preprocessing import (
    build_feature_matrix,
    load_data,
    scale_features,
    split_data,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config(path: str = "configs/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def compute_metrics(y_true, y_pred, y_prob) -> dict:
    """Compute classification metrics for imbalanced problem."""
    return {
        "roc_auc": roc_auc_score(y_true, y_prob),
        "pr_auc": average_precision_score(y_true, y_prob),
        "f1": f1_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "fraud_detected": int(y_pred[y_true == 1].sum()),
        "total_fraud": int(y_true.sum()),
    }


def train_random_forest(X_train, y_train, params: dict):
    logger.info("Training Random Forest...")
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    return model


def train_xgboost(X_train, y_train, params: dict):
    logger.info("Training XGBoost...")
    model = XGBClassifier(**params)
    model.fit(X_train, y_train)
    return model


def run_experiment(model_name: str, model, params: dict, X_test, y_test,
                   experiment_name: str) -> tuple[dict, str]:
    """Run one MLflow experiment run and return metrics + run_id."""
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=model_name) as run:
        mlflow.log_params(params)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        metrics = compute_metrics(y_test, y_pred, y_prob)

        mlflow.log_metrics(metrics)
        logger.info(f"{model_name} | PR-AUC: {metrics['pr_auc']:.4f} | "
                    f"Recall: {metrics['recall']:.4f} | F1: {metrics['f1']:.4f}")
        logger.info(f"Fraud detected: {metrics['fraud_detected']}/{metrics['total_fraud']}")

        # Log model artifact
        if "XGB" in model_name or "xgb" in model_name.lower():
            mlflow.xgboost.log_model(model, artifact_path="model")
        else:
            mlflow.sklearn.log_model(model, artifact_path="model")

        return metrics, run.info.run_id


def register_best_model(best_run_id: str, registry_name: str, tracking_uri: str):
    """Register the winning run in MLflow Model Registry."""
    model_uri = f"runs:/{best_run_id}/model"
    result = mlflow.register_model(model_uri, registry_name)
    logger.info(f"Model registered: {registry_name} v{result.version}")
    return result


def main():
    # Setup
    os.chdir(Path(__file__).resolve().parents[2])
    cfg = load_config()

    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])

    # Data
    df = load_data(cfg["data"]["raw_path"])
    X, y = build_feature_matrix(df, cfg["features"])
    X_train, X_test, y_train, y_test = split_data(
        X, y, cfg["data"]["test_size"], cfg["data"]["random_state"]
    )
    X_train_sc, X_test_sc, scaler = scale_features(
        X_train, X_test, cfg["features"]["numerical_cols"]
    )

    # Save scaler
    Path("models").mkdir(exist_ok=True)
    joblib.dump(scaler, "models/scaler.pkl")

    results = {}

    # --- Random Forest ---
    rf_params = cfg["model"]["random_forest"]
    rf = train_random_forest(X_train_sc, y_train, rf_params)
    rf_metrics, rf_run_id = run_experiment(
        "RandomForest", rf, rf_params, X_test_sc, y_test,
        cfg["mlflow"]["experiment_name"]
    )
    results["RandomForest"] = {"metrics": rf_metrics, "run_id": rf_run_id, "model": rf}

    # --- XGBoost ---
    xgb_params = cfg["model"]["xgboost"]
    xgb = train_xgboost(X_train_sc, y_train, xgb_params)
    xgb_metrics, xgb_run_id = run_experiment(
        "XGBoost", xgb, xgb_params, X_test_sc, y_test,
        cfg["mlflow"]["experiment_name"]
    )
    results["XGBoost"] = {"metrics": xgb_metrics, "run_id": xgb_run_id, "model": xgb}

    # --- Select best by PR-AUC (better for imbalanced) ---
    best_name = max(results, key=lambda k: results[k]["metrics"]["pr_auc"])
    best = results[best_name]
    logger.info(f"\nBest model: {best_name} | PR-AUC: {best['metrics']['pr_auc']:.4f}")

    # Save best model locally
    joblib.dump(best["model"], "models/best_model.pkl")
    logger.info("Best model saved to models/best_model.pkl")

    # Register in MLflow Model Registry
    register_best_model(
        best["run_id"],
        cfg["mlflow"]["model_registry_name"],
        cfg["mlflow"]["tracking_uri"]
    )

    # Print final report
    y_pred = best["model"].predict(X_test_sc)
    print("\n=== Classification Report ===")
    print(classification_report(y_test, y_pred, target_names=["Legit", "Fraud"]))
    print("\n=== Confusion Matrix ===")
    print(confusion_matrix(y_test, y_pred))


if __name__ == "__main__":
    main()
