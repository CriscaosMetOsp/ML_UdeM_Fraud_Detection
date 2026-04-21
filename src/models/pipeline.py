"""
Prefect pipeline completo: preprocesamiento → HPO con Optuna → evaluación → registro.
Run: python src/models/pipeline.py [--hpo] [--trials 30] [--model xgboost]
Run: uv run src/models/pipeline.py [--hpo] [--trials 30] [--model xgboost]
Levantar el servidor de observabilidad con prefect: uv run prefect server start
Ver resultado en mlflow: uv run mlflow ui --port 5000
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import joblib
import yaml
from prefect import flow, get_run_logger, task

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.preprocessing import (
    build_feature_matrix,
    load_data,
    scale_features,
    split_data,
)
from src.models.train import (
    register_best_model,
    run_experiment,
    train_random_forest,
    train_xgboost,
)


def load_config(path="configs/config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


# ── Tasks ──────────────────────────────────────────────────────────────────────


@task(name="load-data", retries=2, retry_delay_seconds=10)
def task_load_data(cfg):
    logger = get_run_logger()
    df = load_data(cfg["data"]["raw_path"])
    logger.info(f"Datos cargados: {len(df):,} filas")
    return df


@task(name="feature-engineering")
def task_feature_engineering(df, cfg):
    logger = get_run_logger()
    X, y = build_feature_matrix(df, cfg["features"])
    logger.info(f"Feature matrix: {X.shape} | Fraude: {y.sum()} ({y.mean():.4%})")
    return X, y


@task(name="split-and-scale")
def task_split_scale(X, y, cfg):
    logger = get_run_logger()
    X_train, X_test, y_train, y_test = split_data(
        X, y, cfg["data"]["test_size"], cfg["data"]["random_state"]
    )
    X_train_sc, X_test_sc, scaler = scale_features(
        X_train, X_test, cfg["features"]["numerical_cols"]
    )
    Path("models").mkdir(exist_ok=True)
    joblib.dump(scaler, "models/scaler.pkl")
    logger.info(f"Train: {len(X_train_sc):,} | Test: {len(X_test_sc):,}")
    return X_train_sc, X_test_sc, y_train, y_test


@task(name="baseline-training")
def task_baseline(X_train, y_train, X_test, y_test, cfg):
    """Entrena RF y XGB con parámetros del config (sin HPO)."""
    logger = get_run_logger()

    rf = train_random_forest(X_train, y_train, cfg["model"]["random_forest"])
    rf_metrics, rf_run_id = run_experiment(
        "RandomForest_baseline",
        rf,
        cfg["model"]["random_forest"],
        X_test,
        y_test,
        cfg["mlflow"]["experiment_name"],
    )
    logger.info(
        f"RF  | PR-AUC: {rf_metrics['pr_auc']:.4f} | Recall: {rf_metrics['recall']:.4f}"
    )

    xgb = train_xgboost(X_train, y_train, cfg["model"]["xgboost"])
    xgb_metrics, xgb_run_id = run_experiment(
        "XGBoost_baseline",
        xgb,
        cfg["model"]["xgboost"],
        X_test,
        y_test,
        cfg["mlflow"]["experiment_name"],
    )
    logger.info(
        f"XGB | PR-AUC: {xgb_metrics['pr_auc']:.4f} | Recall: {xgb_metrics['recall']:.4f}"
    )

    if rf_metrics["pr_auc"] >= xgb_metrics["pr_auc"]:
        return rf, rf_metrics, rf_run_id, "RandomForest"
    return xgb, xgb_metrics, xgb_run_id, "XGBoost"


@task(name="hpo-optuna", timeout_seconds=3600)
def task_hpo(X_train, y_train, X_test, y_test, cfg, model_type: str, n_trials: int):
    """HPO con Optuna — cada trial queda trazado en MLflow."""
    logger = get_run_logger()
    logger.info(f"Iniciando HPO: {model_type.upper()} | {n_trials} trials")

    from src.models.hpo import run_hpo

    best_model, best_params, best_metrics, best_run_id = run_hpo(
        model_type=model_type,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        cfg=cfg,
        n_trials=n_trials,
    )
    logger.info(
        f"HPO listo | PR-AUC: {best_metrics['pr_auc']:.4f} "
        f"| Recall: {best_metrics['recall']:.4f}"
    )
    return best_model, best_params, best_metrics, best_run_id


@task(name="compare-and-register")
def task_compare_register(baseline_result, hpo_result, cfg):
    """Compara baseline vs HPO y registra el ganador en MLflow Model Registry."""
    logger = get_run_logger()
    baseline_model, baseline_metrics, baseline_run_id, baseline_name = baseline_result
    hpo_model, hpo_params, hpo_metrics, hpo_run_id = hpo_result

    improvement = (
        (hpo_metrics["pr_auc"] - baseline_metrics["pr_auc"])
        / baseline_metrics["pr_auc"]
        * 100
    )
    logger.info(f"Baseline PR-AUC : {baseline_metrics['pr_auc']:.4f}")
    logger.info(f"HPO      PR-AUC : {hpo_metrics['pr_auc']:.4f}  ({improvement:+.2f}%)")

    if hpo_metrics["pr_auc"] >= baseline_metrics["pr_auc"]:
        winner_model, winner_run_id, winner_name = (
            hpo_model,
            hpo_run_id,
            "HPO_optimizado",
        )
        winner_metrics = hpo_metrics
    else:
        winner_model, winner_run_id, winner_name = (
            baseline_model,
            baseline_run_id,
            baseline_name,
        )
        winner_metrics = baseline_metrics
        logger.warning("Baseline supera al HPO — desplegando baseline")

    joblib.dump(winner_model, "models/best_model.pkl")
    register_best_model(
        winner_run_id,
        cfg["mlflow"]["model_registry_name"],
        cfg["mlflow"]["tracking_uri"],
    )
    logger.info(f"✅ Registrado: {winner_name}")
    return winner_name, winner_metrics


# ── Flow ───────────────────────────────────────────────────────────────────────


@flow(name="fraud-detection-full-pipeline", log_prints=True)
def full_pipeline(
    run_hpo: bool = True, hpo_model: str = "xgboost", hpo_trials: int = 30
):
    os.chdir(Path(__file__).resolve().parents[2])
    cfg = load_config()

    import mlflow

    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])

    df = task_load_data(cfg)
    X, y = task_feature_engineering(df, cfg)
    X_train, X_test, y_train, y_test = task_split_scale(X, y, cfg)
    baseline_result = task_baseline(X_train, y_train, X_test, y_test, cfg)

    if run_hpo:
        hpo_result = task_hpo(
            X_train, y_train, X_test, y_test, cfg, hpo_model, hpo_trials
        )
        winner_name, winner_metrics = task_compare_register(
            baseline_result, hpo_result, cfg
        )
    else:
        baseline_model, baseline_metrics, baseline_run_id, baseline_name = (
            baseline_result
        )
        joblib.dump(baseline_model, "models/best_model.pkl")
        register_best_model(
            baseline_run_id,
            cfg["mlflow"]["model_registry_name"],
            cfg["mlflow"]["tracking_uri"],
        )
        winner_name, winner_metrics = baseline_name, baseline_metrics

    print(f"\n{'='*55}")
    print(f"✅ Pipeline completo — Modelo: {winner_name}")
    print(f"   PR-AUC  : {winner_metrics['pr_auc']:.4f}")
    print(f"   Recall  : {winner_metrics['recall']:.4f}")
    print(f"   F1      : {winner_metrics['f1']:.4f}")
    print(
        f"   Fraudes : {winner_metrics['fraud_detected']}/{winner_metrics['total_fraud']}"
    )
    print(f"{'='*55}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hpo", action="store_true", default=True)
    parser.add_argument("--no-hpo", dest="hpo", action="store_false")
    parser.add_argument(
        "--model", default="xgboost", choices=["xgboost", "random_forest", "both"]
    )
    parser.add_argument("--trials", type=int, default=30)
    args = parser.parse_args()
    full_pipeline(run_hpo=args.hpo, hpo_model=args.model, hpo_trials=args.trials)
