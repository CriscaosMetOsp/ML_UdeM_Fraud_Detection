"""
Optimización de hiperparámetros con Optuna + MLflow.

Estrategia:
- Optuna busca los mejores hiperparámetros mediante pruning (MedianPruner).
- Cada trial queda registrado como un run en MLflow.
- Al terminar, el mejor modelo se registra en el Model Registry.
- Métrica objetivo: PR-AUC (robusta ante el desbalanceo severo del dataset).

Uso:
    python src/models/hpo.py --model xgboost --trials 30
    python src/models/hpo.py --model random_forest --trials 20
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import optuna
import yaml
from optuna.integration.mlflow import MLflowCallback
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data.preprocessing import (
    build_feature_matrix,
    load_data,
    scale_features,
    split_data,
)
from src.models.train import compute_metrics, run_experiment

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)


def load_config(path="configs/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ── Search spaces ─────────────────────────────────────────────────────────────

def _suggest_xgboost_params(trial: optuna.Trial) -> dict:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 600, step=50),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 0.0, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        "scale_pos_weight": trial.suggest_int("scale_pos_weight", 50, 300),
        "random_state": 42,
        "eval_metric": "aucpr",
    }


def _suggest_rf_params(trial: optuna.Trial) -> dict:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
        "max_depth": trial.suggest_int("max_depth", 5, 30),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.3, 0.5]),
        "class_weight": trial.suggest_categorical("class_weight", ["balanced", "balanced_subsample"]),
        "random_state": 42,
        "n_jobs": -1,
    }


# ── Objective functions ───────────────────────────────────────────────────────

def _make_objective(model_type: str, X_train, y_train):
    """
    Retorna una función objective de Optuna que usa CV para evaluar PR-AUC.
    El uso de CV (en lugar de un único split) da una estimación más robusta,
    especialmente importante con datos muy desbalanceados.
    """
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    def objective(trial: optuna.Trial) -> float:
        if model_type == "xgboost":
            params = _suggest_xgboost_params(trial)
            model = XGBClassifier(**params, verbosity=0)
        else:
            params = _suggest_rf_params(trial)
            model = RandomForestClassifier(**params)

        # PR-AUC via cross_val_score (average_precision)
        scores = cross_val_score(
            model, X_train, y_train,
            cv=cv,
            scoring="average_precision",
            n_jobs=-1,
        )
        mean_pr_auc = scores.mean()

        # Optuna pruning: elimina trials malos temprano
        trial.report(mean_pr_auc, step=0)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        return mean_pr_auc

    return objective


# ── Study runner ──────────────────────────────────────────────────────────────

def run_hpo(
    model_type: str,
    X_train,
    y_train,
    X_test,
    y_test,
    cfg: dict,
    n_trials: int = 30,
) -> tuple:
    """
    Ejecuta el estudio de Optuna con callbacks de MLflow.

    Returns
    -------
    best_model : modelo entrenado con los mejores params
    best_params : dict de hiperparámetros óptimos
    best_metrics : métricas en test set
    best_run_id : run ID de MLflow del mejor trial
    """
    experiment_name = f"{cfg['mlflow']['experiment_name']}-hpo-{model_type}"
    mlflow.set_experiment(experiment_name)

    # MLflow callback: cada trial → un run en MLflow
    mlflow_cb = MLflowCallback(
        tracking_uri=cfg["mlflow"]["tracking_uri"],
        metric_name="pr_auc_cv",
        create_experiment=False,
        mlflow_kwargs={"experiment_id": mlflow.get_experiment_by_name(experiment_name).experiment_id}
        if mlflow.get_experiment_by_name(experiment_name)
        else {},
    )

    study = optuna.create_study(
        study_name=f"fraud-{model_type}-hpo",
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
        sampler=optuna.samplers.TPESampler(seed=42),
    )

    objective = _make_objective(model_type, X_train, y_train)

    logger.info(
        f"Iniciando HPO para {model_type.upper()} | "
        f"{n_trials} trials | métrica objetivo: PR-AUC (CV)"
    )

    study.optimize(
        objective,
        n_trials=n_trials,
        callbacks=[mlflow_cb],
        show_progress_bar=True,
        catch=(Exception,),
    )

    # ── Resultado del estudio ──
    best_trial = study.best_trial
    logger.info(
        f"\n{'='*60}\n"
        f"Mejor trial #{best_trial.number}\n"
        f"  PR-AUC CV : {best_trial.value:.4f}\n"
        f"  Params    : {best_trial.params}\n"
        f"{'='*60}"
    )

    # Registrar importancia de hiperparámetros
    try:
        importance = optuna.importance.get_param_importances(study)
        logger.info(f"Importancia de hiperparámetros:\n" +
                    "\n".join(f"  {k}: {v:.3f}" for k, v in importance.items()))
    except Exception:
        pass

    # ── Entrenar modelo final con mejores params en todos los datos de train ──
    best_params = best_trial.params
    if model_type == "xgboost":
        best_params.update({"random_state": 42, "eval_metric": "aucpr"})
        best_model = XGBClassifier(**best_params)
    else:
        best_params.update({"random_state": 42, "n_jobs": -1})
        best_model = RandomForestClassifier(**best_params)

    best_model.fit(X_train, y_train)

    # ── Evaluar en test y registrar run final ──
    best_metrics, best_run_id = run_experiment(
        f"{model_type.upper()}_HPO_best",
        best_model,
        best_params,
        X_test,
        y_test,
        experiment_name,
    )

    logger.info(
        f"Modelo final en test set:\n"
        f"  PR-AUC : {best_metrics['pr_auc']:.4f}\n"
        f"  Recall : {best_metrics['recall']:.4f}\n"
        f"  F1     : {best_metrics['f1']:.4f}\n"
        f"  Fraudes detectados: {best_metrics['fraud_detected']}/{best_metrics['total_fraud']}"
    )

    return best_model, best_params, best_metrics, best_run_id


# ── CLI entry point ───────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="HPO con Optuna para el modelo de detección de fraude"
    )
    parser.add_argument(
        "--model",
        choices=["xgboost", "random_forest", "both"],
        default="xgboost",
        help="Modelo a optimizar (default: xgboost)",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=30,
        help="Número de trials de Optuna (default: 30)",
    )
    parser.add_argument(
        "--config",
        default="configs/config.yaml",
        help="Ruta al archivo de configuración",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.chdir(Path(__file__).resolve().parents[2])
    cfg = load_config(args.config)
    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])

    # Preparar datos
    df = load_data(cfg["data"]["raw_path"])
    X, y = build_feature_matrix(df, cfg["features"])
    X_train, X_test, y_train, y_test = split_data(
        X, y, cfg["data"]["test_size"], cfg["data"]["random_state"]
    )
    X_train_sc, X_test_sc, scaler = scale_features(
        X_train, X_test, cfg["features"]["numerical_cols"]
    )

    models_to_run = (
        ["xgboost", "random_forest"] if args.model == "both" else [args.model]
    )

    all_results = {}
    for model_type in models_to_run:
        best_model, best_params, best_metrics, best_run_id = run_hpo(
            model_type=model_type,
            X_train=X_train_sc,
            y_train=y_train,
            X_test=X_test_sc,
            y_test=y_test,
            cfg=cfg,
            n_trials=args.trials,
        )
        all_results[model_type] = {
            "model": best_model,
            "params": best_params,
            "metrics": best_metrics,
            "run_id": best_run_id,
        }

    # Si se corrieron ambos, seleccionar el mejor global
    if len(all_results) > 1:
        best_name = max(all_results, key=lambda k: all_results[k]["metrics"]["pr_auc"])
        logger.info(f"\n🏆 Ganador global: {best_name.upper()}")
        best_run_id = all_results[best_name]["run_id"]
        best_model = all_results[best_name]["model"]
    else:
        best_name = models_to_run[0]
        best_run_id = all_results[best_name]["run_id"]
        best_model = all_results[best_name]["model"]

    # Guardar modelo optimizado
    Path("models").mkdir(exist_ok=True)
    joblib.dump(best_model, "models/best_model.pkl")
    joblib.dump(scaler, "models/scaler.pkl")
    logger.info("Modelo optimizado guardado en models/best_model.pkl")

    # Registrar en MLflow Model Registry
    from src.models.train import register_best_model
    register_best_model(
        best_run_id,
        cfg["mlflow"]["model_registry_name"],
        cfg["mlflow"]["tracking_uri"],
    )

    print(f"\n✅ HPO completo. Mejor modelo: {best_name.upper()}")
    print(f"   PR-AUC test : {all_results[best_name]['metrics']['pr_auc']:.4f}")
    print(f"   Recall test : {all_results[best_name]['metrics']['recall']:.4f}")
    print(f"   F1 test     : {all_results[best_name]['metrics']['f1']:.4f}")


if __name__ == "__main__":
    main()
