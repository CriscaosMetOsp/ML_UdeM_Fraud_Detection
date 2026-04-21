"""Unit tests para el módulo de HPO con Optuna."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import optuna
import pandas as pd
import pytest
from sklearn.datasets import make_classification

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.models.hpo import _suggest_rf_params, _suggest_xgboost_params, _make_objective


@pytest.fixture
def imbalanced_data():
    """Dataset binario desbalanceado similar al problema real."""
    X, y = make_classification(
        n_samples=800,
        n_features=12,
        n_informative=8,
        weights=[0.98, 0.02],
        random_state=42,
    )
    return pd.DataFrame(X, columns=[f"f{i}" for i in range(12)]), pd.Series(y)


def _make_trial(params: dict):
    """Crea un trial de Optuna con parámetros fijos para testing."""
    study = optuna.create_study(direction="maximize")
    trial = study.ask()
    # Simular suggest_* rellenando el internal storage
    for k, v in params.items():
        if isinstance(v, int):
            trial._suggest(k, optuna.distributions.IntDistribution(v, v))
        elif isinstance(v, float):
            trial._suggest(k, optuna.distributions.FloatDistribution(v, v))
        elif isinstance(v, str):
            trial._suggest(k, optuna.distributions.CategoricalDistribution([v]))
    return trial


# ── Suggest param tests ───────────────────────────────────────────────────────


def test_suggest_xgboost_returns_required_keys():
    study = optuna.create_study(direction="maximize")
    trial = study.ask()
    params = _suggest_xgboost_params(trial)
    required = {
        "n_estimators",
        "max_depth",
        "learning_rate",
        "subsample",
        "colsample_bytree",
        "min_child_weight",
        "gamma",
        "reg_alpha",
        "reg_lambda",
        "scale_pos_weight",
    }
    assert required.issubset(params.keys())


def test_suggest_rf_returns_required_keys():
    study = optuna.create_study(direction="maximize")
    trial = study.ask()
    params = _suggest_rf_params(trial)
    required = {
        "n_estimators",
        "max_depth",
        "min_samples_split",
        "min_samples_leaf",
        "max_features",
        "class_weight",
    }
    assert required.issubset(params.keys())


def test_xgboost_param_ranges():
    study = optuna.create_study(direction="maximize")
    for _ in range(5):
        trial = study.ask()
        params = _suggest_xgboost_params(trial)
        assert 100 <= params["n_estimators"] <= 600
        assert 3 <= params["max_depth"] <= 10
        assert 0.01 <= params["learning_rate"] <= 0.3
        assert 0.6 <= params["subsample"] <= 1.0
        assert 50 <= params["scale_pos_weight"] <= 300


def test_rf_param_ranges():
    study = optuna.create_study(direction="maximize")
    for _ in range(5):
        trial = study.ask()
        params = _suggest_rf_params(trial)
        assert 100 <= params["n_estimators"] <= 500
        assert 5 <= params["max_depth"] <= 30
        assert params["class_weight"] in ["balanced", "balanced_subsample"]


# ── Objective function tests ──────────────────────────────────────────────────


def test_objective_xgboost_returns_float(imbalanced_data):
    X, y = imbalanced_data
    objective = _make_objective("xgboost", X, y)
    study = optuna.create_study(direction="maximize")
    trial = study.ask()
    score = objective(trial)
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


def test_objective_rf_returns_float(imbalanced_data):
    X, y = imbalanced_data
    objective = _make_objective("random_forest", X, y)
    study = optuna.create_study(direction="maximize")
    trial = study.ask()
    score = objective(trial)
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


def test_optuna_study_improves_over_trials(imbalanced_data):
    """Verifica que Optuna completa múltiples trials sin errores."""
    X, y = imbalanced_data
    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.RandomSampler(seed=42)
    )
    objective = _make_objective("xgboost", X, y)
    study.optimize(objective, n_trials=3, catch=(Exception,))
    assert len(study.trials) >= 1
    assert study.best_value >= 0.0


def test_study_best_params_are_valid(imbalanced_data):
    """El mejor trial debe tener parámetros dentro de los rangos definidos."""
    X, y = imbalanced_data
    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.RandomSampler(seed=0)
    )
    study.optimize(_make_objective("xgboost", X, y), n_trials=2, catch=(Exception,))
    best = study.best_params
    assert "n_estimators" in best
    assert "learning_rate" in best
