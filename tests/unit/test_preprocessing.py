"""Unit tests for preprocessing module."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.preprocessing import (
    _haversine,
    encode_categoricals,
    engineer_features,
    split_data,
)


@pytest.fixture
def sample_df():
    return pd.DataFrame(
        {
            "trans_date_trans_time": ["2019-06-15 14:30:00", "2020-01-01 08:00:00"],
            "merchant": ["ShopA", "ShopB"],
            "category": ["grocery_pos", "entertainment"],
            "amt": [50.0, 200.0],
            "city": ["Medellín", "Bogotá"],
            "state": ["CO", "CO"],
            "lat": [6.2442, 4.7110],
            "long": [-75.5812, -74.0721],
            "city_pop": [3900000, 8000000],
            "job": ["Engineer", "Doctor"],
            "dob": ["1985-03-10", "1990-07-22"],
            "trans_num": ["abc123", "def456"],
            "merch_lat": [6.2500, 4.7200],
            "merch_long": [-75.5900, -74.0800],
            "is_fraud": [0, 1],
        }
    )


def test_engineer_features_creates_columns(sample_df):
    result = engineer_features(sample_df)
    for col in ["hour", "day_of_week", "age", "distance_km"]:
        assert col in result.columns, f"Missing column: {col}"


def test_hour_range(sample_df):
    result = engineer_features(sample_df)
    assert result["hour"].between(0, 23).all()


def test_day_of_week_range(sample_df):
    result = engineer_features(sample_df)
    assert result["day_of_week"].between(0, 6).all()


def test_age_positive(sample_df):
    result = engineer_features(sample_df)
    assert (result["age"] > 0).all()


def test_distance_non_negative(sample_df):
    result = engineer_features(sample_df)
    assert (result["distance_km"] >= 0).all()


def test_haversine_same_point():
    dist = _haversine(
        pd.Series([6.2442]),
        pd.Series([-75.5812]),
        pd.Series([6.2442]),
        pd.Series([-75.5812]),
    )
    assert float(dist.iloc[0]) == pytest.approx(0.0, abs=1e-6)


def test_haversine_known_distance():
    # Medellín to Bogotá ≈ 240 km
    dist = _haversine(
        pd.Series([6.2442]),
        pd.Series([-75.5812]),
        pd.Series([4.7110]),
        pd.Series([-74.0721]),
    )
    assert 220 < float(dist.iloc[0]) < 260


def test_encode_categoricals(sample_df):
    encoded = encode_categoricals(sample_df, ["category", "state"])
    assert encoded["category"].dtype in [int, np.int64, np.int32]
    assert encoded["state"].dtype in [int, np.int64, np.int32]


def test_split_data_stratified():
    X = pd.DataFrame({"a": range(100)})
    y = pd.Series([0] * 95 + [1] * 5)
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2, random_state=42)
    assert len(X_train) + len(X_test) == 100
    # Fraud rate should be preserved approximately
    assert abs(y_test.mean() - y.mean()) < 0.02


def test_split_data_sizes():
    X = pd.DataFrame({"a": range(1000)})
    y = pd.Series([0] * 950 + [1] * 50)
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2)
    assert len(X_test) == pytest.approx(200, abs=5)
