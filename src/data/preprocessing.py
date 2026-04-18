"""
Data loading and preprocessing module.
"""
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_data(path: str) -> pd.DataFrame:
    """Load raw CSV data."""
    logger.info(f"Loading data from {path}")
    df = pd.read_csv(path)
    logger.info(f"Loaded {len(df):,} rows, {df.shape[1]} columns")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create new features from raw columns.

    Features created:
    - hour: hour of the transaction
    - day_of_week: day of week (0=Monday)
    - age: age of the cardholder in years
    - distance_km: haversine distance between customer and merchant
    """
    df = df.copy()

    # Datetime features
    df["trans_dt"] = pd.to_datetime(df["trans_date_trans_time"])
    df["hour"] = df["trans_dt"].dt.hour
    df["day_of_week"] = df["trans_dt"].dt.dayofweek

    # Age in years at transaction time
    df["dob_dt"] = pd.to_datetime(df["dob"])
    df["age"] = ((df["trans_dt"] - df["dob_dt"]).dt.days / 365.25).astype(int)

    # Haversine distance between customer location and merchant location
    df["distance_km"] = _haversine(
        df["lat"], df["long"], df["merch_lat"], df["merch_long"]
    )

    return df


def _haversine(lat1, lon1, lat2, lon2) -> pd.Series:
    """Compute haversine distance in km between two lat/lon pairs."""
    R = 6371.0
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def encode_categoricals(df: pd.DataFrame, categorical_cols: list) -> pd.DataFrame:
    """Label-encode categorical columns."""
    df = df.copy()
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
    return df


def build_feature_matrix(df: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, pd.Series]:
    """
    Apply feature engineering and return X, y ready for modelling.

    Parameters
    ----------
    df : raw dataframe
    config : dict with keys 'target', 'drop_cols', 'categorical_cols'

    Returns
    -------
    X : feature dataframe
    y : target series
    """
    logger.info("Engineering features...")
    df = engineer_features(df)
    df = encode_categoricals(df, config["categorical_cols"])

    drop = config["drop_cols"] + ["trans_dt", "dob_dt"]
    drop_existing = [c for c in drop if c in df.columns]
    df = df.drop(columns=drop_existing)

    y = df[config["target"]]
    X = df.drop(columns=[config["target"]])

    logger.info(f"Feature matrix: {X.shape} | Fraud rate: {y.mean():.4%}")
    return X, y


def split_data(X, y, test_size=0.2, random_state=42):
    """Stratified train/test split preserving class balance."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    logger.info(
        f"Train: {len(X_train):,} | Test: {len(X_test):,} | "
        f"Fraud train: {y_train.sum()} | Fraud test: {y_test.sum()}"
    )
    return X_train, X_test, y_train, y_test


def scale_features(X_train, X_test, numerical_cols: list):
    """Fit StandardScaler on train, apply to both splits."""
    scaler = StandardScaler()
    X_train = X_train.copy()
    X_test = X_test.copy()
    cols = [c for c in numerical_cols if c in X_train.columns]
    X_train[cols] = scaler.fit_transform(X_train[cols])
    X_test[cols] = scaler.transform(X_test[cols])
    return X_train, X_test, scaler
