"""Run a monitoring check using the trained model against the test set."""
import sys
import yaml
import joblib
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.data.preprocessing import build_feature_matrix, load_data, scale_features, split_data
from src.monitoring.monitor import run_monitoring_check

cfg = yaml.safe_load(open("configs/config.yaml"))
df = load_data(cfg["data"]["raw_path"])
X, y = build_feature_matrix(df, cfg["features"])
X_train, X_test, y_train, y_test = split_data(X, y, cfg["data"]["test_size"], cfg["data"]["random_state"])
X_train_sc, X_test_sc, scaler = scale_features(X_train, X_test, cfg["features"]["numerical_cols"])

model = joblib.load("models/best_model.pkl")

# Use original amt for business impact calc
_, df_test_idx = (
    X_train.index.tolist(),
    X_test.index.tolist(),
)
amounts = df.loc[X_test.index, "amt"]

report = run_monitoring_check(
    model, scaler,
    X_train_sc, y_train,
    X_test_sc, y_test,
    amounts,
    cfg["features"]["numerical_cols"]
)
