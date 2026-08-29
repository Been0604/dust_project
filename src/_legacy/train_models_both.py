# src/train_models_both.py

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)


def print_metrics(tag, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"[{tag}] MAE={mae:.3f}, RMSE={rmse:.3f}, R2={r2:.3f}")


def train_for_target(pollutant: str):
    feats_path = DATA_DIR / f"features_full_{pollutant}.parquet"
    if not feats_path.exists():
        raise FileNotFoundError(feats_path)

    df = pd.read_parquet(feats_path)
    df["ts_kst"] = pd.to_datetime(df["ts_kst"])

    target_col = f"target_{pollutant}"

    drop_cols = {"ts_kst", "station_id", "target_pm10", "target_pm25"}
    feature_cols = [c for c in df.columns if c not in drop_cols]

    # 연도 분할: 2021–22 train, 2023 val
    train_mask = df["year"].isin([2021, 2022])
    val_mask = df["year"] == 2023

    train = df[train_mask].copy()
    val = df[val_mask].copy()

    X_train = train[feature_cols]
    y_train = train[target_col]
    X_val = val[feature_cols]
    y_val = val[target_col]

    print(f"\n=== {pollutant.upper()} RF ===")
    rf = RandomForestRegressor(
        n_estimators=300,
        n_jobs=-1,
        random_state=42,
    )
    rf.fit(X_train, y_train)
    print_metrics(f"{pollutant}_RF_train", y_train, rf.predict(X_train))
    print_metrics(f"{pollutant}_RF_val", y_val, rf.predict(X_val))

    rf_path = MODELS_DIR / f"random_forest_{pollutant}.joblib"
    joblib.dump(rf, rf_path)
    print("saved:", rf_path)

    print(f"\n=== {pollutant.upper()} XGB ===")
    xgb = XGBRegressor(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
    )
    xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    print_metrics(f"{pollutant}_XGB_train", y_train, xgb.predict(X_train))
    print_metrics(f"{pollutant}_XGB_val", y_val, xgb.predict(X_val))

    xgb_path = MODELS_DIR / f"xgb_{pollutant}.joblib"
    joblib.dump(xgb, xgb_path)
    print("saved:", xgb_path)


def main():
    train_for_target("pm10")
    train_for_target("pm25")


if __name__ == "__main__":
    main()
