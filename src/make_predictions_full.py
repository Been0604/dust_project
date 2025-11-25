# src/make_predictions_full.py

from pathlib import Path
import pandas as pd
from joblib import load

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"


def load_features(pollutant: str) -> pd.DataFrame:
    path = DATA_DIR / f"features_full_{pollutant}.parquet"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_parquet(path)
    df["ts_kst"] = pd.to_datetime(df["ts_kst"])
    return df


def main():
    # PM10
    feats10 = load_features("pm10")
    drop_cols = {"ts_kst", "station_id", "target_pm10", "target_pm25"}
    feature_cols10 = [c for c in feats10.columns if c not in drop_cols]

    rf10 = load(MODELS_DIR / "random_forest_pm10.joblib")
    xgb10 = load(MODELS_DIR / "xgb_pm10.joblib")

    feats10["y_pred_rf"] = rf10.predict(feats10[feature_cols10])
    feats10["y_pred_xgb"] = xgb10.predict(feats10[feature_cols10])
    feats10["residual_rf"] = feats10["y_pred_rf"] - feats10["target_pm10"]
    feats10["residual_xgb"] = feats10["y_pred_xgb"] - feats10["target_pm10"]

    pred10 = feats10[[
        "ts_kst",
        "station_id",
        "target_pm10",
        "y_pred_rf",
        "y_pred_xgb",
        "residual_rf",
        "residual_xgb",
    ]].copy()

    # PM25
    feats25 = load_features("pm25")
    feature_cols25 = [c for c in feats25.columns if c not in drop_cols]

    rf25 = load(MODELS_DIR / "random_forest_pm25.joblib")
    xgb25 = load(MODELS_DIR / "xgb_pm25.joblib")

    feats25["y_pred_rf_pm25"] = rf25.predict(feats25[feature_cols25])
    feats25["y_pred_xgb_pm25"] = xgb25.predict(feats25[feature_cols25])
    feats25["residual_rf_pm25"] = feats25["y_pred_rf_pm25"] - feats25["target_pm25"]
    feats25["residual_xgb_pm25"] = feats25["y_pred_xgb_pm25"] - feats25["target_pm25"]

    pred25 = feats25[[
        "ts_kst",
        "station_id",
        "target_pm25",
        "y_pred_rf_pm25",
        "y_pred_xgb_pm25",
        "residual_rf_pm25",
        "residual_xgb_pm25",
    ]].copy()

    preds = pred10.merge(pred25, on=["ts_kst", "station_id"], how="outer")

    out_path = DATA_DIR / "predictions_full_rf_xgb.parquet"
    preds.to_parquet(out_path, index=False)
    print("saved:", out_path, "shape=", preds.shape)


if __name__ == "__main__":
    main()
