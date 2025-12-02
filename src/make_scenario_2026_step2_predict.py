import os
import glob
import pandas as pd
from joblib import load


SCENARIOS = ["A", "B", "C"]

# RF 모델 파일명은 이미 쓰던 걸 그대로 사용
RF_PM10_PATH = "models/random_forest_pm10.joblib"
RF_PM25_PATH = "models/random_forest_pm25.joblib"


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def find_xgb_model(pollutant: str):
    """
    pollutant: 'pm10' 또는 'pm25'
    models 폴더에서 'xgb' 또는 'xgboost' + pollutant 가 들어간 .joblib 파일을 자동 탐색.
    없으면 None 리턴 (그럼 RF만 사용).
    """
    pattern = os.path.join("models", f"*{pollutant}*.joblib")
    candidates = glob.glob(pattern)

    candidates = [
        c for c in candidates
        if ("xgb" in os.path.basename(c).lower())
        or ("xgboost" in os.path.basename(c).lower())
    ]

    if not candidates:
        print(f"[WARN] No XGB model found for {pollutant}. RF only for this pollutant.")
        return None

    if len(candidates) > 1:
        print(f"[WARN] Multiple XGB candidates for {pollutant}:")
        for c in candidates:
            print(f"       - {c}")
        print(f"       Using first one: {candidates[0]}")

    return candidates[0]


def load_models():
    print(f"[INFO] Loading RF models...")
    rf_pm10 = load(RF_PM10_PATH)
    rf_pm25 = load(RF_PM25_PATH)

    # RF에서 feature_names_in_ 가져와서 공식 피처 세트로 사용
    feature_cols_pm10 = list(rf_pm10.feature_names_in_)
    feature_cols_pm25 = list(rf_pm25.feature_names_in_)

    print(f"  - PM10 features: {len(feature_cols_pm10)} columns")
    print(f"  - PM25 features: {len(feature_cols_pm25)} columns")

    # XGB는 있으면 로드, 없으면 None
    print(f"[INFO] Trying to load XGB models (optional)...")

    xgb_pm10_path = find_xgb_model("pm10")
    xgb_pm25_path = find_xgb_model("pm25")

    xgb_pm10 = load(xgb_pm10_path) if xgb_pm10_path is not None else None
    xgb_pm25 = load(xgb_pm25_path) if xgb_pm25_path is not None else None

    if xgb_pm10 is not None:
        print(f"  - Loaded XGB PM10 model from: {xgb_pm10_path}")
    if xgb_pm25 is not None:
        print(f"  - Loaded XGB PM25 model from: {xgb_pm25_path}")

    return rf_pm10, rf_pm25, xgb_pm10, xgb_pm25, feature_cols_pm10, feature_cols_pm25


def make_predictions_for_pollutant(
    pollutant: str,
    model_rf,
    model_xgb,
    feature_cols,
    scenario_dir: str,
    output_path: str,
) -> None:
    """
    pollutant: 'pm10' 또는 'pm25'
    model_xgb가 None이면 RF만 예측.
    """
    all_list = []

    for scenario_id in SCENARIOS:
        feat_path = os.path.join(
            scenario_dir,
            f"scenario_2026_features_{pollutant}_{scenario_id}.parquet",
        )
        print(f"[INFO] Loading features for {pollutant.upper()}, Scenario {scenario_id}: {feat_path}")
        df = pd.read_parquet(feat_path)

        # datetime 캐스팅
        for col in ["ts_kst", "ts_kst_original", "ts_kst_2026"]:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col])

        # 모델 입력 X
        X = df[feature_cols]
        print(f"  - Shape X: {X.shape}")

        # RF 예측
        print("  - Predicting RF...")
        pred_rf = model_rf.predict(X)

        # 메타 컬럼
        meta_cols = [
            col for col in [
                "ts_kst_original",
                "ts_kst_2026",
                "ts_kst",
                "station_id",
                "area",
                "scenario_id",
                "scenario_base_year",
            ]
            if col in df.columns
        ]
        out = df[meta_cols].copy()

        if pollutant == "pm10":
            out["pred_pm10_rf"] = pred_rf
        else:
            out["pred_pm25_rf"] = pred_rf

        # XGB 있으면 추가 예측
        if model_xgb is not None:
            print("  - Predicting XGB...")
            pred_xgb = model_xgb.predict(X)
            if pollutant == "pm10":
                out["pred_pm10_xgb"] = pred_xgb
            else:
                out["pred_pm25_xgb"] = pred_xgb
        else:
            print("  - XGB model not available, skipping XGB predictions.")

        all_list.append(out)

    result = pd.concat(all_list, ignore_index=True)
    print(f"[INFO] Final shape for {pollutant.upper()} predictions: {result.shape}")
    result.to_parquet(output_path, index=False)
    print(f"[INFO] Saved predictions to: {output_path}")


def main():
    scenario_dir = "data/scenario"
    output_pm10 = os.path.join(scenario_dir, "scenario_2026_preds_pm10_rf_xgb.parquet")
    output_pm25 = os.path.join(scenario_dir, "scenario_2026_preds_pm25_rf_xgb.parquet")

    ensure_dir(scenario_dir)

    (
        rf_pm10,
        rf_pm25,
        xgb_pm10,
        xgb_pm25,
        feature_cols_pm10,
        feature_cols_pm25,
    ) = load_models()

    # PM10
    make_predictions_for_pollutant(
        pollutant="pm10",
        model_rf=rf_pm10,
        model_xgb=xgb_pm10,
        feature_cols=feature_cols_pm10,
        scenario_dir=scenario_dir,
        output_path=output_pm10,
    )

    # PM25
    make_predictions_for_pollutant(
        pollutant="pm25",
        model_rf=rf_pm25,
        model_xgb=xgb_pm25,
        feature_cols=feature_cols_pm25,
        scenario_dir=scenario_dir,
        output_path=output_pm25,
    )


if __name__ == "__main__":
    main()
