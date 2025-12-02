import os
import numpy as np
import pandas as pd


SCENARIOS = ["A", "B", "C"]


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_preds_and_mapping():
    """2026 시나리오 예측 + station↔행정동 매핑 로드"""
    pm10_path = "data/scenario/scenario_2026_preds_pm10_rf_xgb.parquet"
    pm25_path = "data/scenario/scenario_2026_preds_pm25_rf_xgb.parquet"
    mapping_path = "data/stations_emd_mapping.csv"

    print(f"[INFO] Loading PM10 preds: {pm10_path}")
    pm10 = pd.read_parquet(pm10_path)

    print(f"[INFO] Loading PM25 preds: {pm25_path}")
    pm25 = pd.read_parquet(pm25_path)

    print(f"[INFO] Loading station↔EMD mapping: {mapping_path}")
    # ★ 인코딩 자동 탐지: UTF-8 시도 → 안 되면 cp949
    try:
        map_df = pd.read_csv(mapping_path, encoding="utf-8")
    except UnicodeDecodeError:
        print("[WARN] UTF-8 decode failed, trying cp949...")
        map_df = pd.read_csv(mapping_path, encoding="cp949")

    # station_id 타입 통일
    pm10["station_id"] = pm10["station_id"].astype(str)
    pm25["station_id"] = pm25["station_id"].astype(str)
    map_df["station_id"] = map_df["station_id"].astype(str)

    if "EMD_CD" not in map_df.columns:
        raise KeyError("stations_emd_mapping.csv 에 'EMD_CD' 컬럼이 없습니다.")

    # 동/서부 컬럼 이름 결정
    area_col = None
    if "area_ew" in map_df.columns:
        area_col = "area_ew"
    elif "area" in map_df.columns:
        area_col = "area"
    else:
        if "area_ew" in pm10.columns:
            area_col = "area_ew"
        elif "area" in pm10.columns:
            area_col = "area"

    if area_col is None:
        print("[WARN] 동/서부 구분 컬럼(area_ew/area)을 찾을 수 없습니다. "
              "area_ew 요약은 건너뜁니다.")

    return pm10, pm25, map_df, area_col


def merge_with_mapping(preds: pd.DataFrame, map_df: pd.DataFrame) -> pd.DataFrame:
    """예측값과 station↔행정동 매핑 병합"""
    df = preds.merge(map_df, on="station_id", how="left", suffixes=("", "_map"))
    if "area_map" in df.columns and "area" not in df.columns:
        df.rename(columns={"area_map": "area"}, inplace=True)
    return df


def add_threshold_flags_pm10(df: pd.DataFrame) -> pd.DataFrame:
    """PM10 기준 초과 플래그 추가 (RF/XGB 둘 다)"""
    df = df.copy()
    for model_col in ["pred_pm10_rf", "pred_pm10_xgb"]:
        if model_col not in df.columns:
            continue
        col_over50 = f"{model_col}_over50"
        col_over70 = f"{model_col}_over70"
        df[col_over50] = (df[model_col] > 50).astype(float)
        df[col_over70] = (df[model_col] > 70).astype(float)
    return df


def add_threshold_flags_pm25(df: pd.DataFrame) -> pd.DataFrame:
    """PM25 기준 초과 플래그 추가 (RF/XGB 둘 다)"""
    df = df.copy()
    for model_col in ["pred_pm25_rf", "pred_pm25_xgb"]:
        if model_col not in df.columns:
            continue
        col_over15 = f"{model_col}_over15"
        col_over25 = f"{model_col}_over25"
        df[col_over15] = (df[model_col] > 15).astype(float)
        df[col_over25] = (df[model_col] > 25).astype(float)
    return df


def summarize_admin_pm10(df: pd.DataFrame) -> pd.DataFrame:
    """행정동 단위 PM10 요약 (이름 컬럼은 여기서 안 만든다)"""
    df = add_threshold_flags_pm10(df)

    # 이름 없이 코드만 기준으로 집계
    group_cols = ["scenario_id", "scenario_base_year", "EMD_CD"]

    g = df.groupby(group_cols, dropna=False)

    out = g.agg(
        mean_pm10_rf=("pred_pm10_rf", "mean"),
        mean_pm10_xgb=("pred_pm10_xgb", "mean"),
        p95_pm10_rf=("pred_pm10_rf", lambda x: np.percentile(x, 95)),
        p95_pm10_xgb=("pred_pm10_xgb", lambda x: np.percentile(x, 95)),
        frac_over50_rf=("pred_pm10_rf_over50", "mean"),
        frac_over50_xgb=("pred_pm10_xgb_over50", "mean"),
        frac_over70_rf=("pred_pm10_rf_over70", "mean"),
        frac_over70_xgb=("pred_pm10_xgb_over70", "mean"),
        n_hours=("pred_pm10_rf", "count"),
    ).reset_index()

    return out


def summarize_admin_pm25(df: pd.DataFrame) -> pd.DataFrame:
    """행정동 단위 PM25 요약 (이름 컬럼은 여기서 안 만든다)"""
    df = add_threshold_flags_pm25(df)

    group_cols = ["scenario_id", "scenario_base_year", "EMD_CD"]

    g = df.groupby(group_cols, dropna=False)

    out = g.agg(
        mean_pm25_rf=("pred_pm25_rf", "mean"),
        mean_pm25_xgb=("pred_pm25_xgb", "mean"),
        p95_pm25_rf=("pred_pm25_rf", lambda x: np.percentile(x, 95)),
        p95_pm25_xgb=("pred_pm25_xgb", lambda x: np.percentile(x, 95)),
        frac_over15_rf=("pred_pm25_rf_over15", "mean"),
        frac_over15_xgb=("pred_pm25_xgb_over15", "mean"),
        frac_over25_rf=("pred_pm25_rf_over25", "mean"),
        frac_over25_xgb=("pred_pm25_xgb_over25", "mean"),
        n_hours=("pred_pm25_rf", "count"),
    ).reset_index()

    return out


def summarize_area_pm(df: pd.DataFrame, pollutant: str, area_col: str) -> pd.DataFrame:
    """동/서부 단위 요약 (PM10/25 공용)"""
    if area_col not in df.columns:
        raise KeyError(f"'{area_col}' 컬럼이 없습니다. area_ew/area 매핑을 확인하세요.")

    group_cols = ["scenario_id", "scenario_base_year", area_col]

    g = df.groupby(group_cols, dropna=False)

    if pollutant == "pm10":
        out = g.agg(
            mean_pm10_rf=("pred_pm10_rf", "mean"),
            mean_pm10_xgb=("pred_pm10_xgb", "mean"),
            n_hours=("pred_pm10_rf", "count"),
        ).reset_index()
    else:
        out = g.agg(
            mean_pm25_rf=("pred_pm25_rf", "mean"),
            mean_pm25_xgb=("pred_pm25_xgb", "mean"),
            n_hours=("pred_pm25_rf", "count"),
        ).reset_index()

    out.rename(columns={area_col: "area_ew"}, inplace=True)
    return out


def main():
    scenario_dir = "data/scenario"
    ensure_dir(scenario_dir)

    pm10, pm25, map_df, area_col = load_preds_and_mapping()

    # 행정동 매핑 병합
    pm10_m = merge_with_mapping(pm10, map_df)
    pm25_m = merge_with_mapping(pm25, map_df)

    print("[INFO] Summarizing PM10 by EMD...")
    admin_pm10 = summarize_admin_pm10(pm10_m)
    admin_pm10_path = os.path.join(scenario_dir, "bi_admin_pm_stats_2026_pm10.parquet")
    admin_pm10.to_parquet(admin_pm10_path, index=False)
    print(f"  - Saved: {admin_pm10_path} (shape={admin_pm10.shape})")

    print("[INFO] Summarizing PM25 by EMD...")
    admin_pm25 = summarize_admin_pm25(pm25_m)
    admin_pm25_path = os.path.join(scenario_dir, "bi_admin_pm_stats_2026_pm25.parquet")
    admin_pm25.to_parquet(admin_pm25_path, index=False)
    print(f"  - Saved: {admin_pm25_path} (shape={admin_pm25.shape})")

    # 동/서부 요약
    if area_col is not None:
        print(f"[INFO] Summarizing PM10 by {area_col} (동/서부)...")
        area_pm10 = summarize_area_pm(pm10_m, pollutant="pm10", area_col=area_col)
        area_pm10_path = os.path.join(scenario_dir, "bi_area_ew_stats_2026_pm10.parquet")
        area_pm10.to_parquet(area_pm10_path, index=False)
        print(f"  - Saved: {area_pm10_path} (shape={area_pm10.shape})")

        print(f"[INFO] Summarizing PM25 by {area_col} (동/서부)...")
        area_pm25 = summarize_area_pm(pm25_m, pollutant="pm25", area_col=area_col)
        area_pm25_path = os.path.join(scenario_dir, "bi_area_ew_stats_2026_pm25.parquet")
        area_pm25.to_parquet(area_pm25_path, index=False)
        print(f"  - Saved: {area_pm25_path} (shape={area_pm25.shape})")
    else:
        print("[WARN] area_ew/area 컬럼이 없어 동/서부 요약은 생략했습니다.")


if __name__ == "__main__":
    main()
