import os

import numpy as np
import pandas as pd
import geopandas as gpd


SCENARIO_PATH = "data/scenario/umd_pm_stats_2026_scenarios.gpkg"
OUTPUT_PATH = "data/scenario/umd_pm_stats_2026_scenarios_with_classes.gpkg"


def classify_pm10(values: pd.Series) -> pd.Series:
    """PM10 구간 코드: 1~5"""
    conds = [
        values <= 15,                  # 1: <= 15
        (values > 15) & (values <= 30),  # 2: 15–30
        (values > 30) & (values <= 50),  # 3: 30–50
        (values > 50) & (values <= 70),  # 4: 50–70
        values > 70,                   # 5: > 70
    ]
    choices = [1, 2, 3, 4, 5]
    return pd.Series(np.select(conds, choices, default=np.nan), index=values.index)


def classify_pm25(values: pd.Series) -> pd.Series:
    """PM2.5 구간 코드: 1~5"""
    conds = [
        values <= 5,                    # 1: <= 5
        (values > 5) & (values <= 10),  # 2: 5–10
        (values > 10) & (values <= 15), # 3: 10–15
        (values > 15) & (values <= 25), # 4: 15–25
        values > 25,                    # 5: > 25
    ]
    choices = [1, 2, 3, 4, 5]
    return pd.Series(np.select(conds, choices, default=np.nan), index=values.index)


def main():
    print(f"[INFO] Loading: {SCENARIO_PATH}")
    gdf = gpd.read_file(SCENARIO_PATH)

    # 필드 존재 여부 체크
    required_cols = [
        "mean_pm10_rf",
        "mean_pm10_xgb",
        "mean_pm25_rf",
        "mean_pm25_xgb",
    ]
    for c in required_cols:
        if c not in gdf.columns:
            raise KeyError(f"Missing column: {c}")

    # 클래스 컬럼 추가
    print("[INFO] Creating class columns...")
    gdf["class_pm10_rf"] = classify_pm10(gdf["mean_pm10_rf"])
    gdf["class_pm10_xgb"] = classify_pm10(gdf["mean_pm10_xgb"])
    gdf["class_pm25_rf"] = classify_pm25(gdf["mean_pm25_rf"])
    gdf["class_pm25_xgb"] = classify_pm25(gdf["mean_pm25_xgb"])

    # 정수형으로 캐스팅 (NaN은 그대로 둠)
    for c in ["class_pm10_rf", "class_pm10_xgb", "class_pm25_rf", "class_pm25_xgb"]:
        gdf[c] = gdf[c].astype("Int64")  # pandas nullable int

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    print(f"[INFO] Saving to: {OUTPUT_PATH}")
    gdf.to_file(OUTPUT_PATH, driver="GPKG")
    print("[INFO] Done.")


if __name__ == "__main__":
    main()
