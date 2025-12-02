import os

import numpy as np
import pandas as pd
import geopandas as gpd

SRC_GPKG = "data/scenario/umd_pm_stats_2026_scenarios_with_classes.gpkg"
OUT_GPKG = "data/scenario/umd_ensemble_2026_pm10_pm25.gpkg"
OUT_PARQUET = "data/scenario/bi_ensemble_2026_pm10_pm25.parquet"


def main():
    print(f"[INFO] Loading: {SRC_GPKG}")
    gdf = gpd.read_file(SRC_GPKG)

    # 필수 컬럼: 최소한 코드랑 클래스만 있으면 됨
    required = [
        "EMD_CD",
        "scenario_id",
        "class_pm10_rf",
        "class_pm10_xgb",
        "class_pm25_rf",
        "class_pm25_xgb",
    ]
    for c in required:
        if c not in gdf.columns:
            raise KeyError(f"Missing column: {c}")

    # 1) 고농도 플래그 (클래스 4,5 = 기준 초과)
    print("[INFO] Creating high-concentration flags...")
    gdf["high_pm10_rf"] = gdf["class_pm10_rf"] >= 4
    gdf["high_pm10_xgb"] = gdf["class_pm10_xgb"] >= 4
    gdf["high_pm25_rf"] = gdf["class_pm25_rf"] >= 4
    gdf["high_pm25_xgb"] = gdf["class_pm25_xgb"] >= 4

    # 2) 행정동 단위 집계
    group_cols = ["EMD_CD"]
    # 있으면 같이 묶어주는 옵션 컬럼들
    if "area_ew" in gdf.columns:
        group_cols.append("area_ew")
    if "EMD_NM" in gdf.columns:
        group_cols.append("EMD_NM")

    print("[INFO] Aggregating by EMD...")
    agg = (
        gdf.groupby(group_cols)
        .agg(
            n_scenarios=("scenario_id", "nunique"),
            n_high_pm10_rf=("high_pm10_rf", "sum"),
            n_high_pm10_xgb=("high_pm10_xgb", "sum"),
            n_high_pm25_rf=("high_pm25_rf", "sum"),
            n_high_pm25_xgb=("high_pm25_xgb", "sum"),
        )
        .reset_index()
    )

    # 3) RF+XGB 합산 및 ensemble 등급
    agg["n_high_pm10_all"] = agg["n_high_pm10_rf"] + agg["n_high_pm10_xgb"]
    agg["n_high_pm25_all"] = agg["n_high_pm25_rf"] + agg["n_high_pm25_xgb"]

    def classify_ensemble(n):
        if n == 0:
            return 0          # 거의 낮음
        elif n <= 2:
            return 1          # 가끔 고농도
        elif n <= 4:
            return 2          # 자주 고농도
        else:
            return 3          # 거의 항상 고농도

    agg["risk_pm10_all"] = agg["n_high_pm10_all"].apply(classify_ensemble)
    agg["risk_pm25_all"] = agg["n_high_pm25_all"].apply(classify_ensemble)

    # 4) geometry 붙이기 (행정동별로 하나만)
    geom = gdf[["EMD_CD", "geometry"]].drop_duplicates(subset="EMD_CD")
    g_agg = geom.merge(agg, on="EMD_CD", how="right")

    os.makedirs(os.path.dirname(OUT_PARQUET), exist_ok=True)

    print(f"[INFO] Saving table to: {OUT_PARQUET}")
    agg.to_parquet(OUT_PARQUET, index=False)

    print(f"[INFO] Saving GeoPackage to: {OUT_GPKG}")
    g_agg.to_file(OUT_GPKG, driver="GPKG")

    print("[INFO] Done.")


if __name__ == "__main__":
    main()
