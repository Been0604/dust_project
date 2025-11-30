# src/make_admin_stats_2021_2023.py
#
# 2021–2023 시간·공간 데이터를 이용해서
# 행정동(umd) 단위 취약성 지표 테이블을 만드는 스크립트 (최신 버전).
#
# 흐름:
#   1) stations.csv (위도/경도 좌표) → EPSG:4326으로 읽고 EPSG:5186으로 변환
#   2) daejeon_umd SHP(EPSG:5186)와 공간조인해서 station_id → 행정동 매핑
#   3) for_qgis_2021_2023.parquet에 station_id 기준으로 매핑을 붙임
#   4) 행정동 단위로 PM/경보 지표 집계
#
# 출력:
#   - data/admin_stats_2021_2023.parquet

from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"


def main() -> None:
    # --------------------------------------------------
    # 1. 경로 설정
    # --------------------------------------------------
    qgis_path = DATA_DIR / "for_qgis_2021_2023.parquet"
    stations_path = DATA_DIR / "stations.csv"
    umd_path = ROOT / "daejeon_umd" / "LSMD_ADM_SECT_UMD_30_202511.shp"
    out_path = DATA_DIR / "admin_stats_summary_2021_2023.parquet"

    print(f"[INFO] for_qgis_2021_2023: {qgis_path}")
    print(f"[INFO] stations.csv:       {stations_path}")
    print(f"[INFO] UMD shp:            {umd_path}")

    # --------------------------------------------------
    # 2. 테이블 / SHP 불러오기
    # --------------------------------------------------
    df = pd.read_parquet(qgis_path)
    stations = pd.read_csv(stations_path)
    print(f"[INFO] for_qgis_2021_2023 shape: {df.shape}")
    print(f"[INFO] stations shape:            {stations.shape}")

    gdf_umd = gpd.read_file(umd_path)
    if gdf_umd.crs is None:
        print("[WARN] UMD shp에 CRS 정보가 없어 EPSG:5186으로 설정합니다.")
        gdf_umd.set_crs("EPSG:5186", inplace=True)
    else:
        gdf_umd = gdf_umd.to_crs("EPSG:5186")
    print(f"[INFO] UMD CRS: {gdf_umd.crs}")
    print(f"[INFO] UMD layer columns: {list(gdf_umd.columns)}")

    # 실제 컬럼 이름 매핑
    col_code = "EMD_CD"      # 행정동 코드
    col_emd_name = "EMD_NM"  # 행정동 이름

    # 구 이름 컬럼이 없으므로, 임시로 '대전광역시'로 채운다.
    if "SIG_KOR_NM" not in gdf_umd.columns:
        gdf_umd["SIG_KOR_NM"] = "대전광역시"
    col_gu_name = "SIG_KOR_NM"

    # --------------------------------------------------
    # 3. stations: 4326 → 5186 변환 후 UMD와 공간조인
    # --------------------------------------------------
    # dmY = 경도, dmX = 위도로 보이므로 그대로 사용
    gdf_stations = gpd.GeoDataFrame(
        stations,
        geometry=gpd.points_from_xy(stations["dmY"], stations["dmX"]),
        crs="EPSG:4326",
    )
    print(f"[INFO] stations CRS (before): {gdf_stations.crs}")
    print(f"[INFO] stations total_bounds (4326): {gdf_stations.total_bounds}")

    gdf_stations_5186 = gdf_stations.to_crs(gdf_umd.crs)
    print(f"[INFO] stations CRS (after):  {gdf_stations_5186.crs}")
    print(f"[INFO] stations total_bounds (5186): {gdf_stations_5186.total_bounds}")
    print(f"[INFO] UMD total_bounds:             {gdf_umd.total_bounds}")

    print("[INFO] Spatial join (stations ↔ UMD, within)...")
    stations_joined = gpd.sjoin(
        gdf_stations_5186,
        gdf_umd[[col_code, col_emd_name, col_gu_name, "geometry"]],
        how="left",
        predicate="within",
    )

    print(f"[INFO] stations_joined shape: {stations_joined.shape}")

    stations_admin = stations_joined.rename(
        columns={
            col_code: "umd_code",
            col_emd_name: "umd_name",
            col_gu_name: "gu_name",
        }
    )[["station_id", "umd_code", "umd_name", "gu_name"]].copy()

    print("[INFO] station → 행정동 매핑:")
    print(stations_admin)

    # --------------------------------------------------
    # 4. 시간 테이블에 행정동 정보 붙이기
    # --------------------------------------------------
    df_admin = df.merge(
        stations_admin,
        on="station_id",
        how="left",
        validate="many_to_one",
    )

    missing_admin = df_admin["umd_code"].isna().sum()
    if missing_admin > 0:
        print(f"[WARN] 행정동 정보가 비어 있는 행이 {missing_admin}개 있습니다.")

    # --------------------------------------------------
    # 5. 집계 함수 정의
    # --------------------------------------------------
    def q95(x: pd.Series) -> float:
        return float(x.quantile(0.95))

    def q99(x: pd.Series) -> float:
        return float(x.quantile(0.99))

    def rate_over(x: pd.Series, threshold: float) -> float:
        if len(x) == 0:
            return np.nan
        return float((x >= threshold).mean())

    def rate_alert(x: pd.Series) -> float:
        if len(x) == 0:
            return np.nan
        return float((x > 0).mean())

    group_cols = ["gu_name", "umd_code", "umd_name"]

    # --------------------------------------------------
    # 6. 1차 집계 (PM/경보 관련 지표)
    # --------------------------------------------------
    agg = df_admin.groupby(group_cols).agg(
        n_hours=("pm10", "size"),
        mean_pm10=("pm10", "mean"),
        p95_pm10=("pm10", q95),
        p99_pm10=("pm10", q99),
        high_pm10_rate_80=("pm10", lambda x: rate_over(x, 80.0)),
        mean_pm25=("pm25", "mean"),
        p95_pm25=("pm25", q95),
        p99_pm25=("pm25", q99),
        high_pm25_rate_35=("pm25", lambda x: rate_over(x, 35.0)),
        loc_alert_rate_pm10=("y_loc_pm10", rate_alert),
        loc_alert_rate_pm25=("y_loc_pm25", rate_alert),
    ).reset_index()

    # --------------------------------------------------
    # 7. MAE 계산 (예측 값 있을 때만)
    # --------------------------------------------------
    if {"y_pred_xgb_pm10", "y_pred_xgb_pm25"}.issubset(df_admin.columns):
        df_admin["abs_err_pm10"] = (df_admin["y_pred_xgb_pm10"] - df_admin["pm10"]).abs()
        df_admin["abs_err_pm25"] = (df_admin["y_pred_xgb_pm25"] - df_admin["pm25"]).abs()

        mae_table = df_admin.groupby(group_cols).agg(
            mae_pm10=("abs_err_pm10", "mean"),
            mae_pm25=("abs_err_pm25", "mean"),
        ).reset_index()

        agg = agg.merge(mae_table, on=group_cols, how="left")
    else:
        print("[WARN] y_pred_xgb_pm10 / y_pred_xgb_pm25 컬럼이 없어 MAE는 계산하지 않습니다.")

    # --------------------------------------------------
    # 8. 정렬 및 저장
    # --------------------------------------------------
    agg = agg.sort_values(["gu_name", "umd_name"]).reset_index(drop=True)

    print(f"[INFO] admin_stats_2021_2023 shape: {agg.shape}")
    print(f"[INFO] Saving to {out_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    agg.to_parquet(out_path, index=False)

    print("[INFO] Done.")


if __name__ == "__main__":
    main()
