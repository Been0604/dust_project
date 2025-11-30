# src/make_umd_time_pm10_2021_2023.py
#
# 2021–2023 시간 × 행정동 단위 PM10/PM25 평균을 만드는 스크립트.
# 입력:
#   - data/for_qgis_2021_2023.parquet  (시간 × 측정소 PM + 예측값 등)
#   - data/stations.csv                (station_id, dmX, dmY = 위경도)
#   - daejeon_umd/LSMD_ADM_SECT_UMD_30_202511.shp  (행정동 경계, EPSG:5186)
# 출력:
#   - data/umd_time_pm_2021_2023.gpkg  (레이어 이름: umd_time_pm)

from pathlib import Path

import pandas as pd
import geopandas as gpd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def main():
    print("========== make_umd_time_pm10_2021_2023 ==========")
    print("[INFO] ROOT :", ROOT)
    print("[INFO] DATA :", DATA)

    # --------------------------------------------------
    # 1. 시간 × 측정소 데이터 + stations 좌표 merge
    # --------------------------------------------------
    qgis_path = DATA / "for_qgis_2021_2023.parquet"
    stations_path = DATA / "stations.csv"

    print(f"[INFO] loading for_qgis_2021_2023: {qgis_path}")
    df = pd.read_parquet(qgis_path)
    print("[INFO] for_qgis_2021_2023 shape:", df.shape)

    print(f"[INFO] loading stations: {stations_path}")
    stations = pd.read_csv(stations_path)
    stations = stations[["station_id", "dmX", "dmY"]]
    print("[INFO] stations shape:", stations.shape)

    # station_id 기준으로 좌표 붙이기
    df = df.merge(stations, on="station_id", how="left")
    missing_coords = df["dmX"].isna().sum()
    print(f"[INFO] merged df shape: {df.shape} (missing coords: {missing_coords})")

    # 좌표 없는 행 버리기
    if missing_coords > 0:
        df = df.dropna(subset=["dmX", "dmY"])
        print("[INFO] dropped rows with missing coords. new shape:", df.shape)

    # --------------------------------------------------
    # 2. 포인트 GeoDataFrame (먼저 4326, 그다음 5186 변환)
    #    dmX, dmY = 위경도(WGS84) 라서 먼저 EPSG:4326로 잡는다.
    #    QGIS에서 했던 것처럼 X=dmY, Y=dmX 사용.
    # --------------------------------------------------
    print("[INFO] building point GeoDataFrame (CRS=EPSG:4326 → 5186)...")
    gdf_pts_wgs84 = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["dmY"], df["dmX"]),
        crs="EPSG:4326",
    )
    print("[INFO] stations_total_bounds (4326):", gdf_pts_wgs84.total_bounds)

    # --------------------------------------------------
    # 3. 행정동 SHP 읽기 (ROOT/daejeon_umd/...)
    # --------------------------------------------------
    umd_shp_path = ROOT / "daejeon_umd" / "LSMD_ADM_SECT_UMD_30_202511.shp"
    print(f"[INFO] loading UMD shp: {umd_shp_path}")
    gdf_umd = gpd.read_file(umd_shp_path)
    print("[INFO] UMD shape:", gdf_umd.shape)
    print("[INFO] UMD CRS  :", gdf_umd.crs)
    print("[INFO] UMD total_bounds:", gdf_umd.total_bounds)

    # CRS 정리
    if gdf_umd.crs is None:
        print("[INFO] UMD CRS is None → setting to EPSG:5186")
        gdf_umd.set_crs(epsg=5186, inplace=True)

    # 포인트를 UMD CRS(5186)로 변환
    gdf_pts = gdf_pts_wgs84.to_crs(gdf_umd.crs)
    print("[INFO] stations_total_bounds ({}):".format(gdf_umd.crs), gdf_pts.total_bounds)

    # 필요한 컬럼만 유지
    gdf_umd = gdf_umd[["EMD_CD", "EMD_NM", "COL_ADM_SE", "geometry"]]

    # --------------------------------------------------
    # 4. spatial join: 포인트 → 행정동 (within)
    # --------------------------------------------------
    print("[INFO] spatial join (points within UMD polygons)...")
    gdf_joined = gpd.sjoin(
        gdf_pts,
        gdf_umd,
        how="inner",
        predicate="within",
    )
    print("[INFO] gdf_joined shape:", gdf_joined.shape)

    if gdf_joined.empty:
        print("[WARN] gdf_joined is EMPTY. check CRS/coordinates.")
        # 디버깅용으로 빈 레이어라도 저장
        out_path = DATA / "umd_time_pm_2021_2023.gpkg"
        if out_path.exists():
            out_path.unlink()
        gdf_joined.to_file(out_path, layer="umd_time_pm", driver="GPKG")
        print("[INFO] Saved EMPTY layer (for debug).")
        return

    # --------------------------------------------------
    # 5. ts_kst × 행정동 단위 집계
    # --------------------------------------------------
    group_cols = ["ts_kst", "EMD_CD", "EMD_NM", "COL_ADM_SE"]
    print("[INFO] grouping by:", group_cols)

    grouped = (
        gdf_joined
        .groupby(group_cols, as_index=False)
        .agg(
            pm10_mean=("pm10", "mean"),
            pm25_mean=("pm25", "mean"),
            n_obs=("pm10", "size"),
        )
    )

    print("[INFO] grouped shape:", grouped.shape)

    # --------------------------------------------------
    # 6. 행정동 geometry 다시 붙여서 폴리곤 GeoDataFrame 만들기
    # --------------------------------------------------
    gdf_umd_geom = gdf_umd.drop_duplicates(subset=["EMD_CD"])[["EMD_CD", "geometry"]]

    gdf_time_umd = grouped.merge(gdf_umd_geom, on="EMD_CD", how="left")

    gdf_time_umd = gpd.GeoDataFrame(
        gdf_time_umd,
        geometry="geometry",
        crs=gdf_umd.crs,
    )

    gdf_time_umd = gdf_time_umd.sort_values(["ts_kst", "EMD_CD"]).reset_index(
        drop=True
    )

    print("[INFO] final gdf_time_umd shape:", gdf_time_umd.shape)

    # --------------------------------------------------
    # 7. GeoPackage 저장
    # --------------------------------------------------
    out_path = DATA / "umd_time_pm_2021_2023.gpkg"
    layer_name = "umd_time_pm"

    if out_path.exists():
        print(f"[INFO] removing existing file: {out_path}")
        out_path.unlink()

    print(f"[INFO] saving to {out_path} (layer='{layer_name}')")
    gdf_time_umd.to_file(out_path, layer=layer_name, driver="GPKG")

    print("[INFO] Done.")
    print("===============================================")


if __name__ == "__main__":
    main()
