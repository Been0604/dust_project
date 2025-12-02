import os
import pandas as pd
import geopandas as gpd


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_admin_stats_2026():
    """2026 시나리오 행정동 요약 (PM10/PM25) 로드 & 머지"""
    pm10_path = "data/scenario/bi_admin_pm_stats_2026_pm10.parquet"
    pm25_path = "data/scenario/bi_admin_pm_stats_2026_pm25.parquet"

    print(f"[INFO] Loading admin PM10 stats: {pm10_path}")
    a10 = pd.read_parquet(pm10_path)

    print(f"[INFO] Loading admin PM25 stats: {pm25_path}")
    a25 = pd.read_parquet(pm25_path)

    # EMD_CD + 시나리오 키로만 머지 (이름은 건드리지 않음)
    key_cols = ["scenario_id", "scenario_base_year", "EMD_CD"]

    admin = a10.merge(
        a25,
        on=key_cols,
        how="inner",
        suffixes=("_pm10", "_pm25"),
    )

    admin = admin.sort_values(["scenario_id", "EMD_CD"]).reset_index(drop=True)
    admin["EMD_CD"] = admin["EMD_CD"].astype(str)

    print(f"[INFO] Merged admin stats shape: {admin.shape}")
    return admin


def load_emd_geometry():
    """
    행정동 geometry 로드.
    이름 컬럼은 사용하지 않고 EMD_CD + geometry만 사용.
    """
    umd_path = "data/umd_time_pm_2021_2023.gpkg"
    print(f"[INFO] Loading EMD geometry from: {umd_path}")
    g_umd = gpd.read_file(umd_path)

    if "EMD_CD" not in g_umd.columns:
        raise KeyError("umd_time_pm_2021_2023.gpkg 에 'EMD_CD' 컬럼이 없습니다.")

    g_emd = g_umd[["EMD_CD", "geometry"]].drop_duplicates(subset="EMD_CD").copy()
    g_emd["EMD_CD"] = g_emd["EMD_CD"].astype(str)

    print(f"[INFO] Unique EMD geometries: {len(g_emd)}")
    return g_emd


def make_umd_stats_2026_gpkg():
    admin = load_admin_stats_2026()
    g_emd = load_emd_geometry()

    print("[INFO] Merging admin stats with geometry...")
    g_out = g_emd.merge(admin, on="EMD_CD", how="right")

    print(f"[INFO] Output GeoDataFrame shape: {g_out.shape}")

    out_dir = "data/scenario"
    ensure_dir(out_dir)
    out_path = os.path.join(out_dir, "umd_pm_stats_2026_scenarios.gpkg")

    print(f"[INFO] Saving to: {out_path}")
    g_out.to_file(out_path, driver="GPKG")
    print("[INFO] Done.")


def main():
    make_umd_stats_2026_gpkg()


if __name__ == "__main__":
    main()
