# src/make_bi_admin_pm_stats_2021_2023.py
#
# processed_with_preds_both.parquet + stations_emd_mapping.csv + emd_codes.csv
# → 행정동×연도×계절 단위 PM 통계 요약 (bi_admin_pm_stats_2021_2023.*)

from pathlib import Path
import pandas as pd
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
BI_DIR = DATA_DIR / "bi"


def add_year_season(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["year"] = df["ts_kst"].dt.year
    month = df["ts_kst"].dt.month

    conditions = [
        month.isin([12, 1, 2]),
        month.isin([3, 4, 5]),
        month.isin([6, 7, 8]),
        month.isin([9, 10, 11]),
    ]
    choices = ["겨울", "봄", "여름", "가을"]
    df["season"] = np.select(conditions, choices, default="기타")
    return df


def main():
    BI_DIR.mkdir(parents=True, exist_ok=True)

    pm_path = DATA_DIR / "processed_with_preds_both.parquet"
    map_path = DATA_DIR / "stations_emd_mapping.csv"
    codes_path = BI_DIR / "emd_codes.csv"

    if not pm_path.exists():
        raise FileNotFoundError(pm_path)
    if not map_path.exists():
        raise FileNotFoundError(map_path)
    if not codes_path.exists():
        raise FileNotFoundError(codes_path)

    df = pd.read_parquet(pm_path)
    mapping = pd.read_csv(map_path)
    codes = pd.read_csv(codes_path)

    required_pm_cols = {"ts_kst", "station_id", "pm10", "pm25"}
    missing_pm = required_pm_cols - set(df.columns)
    if missing_pm:
        raise ValueError(f"processed_with_preds_both에 없는 컬럼: {missing_pm}")

    required_map_cols = {"station_id", "EMD_CD"}
    missing_map = required_map_cols - set(mapping.columns)
    if missing_map:
        raise ValueError(f"stations_emd_mapping에 없는 컬럼: {missing_map}")

    required_codes_cols = {"EMD_CD", "EMD_NM"}
    missing_codes = required_codes_cols - set(codes.columns)
    if missing_codes:
        raise ValueError(f"emd_codes에 없는 컬럼: {missing_codes}")

    # 2021–2023만 사용
    df = df[df["ts_kst"].dt.year.between(2021, 2023)].copy()

    # station_id → 행정동 코드
    df = df.merge(
        mapping[["station_id", "EMD_CD"]],
        on="station_id",
        how="left",
    )

    if df["EMD_CD"].isna().any():
        n_missing = df["EMD_CD"].isna().sum()
        raise ValueError(f"행정동 코드가 매핑되지 않은 관측치가 {n_missing}개 있습니다.")

    df = add_year_season(df)

    PM10_KR_LIMIT = 50.0
    PM25_KR_LIMIT = 15.0

    group_cols = ["year", "season", "EMD_CD"]

    def agg_func(x: pd.Series, limit: float | None = None):
        res = {
            "mean": x.mean(),
            "p90": x.quantile(0.9),
            "n": x.notna().sum(),
        }
        if limit is not None:
            res["over_ratio"] = (x > limit).mean()
        return pd.Series(res)

    rows = []

    for (year, season, emd_cd), g in df.groupby(group_cols):
        pm10_stats = agg_func(g["pm10"], PM10_KR_LIMIT)
        pm25_stats = agg_func(g["pm25"], PM25_KR_LIMIT)

        rows.append(
            {
                "year": year,
                "season": season,
                "EMD_CD": emd_cd,
                "n_hours": int(pm10_stats["n"]),
                "pm10_mean": pm10_stats["mean"],
                "pm10_p90": pm10_stats["p90"],
                "pm10_over_kr_ratio": pm10_stats["over_ratio"],
                "pm25_mean": pm25_stats["mean"],
                "pm25_p90": pm25_stats["p90"],
                "pm25_over_kr_ratio": pm25_stats["over_ratio"],
            }
        )

    result = pd.DataFrame(rows)

    # 깨끗한 이름 붙이기
    result = result.merge(
        codes[["EMD_CD", "EMD_NM"]],
        on="EMD_CD",
        how="left",
    )

    if result["EMD_NM"].isna().any():
        n_missing = result["EMD_NM"].isna().sum()
        raise ValueError(f"emd_codes에서 이름을 못 찾은 행정동이 {n_missing}개 있습니다.")

    result = result[
        [
            "year",
            "season",
            "EMD_CD",
            "EMD_NM",
            "n_hours",
            "pm10_mean",
            "pm10_p90",
            "pm10_over_kr_ratio",
            "pm25_mean",
            "pm25_p90",
            "pm25_over_kr_ratio",
        ]
    ].sort_values(["EMD_NM", "year", "season"]).reset_index(drop=True)

    out_csv = BI_DIR / "bi_admin_pm_stats_2021_2023.csv"
    out_parquet = BI_DIR / "bi_admin_pm_stats_2021_2023.parquet"

    result.to_csv(out_csv, index=False, encoding="utf-8-sig")
    result.to_parquet(out_parquet, index=False)

    print(f"Saved: {out_csv}")
    print(f"Saved: {out_parquet}")


if __name__ == "__main__":
    main()
