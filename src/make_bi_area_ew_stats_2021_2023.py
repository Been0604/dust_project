# src/make_bi_area_ew_stats_2021_2023.py
#
# processed_with_preds_both.parquet → 동부권/서부권 × 연도 요약
# (bi_area_ew_stats_2021_2023.*)

from pathlib import Path
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
BI_DIR = DATA_DIR / "bi"


def main():
    BI_DIR.mkdir(parents=True, exist_ok=True)

    pm_path = DATA_DIR / "processed_with_preds_both.parquet"
    if not pm_path.exists():
        raise FileNotFoundError(pm_path)

    df = pd.read_parquet(pm_path)

    required_cols = {"ts_kst", "area", "pm10", "pm25"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"processed_with_preds_both에 없는 컬럼: {missing}")

    # 2021–2023만 사용
    df = df[df["ts_kst"].dt.year.between(2021, 2023)].copy()

    # 연도 + 동부/서부권 이름
    df["year"] = df["ts_kst"].dt.year
    # area: 동부 / 서부 → area_ew: 동부권 / 서부권
    df["area_ew"] = df["area"].map({"동부": "동부권", "서부": "서부권"}).fillna(df["area"])

    PM10_KR_LIMIT = 50.0
    PM25_KR_LIMIT = 15.0

    group_cols = ["year", "area_ew"]

    grouped = []
    for (year, area_ew), g in df.groupby(group_cols):
        pm10 = g["pm10"]
        pm25 = g["pm25"]

        row = {
            "year": year,
            "area_ew": area_ew,
            "n_hours": int(pm10.notna().sum()),
            "pm10_mean": pm10.mean(),
            "pm10_p90": pm10.quantile(0.9),
            "pm10_over_kr_ratio": (pm10 > PM10_KR_LIMIT).mean(),
            "pm25_mean": pm25.mean(),
            "pm25_p90": pm25.quantile(0.9),
            "pm25_over_kr_ratio": (pm25 > PM25_KR_LIMIT).mean(),
        }
        grouped.append(row)

    result = (
        pd.DataFrame(grouped)
        .sort_values(["year", "area_ew"])
        .reset_index(drop=True)
    )

    out_csv = BI_DIR / "bi_area_ew_stats_2021_2023.csv"
    out_parquet = BI_DIR / "bi_area_ew_stats_2021_2023.parquet"

    result.to_csv(out_csv, index=False, encoding="utf-8-sig")
    result.to_parquet(out_parquet, index=False)

    print(f"Saved: {out_csv}")
    print(f"Saved: {out_parquet}")


if __name__ == "__main__":
    main()
