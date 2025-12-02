from pathlib import Path
import pandas as pd

# 프로젝트 기준 경로
BASE_DIR = Path("data/scenario")

FILES = [
    "bi_admin_pm_stats_2026_pm10.parquet",
    "bi_admin_pm_stats_2026_pm25.parquet",
    "bi_area_ew_stats_2026_pm10.parquet",
    "bi_area_ew_stats_2026_pm25.parquet",
    "bi_ensemble_2026_pm10_pm25.parquet",
]

for fname in FILES:
    parquet_path = BASE_DIR / fname
    csv_path = BASE_DIR / fname.replace(".parquet", ".csv")

    print(f"읽는 중: {parquet_path}")
    df = pd.read_parquet(parquet_path)

    print(f"저장 중: {csv_path}")
    df.to_csv(csv_path, index=False)

print("모든 변환 완료.")
