# src/make_processed_final.py
#
# 목표:
# - data/pm_with_alerts.parquet (PM + y_loc)
# - data/raw_weather_final.parquet (ASOS 날씨)
# 를 ts_kst 기준으로 조인해서
# data/processed_final.parquet 으로 저장한다.

from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(r"C:\Users\wachu\dust_project")
DATA_DIR = PROJECT_ROOT / "data"


def load_pm_with_alerts() -> pd.DataFrame:
    path = DATA_DIR / "pm_with_alerts.parquet"
    if not path.exists():
        raise FileNotFoundError(f"pm_with_alerts.parquet not found at {path}")

    df = pd.read_parquet(path)
    # ts_kst를 datetime으로 통일
    df["ts_kst"] = pd.to_datetime(df["ts_kst"])
    return df


def load_weather() -> pd.DataFrame:
    path = DATA_DIR / "raw_weather_final.parquet"
    if not path.exists():
        raise FileNotFoundError(f"raw_weather_final.parquet not found at {path}")

    df = pd.read_parquet(path)
    df["ts_kst"] = pd.to_datetime(df["ts_kst"])
    return df


def make_processed(df_pm: pd.DataFrame, df_w: pd.DataFrame) -> pd.DataFrame:
    """
    ts_kst 기준으로 left join (PM 기준으로 날씨를 붙인다).
    """
    # 혹시 중복 ts_kst가 있으면 평균/첫 값으로 줄이는 것도 가능하지만,
    # 지금 구조에서는 ASOS 1지점(133)이라 중복 거의 없다고 가정.
    df_w_nodup = df_w.drop_duplicates(subset=["ts_kst"]).copy()

    df = df_pm.merge(df_w_nodup, on="ts_kst", how="left", suffixes=("", "_weather"))

    # 정렬
    df = df.sort_values(["ts_kst", "station_id"]).reset_index(drop=True)

    # 기본 정보 출력용
    print("processed_final shape:", df.shape)
    print("ts range:", df["ts_kst"].min(), "→", df["ts_kst"].max())

    # 결측 비율 확인 (특히 날씨 컬럼)
    weather_cols = [c for c in ["temp", "rh", "wind_spd", "wind_dir", "rain", "pressure"] if c in df.columns]
    print("\n날씨 컬럼 결측 비율:")
    print(df[weather_cols].isna().mean())

    return df


def main():
    df_pm = load_pm_with_alerts()
    df_w = load_weather()

    print("pm_with_alerts shape:", df_pm.shape)
    print("raw_weather_final shape:", df_w.shape)

    df_proc = make_processed(df_pm, df_w)

    out_path = DATA_DIR / "processed_final.parquet"
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df_proc.to_parquet(out_path, index=False)
    print("\nsaved:", out_path)


if __name__ == "__main__":
    main()
