# src/features.py
# processed_final.parquet 기준으로
# PM10 + PM25 둘 다에 대한 피처/타깃 생성

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df["ts_kst"] = pd.to_datetime(df["ts_kst"])
    df["year"] = df["ts_kst"].dt.year
    df["month"] = df["ts_kst"].dt.month
    df["weekday"] = df["ts_kst"].dt.weekday
    df["hour"] = df["ts_kst"].dt.hour
    df["is_weekend"] = (df["weekday"] >= 5).astype(int)
    return df


def main():
    path = DATA_DIR / "processed_final.parquet"
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_parquet(path)

    # 시간 피처 + 정렬
    df = add_time_features(df)
    df = df.sort_values(["station_id", "ts_kst"]).reset_index(drop=True)

    g = df.groupby("station_id", group_keys=False)

    # PM10 / PM25 lag + rolling
    def add_lag_and_roll(col: str):
        if col not in df.columns:
            return
        for lag in [1, 2, 3]:
            df[f"{col}_lag{lag}"] = g[col].shift(lag)
        for win in [3, 6, 12, 24]:
            df[f"{col}_roll{win}h"] = (
                g[col]
                .rolling(window=win, min_periods=1)
                .mean()
                .reset_index(level=0, drop=True)
            )

    add_lag_and_roll("pm10")
    add_lag_and_roll("pm25")

    # 날씨 lag1
    for col in ["temp", "rh", "wind_spd", "wind_dir", "rain", "pressure"]:
        if col in df.columns:
            df[f"{col}_lag1"] = g[col].shift(1)

    # 타깃: 1시간 뒤 PM10 / PM25
    if "pm10" not in df.columns or "pm25" not in df.columns:
        raise KeyError("pm10/pm25 컬럼이 processed_final에 있어야 합니다.")

    df["target_pm10"] = g["pm10"].shift(-1)
    df["target_pm25"] = g["pm25"].shift(-1)

    # 피처로 쓸 컬럼 모으기
    base_cols = [
        "ts_kst",
        "station_id",
        "year",
        "month",
        "weekday",
        "hour",
        "is_weekend",
        "pm10",
        "pm25",
        "temp",
        "rh",
        "wind_spd",
        "wind_dir",
        "rain",
        "pressure",
    ]
    lag_cols = [c for c in df.columns if "lag" in c]
    roll_cols = [c for c in df.columns if "roll" in c]

    feature_cols = list(
        dict.fromkeys(
            base_cols + lag_cols + roll_cols + ["target_pm10", "target_pm25"]
        )
    )

    feats = df[feature_cols].copy()

    # 타깃별로 결측 있는 행 제거해서 각각 저장
    feats_pm10 = feats.dropna(subset=["target_pm10"]).copy()
    feats_pm25 = feats.dropna(subset=["target_pm25"]).copy()

    out10 = DATA_DIR / "features_full_pm10.parquet"
    out25 = DATA_DIR / "features_full_pm25.parquet"

    feats_pm10.to_parquet(out10, index=False)
    feats_pm25.to_parquet(out25, index=False)

    print("features_full_pm10:", feats_pm10.shape, "→", out10)
    print("features_full_pm25:", feats_pm25.shape, "→", out25)


if __name__ == "__main__":
    main()
