from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
FEAT_DIR = ROOT / "features"


def load_and_merge() -> pd.DataFrame:
    """processed + y_loc + y_nat을 하나의 테이블로 병합."""
    processed_path = DATA_DIR / "processed.parquet"
    y_loc_path = DATA_DIR / "y_loc.parquet"
    y_nat_path = DATA_DIR / "y_nat.parquet"

    df = pd.read_parquet(processed_path)
    y_loc = pd.read_parquet(y_loc_path)
    y_nat = pd.read_parquet(y_nat_path)

    # ts_kst를 datetime으로
    for d in (df, y_loc, y_nat):
        if "ts_kst" in d.columns:
            d["ts_kst"] = pd.to_datetime(d["ts_kst"])

    # y_loc merge
    if "station_id" in y_loc.columns:
        df = df.merge(
            y_loc[["ts_kst", "station_id", "y_loc"]],
            on=["ts_kst", "station_id"],
            how="left",
        )
    else:
        df = df.merge(
            y_loc[["ts_kst", "y_loc"]],
            on="ts_kst",
            how="left",
        )

    # y_nat merge
    join_keys = ["ts_kst", "station_id"] if "station_id" in y_nat.columns else ["ts_kst"]
    df = df.merge(
        y_nat[join_keys + ["y_nat"]],
        on=join_keys,
        how="left",
    )

    # 라벨 타입 정리
    df["y_loc"] = df["y_loc"].astype("Int64")
    df["y_nat"] = df["y_nat"].fillna(0).astype(int)

    df = df.sort_values(["ts_kst", "station_id"])
    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """시간 관련 피처 추가."""
    df["year"] = df["ts_kst"].dt.year
    df["month"] = df["ts_kst"].dt.month
    df["weekday"] = df["ts_kst"].dt.weekday
    df["hour"] = df["ts_kst"].dt.hour

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24.0)

    return df


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """측정소별 lag / rolling 피처 생성."""
    df = df.sort_values(["station_id", "ts_kst"])
    group = df.groupby("station_id", group_keys=False)

    def make_lags(g: pd.DataFrame) -> pd.DataFrame:
        g = g.copy()
        if "pm10" in g.columns:
            g["pm10_lag1"] = g["pm10"].shift(1)
            g["pm10_lag2"] = g["pm10"].shift(2)
            g["pm10_lag3"] = g["pm10"].shift(3)
            g["pm10_roll3_mean"] = g["pm10"].rolling(3).mean()
            g["pm10_roll6_mean"] = g["pm10"].rolling(6).mean()

        for col in ["temp", "rh", "wind_spd"]:
            if col in g.columns:
                g[f"{col}_lag1"] = g[col].shift(1)

        return g

    df = group.apply(make_lags)
    return df


def make_train_val_and_save(df: pd.DataFrame) -> None:
    """train/val 분할 + features_full 저장."""
    df = df.sort_values(["ts_kst", "station_id"])

    drop_cols = ["ts_kst", "station_id", "y_loc", "y_nat"]
    feature_cols = [c for c in df.columns if c not in drop_cols]

    # 라벨/피처 NaN 제거
    df_model = df.dropna(subset=feature_cols + ["y_loc"])

    n = len(df_model)
    if n == 0:
        print("[경고] 유효한 행이 없습니다. 기간이나 lag 윈도우를 확인해 주세요.")
        return

    split_idx = int(n * 0.8)
    train = df_model.iloc[:split_idx]
    val = df_model.iloc[split_idx:]

    X_train = train[feature_cols]
    y_train = train[["y_loc"]]
    X_val = val[feature_cols]
    y_val = val[["y_loc"]]

    FEAT_DIR.mkdir(exist_ok=True)

    X_train.to_parquet(FEAT_DIR / "X_train.parquet")
    X_val.to_parquet(FEAT_DIR / "X_val.parquet")
    y_train.to_parquet(FEAT_DIR / "y_train.parquet")
    y_val.to_parquet(FEAT_DIR / "y_val.parquet")

    # 연도 홀드아웃 / 워크포워드용 전체 테이블
    df_model.to_parquet(FEAT_DIR / "features_full.parquet")

    print(f"[정보] 전체 학습용 행 수: {n}")
    print(f"[정보] train: {len(train)}, val: {len(val)}")
    print(f"[정보] 피처 수: {len(feature_cols)}")
    print(f"[정보] 저장 위치: {FEAT_DIR}")


def main():
    df = load_and_merge()
    df = add_time_features(df)
    df = add_lag_features(df)
    make_train_val_and_save(df)


if __name__ == "__main__":
    main()
