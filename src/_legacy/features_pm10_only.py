# src/features.py

from pathlib import Path
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
FEATURE_DIR = ROOT / "features"


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["year"] = df["ts_kst"].dt.year
    df["month"] = df["ts_kst"].dt.month
    df["day"] = df["ts_kst"].dt.day
    df["weekday"] = df["ts_kst"].dt.weekday  # 0=월, 6=일
    df["hour"] = df["ts_kst"].dt.hour
    df["is_weekend"] = df["weekday"].isin([5, 6]).astype(int)
    return df


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    station_id별로 정렬한 뒤 lag / rolling 피처 생성.
    """
    df = df.copy()
    df = df.sort_values(["station_id", "ts_kst"])

    group = df.groupby("station_id", group_keys=False)

    # 기본 PM10 lag
    for lag in [1, 2, 3]:
        df[f"pm10_lag{lag}"] = group["pm10"].shift(lag)

    # PM10 rolling mean
    roll_windows = {
        "pm10_roll3h": 3,
        "pm10_roll6h": 6,
        "pm10_roll12h": 12,
        "pm10_roll24h": 24,
    }
    for name, window in roll_windows.items():
        df[name] = (
            group["pm10"]
            .rolling(window, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )

    # PM2.5도 참고 피처로 몇 개
    df["pm25_lag1"] = group["pm25"].shift(1)
    df["pm25_roll6h"] = (
        group["pm25"]
        .rolling(6, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )

    # 날씨 lag (필요 최소만)
    for col in ["temp", "rh", "wind_spd"]:
        df[f"{col}_lag1"] = group[col].shift(1)

    return df


def define_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    target_pm10 = 1시간 뒤 pm10
    """
    df = df.copy()
    df = df.sort_values(["station_id", "ts_kst"])
    df["target_pm10"] = df.groupby("station_id")["pm10"].shift(-1)
    return df


def main():
    FEATURE_DIR.mkdir(exist_ok=True)

    # 1) 데이터 로드
    processed_path = DATA_DIR / "processed_final.parquet"
    df = pd.read_parquet(processed_path)

    # ts_kst datetime 보장
    if not pd.api.types.is_datetime64_any_dtype(df["ts_kst"]):
        df["ts_kst"] = pd.to_datetime(df["ts_kst"])

    # 2) 시간 피처
    df = add_time_features(df)

    # 3) lag / rolling 피처
    df = add_lag_features(df)

    # 4) 타깃 정의
    df = define_target(df)

    # 5) 피처/타깃/메타 컬럼 정리
    id_cols = ["ts_kst", "station_id", "area"]
    base_cols = [
        "pm10", "pm25",
        "temp", "rh", "wind_spd", "wind_dir", "rain", "pressure",
    ]
    time_cols = ["year", "month", "day", "weekday", "hour", "is_weekend"]

    lag_cols = [
        "pm10_lag1", "pm10_lag2", "pm10_lag3",
        "pm10_roll3h", "pm10_roll6h", "pm10_roll12h", "pm10_roll24h",
        "pm25_lag1", "pm25_roll6h",
        "temp_lag1", "rh_lag1", "wind_spd_lag1",
    ]

    # 모델 입력 피처
    feature_cols = base_cols + time_cols + lag_cols

    # 보조 라벨 (나중 분석용)
    label_cols = ["y_loc_pm10", "y_loc_pm25"]

    target_col = "target_pm10"

    # 6) NA 처리: 피처/타깃에 NaN 있는 행 제거
    df_model = df[id_cols + feature_cols + label_cols + [target_col]].copy()
    df_model = df_model.dropna(subset=feature_cols + [target_col])

    # 7) 연도 기준 Train / Val / Test 분리
    years = df_model["ts_kst"].dt.year
    mask_train = years.isin([2021, 2022])
    mask_val = years == 2023
    mask_test = years == 2024  # 외부 테스트

    # 전체 테이블 저장 (연도 분할/워크포워드용)
    features_full_path = FEATURE_DIR / "features_full_pm10.parquet"
    df_model.to_parquet(features_full_path, index=False)

    # Train/Val/Test X, y 저장
    X_train = df_model.loc[mask_train, feature_cols]
    y_train = df_model.loc[mask_train, [target_col]]

    X_val = df_model.loc[mask_val, feature_cols]
    y_val = df_model.loc[mask_val, [target_col]]

    X_test = df_model.loc[mask_test, feature_cols]
    y_test = df_model.loc[mask_test, [target_col]]

    X_train_path = FEATURE_DIR / "X_train_pm10.parquet"
    y_train_path = FEATURE_DIR / "y_train_pm10.parquet"
    X_val_path = FEATURE_DIR / "X_val_pm10.parquet"
    y_val_path = FEATURE_DIR / "y_val_pm10.parquet"
    X_test_path = FEATURE_DIR / "X_test_pm10.parquet"
    y_test_path = FEATURE_DIR / "y_test_pm10.parquet"

    X_train.to_parquet(X_train_path, index=False)
    y_train.to_parquet(y_train_path, index=False)
    X_val.to_parquet(X_val_path, index=False)
    y_val.to_parquet(y_val_path, index=False)
    X_test.to_parquet(X_test_path, index=False)
    y_test.to_parquet(y_test_path, index=False)

    print("features_full_pm10:", df_model.shape)
    print("X_train_pm10:", X_train.shape, "| X_val_pm10:", X_val.shape, "| X_test_pm10:", X_test.shape)
    print("저장 완료")


if __name__ == "__main__":
    main()
