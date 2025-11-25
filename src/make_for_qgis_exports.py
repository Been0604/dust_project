# src/make_for_qgis_exports.py

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """ts_kst 기준으로 year, month, hour, season 컬럼을 추가한다."""
    if not pd.api.types.is_datetime64_any_dtype(df["ts_kst"]):
        df["ts_kst"] = pd.to_datetime(df["ts_kst"])

    df["year"] = df["ts_kst"].dt.year
    df["month"] = df["ts_kst"].dt.month
    df["hour"] = df["ts_kst"].dt.hour

    def month_to_season(m: int) -> str:
        if m in (12, 1, 2):
            return "winter"
        elif m in (3, 4, 5):
            return "spring"
        elif m in (6, 7, 8):
            return "summer"
        else:
            return "autumn"

    df["season"] = df["month"].apply(month_to_season)
    return df


def main() -> None:
    input_path = DATA_DIR / "processed_with_preds_both.parquet"

    if not input_path.exists():
        raise FileNotFoundError(f"입력 파일을 찾을 수 없습니다: {input_path}")

    print(f"[INFO] 입력 로드: {input_path}")
    df = pd.read_parquet(input_path)

    # 시간 파생변수 추가
    df = add_time_features(df)

    # 연도별 분리
    df_2123 = df[df["year"].between(2021, 2023)].copy()
    df_2024 = df[df["year"] == 2024].copy()

    print(f"[INFO] 전체 행 수: {len(df):,}")
    print(f"[INFO] 2021-2023 행 수: {len(df_2123):,}")
    print(f"[INFO] 2024 행 수: {len(df_2024):,}")

    # 공통 컬럼
    cols_common = [
        "ts_kst",
        "station_id",
        "station_name",
        "area",
        "pm10",
        "pm25",
        "y_pred_xgb",
        "y_pred_xgb_pm25",
        "year",
        "month",
        "hour",
        "season",
    ]

    # 2021-2023 전용 라벨
    cols_2123_extra = [
        "y_loc_pm10",
        "y_loc_pm25",
        "dust_stage",
    ]

    # ---------- 2021–2023용: NaN → 0 처리 ----------
    cols_2123 = cols_common + cols_2123_extra
    missing_2123 = [c for c in cols_2123 if c not in df_2123.columns]
    if missing_2123:
        raise ValueError(f"2021-2023 export에서 누락된 컬럼: {missing_2123}")

    # 경보/황사 라벨 결측값을 0으로 채움 (QGIS 시각화용)
    for col in ["y_loc_pm10", "y_loc_pm25", "dust_stage"]:
        if col in df_2123.columns:
            df_2123[col] = df_2123[col].fillna(0)

    out_2123 = df_2123[cols_2123].sort_values(["ts_kst", "station_id"])
    out_2123_path = DATA_DIR / "for_qgis_2021_2023.parquet"
    out_2123.to_parquet(out_2123_path, index=False)
    print(f"[INFO] 저장 완료: {out_2123_path} (rows={len(out_2123):,})")

    # ---------- 2024용: 경보/황사 라벨 없이 ----------
    cols_2024 = cols_common
    missing_2024 = [c for c in cols_2024 if c not in df_2024.columns]
    if missing_2024:
        raise ValueError(f"2024 export에서 누락된 컬럼: {missing_2024}")

    out_2024 = df_2024[cols_2024].sort_values(["ts_kst", "station_id"])
    out_2024_path = DATA_DIR / "for_qgis_2024.parquet"
    out_2024.to_parquet(out_2024_path, index=False)
    print(f"[INFO] 저장 완료: {out_2024_path} (rows={len(out_2024):,})")


if __name__ == "__main__":
    main()
