# src/make_processed_with_preds_both.py
#
# processed_with_dust.parquet  +  predictions_full_rf_xgb.parquet
# → PM10 + PM25 예측/타깃/라벨이 모두 들어간 마스터 테이블 생성

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"


def load_base() -> pd.DataFrame:
    """processed_with_dust.parquet 로드."""
    path = DATA_DIR / "processed_with_dust.parquet"
    if not path.exists():
        raise FileNotFoundError(f"processed_with_dust.parquet 없음: {path}")

    df = pd.read_parquet(path)
    df["ts_kst"] = pd.to_datetime(df["ts_kst"])
    return df


def load_predictions() -> pd.DataFrame:
    """predictions_full_rf_xgb.parquet 로드."""
    path = DATA_DIR / "predictions_full_rf_xgb.parquet"
    if not path.exists():
        raise FileNotFoundError(f"predictions_full_rf_xgb.parquet 없음: {path}")

    df = pd.read_parquet(path)
    df["ts_kst"] = pd.to_datetime(df["ts_kst"])
    return df


def main():
    print("=== 1) processed_with_dust 로딩 중 ===")
    df_proc = load_base()
    print("processed_with_dust shape:", df_proc.shape)

    print("\n=== 2) predictions_full_rf_xgb 로딩 중 ===")
    df_pred = load_predictions()
    print("predictions_full shape:", df_pred.shape)
    print("predictions_full columns:", list(df_pred.columns))

    # preds 쪽에 area가 있으면, base 쪽 area를 기준으로 쓸 거니까 제거
    if "area" in df_pred.columns:
        df_pred = df_pred.drop(columns=["area"])

    # base와 preds 간에 겹치는 컬럼이 있으면 preds 쪽 것을 드롭
    # (ts_kst, station_id 는 merge key라 예외)
    overlap = [
        c
        for c in df_pred.columns
        if c in df_proc.columns and c not in ["ts_kst", "station_id"]
    ]
    if overlap:
        print(f"경고: 겹치는 컬럼이 있어서 preds 쪽에서 드롭합니다: {overlap}")
        df_pred = df_pred.drop(columns=overlap)

    print("\n=== 3) merge 중 (ts_kst + station_id 기준) ===")
    merged = df_proc.merge(
        df_pred,
        on=["ts_kst", "station_id"],
        how="left",
        # validate="one_to_one",  # 필요하면 다시 켜기
    )

    print("merge 완료, shape:", merged.shape)

    out_path = DATA_DIR / "processed_with_preds_both.parquet"
    merged.to_parquet(out_path, index=False)

    print(f"\n최종 마스터 저장 완료: {out_path}")
    print("최종 마스터 shape:", merged.shape)

    # 간단 sanity check (PM10 / PM25 둘 다)
    if "y_pred_xgb" in merged.columns:
        n_missing_pm10 = merged["y_pred_xgb"].isna().sum()
        print(f"XGB PM10 예측값이 NaN인 행 수: {n_missing_pm10}")
    else:
        print("주의: y_pred_xgb 컬럼이 없습니다 (PM10 XGB 예측 없음)")

    if "y_pred_xgb_pm25" in merged.columns:
        n_missing_pm25 = merged["y_pred_xgb_pm25"].isna().sum()
        print(f"XGB PM25 예측값이 NaN인 행 수: {n_missing_pm25}")
    else:
        print("주의: y_pred_xgb_pm25 컬럼이 없습니다 (PM25 XGB 예측 없음)")


if __name__ == "__main__":
    main()
