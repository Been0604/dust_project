# src/analyze_pm10_performance.py

from pathlib import Path
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
REPORTS_DIR = ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def load_df():
    path = DATA_DIR / "processed_with_preds_both.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found")

    df = pd.read_parquet(path)
    df["ts_kst"] = pd.to_datetime(df["ts_kst"])

    # 시간 관련 컬럼 추가
    df["year"] = df["ts_kst"].dt.year
    df["month"] = df["ts_kst"].dt.month
    df["hour"] = df["ts_kst"].dt.hour
    df["weekday"] = df["ts_kst"].dt.weekday

    season_map = {
        12: "DJF", 1: "DJF", 2: "DJF",
        3: "MAM", 4: "MAM", 5: "MAM",
        6: "JJA", 7: "JJA", 8: "JJA",
        9: "SON", 10: "SON", 11: "SON",
    }
    df["season"] = df["month"].map(season_map)

    return df


def compute_metrics(sub: pd.DataFrame, pred_col: str) -> pd.Series:
    y_true = sub["target_pm10"]
    y_pred = sub[pred_col]

    err = y_pred - y_true
    bias = err.mean()
    mae = err.abs().mean()
    rmse = np.sqrt((err ** 2).mean())

    sst = ((y_true - y_true.mean()) ** 2).sum()
    sse = (err ** 2).sum()
    r2 = 1 - sse / sst if sst > 0 else np.nan

    return pd.Series(
        {
            "n": len(sub),
            "bias": bias,
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
        }
    )


def make_basic_metrics(df: pd.DataFrame, pred_col: str = "y_pred_xgb"):
    # 예측/타깃 있는 행만 사용
    df_valid = df[df[pred_col].notna() & df["target_pm10"].notna()].copy()
    print(f"[{pred_col}] 유효 행 수: {len(df_valid)}")

    # 1) 전체 성능
    overall = compute_metrics(df_valid, pred_col).to_frame().T
    overall.insert(0, "model", pred_col)
    overall.to_csv(REPORTS_DIR / f"metrics_overall_{pred_col}.csv", index=False)

    # 2) 연도별 성능
    by_year = (
        df_valid
        .groupby("year")[["target_pm10", pred_col]]
        .apply(lambda sub: compute_metrics(sub, pred_col))
        .reset_index()
    )
    by_year.insert(0, "model", pred_col)
    by_year.to_csv(REPORTS_DIR / f"metrics_by_year_{pred_col}.csv", index=False)

    # 3) 계절별 성능
    by_season = (
        df_valid
        .groupby("season")[["target_pm10", pred_col]]
        .apply(lambda sub: compute_metrics(sub, pred_col))
        .reset_index()
    )
    by_season.insert(0, "model", pred_col)
    by_season.to_csv(REPORTS_DIR / f"metrics_by_season_{pred_col}.csv", index=False)

    # 4) 시간대별 성능
    by_hour = (
        df_valid
        .groupby("hour")[["target_pm10", pred_col]]
        .apply(lambda sub: compute_metrics(sub, pred_col))
        .reset_index()
    )
    by_hour.insert(0, "model", pred_col)
    by_hour.to_csv(REPORTS_DIR / f"metrics_by_hour_{pred_col}.csv", index=False)

    print("기본 성능 요약 CSV 저장 완료")


def make_event_metrics(df: pd.DataFrame, pred_col: str = "y_pred_xgb"):
    """
    2021–2023 구간에서:
      - y_loc_pm10 별
      - dust_stage 별
      - 고농도 이벤트( target_pm10 >= 100 ) 별
      - station_id 별 성능
    """
    df_ev = df[(df["year"] >= 2021) & (df["year"] <= 2023)].copy()
    df_ev = df_ev[df_ev[pred_col].notna() & df_ev["target_pm10"].notna()]

    print(f"[{pred_col}] 2021–2023 유효 행 수: {len(df_ev)}")

    # 고농도 이벤트 플래그
    df_ev["high_event"] = (df_ev["target_pm10"] >= 100).astype(int)

    group_defs = [
        ("station_id", "by_station"),
        ("y_loc_pm10", "by_y_loc_pm10"),
        ("dust_stage", "by_dust_stage"),
        ("high_event", "by_high_event"),
    ]

    for col, suffix in group_defs:
        if col not in df_ev.columns:
            print(f"컬럼 {col} 없음 → 건너뜀")
            continue

        out = (
            df_ev
            .groupby(col)[["target_pm10", pred_col]]
            .apply(lambda sub: compute_metrics(sub, pred_col))
            .reset_index()
        )
        out.insert(0, "model", pred_col)
        out.to_csv(
            REPORTS_DIR / f"metrics_{suffix}_{pred_col}_2021_2023.csv",
            index=False,
        )
        print(f"{col} 기준 이벤트 성능 테이블 저장 완료")


def main():
    df = load_df()

    # XGB 기준 분석
    pred_col = "y_pred_xgb"
    if pred_col not in df.columns:
        raise KeyError(f"{pred_col} 컬럼이 없습니다. 컬럼명을 확인하세요.")

    make_basic_metrics(df, pred_col)
    make_event_metrics(df, pred_col)

    # RF도 같이 보고 싶으면
    if "y_pred_rf" in df.columns:
        print("\n--- RF 모델에 대해서도 동일 분석 실행 ---\n")
        make_basic_metrics(df, pred_col="y_pred_rf")
        make_event_metrics(df, pred_col="y_pred_rf")


if __name__ == "__main__":
    main()
