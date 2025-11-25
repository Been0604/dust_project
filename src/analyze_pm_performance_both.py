# src/analyze_pm_performance_both.py
#
# processed_with_preds_both.parquet 기준으로
# PM10 + PM25에 대해 연도/계절/시간대/측정소/경보/황사/고농도 이벤트별 성능을
# CSV로 저장하는 스크립트

from pathlib import Path
import warnings

import numpy as np
import pandas as pd

# 귀찮은 FutureWarning 안 보이게
warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
REPORTS_DIR = ROOT / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

# [가정] 고농도 이벤트 기준
# PM10: 100 이상, PM25: 50 이상
HIGH_THRESH = {"pm10": 100.0, "pm25": 50.0}


def load_master() -> pd.DataFrame:
    """최종 마스터 processed_with_preds_both.parquet 로드."""
    path = DATA_DIR / "processed_with_preds_both.parquet"
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_parquet(path)
    df["ts_kst"] = pd.to_datetime(df["ts_kst"])

    # 시간 파생 컬럼 없으면 생성
    if "year" not in df.columns:
        df["year"] = df["ts_kst"].dt.year
    if "month" not in df.columns:
        df["month"] = df["ts_kst"].dt.month
    if "hour" not in df.columns:
        df["hour"] = df["ts_kst"].dt.hour

    # 계절 컬럼
    def month_to_season(m: int) -> str:
        if m in [12, 1, 2]:
            return "winter"
        if m in [3, 4, 5]:
            return "spring"
        if m in [6, 7, 8]:
            return "summer"
        return "fall"

    df["season"] = df["month"].map(month_to_season)
    return df


def compute_metrics(sub: pd.DataFrame, target_col: str, pred_col: str) -> pd.Series:
    """MAE / RMSE / Bias / R2 계산."""
    y_true = sub[target_col].to_numpy()
    y_pred = sub[pred_col].to_numpy()

    err = y_pred - y_true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    bias = float(np.mean(err))

    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan

    return pd.Series(
        {
            "n": len(sub),
            "mae": mae,
            "rmse": rmse,
            "bias": bias,
            "r2": r2,
        }
    )


def make_table(
    df: pd.DataFrame,
    group_cols,
    target_col: str,
    pred_col: str,
) -> pd.DataFrame:
    """group별 성능 테이블 생성."""
    use_cols = []
    if group_cols is not None:
        if isinstance(group_cols, (list, tuple)):
            use_cols.extend(group_cols)
        else:
            use_cols.append(group_cols)
    use_cols.extend([target_col, pred_col])

    sub = df[use_cols].dropna(subset=[target_col, pred_col])

    # 전체(overall)일 때는 groupby 필요 없음
    if group_cols is None:
        s = compute_metrics(sub, target_col, pred_col)
        out = s.to_frame().T
        return out

    if isinstance(group_cols, (list, tuple)):
        by = group_cols
    else:
        by = [group_cols]

    # 여기서 경고 나던 부분 수정:
    # groupby 후 target/pred 두 컬럼만 가지고 apply
    metrics = (
        sub.groupby(by, dropna=False)[[target_col, pred_col]]
        .apply(lambda g: compute_metrics(g, target_col, pred_col))
        .reset_index()
    )
    return metrics


def save_table(df: pd.DataFrame, metric_tag: str, pollutant: str, model_tag: str):
    """reports/ 아래 CSV 저장."""
    fname = f"metrics_{metric_tag}_{pollutant}_y_pred_{model_tag}.csv"
    out_path = REPORTS_DIR / fname
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print("  저장 완료:", out_path, "shape=", df.shape)


def analyze_for_pollutant(df_master: pd.DataFrame, pollutant: str):
    """
    pollutant: 'pm10' 또는 'pm25'
    """
    if pollutant == "pm10":
        target_col = "target_pm10"
        model_to_col = {
            "rf": "y_pred_rf",
            "xgb": "y_pred_xgb",
        }
        y_loc_col = "y_loc_pm10"
    elif pollutant == "pm25":
        target_col = "target_pm25"
        model_to_col = {
            "rf": "y_pred_rf_pm25",
            "xgb": "y_pred_xgb_pm25",
        }
        y_loc_col = "y_loc_pm25"
    else:
        raise ValueError("pollutant는 'pm10' 또는 'pm25'만 허용")

    print(f"\n=== {pollutant.upper()} 분석 시작 ===")

    # 타깃이 있는 행만 사용
    df_all = df_master[df_master[target_col].notna()].copy()

    # 2021–2023 구간 (경보/황사 라벨 있는 구간)
    df_2123 = df_all[df_all["year"].between(2021, 2023)].copy()

    for model_tag, pred_col in model_to_col.items():
        if pred_col not in df_all.columns:
            print(f"  경고: {pred_col} 컬럼이 없어서 {model_tag} 모델 스킵")
            continue

        print(f"\n--- {pollutant.upper()} / 모델={model_tag} ({pred_col}) ---")

        df_valid = df_all[df_all[pred_col].notna()].copy()
        if df_valid.empty:
            print("  예측값이 없어 스킵")
            continue

        # 1) 전체(overall)
        tbl_overall = make_table(df_valid, None, target_col, pred_col)
        save_table(tbl_overall, "overall", pollutant, model_tag)

        # 2) 연도별
        tbl_year = make_table(df_valid, "year", target_col, pred_col)
        save_table(tbl_year, "by_year", pollutant, model_tag)

        # 3) 계절별
        tbl_season = make_table(df_valid, "season", target_col, pred_col)
        save_table(tbl_season, "by_season", pollutant, model_tag)

        # 4) 시간대별
        tbl_hour = make_table(df_valid, "hour", target_col, pred_col)
        save_table(tbl_hour, "by_hour", pollutant, model_tag)

        # 5) 측정소별 (2021–2023 + 예측있는 행)
        df_station = df_2123[df_2123[pred_col].notna()].copy()
        if not df_station.empty:
            tbl_station = make_table(
                df_station,
                ["station_id"],
                target_col,
                pred_col,
            )
            save_table(
                tbl_station,
                "by_station_2021_2023",
                pollutant,
                model_tag,
            )

        # 6) 로컬 경보 라벨별 (y_loc_*), 2021–2023
        if y_loc_col in df_2123.columns:
            df_y_loc = df_2123[
                df_2123[[y_loc_col, pred_col, target_col]].notna().all(axis=1)
            ].copy()
            if not df_y_loc.empty:
                tbl_y_loc = make_table(
                    df_y_loc,
                    [y_loc_col],
                    target_col,
                    pred_col,
                )
                save_table(
                    tbl_y_loc,
                    "by_y_loc_2021_2023",
                    pollutant,
                    model_tag,
                )

        # 7) 황사 단계별 (dust_stage), 2021–2023
        if "dust_stage" in df_2123.columns:
            df_dust = df_2123[
                df_2123[["dust_stage", pred_col, target_col]].notna().all(axis=1)
            ].copy()
            if not df_dust.empty:
                tbl_dust = make_table(
                    df_dust,
                    ["dust_stage"],
                    target_col,
                    pred_col,
                )
                save_table(
                    tbl_dust,
                    "by_dust_stage_2021_2023",
                    pollutant,
                    model_tag,
                )

        # 8) 고농도 이벤트별 (high_event), 2021–2023
        thr = HIGH_THRESH[pollutant]
        df_high = df_2123[df_2123[pred_col].notna()].copy()
        if not df_high.empty:
            df_high["high_event"] = (df_high[target_col] >= thr).astype(int)
            tbl_high = make_table(
                df_high,
                ["high_event"],
                target_col,
                pred_col,
            )
            save_table(
                tbl_high,
                "by_high_event_2021_2023",
                pollutant,
                model_tag,
            )


def main():
    print("=== processed_with_preds_both 로드 ===")
    df = load_master()
    print("master shape:", df.shape)

    analyze_for_pollutant(df, "pm10")
    analyze_for_pollutant(df, "pm25")

    print("\n=== 모든 분석 완료 ===")


if __name__ == "__main__":
    main()
