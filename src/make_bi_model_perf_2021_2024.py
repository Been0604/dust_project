# src/make_bi_model_perf_2021_2024.py
#
# processed_with_preds_both.parquet을 이용해서
# RF/XGB, PM10/PM2.5에 대한 연도·조건별 RMSE/MAE/R² 요약 테이블 생성.
#
# 출력:
#   data/bi/bi_model_perf_2021_2024.csv
#   data/bi/bi_model_perf_2021_2024.parquet

from pathlib import Path
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
BI_DIR = DATA_DIR / "bi"

# 예측 컬럼 이름 매핑 (processed_with_preds_both 기준)
PRED_COLS = {
    "pm10": {
        "y": "target_pm10",
        "rf": "y_pred_rf",          # PM10 RF 예측
        "xgb": "y_pred_xgb",        # PM10 XGB 예측
    },
    "pm25": {
        "y": "target_pm25",
        "rf": "y_pred_rf_pm25",     # PM2.5 RF 예측
        "xgb": "y_pred_xgb_pm25",   # PM2.5 XGB 예측
    },
}


def compute_metrics(y_true, y_pred) -> dict:
    """RMSE, MAE, R², n 계산 (옛날 sklearn 버전 대응)."""
    # NaN 제거
    import numpy as np

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    if len(y_true) == 0:
        return {"n_samples": 0, "rmse": None, "mae": None, "r2": None}

    mse = mean_squared_error(y_true, y_pred)  # squared=False 안 씀
    rmse = mse ** 0.5
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    return {
        "n_samples": int(len(y_true)),
        "rmse": float(rmse),
        "mae": float(mae),
        "r2": float(r2),
    }


def main():
    BI_DIR.mkdir(parents=True, exist_ok=True)

    pm_path = DATA_DIR / "processed_with_preds_both.parquet"
    if not pm_path.exists():
        raise FileNotFoundError(pm_path)

    df = pd.read_parquet(pm_path)

    # 필요한 컬럼 체크
    required_cols = {"ts_kst", "y_loc_pm10", "y_loc_pm25", "dust_stage"}
    for t_name, cols in PRED_COLS.items():
        required_cols.add(cols["y"])
        required_cols.add(cols["rf"])
        required_cols.add(cols["xgb"])

    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"processed_with_preds_both에 없는 컬럼: {missing}")

    # 연도 추출
    df["year"] = df["ts_kst"].dt.year

    # 라벨 NaN을 0으로 치환 (마스크 계산 편하게)
    df["y_loc_pm10"] = df["y_loc_pm10"].fillna(0)
    df["y_loc_pm25"] = df["y_loc_pm25"].fillna(0)
    df["dust_stage"] = df["dust_stage"].fillna(0)

    # subset 정의
    def subset_all(d, t_name):
        return pd.Series(True, index=d.index)

    def subset_loc_alert(d, t_name):
        col = "y_loc_pm10" if t_name == "pm10" else "y_loc_pm25"
        return d[col] > 0

    def subset_no_loc_alert(d, t_name):
        col = "y_loc_pm10" if t_name == "pm10" else "y_loc_pm25"
        return d[col] == 0

    def subset_dust(d, t_name):
        return d["dust_stage"] > 0

    def subset_no_dust(d, t_name):
        return d["dust_stage"] == 0

    subsets = [
        ("all", subset_all),
        ("loc_alert", subset_loc_alert),
        ("no_loc_alert", subset_no_loc_alert),
        ("dust", subset_dust),
        ("no_dust", subset_no_dust),
    ]

    rows = []

    for year in sorted(df["year"].unique()):
        if year < 2021 or year > 2024:
            continue

        df_y = df[df["year"] == year]

        for t_name, cols in PRED_COLS.items():
            y_true = df_y[cols["y"]]

            for model_name in ["rf", "xgb"]:
                pred_col = cols[model_name]

                for subset_name, subset_fn in subsets:
                    # 2024는 라벨 기반 subset 제외 (all만)
                    if year == 2024 and subset_name != "all":
                        continue

                    mask = subset_fn(df_y, t_name)
                    y = y_true[mask]
                    y_hat = df_y.loc[mask, pred_col]

                    metrics = compute_metrics(y, y_hat)
                    if metrics["n_samples"] == 0:
                        continue

                    rows.append(
                        {
                            "year": year,
                            "subset": subset_name,
                            "target": t_name,   # 'pm10' or 'pm25'
                            "model": model_name,  # 'rf' or 'xgb'
                            **metrics,
                        }
                    )

    result = pd.DataFrame(rows).sort_values(
        ["target", "model", "subset", "year"]
    ).reset_index(drop=True)

    out_csv = BI_DIR / "bi_model_perf_2021_2024.csv"
    out_parquet = BI_DIR / "bi_model_perf_2021_2024.parquet"

    result.to_csv(out_csv, index=False, encoding="utf-8-sig")
    result.to_parquet(out_parquet, index=False)

    print(f"Saved: {out_csv}")
    print(f"Saved: {out_parquet}")


if __name__ == "__main__":
    main()
