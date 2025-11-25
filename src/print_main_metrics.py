# src/print_main_metrics.py
#
# reports 폴더에 저장된 성능 지표 CSV를 읽어서
# MAE / RMSE / R2를 깔끔하게 출력하는 스크립트.
#
# 현재 대상:
#   - PM10 XGB
#   - PM25 XGB
#   - PM10 RF
#   - PM25 RF

from pathlib import Path
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT / "reports"

# ============================================================
# [여기]만 바꾸면, 어떤 CSV를 읽을지 / 라벨을 어떻게 붙일지 조정 가능
# ============================================================
METRIC_FILES = {
    "PM10 XGB (overall)": "metrics_overall_pm10_y_pred_xgb.csv",
    "PM25 XGB (overall)": "metrics_overall_pm25_y_pred_xgb.csv",
    "PM10 RF (overall)":  "metrics_overall_pm10_y_pred_rf.csv",
    "PM25 RF (overall)":  "metrics_overall_pm25_y_pred_rf.csv",
}
# 파일명이 다르면 오른쪽 값만 실제 이름에 맞게 고쳐주면 됨.
# 예: "metrics_overall_pm10_y_pred_random_forest.csv" 처럼.


def load_metrics(filename: str) -> pd.DataFrame:
    path = REPORTS_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {path}")
    df = pd.read_csv(path)
    print(f"[INFO] 로드 완료: {path}")
    print(f"[INFO] 컬럼: {list(df.columns)}")
    return df


def print_from_long_format(df: pd.DataFrame, label: str) -> bool:
    """
    long 형식 시도:
    - metric / value (또는 score) 구조일 때 사용
    성공하면 True, 아니면 False 반환
    """
    lower_cols = {c.lower(): c for c in df.columns}

    if "metric" not in lower_cols:
        return False

    # 값 컬럼 후보
    value_col_name = None
    for cand in ["value", "score"]:
        if cand in lower_cols:
            value_col_name = lower_cols[cand]
            break

    if value_col_name is None:
        return False

    metric_col_name = lower_cols["metric"]

    target_metrics = ["MAE", "RMSE", "R2"]
    sub = df[df[metric_col_name].str.upper().isin(target_metrics)].copy()
    if sub.empty:
        return False

    print(f"\n=== {label} (long format) ===")
    for m in target_metrics:
        row = sub[sub[metric_col_name].str.upper() == m]
        if not row.empty:
            v = row[value_col_name].iloc[0]
            print(f"{m}: {v:.4f}")
    print()
    return True


def print_from_wide_format(df: pd.DataFrame, label: str) -> bool:
    """
    wide 형식 시도:
    - mae / rmse / r2 같은 컬럼이 가로로 있을 때
    """
    lower_cols = {c.lower(): c for c in df.columns}

    def find_col(candidates):
        for cand in candidates:
            if cand in lower_cols:
                return lower_cols[cand]
        return None

    mae_col = find_col(["mae", "mean_absolute_error"])
    rmse_col = find_col(["rmse", "root_mean_squared_error", "rmse_test"])
    r2_col = find_col(["r2", "r2_score", "r_squared"])

    if not any([mae_col, rmse_col, r2_col]):
        return False

    row = df.iloc[0]

    print(f"\n=== {label} (wide format) ===")
    if mae_col:
        print(f"MAE: {row[mae_col]:.4f}")
    if rmse_col:
        print(f"RMSE: {row[rmse_col]:.4f}")
    if r2_col:
        print(f"R2: {row[r2_col]:.4f}")
    print()
    return True


def extract_main_metrics(df: pd.DataFrame, label: str) -> None:
    """
    long 형식 → wide 형식 순으로 시도.
    둘 다 실패하면 컬럼/앞부분 샘플만 보여준다.
    """
    if print_from_long_format(df, label):
        return
    if print_from_wide_format(df, label):
        return

    # 둘 다 실패한 경우: 디버그용 출력
    print(f"\n[WARN] {label}에서 MAE/RMSE/R2를 자동으로 찾지 못했습니다.")
    print("[WARN] CSV 구조를 한 번 눈으로 확인해봐야 합니다.")
    print(f"[DEBUG] 컬럼 목록: {list(df.columns)}")
    print("[DEBUG] head(5):")
    print(df.head(5))
    print()


def main():
    for label, filename in METRIC_FILES.items():
        df = load_metrics(filename)
        extract_main_metrics(df, label)


if __name__ == "__main__":
    main()
