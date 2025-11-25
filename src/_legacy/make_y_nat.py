# src/make_y_nat.py

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"


def main():
    proc_path = DATA_DIR / "processed.parquet"
    events_path = DATA_DIR / "national_alert_events.csv"

    if not proc_path.exists():
        raise FileNotFoundError(f"{proc_path} 가 없습니다. 먼저 make_processed.py 를 실행하세요.")
    if not events_path.exists():
        raise FileNotFoundError(f"{events_path} 가 없습니다. 먼저 make_national_events.py 를 실행하세요.")

    # 1) 공통 테이블 읽기
    df = pd.read_parquet(proc_path).sort_values("ts_kst")

    # ts_kst가 혹시 스트링이면 datetime으로 변환
    if not pd.api.types.is_datetime64_any_dtype(df["ts_kst"]):
        df["ts_kst"] = pd.to_datetime(df["ts_kst"])

    # 2) 이벤트 테이블 읽기
    events = pd.read_csv(
        events_path,
        parse_dates=["start_kst", "end_kst"],
    )

    # 3) 기본 y_nat = 0
    y_nat = df[["ts_kst"]].copy()
    y_nat["y_nat"] = 0

    # 4) 이벤트 구간마다 y_nat 덮어쓰기
    for _, ev in events.iterrows():
        mask = (y_nat["ts_kst"] >= ev["start_kst"]) & (y_nat["ts_kst"] <= ev["end_kst"])
        y_nat.loc[mask, "y_nat"] = int(ev["stage"])

    # 5) 저장 + 분포 확인용 출력
    out = DATA_DIR / "y_nat.parquet"
    y_nat.to_parquet(out, index=False)
    print(f"saved: {out}")
    print("y_nat distribution:")
    print(y_nat["y_nat"].value_counts(dropna=False, normalize=True))


if __name__ == "__main__":
    main()
