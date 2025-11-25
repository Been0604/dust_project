# src/make_national_events.py

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"


def main():
    raw_path = DATA_DIR / "national_alert_raw.csv"
    if not raw_path.exists():
        raise FileNotFoundError(
            f"{raw_path} 가 없습니다. 먼저 fetch_national_alerts.py 를 실행하세요."
        )

    df = pd.read_csv(raw_path)

    required_cols = ["dataTime", "tmArea"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(
                f"'{col}' 컬럼이 없습니다. 실제 컬럼: {df.columns.tolist()}"
            )

    # 🔹 1) 전체 행 수 체크
    print("총 행 수:", len(df))

    # 🔹 2) '대전' 포함된 행만 필터
    mask_daejeon = df["tmArea"].str.contains("대전", na=False)
    print("대전 포함 행 수:", mask_daejeon.sum())

    df = df[mask_daejeon].copy()

    if df.empty:
        print("⚠ 대전이 포함된 황사주의보 기록이 없습니다.")
        events_empty = pd.DataFrame(
            columns=["start_kst", "end_kst", "stage", "note"]
        )
        out_empty = DATA_DIR / "national_alert_events.csv"
        events_empty.to_csv(out_empty, index=False)
        print(f"saved EMPTY events: {out_empty}")
        return

    # 3) 날짜 파싱
    df["date"] = pd.to_datetime(df["dataTime"], format="%Y-%m-%d")

    # 4) 날짜별 1행으로 축약
    days = (
        df.sort_values("date")
          .groupby("date", as_index=False)
          .agg({"tmCnt": "max", "tmArea": "first"})
    )

    # 5) 하루 전체를 이벤트로
    days["start_kst"] = days["date"].dt.normalize()
    days["end_kst"] = (
        days["start_kst"] + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    )

    # 6) stage = 1
    days["stage"] = 1

    # 7) note = tmArea
    days["note"] = days["tmArea"].fillna("")

    events = days[["start_kst", "end_kst", "stage", "note"]].copy()

    out = DATA_DIR / "national_alert_events.csv"
    events.to_csv(out, index=False)
    print(f"saved events: {out}")
    print(events.head())


if __name__ == "__main__":
    main()
