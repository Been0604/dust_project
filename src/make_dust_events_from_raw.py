# src/make_dust_events_from_raw.py
#
# 목적:
# - data/national_alert_raw.csv 에서
#   "tmArea"에 '대전'이 포함된 황사주의보 발생정보만 추출
# - dataTime을 ts_kst(datetime)으로 바꾸고
#   dust_stage (0/1) 라벨을 만든다.
#   ※ 이 API는 애초에 "황사주의보 발생정보"라서,
#      단계 구분(주의보/경보)이 아니라 "황사 특보 발생 여부"만 본다고 가정.
#
# 출력:
# - data/national_alert_events_daejeon.parquet
#   (컬럼: ts_kst, dust_stage)

from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(r"C:\Users\wachu\dust_project")
DATA_DIR = PROJECT_ROOT / "data"


def load_raw() -> pd.DataFrame:
    path = DATA_DIR / "national_alert_raw.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} 가 없습니다. fetch_national_alerts 먼저 실행해 주세요.")
    df = pd.read_csv(path)
    return df


def filter_daejeon(df: pd.DataFrame) -> pd.DataFrame:
    # tmArea, dataTime, year 컬럼이 있다고 가정
    required_cols = ["tmArea", "dataTime", "year"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(f"다음 컬럼이 없습니다: {missing}")

    # '대전'이 포함된 tmArea만 필터 (예: '대전, 세종, 충남' 이런 식도 포함)
    mask_dj = df["tmArea"].astype(str).str.contains("대전", na=False)
    dj = df[mask_dj].copy()

    # dataTime → ts_kst(datetime)
    dj["ts_kst"] = pd.to_datetime(dj["dataTime"])

    # 황사주의보 발생 여부 라벨: 이 테이블에 있는 시점은 모두 '황사 특보 있음'
    dj["dust_stage"] = 1  # 1 = 황사 특보 존재, 0 = 없음 (나중에 merge 후 fillna로 처리)

    # 중복 ts_kst가 있으면 한 시각에 dust_stage=1만 유지하면 되므로 unique 처리
    dj = (
        dj[["ts_kst", "dust_stage"]]
        .drop_duplicates()
        .sort_values("ts_kst")
        .reset_index(drop=True)
    )

    return dj


def main():
    print("▶ national_alert_raw.csv 로드")
    df_raw = load_raw()
    print("raw shape:", df_raw.shape)

    print("▶ tmArea에 '대전'이 포함된 행만 필터")
    dj_events = filter_daejeon(df_raw)
    print("daejeon events shape:", dj_events.shape)
    print(dj_events.head())

    out_path = DATA_DIR / "national_alert_events_daejeon.parquet"
    dj_events.to_parquet(out_path, index=False)
    print("\n✅ 저장 완료:", out_path)


if __name__ == "__main__":
    main()
