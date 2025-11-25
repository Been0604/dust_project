"""
make_local_alert_events_pm10.py

에어코리아 '미세먼지(PM10) 경보제 발령 현황' 엑셀들을 모아서
→ 대전만 필터
→ 2021~2023 구간만 사용
→ 24시(24:00)는 다음날 00시로 보정
→ 시간 단위(ts_kst)로 펼친 이벤트 테이블을 만들어

data/local_alert_events_pm10.parquet 으로 저장한다.

최종 컬럼:
- ts_kst      : 경보가 유효한 시각 (KST, 1시간 단위)
- area        : 권역 (동부/서부 등)
- y_loc_pm10  : 로컬 PM10 경보 단계 (1=주의보, 2=경보)
  ※ 0은 나중에 PM 시계열과 조인할 때 "경보 없음"으로 채워지는 값.
"""

from pathlib import Path
import pandas as pd


# ---------- 경로 설정 ----------
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
ALERT_DIR = DATA_DIR / "airkorea_pm_alerts"

# PM10 경보 엑셀 파일 목록
ALERT_FILES_PM10 = [
    ALERT_DIR / "airkorea_pm10_alerts_2015_2021.xlsx",
    ALERT_DIR / "airkorea_pm10_alerts_2022.xlsx",
    ALERT_DIR / "airkorea_pm10_alerts_2023.xlsx",
]

# 한글 헤더 → 영어 헤더 매핑
RENAME_COLS = {
    "시도": "province",
    "권역": "area",
    "경보단계": "alert_step",
    "발령날짜": "start_date",
    "발령시각": "start_hour",
    "발령농도": "start_conc",
    "발령기준": "start_basis",
    "해제날짜": "end_date",
    "해제시각": "end_hour",
    "해제농도": "end_conc",
    "해제기준": "end_basis",
    "경과시간": "duration_hours",
    "처리구분": "status",
}


# ---------- 헬퍼 함수들 ----------

def load_alert_excel(path: Path) -> pd.DataFrame:
    """
    에어코리아 '미세먼지 경보제 발령 현황' 엑셀을 읽어서
    - '시도'가 있는 행을 헤더로 사용
    - 그 아래 데이터만 DataFrame으로 반환
    """
    print(f"[정보] 엑셀 로드: {path}")
    raw = pd.read_excel(path, header=None)

    # '시도'가 포함된 행을 찾아서 헤더로 사용
    matches = raw.eq("시도").any(axis=1)
    if not matches.any():
        raise ValueError(f"[에러] 파일 {path}에서 '시도' 헤더 행을 찾지 못했습니다.")

    header_row_idx = raw.index[matches][0]
    header = raw.iloc[header_row_idx]
    df = raw.iloc[header_row_idx + 1 :].copy()
    df.columns = header

    # 전부 NaN인 행 제거
    df = df.dropna(how="all")

    return df


def build_ts_with_24(date_series: pd.Series, hour_series: pd.Series) -> pd.Series:
    """
    날짜 + 시(hour) → timestamp
    - hour가 24인 경우: 다음날 00시로 보정
      예: 2023-03-23, 24 → 2023-03-24 00:00
    """
    base_date = pd.to_datetime(date_series, errors="coerce").dt.normalize()
    hour = pd.to_numeric(hour_series, errors="coerce")

    mask_24 = hour == 24
    hour = hour.where(~mask_24, 0)

    ts = base_date + pd.to_timedelta(hour, unit="h")
    ts.loc[mask_24] = ts.loc[mask_24] + pd.Timedelta(days=1)

    return ts


def preprocess_pm10_alerts(df: pd.DataFrame) -> pd.DataFrame:
    """
    - 대전만 필터
    - 컬럼 영어로 리네임
    - start_ts / end_ts / y_loc_pm10 생성
    - 2021~2023만 사용
    """
    if "시도" not in df.columns:
        raise ValueError("[에러] '시도' 컬럼이 없습니다. 엑셀 구조를 확인해 주세요.")

    # 대전만
    df = df[df["시도"] == "대전"].copy()
    if df.empty:
        print("[경고] 대전 데이터가 비어 있습니다.")
        return df

    # 컬럼 리네임
    df = df.rename(columns=RENAME_COLS)

    # 날짜/시간 컬럼 정리
    df["start_date"] = pd.to_datetime(df["start_date"], errors="coerce")
    df["end_date"] = pd.to_datetime(df["end_date"], errors="coerce")
    df["start_hour"] = pd.to_numeric(df["start_hour"], errors="coerce")
    df["end_hour"] = pd.to_numeric(df["end_hour"], errors="coerce")

    # 유효하지 않은 것 제거
    df = df.dropna(subset=["start_date", "end_date", "start_hour", "end_hour"])

    # 24시 보정 포함해서 timestamp 생성
    df["start_ts"] = build_ts_with_24(df["start_date"], df["start_hour"])
    df["end_ts"] = build_ts_with_24(df["end_date"], df["end_hour"])

    # 2021~2023년 구간만 사용 (발령 시각 기준)
    mask_year = (df["start_ts"].dt.year >= 2021) & (df["start_ts"].dt.year <= 2023)
    df = df[mask_year].copy()
    if df.empty:
        print("[경고] 2021~2023 대전 PM10 경보 데이터가 없습니다.")
        return df

    # 경보단계 → 숫자 라벨 (y_loc_pm10)
    if "alert_step" not in df.columns:
        raise ValueError("[에러] '경보단계' → 'alert_step' 리네임이 안 된 것 같습니다.")

    step_map = {"주의보": 1, "경보": 2}
    df["y_loc_pm10"] = df["alert_step"].map(step_map)
    df["y_loc_pm10"] = df["y_loc_pm10"].fillna(1).astype(int)

    print("[정보] 대전 행 개수:", len(df))
    print(
        df[
            ["province", "area", "alert_step", "start_ts", "end_ts", "duration_hours", "y_loc_pm10"]
        ].head()
    )

    return df


def expand_to_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """
    각 경보 이벤트(발령 ~ 해제)를 시간 단위로 풀어서
    ts_kst, area, y_loc_pm10 컬럼을 가진 테이블 생성.

    주의:
    - 보통 '발령시각 = 5, 해제시각 = 14, 경과시간 = 9' 이런 식이라
      5,6,7,8,9,10,11,12,13 → 9시간으로 보는 게 자연스러움.
      그래서 date_range는 [start_ts, end_ts) (end_ts는 포함 안 함)으로 처리.
    """
    rows = []

    for row in df.itertuples():
        rng = pd.date_range(row.start_ts, row.end_ts, freq="h", inclusive="left")

        for ts in rng:
            rows.append(
                {
                    "ts_kst": ts,
                    "area": row.area,
                    "y_loc_pm10": row.y_loc_pm10,
                }
            )

    events = pd.DataFrame(rows)
    if events.empty:
        print("[경고] 시간대별 이벤트가 비어 있습니다.")
        return events

    # 같은 시각·권역에 중복 있으면 가장 높은 단계만 남기기
    events = (
        events.groupby(["ts_kst", "area"], as_index=False)["y_loc_pm10"]
        .max()
        .sort_values(["ts_kst", "area"])
    )

    print("[정보] 시간대별 이벤트 행 개수:", len(events))
    print(events.head())

    return events


# ---------- main ----------

def main():
    # 1) 여러 엑셀 읽어서 하나로 합치기
    dfs = []
    for path in ALERT_FILES_PM10:
        if path.exists():
            df_raw = load_alert_excel(path)
            dfs.append(df_raw)
        else:
            print(f"[경고] 파일 없음: {path}")

    if not dfs:
        print("[에러] 읽을 수 있는 PM10 경보제 엑셀이 없습니다.")
        return

    df_all = pd.concat(dfs, ignore_index=True)

    # 2) 대전만 추출 + 전처리 (2021~2023)
    df_dj = preprocess_pm10_alerts(df_all)
    if df_dj.empty:
        print("[에러] 대전 PM10 경보 데이터 없음. 중단.")
        return

    # 3) 시간대별로 expand
    events = expand_to_hourly(df_dj)
    if events.empty:
        print("[에러] 시간대별 이벤트 생성 실패. 중단.")
        return

    # 4) 저장
    out_path = DATA_DIR / "local_alert_events_pm10.parquet"
    events.to_parquet(out_path, index=False)
    print(f"[완료] 대전 PM10 시간대별 경보 이벤트 저장: {out_path}")


if __name__ == "__main__":
    main()
