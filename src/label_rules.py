# src/label_rules.py

from pathlib import Path
import pandas as pd


DATA_DIR = Path("data")
PROCESSED_PATH = DATA_DIR / "processed.parquet"
Y_LOC_PATH = DATA_DIR / "y_loc.parquet"
Y_NAT_PATH = DATA_DIR / "y_nat.parquet"

# ---- PM10 단계 기준 (임시 값) ----
# TODO: 실제 황사 주의보/경보 기준으로 나중에 바꿔도 됨
PM10_WARN = 150   # 주의보 가정
PM10_ALERT = 300  # 경보 가정

# 최소 몇 개 측정소가 기준을 넘으면 경보로 볼지
MIN_STATION_FRAC = 0.4  # 전체의 40%
MIN_STATIONS_ABS = 3    # 최소 3개는 넘겨야 함


def classify_pm10_stage(pm10: float) -> int:
    """pm10 값 하나를 0/1/2 단계로 변환."""
    if pd.isna(pm10):
        return 0
    if pm10 >= PM10_ALERT:
        return 2
    if pm10 >= PM10_WARN:
        return 1
    return 0


def make_y_loc() -> pd.DataFrame:
    """
    지역 기준(y_loc) 라벨 생성.

    - processed.parquet에서 ts_kst, station_id, pm10 사용
    - 각 시각마다:
        · 1단계 이상인 측정소 개수
        · 2단계 이상인 측정소 개수
      를 센 뒤,
        · 2단계 조건 만족(연속 2시간)이면 2
        · 아니고 1단계 조건 만족(연속 2시간)이면 1
        · 아니면 0
    """
    df = pd.read_parquet(PROCESSED_PATH)

    # 필요한 컬럼만
    cols_need = ["ts_kst", "station_id", "pm10"]
    for c in cols_need:
        if c not in df.columns:
            raise ValueError(f"processed.parquet에 '{c}' 컬럼이 없습니다.")
    df = df[cols_need].copy()

    # 단계화
    df["stage"] = df["pm10"].apply(classify_pm10_stage)

    # 피벗: index=ts_kst, columns=station_id, values=stage
    pivot = df.pivot(index="ts_kst", columns="station_id", values="stage").sort_index()

    n_stations = pivot.shape[1]
    min_stations = max(MIN_STATIONS_ABS, int(n_stations * MIN_STATION_FRAC))

    # 각 시각별로 1단계 이상 / 2단계 이상인 측정소 숫자
    cnt_ge1 = (pivot >= 1).sum(axis=1)
    cnt_ge2 = (pivot >= 2).sum(axis=1)

    cond1 = cnt_ge1 >= min_stations
    cond2 = cnt_ge2 >= min_stations

    # "연속 2시간 유지" 조건: 현재 시각과 직전 시각이 모두 True인 경우
    cond1_2h = cond1 & cond1.shift(1, fill_value=False)
    cond2_2h = cond2 & cond2.shift(1, fill_value=False)

    # 최종 y_loc 시리즈
    y_loc = pd.Series(0, index=pivot.index, name="y_loc")
    y_loc[cond1_2h] = 1
    y_loc[cond2_2h] = 2

    out = y_loc.reset_index()  # ts_kst, y_loc
    out.to_parquet(Y_LOC_PATH, index=False)
    print(f"saved y_loc: {Y_LOC_PATH} / shape={out.shape}")

    return out


def check_class_dist(path: Path):
    """0/1/2 클래스 분포 간단 확인."""
    df = pd.read_parquet(path)
    col = "y_loc" if "y_loc" in df.columns else df.columns[-1]
    counts = df[col].value_counts(normalize=True).sort_index() * 100
    print(f"class distribution for {col}:")
    for k, v in counts.items():
        print(f"  {k}: {v:.2f}%")


# -------------------- y_nat (전국 특보) 뼈대 --------------------


def make_y_nat() -> pd.DataFrame:
    """
    전국 황사 특보 기반 y_nat 생성 (뼈대).
    TODO: national_alert_events.csv 형식 정한 뒤 구현.

    예상 포맷:
        ts_start, ts_end, level  (level=0/1/2)
    """
    events_path = DATA_DIR / "national_alert_events.csv"
    if not events_path.exists():
        print("national_alert_events.csv 가 아직 없어 y_nat은 건너뜁니다.")
        return pd.DataFrame()

    events = pd.read_csv(events_path)
    events["ts_start"] = pd.to_datetime(events["ts_start"])
    events["ts_end"] = pd.to_datetime(events["ts_end"])

    # 타임라인 생성: processed와 같은 시간축 사용
    base = pd.read_parquet(PROCESSED_PATH)[["ts_kst"]].drop_duplicates().sort_values("ts_kst")
    base["y_nat"] = 0

    for _, row in events.iterrows():
        mask = (base["ts_kst"] >= row["ts_start"]) & (base["ts_kst"] <= row["ts_end"])
        base.loc[mask, "y_nat"] = row["level"]

    base.to_parquet(Y_NAT_PATH, index=False)
    print(f"saved y_nat: {Y_NAT_PATH} / shape={base.shape}")
    return base


if __name__ == "__main__":
    # 1) y_loc 생성 & 분포 확인
    y_loc = make_y_loc()
    check_class_dist(Y_LOC_PATH)

    # 2) y_nat은 national_alert_events.csv 생기면 켜기
    # make_y_nat()
