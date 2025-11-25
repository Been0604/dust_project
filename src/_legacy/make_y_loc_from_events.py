from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

PROCESSED_PATH = DATA_DIR / "processed.parquet"
STATIONS_PATH = DATA_DIR / "stations.csv"
EVENTS_PM10_PATH = DATA_DIR / "local_alert_events_pm10.parquet"


def load_data():
    print(f"[정보] 로드: {PROCESSED_PATH}")
    processed = pd.read_parquet(PROCESSED_PATH)

    print(f"[정보] 로드: {STATIONS_PATH}")
    stations = pd.read_csv(STATIONS_PATH)

    print(f"[정보] 로드: {EVENTS_PM10_PATH}")
    events = pd.read_parquet(EVENTS_PM10_PATH)

    return processed, stations, events


def prepare_processed_with_area(processed: pd.DataFrame, stations: pd.DataFrame) -> pd.DataFrame:
    """processed에 station_id 기준으로 area(동부/서부권역) 붙이기"""

    if "area" not in stations.columns:
        raise ValueError("[에러] stations.csv에 'area' 컬럼이 없습니다. 동부/서부권역을 먼저 채워주세요.")

    # ts_kst를 datetime으로 정리
    if "ts_kst" in processed.columns:
        processed["ts_kst"] = pd.to_datetime(processed["ts_kst"])
    else:
        raise ValueError("[에러] processed.parquet에 ts_kst 컬럼이 없습니다.")

    # station_id 기준으로 area 붙이기
    merged = processed.merge(
        stations[["station_id", "area"]],
        on="station_id",
        how="left",
    )

    if merged["area"].isna().any():
        n_missing = merged["area"].isna().sum()
        print(f"[경고] area가 비어 있는 행이 {n_missing}개 있습니다. stations.csv의 station_id/area를 확인하세요.")

    return merged


def merge_with_events(merged: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """processed+area 테이블에 시간대별 경보 이벤트를 붙여서 y_loc 만들기"""

    # 이벤트 ts_kst도 datetime으로
    events["ts_kst"] = pd.to_datetime(events["ts_kst"])

    # ts_kst + area 기준으로 left join
    df = merged.merge(
        events[["ts_kst", "area", "alert_level"]],
        on=["ts_kst", "area"],
        how="left",
    )

    # 이벤트 없으면 0(평상), 있으면 1/2
    df["alert_level"] = df["alert_level"].fillna(0).astype(int)
    df = df.rename(columns={"alert_level": "y_loc"})

    print("[정보] y_loc 분포:")
    print(df["y_loc"].value_counts(dropna=False).sort_index())

    return df


def save_y_loc(df: pd.DataFrame):
    """y_loc만 추려서 parquet로 저장 (ts_kst, station_id, y_loc)"""

    out = df[["ts_kst", "station_id", "y_loc"]].copy()
    out_path = DATA_DIR / "y_loc.parquet"
    out.to_parquet(out_path, index=False)
    print(f"[완료] y_loc 저장: {out_path} (행 {len(out)}개)")


def main():
    processed, stations, events = load_data()
    merged = prepare_processed_with_area(processed, stations)
    df_with_y = merge_with_events(merged, events)
    save_y_loc(df_with_y)


if __name__ == "__main__":
    main()
