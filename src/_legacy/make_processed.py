# src/make_processed.py

from pathlib import Path
import pandas as pd

# [여기만 나중에 네 파일 이름에 맞게 수정하면 됨]
PM_PATH = Path("data/raw_pm.csv")            # PM10 데이터
WEATHER_PATH = Path("data/raw_weather.csv")  # 기상 데이터
OUT_PATH = Path("data/processed.parquet")


def load_pm(pm_path: Path) -> pd.DataFrame:
    """PM 테이블 로드: ts_kst, station_id, pm10 가정"""
    pm = pd.read_csv(pm_path)

    # ts_kst를 datetime으로 변환
    pm["ts_kst"] = pd.to_datetime(pm["ts_kst"])

    # 시간 & 지점 기준으로 정렬
    pm = pm.sort_values(["ts_kst", "station_id"])

    return pm


def make_hourly_grid(pm: pd.DataFrame) -> pd.DataFrame:
    """
    시간(ts_kst) x station_id 전체 조합으로 1시간 간격 그리드 만들기.
    PM 값은 보간하지 않고 그대로 둔다. (빈 값 NaN 허용)
    """
    # ⬇ 여기 'H' → 'h'
    start = pm["ts_kst"].min().floor("h")
    end = pm["ts_kst"].max().ceil("h")

    # 1시간 간격 시간축
    # ⬇ freq="1H" → "1h"
    full_time_index = pd.date_range(start, end, freq="1h")

    stations = pm["station_id"].unique()

    # 시간 x 지점 전체 그리드
    full_index = pd.MultiIndex.from_product(
        [full_time_index, stations],
        names=["ts_kst", "station_id"],
    )

    pm_full = (
        pm.set_index(["ts_kst", "station_id"])
          .reindex(full_index)
          .reset_index()
    )

    return pm_full


def load_and_interp_weather(weather_path: Path) -> pd.DataFrame:
    """
    기상 테이블 로드 + 1시간 간격으로 재샘플 + 선형 보간.
    ts_kst, temp, rh, wind_spd, wind_dir, rain, pressure 가정.
    """
    w = pd.read_csv(weather_path)

    w["ts_kst"] = pd.to_datetime(w["ts_kst"])
    w = w.sort_values("ts_kst")

    # ts_kst를 인덱스로 설정
    w = w.set_index("ts_kst")

    # 1시간 간격 시간 축으로 재인덱싱
    # ⬇ 여기들도 'h'로
    start = w.index.min().floor("h")
    end = w.index.max().ceil("h")
    full_time_index = pd.date_range(start, end, freq="1h")
    w = w.reindex(full_time_index)

    # 숫자형 컬럼만 선형 보간 (시간 기준)
    num_cols = w.select_dtypes(include="number").columns
    w[num_cols] = w[num_cols].interpolate(method="time", limit_direction="both")

    # 다시 ts_kst 컬럼으로 복원
    w = w.reset_index().rename(columns={"index": "ts_kst"})

    return w


def main():
    # 1) PM 로드
    pm = load_pm(PM_PATH)
    print(f"PM raw: {pm.shape}")

    # 2) 1시간 그리드 만들기 (PM 보간 X)
    pm_full = make_hourly_grid(pm)
    print(f"PM hourly grid: {pm_full.shape}")

    # 3) 기상 로드 + 보간
    weather = load_and_interp_weather(WEATHER_PATH)
    print(f"Weather hourly: {weather.shape}")

    # 4) ts_kst 기준으로 조인
    df = pm_full.merge(weather, on="ts_kst", how="left")

    # 5) 최종 컬럼 순서 정리
    cols = [
        "ts_kst", "station_id", "pm10",
        "temp", "rh", "wind_spd", "wind_dir", "rain", "pressure",
    ]
    cols = [c for c in cols if c in df.columns]
    df = df[cols]

    # 6) Parquet로 저장
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PATH, index=False)
    print(f"saved: {OUT_PATH} / shape={df.shape}")


if __name__ == "__main__":
    main()
