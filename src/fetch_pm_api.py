# src/fetch_pm_api.py
# 에어코리아 실시간 대기오염정보 → data/raw_pm.csv
# 측정소 목록은 data/stations.csv에서 읽어옴

from pathlib import Path
import requests
import pandas as pd

CONFIG_DIR = Path("config")
DATA_DIR = Path("data")

AIRKOREA_KEY = CONFIG_DIR.joinpath("airkorea_key.txt").read_text().strip()

# 에어코리아 실시간 측정소별 대기오염정보 API
BASE_URL = (
    "http://apis.data.go.kr/B552584/"
    "ArpltnInforInqireSvc/getMsrstnAcctoRltmMesureDnsty"
)

def parse_ts(ts_str: str) -> pd.Timestamp:
    """
    dataTime 문자열 -> pandas Timestamp
    '2025-11-19 24:00' 같은 값은 '다음날 00:00'으로 보정.
    """
    date_part, time_part = ts_str.split()  # "2025-11-19", "24:00" 이런 식

    if time_part == "24:00":
        base = pd.to_datetime(date_part)       # 2025-11-19 00:00
        return base + pd.Timedelta(days=1)     # 2025-11-20 00:00
    else:
        # 정상 시간은 그대로 처리
        return pd.to_datetime(ts_str)

def load_stations():
    """
    data/stations.csv에서 station_id, station_name 읽어서
    [("DJ01", "읍내동"), ...] 리스트로 반환.
    """
    path = DATA_DIR / "stations.csv"
    df = pd.read_csv(path)

    pairs = list(
        df[["station_id", "station_name"]].itertuples(index=False, name=None)
    )
    print(f"→ stations.csv에서 측정소 {len(pairs)}개 로드")
    return pairs

def fetch_pm_for_station(station_id: str, station_name: str) -> pd.DataFrame:
    """한 측정소에 대한 최근 PM10 시계열을 DataFrame으로 가져오기."""
    params = {
        "serviceKey": AIRKOREA_KEY,
        "stationName": station_name,
        "dataTerm": "DAILY",      # DAILY / MONTH / 3MONTH 등
        "pageNo": 1,
        "numOfRows": 100,
        "returnType": "json",
        "ver": "1.3",
    }

    print(f"  → 요청: {station_name} ({station_id})")
    resp = requests.get(BASE_URL, params=params)
    resp.raise_for_status()
    data = resp.json()

    items = data["response"]["body"]["items"]

    records = []
    for item in items:
        ts_raw = item["dataTime"]       # "2025-11-19 24:00" 같은 거
        pm10_str = item["pm10Value"]    # "42" or "-"

        if pm10_str in ("-", "", None):
            continue

        ts = parse_ts(ts_raw)

        records.append(
            {
                "ts_kst": ts,
                "station_id": station_id,
                "pm10": float(pm10_str),
            }
        )

    df = pd.DataFrame(records)

    if not df.empty:
        # 이미 Timestamp라 바로 정렬 + 포맷만 문자열로 변환
        df = df.sort_values("ts_kst")
        df["ts_kst"] = df["ts_kst"].dt.strftime("%Y-%m-%d %H:%M")

    print(f"    → {station_name}: {len(df)}행")
    return df

def main():
    stations = load_stations()

    all_df = []
    for station_id, station_name in stations:
        df = fetch_pm_for_station(station_id, station_name)
        if not df.empty:
            all_df.append(df)

    if not all_df:
        print("❌ 어떤 측정소에서도 데이터를 못 가져옴. 파라미터 확인 필요.")
        return

    full = pd.concat(all_df, ignore_index=True)

    out_path = DATA_DIR / "raw_pm.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    full.to_csv(out_path, index=False, encoding="utf-8-sig")

    print(f"\n✅ 저장 완료: {out_path} (행 {len(full)}개)")

if __name__ == "__main__":
    main()
