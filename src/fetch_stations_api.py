# src/fetch_stations_api.py
# 에어코리아 "측정소정보" API로 대전 측정소 목록 → data/stations.csv 저장

from pathlib import Path
import requests
import pandas as pd

CONFIG_DIR = Path("config")
AIRKOREA_KEY = CONFIG_DIR.joinpath("airkorea_key.txt").read_text().strip()

# 측정소정보 API URL (에어코리아)
BASE_URL = (
    "http://apis.data.go.kr/B552584/"
    "MsrstnInfoInqireSvc/getMsrstnList"
)

def main():
    params = {
    "serviceKey": AIRKOREA_KEY,
    "returnType": "json",
    "numOfRows": 100,
    "pageNo": 1,
    "addr": "대전",          # 주소에 '대전'이 들어가는 측정소만 필터
}

    print("→ 대전 측정소 목록 요청 중...")
    resp = requests.get(BASE_URL, params=params)
    resp.raise_for_status()
    data = resp.json()

    items = data["response"]["body"]["items"]

    rows = []
    for idx, item in enumerate(items, start=1):
        rows.append(
            {
                # 우리 쪽에서 쓸 내부 ID (DJ01, DJ02 ...)
                "station_id": f"DJ{idx:02d}",
                # 에어코리아 공식 측정소 이름
                "station_name": item["stationName"],
                # 참고용 정보들 (주소 & TM좌표). 나중에 lon/lat로 변환할 수도 있음.
                "addr": item.get("addr"),
                "dmX": item.get("dmX"),
                "dmY": item.get("dmY"),
            }
        )

    df = pd.DataFrame(rows)

    out_path = Path("data") / "stations.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False, encoding="utf-8-sig")

    print(f"✅ 대전 측정소 {len(df)}개를 {out_path} 에 저장")
    print(df[["station_id", "station_name"]])

if __name__ == "__main__":
    main()
