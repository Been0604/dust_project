from pathlib import Path

import pandas as pd
import requests

# 프로젝트 루트 / data / config 경로
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
CONFIG_DIR = ROOT / "config"

DATA_DIR.mkdir(exist_ok=True)

# 🔑 config/airkorea_key.txt 에서 서비스키 읽기
KEY_PATH = CONFIG_DIR / "airkorea_key.txt"
with open(KEY_PATH, "r", encoding="utf-8") as f:
    SERVICE_KEY = f.read().strip()

# 황사주의보 발생정보 조회 API 엔드포인트
BASE_URL = (
    "http://apis.data.go.kr/B552584/"
    "OzYlwsndOccrrncInforInqireSvc/getYlwsndAdvsryOccrrncInfo"
)


def fetch_year(year: int) -> pd.DataFrame:
    """지정 연도의 황사주의보 발생정보 조회해서 DataFrame으로 반환"""
    params = {
        "serviceKey": SERVICE_KEY,
        "returnType": "json",
        "numOfRows": 100,   # 1년치 100건이면 충분
        "pageNo": 1,
        "year": year,
    }

    r = requests.get(BASE_URL, params=params, timeout=10)
    r.raise_for_status()
    data = r.json()

    # JSON 구조: response -> body -> items -> item
    response = data.get("response", {})
    body = response.get("body", {})
    items = body.get("items")

    if not items:
        print(f"[{year}] no items in response")
        return pd.DataFrame()

    # 보통 items = {"item": [...]} 형태일 가능성이 큼
    if isinstance(items, dict):
        records = items.get("item", [])
    else:
        records = items

    if isinstance(records, dict):
        records = [records]

    df = pd.DataFrame.from_records(records)
    df["year"] = year
    return df


def main():
    # 👉 새 연도 범위: PM/날씨랑 맞춰서 2021~2023으로
    years = [2021, 2022, 2023]

    frames = []
    for y in years:
        print(f"fetching year {y}...")
        df_y = fetch_year(y)
        if not df_y.empty:
            frames.append(df_y)

    if not frames:
        print("no data fetched. check your service key or params.")
        return

    df_all = pd.concat(frames, ignore_index=True)

    out_path = DATA_DIR / "national_alert_raw.csv"
    df_all.to_csv(out_path, index=False)
    print(f"saved raw: {out_path}")
    print(df_all.head())
    print(df_all.columns)


if __name__ == "__main__":   #  이 줄이 핵심
    main()
