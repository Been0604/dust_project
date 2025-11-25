# src/fetch_weather_api.py
# 기상청 ASOS 시간자료 → data/raw_weather.csv
# 에러 응답(resultCode != 00)도 안전하게 처리

from pathlib import Path
import requests
import pandas as pd
from datetime import date, timedelta

CONFIG_DIR = Path("config")
DATA_DIR = Path("data")

KMA_KEY = CONFIG_DIR.joinpath("kma_key.txt").read_text().strip()

# 기상청 ASOS 시간자료 조회서비스
BASE_URL = "http://apis.data.go.kr/1360000/AsosHourlyInfoService/getWthrDataList"

# 대전 ASOS 지점 번호 (133이 대전지점)
STN_ID = "133"


def to_float(v):
    """빈 문자열/None은 NaN으로, 나머지는 float."""
    if v in (None, "", " "):
        return float("nan")
    return float(v)


def fetch_weather(start: date, end: date) -> pd.DataFrame:
    records = []
    cur = start

    while cur <= end:
        params = {
            "serviceKey": KMA_KEY,
            "pageNo": 1,
            "numOfRows": 24,      # 하루 24시간
            "dataType": "JSON",
            "dataCd": "ASOS",
            "dateCd": "HR",
            "startDt": cur.strftime("%Y%m%d"),
            "startHh": "00",
            "endDt": cur.strftime("%Y%m%d"),
            "endHh": "23",
            "stnIds": STN_ID,
        }

        print(f"→ ASOS 요청 {cur.isoformat()} (stn {STN_ID})")
        resp = requests.get(BASE_URL, params=params)
        resp.raise_for_status()
        data = resp.json()

        # 공통 구조: response → header(+body)
        response = data.get("response", {})
        header = response.get("header", {})
        code = header.get("resultCode")
        msg = header.get("resultMsg")

        print(f"   resultCode={code}, resultMsg={msg}")

        body = response.get("body")
        if body is None:
            # body가 없으면 이 날짜는 건너뛰기
            print("   ⚠ body 없음 → 이 날짜는 스킵")
            cur += timedelta(days=1)
            continue

        items_node = body.get("items")
        if not items_node:
            print("   ⚠ items 비어 있음 → 스킵")
            cur += timedelta(days=1)
            continue

        items = items_node.get("item", [])
        if not items:
            print("   ⚠ item 리스트 없음 → 스킵")
            cur += timedelta(days=1)
            continue

        for item in items:
            ts_str = item["tm"]          # "2025-11-20 01:00"

            records.append(
                {
                    "ts_kst": ts_str,
                    "temp": to_float(item.get("ta")),    # 기온
                    "rh": to_float(item.get("hm")),      # 상대습도
                    "wind_spd": to_float(item.get("ws")),
                    "wind_dir": to_float(item.get("wd")),
                    "rain": to_float(item.get("rn")),    # 강수량
                    "pressure": to_float(item.get("pa")),
                }
            )

        cur += timedelta(days=1)

    df = pd.DataFrame(records)

    if not df.empty:
        df["ts_kst"] = pd.to_datetime(df["ts_kst"])
        df = df.sort_values("ts_kst")
        df["ts_kst"] = df["ts_kst"].dt.strftime("%Y-%m-%d %H:%M")

    return df


def main():
    # 테스트용: 어제 ~ 오늘 2일치
    end = date.today()
    start = end - timedelta(days=1)

    df = fetch_weather(start, end)
    if df.empty:
        print("❌ 날씨 데이터가 비어 있음. 위 로그의 resultCode/resultMsg 확인 필요.")
        return

    out_path = DATA_DIR / "raw_weather.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False, encoding="utf-8-sig")

    print(f"\n✅ 저장 완료: {out_path} (행 {len(df)}개)")


if __name__ == "__main__":
    main()
