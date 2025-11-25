# src/make_raw_weather_final_from_api.py
#
# 기상자료개방포털 ASOS 시간자료 API를 이용해서
# 대전(ASOS 지점번호 133) 2021년 이후 시간별 기상자료를 받아
# data/raw_weather_final.parquet 으로 저장한다.
#
# 이번 버전 기준 목표 범위(예시):
# - 2021-01-01 00:00 ~ 2025-12-31 23:00
#   (실제로는 API가 제공하는 마지막 시각까지만 들어옴)
#
# 최종 컬럼:
# - ts_kst   : 관측시각 (datetime, KST)
# - temp     : 기온(ta)
# - rh       : 상대습도(hm)
# - wind_spd : 풍속(ws)
# - wind_dir : 풍향(wd)
# - rain     : 강수량(rn)
# - pressure : 현지기압(pa)

from pathlib import Path
import time

import requests
from requests.exceptions import (
    ReadTimeout,
    ConnectionError as ReqConnectionError,
    HTTPError,
)
import pandas as pd

# 경로 설정 (빈 환경 기준)
PROJECT_ROOT = Path(r"C:\Users\wachu\dust_project")
DATA_DIR = PROJECT_ROOT / "data"
CONFIG_DIR = PROJECT_ROOT / "config"

# 기상청 ASOS 시간자료 조회서비스
API_URL = "http://apis.data.go.kr/1360000/AsosHourlyInfoService/getWthrDataList"

# 대전 ASOS 지점 번호 (133이 대전지점)
STN_ID = "133"

# 요청할 연도 범위 (필요하면 여기만 바꾸면 됨)
START_YEAR = 2021
END_YEAR = 2025  # 2021~2025년까지 요청 시도


def load_kma_key() -> str:
    """
    config/kma_key.txt 에 저장된 인증키 읽기.
    txt 안에는 보통 인코딩된 서비스키 한 줄만 들어 있음.
    """
    key_path = CONFIG_DIR / "kma_key.txt"
    if not key_path.exists():
        raise FileNotFoundError(f"kma_key.txt not found at {key_path}")

    return key_path.read_text(encoding="utf-8").strip()


def fetch_hourly_one_year(year: int) -> pd.DataFrame:
    """
    ASOS 시간자료 API로 특정 연도(1년) 자료를 모두 가져와서 DataFrame으로 반환.

    - 시간 자료(HR)는 1회 조회 최대 기간이 '1년'이라
      startDt = YYYY0101, endDt = YYYY1231 로 한 번에 요청.
    - 페이지네이션(pageNo, numOfRows) 처리 포함.
    - 타임아웃/일시적인 네트워크/서버 에러(5xx)에 대해 최대 3회 재시도.
    """
    service_key = load_kma_key()

    page_no = 1
    num_rows = 999  # 한 페이지 최대치로 크게 잡기
    all_items = []

    while True:
        params = {
            "serviceKey": service_key,
            "pageNo": page_no,
            "numOfRows": num_rows,
            "dataType": "JSON",
            "dataCd": "ASOS",
            "dateCd": "HR",
            "startDt": f"{year}0101",
            "startHh": "00",
            "endDt": f"{year}1231",
            "endHh": "23",
            "stnIds": STN_ID,
        }

        # 타임아웃/연결/서버 오류 대비 재시도
        max_retry = 3
        data = None

        for attempt in range(1, max_retry + 1):
            try:
                print(f"[{year}] requesting page {page_no} (try {attempt}/{max_retry}) ...")
                resp = requests.get(API_URL, params=params, timeout=60)
                resp.raise_for_status()
                data = resp.json()
                break  # 성공하면 재시도 루프 탈출

            except (ReadTimeout, ReqConnectionError) as e:
                # 클라이언트 쪽 타임아웃/연결 문제
                print(f"[{year}] page {page_no} timeout/conn error: {e}")
                if attempt == max_retry:
                    raise
                sleep_sec = 2 * attempt  # 2초, 4초, 6초
                print(f"[{year}] retry after {sleep_sec} sec ...")
                time.sleep(sleep_sec)

            except HTTPError as e:
                # 서버에서 5xx(504 포함) 에러를 주는 경우 재시도
                status = getattr(e.response, "status_code", None)
                print(f"[{year}] page {page_no} HTTP error: {status} {e}")

                # 5xx면 재시도, 그 외(4xx)는 바로 에러
                if status is not None and 500 <= status < 600 and attempt < max_retry:
                    sleep_sec = 2 * attempt
                    print(f"[{year}] retry after {sleep_sec} sec (HTTP {status}) ...")
                    time.sleep(sleep_sec)
                    continue

                # 재시도 다 썼거나 4xx 에러면 그대로 터뜨림
                raise

        # 여기까지 왔으면 data는 정상 응답
        response = data.get("response", {})
        header = response.get("header", {})
        body = response.get("body", {})

        result_code = header.get("resultCode")
        result_msg = header.get("resultMsg")
        if result_code != "00":
            # 예: 아예 데이터가 없는 연도 등
            raise RuntimeError(f"[{year}] API error {result_code}: {result_msg}")

        items_node = body.get("items", {})
        items = items_node.get("item", []) if items_node else []

        if not items:
            print(f"[{year}] no items on page {page_no}, stop.")
            break

        all_items.extend(items)

        total_count = body.get("totalCount", 0)
        print(f"[{year}] page {page_no} items: {len(items)}, total: {total_count}")

        # 마지막 페이지면 종료
        if page_no * num_rows >= total_count:
            break

        page_no += 1

    df = pd.DataFrame(all_items)
    print(f"[{year}] fetched rows:", df.shape[0])
    return df


def to_weather_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
    원본 ASOS 컬럼을 프로젝트에서 쓸 스키마로 매핑.

    주요 원본 컬럼(예시):
      - tm : 'YYYY-MM-DD HH:MM'
      - ta : 기온(℃)
      - hm : 상대습도(%)
      - ws : 풍속(m/s)
      - wd : 풍향(도)
      - rn : 강수량(mm)
      - pa : 현지기압(hPa)
    """
    if "tm" not in df.columns:
        raise KeyError("tm 컬럼이 없음. API 응답 구조가 예상과 다름.")

    out = pd.DataFrame()
    out["ts_kst"] = pd.to_datetime(df["tm"])

    def safe_num(col_name: str):
        if col_name in df.columns:
            return pd.to_numeric(df[col_name], errors="coerce")
        else:
            return pd.Series([pd.NA] * len(df), dtype="float64")

    out["temp"] = safe_num("ta")
    out["rh"] = safe_num("hm")
    out["wind_spd"] = safe_num("ws")
    out["wind_dir"] = safe_num("wd")
    out["rain"] = safe_num("rn")
    out["pressure"] = safe_num("pa")

    # 시간순 정렬
    out = out.sort_values("ts_kst").reset_index(drop=True)

    # ✅ 더 이상 2021~2023으로 자르지 않음 (전체 범위 유지)
    return out


def main():
    years = list(range(START_YEAR, END_YEAR + 1))
    dfs = []

    for y in years:
        try:
            raw_y = fetch_hourly_one_year(y)
        except RuntimeError as e:
            # 예: 아직 데이터가 없어서 API 에러 나는 연도
            print(f"[{y}] skipped due to API error: {e}")
            continue

        if raw_y.empty:
            print(f"[{y}] empty response, skip.")
            continue

        clean_y = to_weather_schema(raw_y)
        print(f"[{y}] after clean:", clean_y.shape)
        dfs.append(clean_y)

    if not dfs:
        raise RuntimeError("No weather data fetched. Check API key or year range.")

    full = pd.concat(dfs, ignore_index=True)
    full = full.sort_values("ts_kst").reset_index(drop=True)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DATA_DIR / "raw_weather_final.parquet"
    full.to_parquet(out_path, index=False)

    print("saved:", out_path)
    print("final shape:", full.shape)
    print("ts range:", full['ts_kst'].min(), "→", full['ts_kst'].max())


if __name__ == "__main__":
    main()
