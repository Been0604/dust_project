from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MONTHLY_DIR = DATA_DIR / "final_monthly"
STATIONS_PATH = DATA_DIR / "stations.csv"

# 우리가 사용할 연도 범위
TARGET_YEARS = {2021, 2022, 2023, 2024}

# PM 시계열에서 사용할 최종 상한 시각
# (날씨 데이터와 공통으로 맞추기 위해 2024-12-31 23:00까지만 사용)
FINAL_END_TS = pd.Timestamp("2024-12-31 23:00:00")


def load_one_month(path: Path) -> pd.DataFrame:
    print(f"[정보] 월별 엑셀 로드: {path.name}")

    # 시트 이름이 보통 'Data'인 경우가 많아서 우선 시도,
    # 아니면 첫 번째 시트를 사용
    xls = pd.ExcelFile(path)
    sheet_name = "Data" if "Data" in xls.sheet_names else xls.sheet_names[0]
    df = pd.read_excel(xls, sheet_name=sheet_name)

    required_cols = ["지역", "망", "측정소명", "측정일시", "PM10", "PM25", "주소"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"[에러] {path.name} 에 필요한 컬럼이 없습니다: {missing}")

    # 1) 대전 + 도시대기만 필터
    mask_city = df["지역"].astype(str).str.startswith("대전")
    mask_net = df["망"] == "도시대기"
    df = df[mask_city & mask_net].copy()

    if df.empty:
        print("  → 대전 도시대기 데이터 없음 (건너뜀)")
        return df

    # 2) 측정일시를 문자열(YYYYMMDDHH)로 정규화 + 연도 필터
    #    예: 2021010101, 2021010124 (24시는 다음날 00시로 처리할 것)
    df["측정일시"] = df["측정일시"].astype(str).str.zfill(10)
    df["year"] = df["측정일시"].str.slice(0, 4).astype(int)

    before = len(df)
    df = df[df["year"].isin(TARGET_YEARS)].copy()
    print(
        f"  → 연도 필터({min(TARGET_YEARS)}-{max(TARGET_YEARS)}): "
        f"{before} → {len(df)} 행"
    )

    if df.empty:
        return df

    # 3) ts_kst (datetime) 생성
    #    앞 8자리 = 날짜(YYYYMMDD), 뒤 2자리 = 시(HH)
    #    날짜 + 시간(hour) timedelta → 24시도 자동으로 다음날 00시로 넘어감
    s = df["측정일시"]
    date_str = s.str.slice(0, 8)
    hour_str = s.str.slice(8, 10)

    base_date = pd.to_datetime(date_str, format="%Y%m%d")
    hour = hour_str.astype(int)

    df["ts_kst"] = base_date + pd.to_timedelta(hour, unit="h")

    # 4) 컬럼 정리 & 이름 통일
    df = df.rename(
        columns={
            "측정소명": "station_name",
            "PM10": "pm10",
            "PM25": "pm25",
        }
    )

    # 필요한 것만 남기기
    df = df[["ts_kst", "station_name", "pm10", "pm25", "지역", "망", "주소"]]

    return df


def attach_station_id_and_area(pm: pd.DataFrame, stations: pd.DataFrame) -> pd.DataFrame:
    """station_name 기준으로 station_id, area 붙이기"""

    merged = pm.merge(
        stations[["station_id", "station_name", "area"]],
        on="station_name",
        how="left",
    )

    missing = merged["station_id"].isna().sum()
    if missing > 0:
        print(f"[경고] station_id 매핑 안 된 행 {missing}개 있습니다.")
        print(
            merged[merged["station_id"].isna()][["station_name", "지역", "주소"]]
            .drop_duplicates()
        )

    return merged


def main():
    # airkorea_hourly_YYYY_MM.xlsx 패턴만 읽기
    paths = sorted(MONTHLY_DIR.glob("airkorea_hourly_*.xlsx"))
    if not paths:
        print(f"[에러] {MONTHLY_DIR} 안에 airkorea_hourly_*.xlsx 파일이 없습니다.")
        return

    print(f"[정보] 발견한 월별 파일 수: {len(paths)}개")

    monthly_dfs: list[pd.DataFrame] = []
    for p in paths:
        df_month = load_one_month(p)
        if not df_month.empty:
            monthly_dfs.append(df_month)

    if not monthly_dfs:
        print("[에러] 대전 도시대기 데이터가 하나도 없습니다.")
        return

    pm_all = pd.concat(monthly_dfs, ignore_index=True)

    # ✅ 최종 상한 시각(FINAL_END_TS) 이후 데이터는 모두 제거
    before = len(pm_all)
    pm_all = pm_all[pm_all["ts_kst"] <= FINAL_END_TS].reset_index(drop=True)
    print(
        f"[정보] {len(TARGET_YEARS)}개년 합친 행 개수(대전 도시대기, "
        f"{FINAL_END_TS} 까지): {before} → {len(pm_all)} 행"
    )

    # stations 정보 로드
    stations = pd.read_csv(STATIONS_PATH)

    pm_with_ids = attach_station_id_and_area(pm_all, stations)

    # 시간 + station_id 기준 정렬
    pm_with_ids = pm_with_ids.sort_values(["ts_kst", "station_id"])

    out_path = DATA_DIR / "raw_pm_final.parquet"
    pm_with_ids.to_parquet(out_path, index=False)
    print(f"[완료] raw_pm_final 저장: {out_path} (행 {len(pm_with_ids)}개)")


if __name__ == "__main__":
    main()
