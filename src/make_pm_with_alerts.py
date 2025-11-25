# src/make_pm_with_alerts.py
#
# raw_pm_final.parquet (2021-2024 PM10/PM2.5, station_id/area 포함)에
# 동부/서부 단위 로컬 PM 경보 타임라인
#   - local_alert_events_pm10.parquet
#   - local_alert_events_pm25.parquet
# 을 붙여서
#   data/pm_with_alerts.parquet
# 을 만든다.
#
# 주의:
# - 경보 이력은 2021~2023만 존재
# - 2024 구간은 y_loc_*가 NaN으로 남는 게 정상 (임의로 0 채우지 않음)

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

PM_PATH = DATA_DIR / "raw_pm_final.parquet"
ALERT10_PATH = DATA_DIR / "local_alert_events_pm10.parquet"
ALERT25_PATH = DATA_DIR / "local_alert_events_pm25.parquet"
OUT_PATH = DATA_DIR / "pm_with_alerts.parquet"


def main():
    # 1) PM 로드
    print(f"[정보] PM 시계열 로드: {PM_PATH}")
    pm = pd.read_parquet(PM_PATH)
    pm["ts_kst"] = pd.to_datetime(pm["ts_kst"])

    # area 컬럼이 있어야 동부/서부 기준으로 경보를 붙일 수 있음
    if "area" not in pm.columns:
        raise KeyError("raw_pm_final.parquet 안에 'area' 컬럼이 없습니다. "
                       "make_raw_pm_from_monthly_excels.py에서 station_id/area 매핑이 되었는지 확인하세요.")

    print("  → PM 시계열 shape:", pm.shape)

    # 2) 로컬 경보 타임라인 로드 (PM10, PM2.5)
    print(f"[정보] PM10 경보 타임라인 로드: {ALERT10_PATH}")
    alert10 = pd.read_parquet(ALERT10_PATH)
    alert10["ts_kst"] = pd.to_datetime(alert10["ts_kst"])

    print(f"[정보] PM2.5 경보 타임라인 로드: {ALERT25_PATH}")
    alert25 = pd.read_parquet(ALERT25_PATH)
    alert25["ts_kst"] = pd.to_datetime(alert25["ts_kst"])

    # 컬럼 이름이 예상과 다를 수도 있으니 정리
    # (ts_kst, area, y_loc_pm10 / y_loc_pm25 형태로 맞춘다고 가정)
    if "y_loc_pm10" not in alert10.columns:
        # 예: 'stage_pm10' 같은 이름이면 여기서 바꿔주면 됨
        raise KeyError("local_alert_events_pm10.parquet 안에 'y_loc_pm10' 컬럼이 없습니다.")

    if "y_loc_pm25" not in alert25.columns:
        raise KeyError("local_alert_events_pm25.parquet 안에 'y_loc_pm25' 컬럼이 없습니다.")

    cols10 = ["ts_kst", "area", "y_loc_pm10"]
    cols25 = ["ts_kst", "area", "y_loc_pm25"]

    alert10 = alert10[cols10].copy()
    alert25 = alert25[cols25].copy()

    print("  → PM10 경보 타임라인 shape:", alert10.shape)
    print("  → PM2.5 경보 타임라인 shape:", alert25.shape)

    # 3) PM + PM10 경보 병합
    #    key = (ts_kst, area)
    #    how = 'left' → PM 시계열(2021~2024)은 그대로 유지, 경보 없는 구간은 NaN
    merged = pm.merge(alert10, on=["ts_kst", "area"], how="left")

    # 4) PM + PM2.5 경보 병합
    merged = merged.merge(alert25, on=["ts_kst", "area"], how="left")

    print("[정보] 경보 병합 후 shape:", merged.shape)

    # 5) 경보 구간/비경보 구간 요약
    print("[정보] y_loc_pm10 값 분포:")
    print(merged["y_loc_pm10"].value_counts(dropna=False))

    print("[정보] y_loc_pm25 값 분포:")
    print(merged["y_loc_pm25"].value_counts(dropna=False))

    # 6) 저장
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(OUT_PATH, index=False)
    print(f"[완료] pm_with_alerts 저장: {OUT_PATH} (행 {len(merged)}개)")


if __name__ == "__main__":
    main()
