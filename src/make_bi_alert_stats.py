# src/make_bi_alert_stats.py
#
# 목적:
#   2021-2023 기간 동안
#   - PM10 경보(y_loc_pm10 == 1)
#   - PM2.5 경보(y_loc_pm25 == 1)
#   - 황사 경보(dust_stage > 0)
#   발생 시간대의
#   "행정동(EMD_CD/EMD_NM) 평균 PM10/PM2.5 + 발생 시간 수"를 계산.
#
# 변경점(2025-11-30):
#   - EMD_NM은 stations_emd_mapping에서 가져오지 않고,
#     data/bi/emd_codes.csv(행정동 SHP 기반 깨끗한 코드표)에서 가져온다.
#
# 출력:
#   data/bi/bi_alert_pm10_alert.(csv, parquet)
#   data/bi/bi_alert_pm25_alert.(csv, parquet)
#   data/bi/bi_alert_dust_alert.(csv, parquet)

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
BI_DIR = DATA_DIR / "bi"

PM_PATH = DATA_DIR / "processed_with_dust.parquet"
STN_EMD_PATH = DATA_DIR / "stations_emd_mapping.csv"
EMD_CODES_PATH = BI_DIR / "emd_codes.csv"


def load_data():
    if not PM_PATH.exists():
        raise FileNotFoundError(f"PM 데이터 없음: {PM_PATH}")
    if not STN_EMD_PATH.exists():
        raise FileNotFoundError(
            f"station ↔ 행정동 매핑 CSV가 없습니다: {STN_EMD_PATH}\n"
            "QGIS에서 stations와 UMD shp를 공간조인해서 stations_emd_mapping.csv를 먼저 만들어야 합니다."
        )
    if not EMD_CODES_PATH.exists():
        raise FileNotFoundError(
            f"행정동 코드표(emD_codes.csv)가 없습니다: {EMD_CODES_PATH}\n"
            "먼저 src.make_emd_codes를 실행해서 emd_codes.csv를 생성하세요."
        )

    print(f"[로드] {PM_PATH}")
    df = pd.read_parquet(PM_PATH)

    print(f"[로드] {STN_EMD_PATH}")
    # 여기서는 EMD_NM 안 쓰고 station_id ↔ EMD_CD만 사용
    stn_emd = pd.read_csv(
        STN_EMD_PATH,
        dtype={"station_id": str, "EMD_CD": str},
    )

    print(f"[로드] {EMD_CODES_PATH}")
    emd_codes = pd.read_csv(
        EMD_CODES_PATH,
        dtype={"EMD_CD": str, "EMD_NM": str},
    )

    return df, stn_emd, emd_codes


def attach_emd(df: pd.DataFrame, stn_emd: pd.DataFrame) -> pd.DataFrame:
    """processed_with_dust에 station_id 기반으로 행정동 코드 붙이기"""

    df = df.copy()
    if "station_id" not in df.columns:
        raise KeyError("processed_with_dust에 station_id 컬럼이 없습니다.")

    df["station_id"] = df["station_id"].astype(str)
    stn_emd["station_id"] = stn_emd["station_id"].astype(str)

    df_out = df.merge(stn_emd[["station_id", "EMD_CD"]], on="station_id", how="left")

    missing = df_out["EMD_CD"].isna().sum()
    total = len(df_out)
    print(f"[체크] 행정동 코드 매핑 안 된 행 개수: {missing} / {total}")

    if missing > 0:
        missing_ids = (
            df_out.loc[df_out["EMD_CD"].isna(), "station_id"]
            .drop_duplicates()
            .tolist()
        )
        print("[경고] 행정동 코드가 매핑되지 않은 station_id 목록:", missing_ids)

    return df_out


def make_hourly_emd_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    ts_kst × EMD_CD 기준으로 시간별 평균 PM + 경보 플래그 집계.
    2021-01-01 ~ 2023-12-31 기간만 사용.
    """

    needed = [
        "ts_kst",
        "EMD_CD",
        "pm10",
        "pm25",
        "y_loc_pm10",
        "y_loc_pm25",
        "dust_stage",
    ]
    for c in needed:
        if c not in df.columns:
            raise KeyError(f"필요 컬럼 없음: {c}")

    df = df.copy()
    df["ts_kst"] = pd.to_datetime(df["ts_kst"])

    mask = (df["ts_kst"] >= "2021-01-01") & (df["ts_kst"] < "2024-01-01")
    df = df.loc[mask].copy()

    print("[집계] ts_kst × EMD_CD 기준 시간별 평균/플래그 계산 중…")
    hourly = (
        df.groupby(["ts_kst", "EMD_CD"])
        .agg(
            pm10_mean=("pm10", "mean"),
            pm25_mean=("pm25", "mean"),
            y_loc_pm10=("y_loc_pm10", "max"),
            y_loc_pm25=("y_loc_pm25", "max"),
            dust_stage=("dust_stage", "max"),
            n_obs=("pm10", "size"),
        )
        .reset_index()
    )

    return hourly


def agg_by_alert(hourly: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    세 가지 경보별로 EMD_CD 단위 평균 PM10/PM25 계산.

    - 'pm10_alert': y_loc_pm10 == 1
    - 'pm25_alert': y_loc_pm25 == 1
    - 'dust_alert': dust_stage > 0
    """

    results: dict[str, pd.DataFrame] = {}

    # PM10 경보
    pm10_alert = hourly[hourly["y_loc_pm10"] == 1]
    agg_pm10 = (
        pm10_alert.groupby("EMD_CD")
        .agg(
            pm10_mean_alert=("pm10_mean", "mean"),
            pm25_mean_alert=("pm25_mean", "mean"),
            n_hours=("ts_kst", "size"),
        )
        .reset_index()
    )
    results["pm10_alert"] = agg_pm10

    # PM2.5 경보
    pm25_alert = hourly[hourly["y_loc_pm25"] == 1]
    agg_pm25 = (
        pm25_alert.groupby("EMD_CD")
        .agg(
            pm10_mean_alert=("pm10_mean", "mean"),
            pm25_mean_alert=("pm25_mean", "mean"),
            n_hours=("ts_kst", "size"),
        )
        .reset_index()
    )
    results["pm25_alert"] = agg_pm25

    # 황사 경보
    dust_alert = hourly[hourly["dust_stage"] > 0]
    agg_dust = (
        dust_alert.groupby("EMD_CD")
        .agg(
            pm10_mean_alert=("pm10_mean", "mean"),
            pm25_mean_alert=("pm25_mean", "mean"),
            n_hours=("ts_kst", "size"),
        )
        .reset_index()
    )
    results["dust_alert"] = agg_dust

    return results


def attach_names_to_results(
    results: dict[str, pd.DataFrame],
    emd_codes: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """각 결과 DF에 emd_codes로부터 EMD_NM 붙이기"""

    out: dict[str, pd.DataFrame] = {}

    for key, df_res in results.items():
        merged = df_res.merge(
            emd_codes[["EMD_CD", "EMD_NM"]],
            on="EMD_CD",
            how="left",
        )

        if merged["EMD_NM"].isna().any():
            n_missing = merged["EMD_NM"].isna().sum()
            print(f"[경고] {key}: emd_codes에서 이름을 못 찾은 행정동 {n_missing}개")

        # 컬럼 순서 정리
        merged = merged[
            [
                "EMD_CD",
                "EMD_NM",
                "pm10_mean_alert",
                "pm25_mean_alert",
                "n_hours",
            ]
        ].sort_values(["EMD_NM"]).reset_index(drop=True)

        out[key] = merged

    return out


def save_results(results: dict[str, pd.DataFrame]) -> None:
    out_dir = BI_DIR
    out_dir.mkdir(exist_ok=True)

    for key, df_res in results.items():
        csv_path = out_dir / f"bi_alert_{key}.csv"
        pq_path = out_dir / f"bi_alert_{key}.parquet"
        df_res.to_csv(csv_path, index=False, encoding="utf-8-sig")
        df_res.to_parquet(pq_path, index=False)
        print(f"[저장] {key}: {csv_path.name}, {pq_path.name}")


def main():
    df, stn_emd, emd_codes = load_data()
    df_with_emd = attach_emd(df, stn_emd)
    hourly = make_hourly_emd_table(df_with_emd)
    results = agg_by_alert(hourly)
    results = attach_names_to_results(results, emd_codes)
    save_results(results)


if __name__ == "__main__":
    main()
