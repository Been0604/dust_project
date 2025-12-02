import os
import pandas as pd


# 시나리오 정의
SCENARIOS = {
    "A": {"base_year": 2021, "shift_years": 5},  # 2021 → 2026
    "B": {"base_year": 2022, "shift_years": 4},  # 2022 → 2026
    "C": {"base_year": 2023, "shift_years": 3},  # 2023 → 2026
}


def ensure_dir(path: str) -> None:
    """폴더 없으면 만들어주는 함수."""
    os.makedirs(path, exist_ok=True)


def add_scenario_columns(
    df: pd.DataFrame,
    scenario_id: str,
    base_year: int,
    shift_years: int,
) -> pd.DataFrame:
    """
    공통 시나리오 컬럼 추가:
    - ts_kst_original
    - ts_kst_2026
    - scenario_id
    - scenario_base_year
    """
    df = df.copy()

    if "ts_kst" not in df.columns:
        raise KeyError("DataFrame에 'ts_kst' 컬럼이 없습니다. 파이프라인을 먼저 확인하세요.")

    df["ts_kst"] = pd.to_datetime(df["ts_kst"])

    df["ts_kst_original"] = df["ts_kst"]
    df["ts_kst_2026"] = df["ts_kst"] + pd.DateOffset(years=shift_years)
    df["scenario_id"] = scenario_id
    df["scenario_base_year"] = base_year

    return df


def make_base_from_processed(processed_path: str, output_dir: str) -> None:
    """
    processed_final에서 2021/2022/2023 베이스 시나리오 parquet 생성.

    출력:
    - data/scenario/scenario_base_2021.parquet
    - data/scenario/scenario_base_2022.parquet
    - data/scenario/scenario_base_2023.parquet
    """
    print(f"[INFO] Loading processed_final from: {processed_path}")
    df = pd.read_parquet(processed_path)

    if "ts_kst" not in df.columns:
        raise KeyError("'ts_kst' 컬럼이 없습니다. processed_final 생성 단계 확인 필요.")

    df["ts_kst"] = pd.to_datetime(df["ts_kst"])

    for scenario_id, cfg in SCENARIOS.items():
        base_year = cfg["base_year"]
        shift_years = cfg["shift_years"]

        print(f"[INFO] Scenario {scenario_id}: base_year={base_year}, shift_years={shift_years}")

        mask = df["ts_kst"].dt.year == base_year
        df_base = df.loc[mask].copy()
        print(f"  - Rows for year {base_year}: {len(df_base):,}")

        df_base = add_scenario_columns(df_base, scenario_id, base_year, shift_years)

        out_path = os.path.join(output_dir, f"scenario_base_{base_year}.parquet")
        df_base.to_parquet(out_path, index=False)
        print(f"  - Saved base file: {out_path}")


def make_features_scenario(
    features_pm10_path: str,
    features_pm25_path: str,
    output_dir: str,
) -> None:
    """
    features_full_pm10/pm25에서 2026 시나리오용 피처 parquet 생성.

    출력:
    - data/scenario/scenario_2026_features_pm10_A/B/C.parquet
    - data/scenario/scenario_2026_features_pm25_A/B/C.parquet
    """
    print(f"[INFO] Loading features_full_pm10 from: {features_pm10_path}")
    f_pm10 = pd.read_parquet(features_pm10_path)

    print(f"[INFO] Loading features_full_pm25 from: {features_pm25_path}")
    f_pm25 = pd.read_parquet(features_pm25_path)

    if "ts_kst" not in f_pm10.columns or "ts_kst" not in f_pm25.columns:
        raise KeyError("features_full 데이터에 'ts_kst' 컬럼이 없습니다. features.py 확인 필요.")

    f_pm10["ts_kst"] = pd.to_datetime(f_pm10["ts_kst"])
    f_pm25["ts_kst"] = pd.to_datetime(f_pm25["ts_kst"])

    for scenario_id, cfg in SCENARIOS.items():
        base_year = cfg["base_year"]
        shift_years = cfg["shift_years"]

        print(f"[INFO] Scenario {scenario_id}: base_year={base_year}, shift_years={shift_years}")

        # PM10
        mask10 = f_pm10["ts_kst"].dt.year == base_year
        f_pm10_base = f_pm10.loc[mask10].copy()
        print(f"  - PM10 rows for year {base_year}: {len(f_pm10_base):,}")
        f_pm10_base = add_scenario_columns(f_pm10_base, scenario_id, base_year, shift_years)

        out_pm10 = os.path.join(output_dir, f"scenario_2026_features_pm10_{scenario_id}.parquet")
        f_pm10_base.to_parquet(out_pm10, index=False)
        print(f"  - Saved PM10 scenario features: {out_pm10}")

        # PM25
        mask25 = f_pm25["ts_kst"].dt.year == base_year
        f_pm25_base = f_pm25.loc[mask25].copy()
        print(f"  - PM25 rows for year {base_year}: {len(f_pm25_base):,}")
        f_pm25_base = add_scenario_columns(f_pm25_base, scenario_id, base_year, shift_years)

        out_pm25 = os.path.join(output_dir, f"scenario_2026_features_pm25_{scenario_id}.parquet")
        f_pm25_base.to_parquet(out_pm25, index=False)
        print(f"  - Saved PM25 scenario features: {out_pm25}")


def main():
    # [사실] 빈 프로젝트 구조 기준 경로
    processed_path = "data/processed_final.parquet"
    features_pm10_path = "data/features_full_pm10.parquet"
    features_pm25_path = "data/features_full_pm25.parquet"
    scenario_dir = "data/scenario"

    ensure_dir(scenario_dir)

    # 1) processed_final 기반 베이스 시나리오
    make_base_from_processed(processed_path, scenario_dir)

    # 2) features_full 기반 2026 시나리오 피처
    make_features_scenario(features_pm10_path, features_pm25_path, scenario_dir)


if __name__ == "__main__":
    main()
