# src/make_processed_with_dust.py
#
# 목적:
# - data/processed_final.parquet (PM + 날씨 + PM 경보)
# - data/national_alert_events_daejeon.parquet (황사 특보 시간)
# 를 ts_kst 기준으로 merge 해서
# data/processed_with_dust.parquet 생성.
#
# dust_stage:
#   0 = 황사 특보 없음
#   1 = 황사 특보 있음

from pathlib import Path
import pandas as pd

# 저장소 최상위 = 이 파일(src/xxx.py)의 부모의 부모
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"


def load_processed() -> pd.DataFrame:
    path = DATA_DIR / "processed_final.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} 가 없습니다. processed_final 먼저 만들어야 합니다.")
    df = pd.read_parquet(path)
    df["ts_kst"] = pd.to_datetime(df["ts_kst"])
    return df


def load_dust_events() -> pd.DataFrame:
    path = DATA_DIR / "national_alert_events_daejeon.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} 가 없습니다. make_dust_events_from_raw 먼저 실행해 주세요.")
    df = pd.read_parquet(path)
    df["ts_kst"] = pd.to_datetime(df["ts_kst"])
    return df


def main():
    print("▶ processed_final 로드")
    df_proc = load_processed()
    print("processed_final shape:", df_proc.shape)

    print("▶ national_alert_events_daejeon 로드")
    df_dust = load_dust_events()
    print("dust events shape:", df_dust.shape)

    print("▶ ts_kst 기준 left join (PM 시계열 기준)")
    df_merged = df_proc.merge(df_dust, on="ts_kst", how="left")

    # 황사 특보 없는 시각은 0으로
    df_merged["dust_stage"] = df_merged["dust_stage"].fillna(0).astype(int)

    df_merged = df_merged.sort_values(["ts_kst", "station_id"]).reset_index(drop=True)

    print("merged shape:", df_merged.shape)
    print("dust_stage 분포:")
    print(df_merged["dust_stage"].value_counts().sort_index())

    out_path = DATA_DIR / "processed_with_dust.parquet"
    df_merged.to_parquet(out_path, index=False)
    print("\n✅ 저장 완료:", out_path)


if __name__ == "__main__":
    main()
