# src/make_emd_codes.py
#
# daejeon_umd 폴더의 행정동 SHP에서
# EMD_CD, EMD_NM만 뽑아서 Power BI / 요약 스크립트에서 쓸
# 매핑 테이블(emD_codes.csv)을 만든다.

from pathlib import Path
import geopandas as gpd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
BI_DIR = DATA_DIR / "bi"


def main():
    BI_DIR.mkdir(parents=True, exist_ok=True)

    # 1) SHP가 들어있는 폴더: C:\Users\wachu\dust_project\daejeon_umd
    shp_dir = ROOT / "daejeon_umd"
    if not shp_dir.exists():
        raise FileNotFoundError(f"폴더가 없음: {shp_dir}")

    # 2) 폴더 안에서 .shp 파일 자동 검색 (1개라고 가정)
    shp_candidates = list(shp_dir.glob("*.shp"))
    if not shp_candidates:
        raise FileNotFoundError(f"{shp_dir} 안에서 .shp 파일을 찾지 못함")

    if len(shp_candidates) > 1:
        print("경고: .shp 파일이 여러 개입니다. 첫 번째 것만 사용합니다.")
    shp_path = shp_candidates[0]
    print(f"Using shapefile: {shp_path}")

    # 3) 한국 SHP 인코딩: cp949 시도
    gdf = gpd.read_file(shp_path, encoding="cp949")

    cols = ["EMD_CD", "EMD_NM"]
    missing = set(cols) - set(gdf.columns)
    if missing:
        raise ValueError(f"SHP에 없는 컬럼: {missing}")

    emd_codes = (
        gdf[cols]
        .drop_duplicates()
        .sort_values("EMD_CD")
        .reset_index(drop=True)
    )

    out_csv = BI_DIR / "emd_codes.csv"
    emd_codes.to_csv(out_csv, index=False, encoding="utf-8-sig")

    print(f"Saved: {out_csv}")


if __name__ == "__main__":
    main()
