from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
STATIONS_PATH = DATA_DIR / "stations.csv"

def main():
    df = pd.read_csv(STATIONS_PATH)

    # 구 이름을 보고 동부/서부 권역 매핑
    def infer_area(addr: str) -> str:
        if isinstance(addr, str):
            if "서구" in addr or "유성구" in addr:
                return "서부권역"
            if "동구" in addr or "중구" in addr or "대덕구" in addr:
                return "동부권역"
        return "기타"

    df["area"] = df["addr"].apply(infer_area)

    print(df[["station_id", "station_name", "addr", "area"]])

    # 덮어쓰기 저장 (백업이 걱정되면 다른 파일명으로 저장해도 됨)
    df.to_csv(STATIONS_PATH, index=False, encoding="utf-8-sig")
    print(f"[완료] area 컬럼이 추가된 stations.csv 저장: {STATIONS_PATH}")

if __name__ == "__main__":
    main()
