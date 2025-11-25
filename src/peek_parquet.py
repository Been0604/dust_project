# src/peek_parquet.py

from pathlib import Path
import pandas as pd
import sys

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

def main():
    # 인자 없으면 사용법 + 파일 목록 출력
    if len(sys.argv) < 2:
        print("사용법: python src/peek_parquet.py <파일이름.parquet>")
        print("\ndata 폴더 안 parquet 목록:")
        for p in DATA_DIR.glob("*.parquet"):
            print(" -", p.name)
        return

    fname = sys.argv[1]
    path = DATA_DIR / fname

    if not path.exists():
        print(f"파일이 없습니다: {path}")
        print("\ndata 폴더 안 parquet 목록:")
        for p in DATA_DIR.glob("*.parquet"):
            print(" -", p.name)
        return

    print(f"읽는 중: {path}")
    df = pd.read_parquet(path)

    print("\n[shape]")
    print(df.shape)

    print("\n[columns]")
    print(df.columns.tolist())

    print("\n[head(5)]")
    print(df.head())

if __name__ == "__main__":
    main()
