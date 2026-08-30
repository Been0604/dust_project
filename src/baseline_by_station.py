# -*- coding: utf-8 -*-
"""
측정소별 persistence 대비 성능 (baseline.py의 측정소 단위 확장)

baseline.py는 11개 측정소를 합친 값을 낸다. 전체 skill이 음수여도
일부 측정소에서는 양수일 수 있으므로, 같은 공통 표본 규칙을
측정소별로 적용해 skill score를 산출한다.

공통 표본 정의는 baseline.py와 동일:
  각 구간에서 target / persistence(현재값) / RF / XGB가 모두 유효한 행만 사용.
  단, 표본은 '구간 전체' 기준으로 먼저 확정한 뒤 측정소별로 나눈다.
  (측정소마다 따로 필터링하면 baseline.py 결과와 합이 맞지 않는다.)

실행:  python src/baseline_by_station.py   (저장소 최상위에서)
출력:  src/baseline_by_station.csv
"""

from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

PARQUET = DATA_DIR / "processed_with_preds_both.parquet"
STATIONS = DATA_DIR / "stations.csv"
OUT_CSV = Path(__file__).resolve().parent / "baseline_by_station.csv"

if not PARQUET.exists():
    raise FileNotFoundError(PARQUET)

df = pd.read_parquet(PARQUET)
df["year"] = df["ts_kst"].dt.year


def to_split(y):
    if y <= 2022:
        return "Train(21-22)"
    if y == 2023:
        return "Val(2023)"
    return "Test(2024)"


df["split"] = df["year"].map(to_split)

# 측정소 이름 붙이기 (있으면)
name_map = {}
if STATIONS.exists():
    st = pd.read_csv(STATIONS)
    if {"station_id", "station_name"} <= set(st.columns):
        name_map = dict(zip(st["station_id"], st["station_name"]))

SPECS = [
    ("PM10",  "target_pm10", "pm10", "y_pred_rf",      "y_pred_xgb"),
    ("PM2.5", "target_pm25", "pm25", "y_pred_rf_pm25", "y_pred_xgb_pm25"),
]
SPLITS = ["Val(2023)", "Test(2024)"]      # 학습 구간은 의미가 적어 제외


def rmse(y, p):
    return float(np.sqrt(((p - y) ** 2).mean()))


rows = []

for label, tgt, per, rf, xgb in SPECS:
    for sp in SPLITS:
        d = df[df["split"] == sp]

        # baseline.py와 동일한 공통 표본
        mask = (d[tgt].notna() & d[per].notna()
                & d[rf].notna() & d[xgb].notna())
        dd = d[mask]

        for sid, g in dd.groupby("station_id"):
            if len(g) < 100:            # 표본이 너무 적은 측정소는 제외
                continue
            r_per = rmse(g[tgt], g[per])
            r_rf = rmse(g[tgt], g[rf])
            r_xgb = rmse(g[tgt], g[xgb])
            rows.append(dict(
                Pollutant=label,
                Split=sp,
                station_id=sid,
                station_name=name_map.get(sid, ""),
                n=len(g),
                mean_obs=round(float(g[per].mean()), 2),   # 해당 측정소 평균 농도
                RMSE_persistence=round(r_per, 2),
                RMSE_RF=round(r_rf, 2),
                RMSE_XGB=round(r_xgb, 2),
                Skill_RF_pct=round((1 - r_rf / r_per) * 100, 2),
                Skill_XGB_pct=round((1 - r_xgb / r_per) * 100, 2),
            ))

res = pd.DataFrame(rows).sort_values(["Pollutant", "Split", "Skill_RF_pct"])

pd.set_option("display.width", 250)
pd.set_option("display.max_rows", 200)

for label, *_ in SPECS:
    for sp in SPLITS:
        sub = res[(res.Pollutant == label) & (res.Split == sp)]
        if sub.empty:
            continue
        print(f"\n=== {label} / {sp} — 측정소별 skill score ===")
        print(sub[["station_id", "station_name", "n", "mean_obs",
                   "RMSE_persistence", "RMSE_RF", "RMSE_XGB",
                   "Skill_RF_pct", "Skill_XGB_pct"]].to_string(index=False))
        pos_rf = (sub.Skill_RF_pct > 0).sum()
        pos_xgb = (sub.Skill_XGB_pct > 0).sum()
        print(f"  → persistence를 넘어선 측정소: RF {pos_rf}/{len(sub)}, "
              f"XGB {pos_xgb}/{len(sub)}")
        # 평균 농도와 skill의 관계 (취약 지역일수록 예측이 어려운가?)
        if len(sub) >= 4:
            c = sub["mean_obs"].corr(sub["Skill_RF_pct"])
            print(f"  → 평균 농도 vs RF skill 상관계수: {c:+.2f}")

res.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
print(f"\n저장: {OUT_CSV}")
