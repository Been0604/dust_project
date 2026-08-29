# -*- coding: utf-8 -*-
"""
persistence 기준 모형 대비 성능 평가 (v2)

v1의 결함:
    모델별로 결측 필터를 따로 걸어 서로 다른 표본에서 RMSE를 계산한 뒤
    그 값들을 나눠 skill score를 산출했다. 분모와 분자의 표본이 달라
    공정한 비교가 아니었다.

v2의 수정:
    구간마다 target / persistence / RF / XGB가 모두 유효한 행만 남긴
    공통 표본(common sample)을 먼저 만들고, 모든 지표를 그 위에서 계산한다.

    persistence 예측:  pred(t+1) = obs(t)
    skill score     :  1 - RMSE_model / RMSE_persistence   (양수면 모형 우세)
"""

import pandas as pd
import numpy as np

PARQUET = 'processed_with_preds_both.parquet'
OUT_CSV = 'baseline_results_v2.csv'

# ---------------------------------------------------------------- 데이터 적재
df = pd.read_parquet(PARQUET)
df['year'] = df['ts_kst'].dt.year


def to_split(y):
    if y <= 2022:
        return 'Train(21-22)'
    if y == 2023:
        return 'Val(2023)'
    return 'Test(2024)'


df['split'] = df['year'].map(to_split)

# 오염물질별 컬럼 정의: (라벨, 타깃, persistence 예측=현재 관측값, RF, XGB)
SPECS = [
    ('PM10',  'target_pm10', 'pm10', 'y_pred_rf',      'y_pred_xgb'),
    ('PM2.5', 'target_pm25', 'pm25', 'y_pred_rf_pm25', 'y_pred_xgb_pm25'),
]

SPLITS = ['Train(21-22)', 'Val(2023)', 'Test(2024)']


# ---------------------------------------------------------------- 지표 계산
def metrics(y, p):
    """공통 표본이 이미 적용된 상태를 전제로 한다."""
    err = p - y
    rmse = float(np.sqrt((err ** 2).mean()))
    mae = float(err.abs().mean())
    r2 = float(1 - (err ** 2).sum() / ((y - y.mean()) ** 2).sum())
    return dict(n=int(len(y)), R2=round(r2, 4),
                RMSE=round(rmse, 2), MAE=round(mae, 2))


rows = []

for label, tgt, per, rf, xgb in SPECS:
    for sp in SPLITS:
        d = df[df['split'] == sp]

        # ★ 핵심: 네 계열이 모두 유효한 행만 남긴다
        mask = (d[tgt].notna() & d[per].notna()
                & d[rf].notna() & d[xgb].notna())
        dd = d[mask]

        if len(dd) < 10:
            continue

        base = metrics(dd[tgt], dd[per])
        for name, col in [('Persistence', per), ('RF', rf), ('XGB', xgb)]:
            m = metrics(dd[tgt], dd[col])
            skill = np.nan if name == 'Persistence' else \
                round((1 - m['RMSE'] / base['RMSE']) * 100, 2)
            rows.append(dict(Pollutant=label, Split=sp, Model=name,
                             **m, Skill_pct=skill))

res = pd.DataFrame(rows)

pd.set_option('display.width', 200)
print('=== 공통 표본 기준 성능 ===')
print(res.to_string(index=False))

print('\n=== Skill score (1 - RMSE_model / RMSE_persistence) ===')
for label, *_ in SPECS:
    for sp in ['Val(2023)', 'Test(2024)']:
        sub = res[(res.Pollutant == label) & (res.Split == sp)].set_index('Model')
        n = int(sub.loc['Persistence', 'n'])
        base_rmse = sub.loc['Persistence', 'RMSE']
        for mdl in ['RF', 'XGB']:
            print(f'{label:6s} {sp:12s} {mdl:4s}: '
                  f'RMSE {sub.loc[mdl, "RMSE"]:6.2f} vs persistence {base_rmse:6.2f}'
                  f'  ->  skill {sub.loc[mdl, "Skill_pct"]:+6.2f}%  (n={n})')

res.to_csv(OUT_CSV, index=False, encoding='utf-8-sig')
print(f'\n저장: {OUT_CSV}')
