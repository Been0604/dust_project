# Daejeon Dust Project

This is a personal project where I built an end-to-end pipeline to:

- forecast **PM10/PM2.5 one hour ahead**,
- map **spatial vulnerability** inside Daejeon, and
- explore a simple **“what if 2026 looks like recent years?”** scenario

using **only public air-quality and meteorological data**.

The work is summarized in a Korean thesis-style report:

> “대전시 공공 대기질 자료를 활용한 미세먼지 예측, 공간 취약성 진단 및 2026년 근미래 시나리오 분석” (2025)

I use this repo mainly as a **code portfolio** and a reproducible record of the pipeline.

---

## 1. What I wanted to know

I structured the project around three questions:

- **RQ1 – Forecasting**

  > With only public PM and weather data and simple time-series features,  
  > can we predict **1-hour-ahead PM10/PM2.5** at 11 urban stations in Daejeon,  
  > in a way that still works when the year changes?

- **RQ2 – Spatial vulnerability (2021–2023)**

  > Which administrative dongs and which side of the city (East vs West)  
  > are structurally more exposed in terms of:
  > - mean PM10/PM2.5,
  > - annual standard exceedance, and
  > - exposure during local PM alerts and national dust advisories?

- **RQ3 – Near-future 2026 scenarios**

  > If the patterns from **2021, 2022, and 2023** were to repeat in **2026**,  
  > are there areas that stay high-risk across:
  > - three “analog” years (2021→A, 2022→B, 2023→C),
  > - two models (Random Forest / XGBoost), and
  > - both pollutants (PM10 / PM2.5)?

---

## 2. Data & study area

- **City:** Daejeon, Korea
- **Stations:** 11 urban air-quality monitoring stations  
- **Spatial units:**
  - administrative dongs
  - East vs West sectors  
    (Dong / Jung / Daedeok vs Seo / Yuseong)

**Data sources (all public):**

- Hourly PM10/PM2.5 from **AirKorea** (monthly finalized data, 2021–2024)
- Hourly weather from **KMA ASOS** (temperature, humidity, wind, precipitation, etc.)
- Local PM10/PM2.5 alert history
- National dust / fine-dust advisories

Raw data files are **not included** in this repo.  
All scripts assume a consistent folder structure under `data/`, so if you download similar data you can re-run the pipeline.

---

## 3. Pipeline overview

Rough structure of the project:

1. **Data pipeline (Python)**
   - build `raw_pm_final` from AirKorea Excel files  
   - fetch and aggregate ASOS weather into `raw_weather_final`  
   - create local alert timelines and dust stages  
   - merge everything into `processed_final` / `processed_with_dust`

2. **Feature engineering & models**
   - time features (year, month, weekday, hour, weekend flag)
   - lag & rolling features for PM10/PM2.5 and key weather variables
   - targets:
     - `target_pm10` = 1-hour-ahead PM10
     - `target_pm25` = 1-hour-ahead PM2.5
   - splits:
     - train: **2021–2022**
     - validation: **2023**
     - external test: **2024**
   - models:
     - RandomForestRegressor
     - XGBoost regressor
   - full-period predictions → `predictions_full_rf_xgb`
   - merged master table → `processed_with_preds_both`

3. **Spatial analysis (QGIS, 2021–2023)**
   - hourly station data → QGIS export
   - station points → administrative dongs (spatial join)
   - dong-level hourly means and exposure indicators
   - maps:
     - 3-year mean PM10 / PM2.5
     - East vs West contrast
     - selected alert / dust events

4. **2026 multi-analog scenarios**
   - Scenario A/B/C:
     - A: 2021 patterns shifted to 2026
     - B: 2022 → 2026
     - C: 2023 → 2026
   - use trained RF/XGBoost to predict 2026 PM10/PM2.5
   - aggregate by dong and East/West
   - build ensemble risk classes:
     - “high-risk” vs “not high-risk” based on frequency of high episodes across
       scenarios, models, and pollutants

---

## 4. Repository layout

High-level structure (simplified):

```text
dust_project/
  ├─ src/                  # all Python scripts
  ├─ config/               # config_example.yml etc. (no secrets)
  ├─ data/                 # raw/processed data (git-ignored)
  ├─ models/               # trained models (git-ignored)
  ├─ reports/              # thesis-style report, figures (mostly git-ignored)
  ├─ environment.yml       # conda environment (or requirements.txt)
  └─ README.md
````

The **core logic** lives in `src/`.
Most heavy outputs (`data/`, `models/`, detailed figures) are not tracked by git.

---

## 5. Running the main pipeline

Basic idea (assuming a conda env named `dust`):

```bash
# Create and activate environment (example)
conda env create -f environment.yml
conda activate dust

# 1) Data tables
python src/make_raw_pm_from_monthly_excels.py
python src/make_raw_weather_final_from_api.py
python src/make_local_alert_events_pm10.py
python src/make_local_alert_events_pm25.py
python src/make_pm_with_alerts.py
python src/make_processed_final.py
python src/make_processed_with_dust.py

# 2) Features & models
python src/features.py
python src/models/random_forest_baseline.py
python src/models/xgboost_baseline.py
python src/make_predictions_full.py
python src/make_processed_with_preds_both.py

# 3) QGIS exports
python src/make_for_qgis_exports.py
python src/make_umd_time_pm10_2021_2023.py

# 4) 2026 scenarios
python src/make_scenario_2026_step1.py
python src/make_scenario_2026_step2_predict.py
python src/make_scenario_2026_step3_admin_stats.py
python src/make_umd_ensemble_2026.py
```

You’ll need to adapt paths and configs if your folder layout or data sources differ.

---

## 6. A quick summary of findings

Very short version of the results:

* **Forecasting (RQ1):**

  * R² around **0.8–0.9** for both PM10 and PM2.5 on 2023 (validation) and 2024 (external test).
  * Errors are clearly larger during high-concentration episodes (local alerts / dust advisories).

* **Spatial vulnerability (RQ2):**

  * Several administrative dongs (e.g. Gwanpyeong-dong, Munpyeong-dong, Noeun-dong, Dunsan-dong, Munchang-dong, Jeongnim-dong) appear as **structurally high-risk**.
  * The East side of Daejeon (Dong/Jung/Daedeok) is generally more vulnerable than the West (Seo/Yuseong).

* **2026 scenarios (RQ3):**

  * For **PM2.5**, almost all stations are classified as high-risk in the ensemble view.
  * For **PM10**, none of the stations are high-risk under the same rule.


```

