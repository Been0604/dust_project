# Daejeon Dust Project

This is a personal project where I built an end-to-end pipeline to:

- forecast **PM10/PM2.5 one hour ahead** and evaluate it **against a persistence baseline**,
- map **spatial vulnerability** across Daejeon (2021–2023), and
- check whether recent-year patterns are **spatially repeatable** (a 2026 "analog-year" view, *not* a future projection)

using **only public air-quality and meteorological data**.

The work is summarized in a Korean thesis-style report:

> “대전시 공공 대기질 자료를 활용한 1시간 뒤 미세먼지 예측 성능 평가와 공간 취약성 진단” (v1 2025.12 forecasting → v2 2026.08 spatial-vulnerability revision)

I use this repo mainly as a **code portfolio** and a reproducible record of the pipeline.

---

## 1. What I wanted to know

I structured the project around three questions:

- **RQ1 – Forecasting vs. a persistence baseline**

  > With only public PM and weather data and simple time-series features,
  > do RandomForest/XGBoost predicting **1-hour-ahead PM10/PM2.5** at 11 urban
  > stations in Daejeon actually **beat a persistence baseline** ("next hour = current value"),
  > and does that hold when the year changes?

- **RQ2 – Spatial vulnerability (2021–2023)**

  > Which administrative dongs and which side of the city (East vs West)
  > are structurally more exposed in terms of:
  > - mean PM10/PM2.5,
  > - annual standard exceedance, and
  > - exposure during local PM alerts and national dust advisories?

- **RQ3 – Spatial repeatability of recent-year patterns**

  > If the observed patterns from **2021, 2022, and 2023** are each treated as an
  > analog year, are there areas that stay high-risk across:
  > - three analog years (2021→A, 2022→B, 2023→C),
  > - two models (Random Forest / XGBoost), and
  > - both pollutants (PM10 / PM2.5)?
  >
  > Because the models use the previous hour's observed concentration as a feature,
  > this is **a check of spatial repeatability of past patterns, not a projection of future concentrations.**

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
   - lag features via `shift(+1/+2/+3)` and trailing rolling means (3/6/12/24h) — **past-and-current information only**
   - weather lag-1 features
   - targets (future label):
     - `target_pm10` = `pm10` shifted by `-1` (1-hour-ahead PM10)
     - `target_pm25` = `pm25` shifted by `-1` (1-hour-ahead PM2.5)
   - **No leakage by construction:** features only use information available at prediction time; only the target reaches into the next hour.
   - splits (**temporal**, to avoid random-split leakage):
     - train: **2021–2022**
     - validation: **2023**
     - external test: **2024**
   - models:
     - RandomForestRegressor
     - XGBoost regressor
   - a **persistence baseline** ("1-hour-ahead = current value") is computed and used as the reference for skill scores
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

4. **2026 analog-year check (spatial repeatability)**
   - Scenario A/B/C:
     - A: 2021 patterns as the analog
     - B: 2022 analog
     - C: 2023 analog
   - apply trained RF/XGBoost across the analog years
   - aggregate by dong and East/West
   - build ensemble risk classes ("high-risk" vs "not") from how often high episodes recur across analog years, models, and pollutants
   - **Interpretation:** this reflects repeatability of past patterns, not a forecast of 2026 concentrations.

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
```

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

# 4) 2026 analog-year check
python src/make_scenario_2026_step1.py
python src/make_scenario_2026_step2_predict.py
python src/make_scenario_2026_step3_admin_stats.py
python src/make_umd_ensemble_2026.py
```

You'll need to adapt paths and configs if your folder layout or data sources differ.

---

## 6. A quick summary of findings

Short version of the results:

* **Forecasting (RQ1):**

  * Judged **against a persistence baseline** (not by R² alone):
    * **PM10 — the models do NOT beat persistence.** Skill score
      (`1 − RMSE_model / RMSE_persistence`) is negative:
      RF −18.5% / XGB −31.9% on 2023 (val), RF −4.5% / XGB −7.2% on 2024 (test).
    * **PM2.5 — only a marginal gain over persistence:** about +4.5–4.6% (2023) and +5.9–7.1% (2024).
  * Training R² was ~0.99 but dropped to **0.86–0.90** on validation/test → clear **overfitting**.
  * Takeaway: a high R² (~0.8–0.9) here is misleading, because a trivial persistence
    prediction already scores that high on strongly autocorrelated hourly PM. R² alone
    overstates the model's contribution; the persistence comparison is what matters.

* **Spatial vulnerability (RQ2):**

  * Several administrative dongs (e.g. Gwanpyeong-dong, Munpyeong-dong, Noeun-dong, Dunsan-dong, Munchang-dong, Jeongnim-dong) appear as **structurally high-risk**.
  * The East side of Daejeon (Dong/Jung/Daedeok) is generally more vulnerable than the West (Seo/Yuseong).

* **2026 analog-year check (RQ3):**

  * In the ensemble view, **PM2.5** classifies nearly all stations as high-risk while **PM10** classifies none.
  * This difference comes largely from the **threshold-to-mean ratio differing by pollutant**, so it should **not** be read as a real difference in risk.
  * Because the models use the previous hour's observed value as a feature, this is a **repeatability check of past patterns, not a 2026 forecast.**

---

## 7. Limitations & next step

* Data-driven, station-based features hit a ceiling: they do not reliably beat persistence for 1-hour-ahead PM, and station points are a coarse proxy for dong-level exposure.
* This limitation is *why* coupling with a physical/chemical transport model (e.g. CMAQ) is a **necessary next step**, not just an optional extension — to add process information the ML setup structurally lacks.
