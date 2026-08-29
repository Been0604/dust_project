# Daejeon Dust Project

An end-to-end pipeline, built with **public data only**, that:

- forecasts **PM10/PM2.5 one hour ahead** and evaluates it **against a persistence baseline**,
- maps **spatial vulnerability** across Daejeon (2021–2023), and
- checks whether recent-year patterns are **spatially repeatable** (a 2026 "analog-year" view, *not* a future projection).

The work is written up in a Korean thesis-style report:

> 「대전시 공공 대기질 자료를 활용한 1시간 뒤 미세먼지 예측 성능 평가와 공간 취약성 진단」
> (v1 2025-12 forecasting → v2 2026-08 revision: spatial vulnerability + common-sample re-evaluation)

This repo is a **code portfolio** and a reproducible record of the pipeline.

---

## 1. What I wanted to know

- **RQ1 — Forecasting vs. a persistence baseline**

  > With only public PM and weather data and simple time-series features, do
  > Random Forest / XGBoost predicting **1-hour-ahead PM10/PM2.5** at 11 urban
  > stations in Daejeon actually **beat a persistence baseline** ("next hour =
  > current value"), and does that hold when the year changes?

- **RQ2 — Spatial vulnerability (2021–2023)**

  > Which administrative dongs and which side of the city (East vs West) are
  > structurally more exposed, in terms of mean PM10/PM2.5, annual-standard
  > exceedance, and exposure during local PM alerts and national dust advisories?

- **RQ3 — Spatial repeatability of recent-year patterns**

  > Treating 2021, 2022 and 2023 each as an analog year, are there areas that stay
  > high-risk across three analog years, two models, and both pollutants?
  >
  > Because the models take the previous hour's *observed* concentration as a
  > feature, this is **a repeatability check on past patterns, not a projection of
  > future concentrations.**

---

## 2. Data & study area

- **City:** Daejeon, Korea — 11 urban air-quality monitoring stations
- **Spatial units:** administrative dongs; East (Dong / Jung / Daedeok) vs West (Seo / Yuseong)
- **Period:** 2021-01-01 01:00 – 2024-12-31 23:00, hourly

| Source | What it provides | How it enters the repo |
|---|---|---|
| AirKorea — monthly finalized hourly files | PM10 / PM2.5 | manual download → `data/final_monthly/*.xlsx` |
| AirKorea — PM alert history | local advisory / warning records | manual download → `data/airkorea_pm_alerts/*.xlsx` |
| AirKorea — station API | station coordinates | `src/fetch_stations_api.py` |
| KMA ASOS API | temperature, humidity, wind, precipitation, pressure | `src/make_raw_weather_final_from_api.py` |
| National dust advisory API | Asian-dust / fine-dust stages | `src/fetch_national_alerts.py` |
| 법정동 경계 (LSMD_ADM_SECT_UMD) | dong polygons | shapefile in `daejeon_umd/` |

**API keys** are read from `config/` at runtime and are **not** in this repo. See §7.

---

## 3. Repository layout

```text
dust_project/
  ├─ src/                 # every script used by the current pipeline
  │   ├─ baseline.py            # persistence-baseline evaluation (authoritative)
  │   ├─ baseline_results.csv   # its output — the only tracked result table
  │   └─ _legacy/               # superseded v1 scripts, kept for history
  ├─ data/                # all inputs and intermediate tables   (git-ignored, ~3.5 GB)
  ├─ features/            # v1 train/val/test splits             (git-ignored, superseded)
  ├─ models/              # trained RF / XGBoost artifacts        (git-ignored, ~7 GB)
  ├─ config/              # API keys, local only                  (git-ignored)
  ├─ daejeon_umd/         # dong boundary shapefile
  ├─ boundaries/          # East/West sector layer
  ├─ bi/                  # Power BI file, dashboard exports (.pbix / .pdf / .png)
  ├─ reports/             # per-condition metric tables; figures are git-ignored
  ├─ dust_pm.qgz          # QGIS project
  └─ README.md
```

**Why `data/`, `features/`, `models/` are excluded.** They are regenerated outputs and
they are large: `models/random_forest_pm10.joblib` alone is 3.7 GB, `random_forest_pm25.joblib`
3.4 GB, and `data/umd_time_pm_2021_2023.gpkg` is 2.0 GB — all far past GitHub's 100 MB
per-file limit. Trained `.joblib` files are also tied to specific scikit-learn / XGBoost
versions, so they would not load reliably elsewhere anyway. Everything needed to rebuild
them is in `src/`; §6 lists what each file is.

---

## 4. Pipeline

Every command is run **from the repository root**. Some scripts resolve paths from their
own location, others from the working directory, so running them from inside `src/` will
fail.

### 4.1 Station table

```bash
python src/fetch_stations_api.py     # → data/stations.csv
python src/assign_station_area.py    # adds East/West sector, rewrites data/stations.csv
```

### 4.2 Raw tables

Place the manually downloaded AirKorea files first:
`data/final_monthly/airkorea_hourly_YYYY_MM.xlsx` (2021-01 … 2024-12) and
`data/airkorea_pm_alerts/*.xlsx`.

```bash
python src/make_raw_pm_from_monthly_excels.py   # final_monthly + stations.csv → data/raw_pm_final.parquet
python src/make_raw_weather_final_from_api.py   # KMA ASOS               → data/raw_weather_final.parquet
python src/fetch_national_alerts.py             #                        → data/national_alert_raw.csv
python src/make_dust_events_from_raw.py         # national_alert_raw.csv → data/national_alert_events_daejeon.parquet
python src/make_local_alert_events_pm10.py      # airkorea_pm_alerts/    → data/local_alert_events_pm10.parquet
python src/make_local_alert_events_pm25.py      # airkorea_pm_alerts/    → data/local_alert_events_pm25.parquet
```

### 4.3 Merged tables

```bash
python src/make_pm_with_alerts.py        # raw_pm_final + local alerts        → data/pm_with_alerts.parquet
python src/make_processed_final.py       # pm_with_alerts + raw_weather_final → data/processed_final.parquet
python src/make_processed_with_dust.py   # processed_final + dust events      → data/processed_with_dust.parquet
```

### 4.4 Features, models, predictions

```bash
python src/features.py                        # processed_final  → data/features_full_pm10.parquet, features_full_pm25.parquet
python src/train_models_both.py               # features_full_*  → models/random_forest_*.joblib, xgb_*.joblib
python src/make_predictions_full.py           # features_full_*  → data/predictions_full_rf_xgb.parquet
python src/make_processed_with_preds_both.py  # processed_with_dust + predictions → data/processed_with_preds_both.parquet
```

`data/processed_with_preds_both.parquet` is the **master table**: observation, target,
persistence input, both model predictions, residuals, and alert/dust labels, per
station-hour.

**Feature construction**

- time features: `year`, `month`, `weekday`, `hour`, `is_weekend`
- PM lags: `shift(1/2/3)` within each station
- trailing rolling means: 3 / 6 / 12 / 24 h
- weather lag-1: temperature, humidity, wind speed/direction, precipitation, pressure
- targets: `target_pm10 = pm10.shift(-1)`, `target_pm25 = pm25.shift(-1)`

**No leakage by construction.** Lags and rolling means look only backward; only the target
reaches into the next hour. Splits are **temporal**, not random:

| Split | Years |
|---|---|
| Train | 2021–2022 |
| Validation | 2023 |
| External test | 2024 |

### 4.5 Evaluation

```bash
python src/baseline.py                    # master table → src/baseline_results.csv
python src/print_main_metrics.py          # prints the headline numbers
python src/analyze_pm_performance_both.py # per year / season / hour / station / alert / dust → reports/metrics_*.csv
```

`src/baseline.py` is the authoritative evaluation. It computes the persistence baseline
(`pred(t+1) = obs(t)`), the RF/XGBoost metrics, and the skill score
`1 − RMSE_model / RMSE_persistence`.

**Common sample.** All four series — target, persistence prediction, RF prediction,
XGBoost prediction — must be present for a station-hour to enter the evaluation
(complete-case analysis). An earlier version filtered missing values per model, so
persistence and the ML models were scored on *different* samples and the resulting skill
scores were not comparable. Every number in §5 and in the report is on the common sample.

| Split | PM10 rows | PM2.5 rows |
|---|---|---|
| Train (2021–22) | 186,064 | 185,611 |
| Validation (2023) | 93,615 | 93,800 |
| External test (2024) | 93,189 | 92,249 |

### 4.6 Spatial analysis (QGIS + GeoPandas, 2021–2023)

```bash
python src/make_emd_codes.py                      # daejeon_umd shapefile → data/bi/emd_codes.csv
python src/make_for_qgis_exports.py               # master table → data/for_qgis_2021_2023.parquet, for_qgis_2024.parquet
python src/make_umd_time_pm10_2021_2023.py        # → data/umd_time_pm_2021_2023.gpkg
python src/make_admin_stats_summary_2021_2023.py  # → data/admin_stats_summary_2021_2023.parquet
```

Station points (WGS84 → EPSG:5186) are spatially joined to dong polygons; hourly values
are aggregated to dong level, then summarised as 3-year means and exceedance shares.

> **Manual step.** `data/stations_emd_mapping.csv` (station → dong lookup, used by the
> Power BI scripts below) is produced in QGIS, not by a script. Regenerating it means
> repeating that spatial join by hand.

### 4.7 Power BI summary tables

```bash
python src/make_bi_admin_pm_stats_2021_2023.py   # → data/bi/bi_admin_pm_stats_2021_2023.{csv,parquet}
python src/make_bi_area_ew_stats_2021_2023.py    # → data/bi/bi_area_ew_stats_2021_2023.*
python src/make_bi_alert_stats.py                # → data/bi/bi_alert_{pm10,pm25,dust}_alert.*
python src/make_bi_model_perf_2021_2024.py       # → data/bi/bi_model_perf_2021_2024.*
```

These feed `bi/dust_project_overview.pbix`. Rendered dashboard pages are in
`bi/dust_project_overview.png/` and `bi/dust_project_overview.pdf`.

### 4.8 2026 analog-year check

```bash
python src/make_scenario_2026_step1.py           # processed_final + features_full_* → data/scenario/scenario_base_20{21,22,23}.parquet, scenario_2026_features_*
python src/make_scenario_2026_step2_predict.py   # apply trained RF / XGBoost        → data/scenario/scenario_2026_preds_*
python src/make_scenario_2026_step3_admin_stats.py  # → data/scenario/bi_admin_pm_stats_2026_*, bi_area_ew_stats_2026_*
python src/make_umd_pm_stats_2026_scenarios.py   # → data/scenario/umd_pm_stats_2026_scenarios.gpkg
python src/add_pm_classes_2026.py                # adds risk classes → ..._with_classes.gpkg
python src/make_umd_ensemble_2026.py             # ensemble over years × models × pollutants → umd_ensemble_2026_pm10_pm25.gpkg
python src/convert_bi_2026_to_csv.py             # parquet → csv for Power BI
```

`src/peek_parquet.py` is a utility for inspecting any parquet under `data/`.

---

## 5. Results

Common-sample values, reproduced by `python src/baseline.py`
(full table: `src/baseline_results.csv`).

### 5.1 RQ1 — Forecasting vs. persistence

| Pollutant | Split | Model | R² | RMSE (µg/m³) | MAE (µg/m³) | Skill |
|---|---|---|---|---|---|---|
| PM10 | Validation (2023) | Persistence | 0.925 | 8.76 | 5.41 | — |
| | | RF | 0.904 | 9.93 | 5.47 | **−13.4 %** |
| | | XGB | 0.879 | 11.17 | 5.62 | **−27.5 %** |
| | External test (2024) | Persistence | 0.874 | 8.01 | 4.79 | — |
| | | RF | 0.864 | 8.32 | 4.67 | **−3.9 %** |
| | | XGB | 0.857 | 8.54 | 4.65 | **−6.6 %** |
| PM2.5 | Validation (2023) | Persistence | 0.859 | 5.62 | 3.52 | — |
| | | RF | 0.876 | 5.27 | 3.36 | **+6.2 %** |
| | | XGB | 0.877 | 5.26 | 3.29 | **+6.4 %** |
| | External test (2024) | Persistence | 0.801 | 4.95 | 3.37 | — |
| | | RF | 0.829 | 4.59 | 3.21 | **+7.3 %** |
| | | XGB | 0.834 | 4.52 | 3.16 | **+8.7 %** |

**PM10 — the models do not beat persistence.** Negative skill in both the validation and
the external-test year. In 2024 the RF's MAE is *lower* than persistence (4.67 vs 4.79)
while its RMSE is *higher* (8.32 vs 8.01): it trims typical errors slightly and pays for
it with large errors during high-concentration episodes.

**PM2.5 — a small but consistent gain**, +6.2 to +8.7 %. The sign holds in both the
validation and the external-test year, so it is unlikely to be noise; the size is too
small to support operational use.

**Overfitting.** Training R² is 0.98–0.99, validation/external-test R² 0.83–0.90.

**Why R² alone is misleading here.** Persistence itself scores R² 0.80–0.93 on the same
data, because hourly PM is strongly autocorrelated. A model at R² 0.86 has added nothing.
**The baseline comparison, not R², is what carries information** — and for PM10 it
reverses the conclusion entirely.

### 5.2 RQ2 — Spatial vulnerability

- Several dongs — Gwanpyeong, Munpyeong, Noeun, Dunsan, Munchang, Jeongnim — are
  structurally high-risk on both mean concentration and share of high-concentration hours.
- The East side (Dong / Jung / Daedeok) is generally more exposed than the West
  (Seo / Yuseong): PM10 ~34–37 vs ~31–35 µg/m³, exceedance-hour share 44 % vs 42 %.
- The same dongs stay high during local alert and dust-advisory periods, so the pattern is
  not an artefact of averaging.

### 5.3 RQ3 — Analog-year repeatability

- Munpyeong, Gwanpyeong, Guseong, Dunsan and Jeongnim recur in the upper exposure range
  across all three analog years and both models.
- In the ensemble view **PM2.5 classifies nearly every station as high-risk while PM10
  classifies none.** This is an artefact of the threshold-to-mean ratio (PM2.5: 25 vs a
  ~17–18 µg/m³ mean; PM10: 50 vs a ~31–37 µg/m³ mean), not a real difference in risk. A
  percentile-based threshold would separate the two effects.
- Because the models consume the previous hour's observed value, this is a repeatability
  check on past patterns, **not a 2026 forecast**.

---

## 6. What `data/` should contain

`data/` is git-ignored, so a fresh clone starts empty. Running §4 in order rebuilds it.

| Path | Produced by | Contents |
|---|---|---|
| `final_monthly/*.xlsx` | manual download | AirKorea monthly hourly files, 2021–2024 |
| `airkorea_pm_alerts/*.xlsx` | manual download | local PM alert history |
| `stations.csv` | `fetch_stations_api.py` + `assign_station_area.py` | station id, coordinates, East/West sector |
| `stations_emd_mapping.csv` | **QGIS spatial join (manual)** | station → dong lookup |
| `raw_pm_final.parquet` | `make_raw_pm_from_monthly_excels.py` | `ts_kst`, `station_id`, `pm10`, `pm25` |
| `raw_weather_final.parquet` | `make_raw_weather_final_from_api.py` | hourly ASOS variables on the same time axis |
| `national_alert_raw.csv` | `fetch_national_alerts.py` | raw advisory records |
| `national_alert_events_daejeon.parquet` | `make_dust_events_from_raw.py` | `dust_stage` per hour, Daejeon only |
| `local_alert_events_pm10/pm25.parquet` | `make_local_alert_events_*.py` | `ts_kst`, `area`, alert label |
| `pm_with_alerts.parquet` | `make_pm_with_alerts.py` | PM + `y_loc_pm10`, `y_loc_pm25` |
| `processed_final.parquet` | `make_processed_final.py` | PM + weather + time features, cleaned |
| `processed_with_dust.parquet` | `make_processed_with_dust.py` | adds `dust_stage` |
| `features_full_pm10/pm25.parquet` | `features.py` | lag / rolling / weather-lag features + targets |
| `predictions_full_rf_xgb.parquet` | `make_predictions_full.py` | RF / XGBoost predictions, 2021–2024 |
| `processed_with_preds_both.parquet` | `make_processed_with_preds_both.py` | **master table** |
| `for_qgis_2021_2023.parquet`, `for_qgis_2024.parquet` | `make_for_qgis_exports.py` | QGIS-ready station-hour tables |
| `umd_time_pm_2021_2023.gpkg`, `umd_pm_stats_2021_2023.gpkg`, `stations_time_2021_2023.gpkg` | QGIS export steps | point and dong-level layers |
| `admin_stats_summary_2021_2023.parquet` | `make_admin_stats_summary_2021_2023.py` | dong-level summary statistics |
| `bi/` | `make_bi_*.py`, `make_emd_codes.py` | Power BI input tables |
| `scenario/` | `make_scenario_2026_step*.py` and the 2026 steps | analog-year features, predictions, dong stats |
| `_legacy/` | v1 pipeline | superseded intermediate tables |

`src/baseline_results.csv` — the evaluation output backing the report's Tables 8 and 9 —
is the one result file kept under version control, because it is small and it is the
evidence for §5.

---

## 7. Setup

**API keys.** Two plain-text files are read at runtime and are never committed:

```text
config/airkorea_key.txt   # AirKorea / 공공데이터포털 service key
config/kma_key.txt        # KMA (기상청) service key
```

Create them yourself with your own keys, one key per file, no quotes. `.gitignore`
excludes everything under `config/` except `*.example` files.

**Environment.** Python 3 with `pandas`, `numpy`, `scikit-learn`, `xgboost`, `joblib`,
`geopandas`, `pyarrow`, `requests`, `openpyxl`.

---

## 8. Limitations & next step

- **Overfitting.** The train-to-test R² gap is 0.09–0.13. Stronger regularisation,
  cross-validated hyperparameter search, and a smaller feature set should come first.
- **Feature ceiling.** Lag, rolling and weather-lag features describe ordinary variation
  but not the sharp rises that matter. The MAE/RMSE split in PM10 2024 is the fingerprint
  of that failure.
- **Horizon.** At a 1-hour horizon persistence is already very strong, so there is
  structurally little room to improve. At 6 h and 24 h persistence degrades quickly and a
  data-driven model has more to contribute; extending the horizon and tracking skill by
  horizon is the obvious next experiment.
- **Spatial resolution.** Eleven station points are a coarse proxy for dong-level exposure.
- **Reproducibility gaps.** Two inputs are not scripted: the AirKorea monthly/alert
  downloads, and the QGIS spatial join that produces `stations_emd_mapping.csv`.

These are the reasons coupling with a physical / chemical transport model (CMAQ,
WRF-Chem) is a **necessary** next step rather than an optional extension: it supplies the
emission, transport and chemistry information this setup structurally lacks — precisely
where the current pipeline fails.
