# War Predictive Dashboard (Bridgewater Macro Model)

Inspired by Ray Dalio’s "Principles," I built this project to function as a "machine" with a systematic approach to processing data into actionable reality. By mapping out cause-effect relationships within the data, this tool predicts future outcomes and provides a front-end interface to translate complex insights into clear, human-centric narratives. 

Building this is a continuous process of 'radical transparency' and evolution. If you have thoughts on the logic or want to discuss how to improve the forecasting model, I’d love to chat!

---

## Project Structure

```
war predictive/
├── app/
│   └── streamlit_app.py        # Main Streamlit application (all logic, no external modules)
├── data/
│   ├── geopolitical_risk_index.csv   # Monthly GPR data 2000-01 to 2025-05
│   └── data_gpr_export.xls          # Full GPR export from Caldara & Iacoviello (optional detail columns)
├── requirements.txt            # Python dependencies
├── run.sh                      # Convenience launcher script
└── config.yml                  # Project config (reserved)
```

---

## Data

**Source:** Caldara & Iacoviello Geopolitical Risk Index (GPR)

| File | Description |
|---|---|
| `data/geopolitical_risk_index.csv` | Two columns: `Date` (monthly, YYYY-MM-DD) and `GPR` (index value). Covers **2000-01 through 2025-05**. |
| `data/data_gpr_export.xls` | Full XLS export — loaded automatically if present; provides additional sub-index columns that are merged into the main dataframe at startup. |

---

## Dependencies

Declared in `requirements.txt`:

| Package | Minimum Version | Role |
|---|---|---|
| streamlit | 1.32.0 | UI framework |
| pandas | 2.0.0 | Data wrangling |
| numpy | 1.26.0 | Numerical computation |
| plotly | 5.20.0 | All charts |
| scipy | 1.12.0 | Statistics, normal PDF, confidence intervals |
| openpyxl | 3.1.0 | Excel (.xlsx) reading |
| xlrd | 2.0.0 | Legacy Excel (.xls) reading |

---

## Application Architecture

### Data Pipeline

```
load_gpr_data()         — reads CSV + optional XLS, returns raw DataFrame indexed by Date
        |
compute_features()      — adds all derived columns (see below), returns feature DataFrame
        |
sidebar()               — returns user-selected date range, z-score threshold, display toggles
        |
[filter by date range]
        |
kpi_row() + 6 tabs
```

### Derived Feature Columns (computed in `compute_features`)

| Column | Description |
|---|---|
| `MA_3`, `MA_6`, `MA_12`, `MA_24` | Rolling moving averages (3, 6, 12, 24 months) |
| `Std_6`, `Std_12` | Rolling standard deviation |
| `BB_Upper`, `BB_Lower` | Bollinger Bands — `MA_12 ± 2 × Std_12` |
| `Z_12`, `Z_6` | Z-score of GPR relative to 12M and 6M rolling mean/std |
| `MoM`, `QoQ`, `YoY` | Month-on-month, quarter-on-quarter, year-on-year % change |
| `Delta_1`, `Delta_3`, `Delta_12` | Absolute level change over 1, 3, 12 months |
| `Percentile` | Percentile rank of each observation across the full history |
| `Hist_Mean`, `Hist_Std`, `Z_Hist` | Expanding (lifetime) mean, std, and z-score |
| `Regime` | Categorical classification based on `Z_12` (see below) |
| `War_Risk_Score` | Composite 0–100 score (see below) |

### Regime Classification (based on `Z_12`)

| Z-Score Range | Regime |
|---|---|
| z > 2.0 | Extreme |
| 1.0 < z ≤ 2.0 | High Alert |
| 0.5 < z ≤ 1.0 | Elevated |
| z < -1.0 | Low Risk |
| Otherwise | Normal |

### War Risk Score Formula

```
z_norm   = clip(Z_12, -3, 3) / 3            # normalised z-score
mom_norm = clip(QoQ / 20, -1, 1)            # normalised quarterly momentum
pct_norm = (Percentile - 50) / 50           # normalised percentile rank

War_Risk_Score = clip((z_norm×0.5 + mom_norm×0.2 + pct_norm×0.3) × 50 + 50,  0, 100)
```

Weights: **50% z-score**, **30% percentile rank**, **20% momentum**.

---

## Dashboard Tabs

### Tab 1 — Historical Analysis
- Full timeline of GPR with 12M and 24M moving averages
- Optional Bollinger Bands overlay
- Regime-coloured background shading
- Year-on-year % change bar chart (below the main chart)
- Annotated vertical lines for 20 key geopolitical events (toggleable)
- Expandable table of all key events with GPR value, z-score, and regime

**Key events annotated:**

| Date | Event |
|---|---|
| Sep 2001 | 9/11 Attacks |
| Mar 2003 | Iraq War Begins |
| Mar 2014 | Crimea Annexation |
| Jan 2020 | Soleimani Strike |
| Mar 2022 | Russia–Ukraine War |
| Oct 2023 | Israel–Hamas War |
| *(+14 more)* | *(see EVENTS dict in source)* |

### Tab 2 — War Risk Signals
- Three-panel chart: War Risk Score / Z-Score (12M) / Momentum (QoQ %)
- Threshold markers on z-score chart; alert scatter for months exceeding threshold
- Sidebar z-score threshold slider (0.5–3.0, default 1.0)
- Live signal summary panel with colour-coded indicators for 9 metrics
- Count of months in alert state + 5 most recent alert months table

### Tab 3 — Statistical Analysis
- GPR distribution histogram with normal distribution fit overlay
- Descriptive statistics table: mean, median, std, min, max, quartiles, skewness, kurtosis
- Percentile rank time-series chart
- Regime frequency table (months per regime, % of history, average GPR)
- Rolling 12M standard deviation chart
- GPR vs. expanding historical mean chart

### Tab 4 — Forecast & Trend
- Adjustable linear trend fit (lookback: 12–120 months, default 36)
- Forward projection (horizon: 3–24 months, default 12)
- Configurable confidence band (80–99%, default 90%)
- Forecast table with upper/lower confidence bounds
- Trend direction indicator (rising/falling, slope in pts/month)
- Momentum signals table for the most recent 24 months

### Tab 5 — Regime Analysis
- GPR scatter coloured by regime classification
- 12M MA line overlay
- Pie/donut chart of regime distribution across full history
- Regime transition spans table (all periods)
- Expandable table filtered to High Alert and Extreme periods only

### Tab 6 — Data Explorer
- Sortable, scrollable table of key columns: GPR, MA_12, Z_12, War_Risk_Score, Regime, Percentile, MoM, QoQ, YoY
- Sorted descending (most recent first)
- CSV download button

---

## Sidebar Controls

| Control | Default | Effect |
|---|---|---|
| Start Date | 2000-01-01 | Filters data shown in all tabs |
| End Date | Latest available | Filters data shown in all tabs |
| Z-Score Alert Threshold | 1.0 | Controls alert highlighting in Tab 2 |
| Show Key Events | On | Toggles event annotations in Tab 1 |
| Show Bollinger Bands | On | Toggles band overlay in Tab 1 |

Note: KPI metrics at the top always use the **full unfiltered dataset** for percentile context, even when a date range is selected.

---

## Workflow: From Raw Data to War Risk Insight

```
1. Raw monthly GPR values (2000–2025)
           |
2. Feature engineering (rolling stats, z-scores, momentum, percentiles, regime labels)
           |
3. Dashboard loads with full dataset; user selects date range via sidebar
           |
4. KPI row shows current-state snapshot: GPR value, z-score, percentile, war risk score, regime
           |
5. Tabs allow drill-down:
   - Historical tab  → where are we in the long-run context?
   - Signals tab     → are current z-scores and momentum triggering alerts?
   - Stats tab       → how unusual is the current level statistically?
   - Forecast tab    → what does the trend imply for the next 3–24 months?
   - Regime tab      → which risk regime are we in, and how long do regimes typically last?
   - Data tab        → inspect and export the full feature table
```

---
