
```markdown
# 📊 Retail Datathon 2026 — Executive BI Suite & Forecasting Engine

An end-to-end analytics and machine learning pipeline designed for retail operational intelligence. This repository features a **5-tier Executive Business Intelligence Dashboard** built with Streamlit and Plotly, alongside a **Hierarchical Time-Series Forecasting Engine** combining linear parametric models and tree-based boosting to project daily revenue and cost of goods sold (COGS) across an 18-month horizon.

---

## 🏛️ Repository Structure

```text
RETAIL_DATATHON_2026/
├── data/                         # Raw historical transaction & dimension tables
│   ├── sales.csv                 # Daily company-wide revenue & COGS (2012–2022)
│   ├── orders.csv                # Enriched order-level transaction logs
│   ├── promotions.csv            # Promotional campaign calendars
│   ├── web_traffic.csv           # Daily session & bounce rate metrics
│   └── inventory.csv             # SKU-level warehouse inventory snapshots
├── scripts/
│   ├── app.py                    # Streamlit Executive Intelligence BI Suite
│   ├── run_forecasting.py        # Ridge + LightGBM Hierarchical Forecasting Engine
│   └── prepare_data.py           # ETL pipeline & data transformation script
├── tableau_data/                 # Aggregated marts for BI & external visualization
├── .gitignore                    # Excludes venv/, system files, and large raw CSVs
├── requirements.txt              # Project dependencies
└── submission.csv                # Final 18-month daily forecast payload (2023–2024)

```

---

## 📈 1. Executive Business Intelligence Suite (`scripts/app.py`)

The BI Suite provides full-stack **Descriptive, Diagnostic, Predictive, and Prescriptive** operational analytics across five functional areas. Built with defensive programming principles, it features cross-version Pandas compatibility (`ME` resampling offsets) and Plotly layout normalization (`barnorm="percent"` for 100% stacked bar charts).

### Dashboard Architecture

| Tab | Area | Core Strategic Questions Addressed | Key Visualizations |
| --- | --- | --- | --- |
| **D1** | **Revenue & Profitability** | What is the historical growth trajectory, and are gross margins sustainable? | • Monthly Revenue & MA30 Overlay<br>
| **D2** | **Customer Segmentation** | Who are our high-value Champions, and which segments are at risk of churning? | • RFM Segment Population Distribution<br>
| **D3** | **Product Analysis** | Which SKUs drive real profitability, and what causes product returns? | • Category Volume Share Pie Chart<br>
| **D4** | **Marketing & Channels** | Which acquisition channels generate the best ROI, and are promos effective? | • Monthly Revenue Share by Channel<br>
| **D5** | **Operations & Supply Chain** | Are regional delivery lead times optimal, and where are stockouts occurring? | • Inventory Health Snapshot by Status<br>

---

## 🤖 2. Time-Series Forecasting Engine (`scripts/run_forecasting.py`)

The forecasting engine generates multi-step daily projections for **Revenue** and **COGS** from **January 1, 2023, to July 1, 2024** (547 days). It is engineered to prevent recursive error drift by relying on deterministic calendar mathematics and hierarchical model ensembling.

### Modeling Methodology & Ensembling Layer

```text
                        [ Deterministic Feature Matrix ]
                         (Calendar, Fourier, Lunar, Trend)
                                        │
             ┌──────────────────────────┴──────────────────────────┐
             ▼                                                     ▼
 [ Family 1: Ridge Regression ]                     [ Family 2: Global LightGBM ]
   (Parametric Linear Anchor)                      (Recency-Weighted Tree Boosting)
  • Alpha = 3.0                                     • sample_weights: 2020–2022 = 1.0
  • Extrapolates long-term growth                   • Captures non-linear seasonality
             │                                                     │
             │                   ┌─────────────────────────────────┘
             │                   ▼
             │       [ Family 3: Quarterly Specialists ]
             │         (4x Isolated LightGBM Models)
             │        • 2.0x weight on target quarter
             │                   │
             │                   ▼
             │     [ Tier 1: Tree Ensemble Blend ]
             │       (60% Specialists + 40% Global)
             │                   │
             └─────────┬─────────┘
                       ▼
       [ Tier 2: Cross-Family Integration ]
          (80% Tree Ensemble + 20% Ridge)
                       │
                       ▼
      [ Tier 3: Growth Calibration Scaling ]
        (Targeted +8% YoY Corporate Growth)
                       │
                       ▼
             [ submission.csv Payload ]

```

### Advanced Feature Engineering (`build_features`)

* **Continuous Trend Index (`time_idx`):** Converts timestamps into elapsed days since project inception (`2011-01-01`). This solves the fundamental limitation of tree-based algorithms, enabling LightGBM to extrapolate YoY revenue growth across multi-year horizons.
* **Fourier Seasonality Terms:** Generates sine and cosine wave transformations for annual (`sin_y1` to `sin_y5`) and weekly (`sin_w1` to `sin_w2`) seasonality, avoiding sparse 365-day one-hot encoding.
* **Lunar Calendar Drift Calculator:** Maps historical and future **Tet (Lunar New Year)** holiday dates (`2011–2026`) and calculates exact `tet_days_diff` and `tet_in_7` proximity flags, protecting Gregorian monthly averages from floating holiday distortions.
* **Deterministic Promotional Windows:** Encodes predictable annual sale campaigns as exogenous flags (`is_spring_sale_window`, `is_midyear_sale_window`, `is_yearend_sale_window`), allowing models to distinguish baseline daily demand from seasonal promotional spikes.
* **Structural Payday Triggers:** Captures salary-driven retail demand surges using edge-of-month (`is_last1..3`) and start-of-month (`is_first1..3`) binary indicators.

---

## 🎯 Model Validation & Performance Scorecard

The script includes an **In-Console Automated Diagnostic Suite** that performs temporal holdout validation (**Train < 2022, Evaluate 2022**) and validates forecast distribution boundaries against historical ceilings.

### Holdout Validation Optimization (Before vs. After)

By introducing the continuous `time_idx` trend feature and applying recency sample weighting (`sample_weights = 1.0` for 2020–2022), the model successfully stopped predicting outdated historical price levels:

| Diagnostic Metric | Baseline Model | Optimized Pipeline | Improvement |
| --- | --- | --- | --- |
| **2022 Holdout MAPE** | 59.50% | **24.00%** | **+35.50 pts error reduction** |
| **2022 Holdout MAE** | $1,624,917.52 | **$749,710.95** | **53.86% error reduction** |
| **Daily Mean Revenue** | $5,544,752.72 *(+73.0% vs 2022)* | **$3,128,676.99** *(–2.38% vs 2022)* | **Eliminated artificial 1.73x inflation** |
| **Daily Peak Revenue** | $20,086,885.89 *(+72.5% vs 2022)* | **$9,691,846.08** *(–16.8% vs 2022)* | **Peak ceiling disciplined to reality** |

---

## 🛠️ Technology Stack

* **Analytics & Dashboarding:** `Streamlit`, `Plotly Express`, `Plotly Graph Objects`
* **Machine Learning & Time-Series:** `LightGBM`, `Scikit-Learn (Ridge)`, `NumPy`, `Pandas`
* **Version Control & CI/CD:** `Git`, `GitHub`

```

```
