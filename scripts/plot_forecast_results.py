import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
import lightgbm as lgb

# ==========================================
# 0. SETUP PATHS & DIRECTORIES
# ==========================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PLOTS_DIR = PROJECT_ROOT / "plots"
SUBMISSION_PATH = PROJECT_ROOT / "submission.csv"

PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================
# 1. LOAD HISTORICAL ACTUALS & SUBMISSION
# ==========================================
print("\n--- [1/3] Loading Data Files ---")
sales_path = DATA_DIR / "sales.csv"

if not sales_path.exists():
    raise FileNotFoundError(f"Missing {sales_path}. Ensure sales.csv is inside the 'data' directory.")
if not SUBMISSION_PATH.exists():
    raise FileNotFoundError(f"Missing {SUBMISSION_PATH}. Please run run_forecasting.py first to generate submission.csv.")

df_sales = pd.read_csv(sales_path, parse_dates=["Date"])
df_sub = pd.read_csv(SUBMISSION_PATH, parse_dates=["Date"])

print(f"Loaded {len(df_sales):,} historical records ({df_sales['Date'].min().date()} to {df_sales['Date'].max().date()}).")
print(f"Loaded {len(df_sub):,} forecasted records ({df_sub['Date'].min().date()} to {df_sub['Date'].max().date()}).")

# ==========================================
# 2. RUN 2022 HOLDOUT VALIDATION FOR PLOTTING
# ==========================================
print("\n--- [2/3] Computing 2022 Holdout Validation Benchmark ---")
TAU = 2 * np.pi
TET_DATES = {
    2011: "2011-02-03", 2012: "2012-01-23", 2013: "2013-02-10", 2014: "2014-01-31",
    2015: "2015-02-19", 2016: "2016-02-08", 2017: "2017-01-28", 2018: "2018-02-16",
    2019: "2019-02-05", 2020: "2020-01-25", 2021: "2021-02-12", 2022: "2022-02-01",
    2023: "2023-01-22", 2024: "2024-02-10", 2025: "2025-01-29", 2026: "2026-02-17"
}

def build_features(dates):
    df = pd.DataFrame({'Date': pd.to_datetime(dates)})
    d = df['Date']
    df['year'] = d.dt.year
    df['month'] = d.dt.month
    df['day'] = d.dt.day
    df['dow'] = d.dt.dayofweek
    df['doy'] = d.dt.dayofyear
    df['quarter'] = d.dt.quarter
    df['is_weekend'] = (df['dow'] >= 5).astype(int)
    df['time_idx'] = (d - pd.Timestamp('2011-01-01')).dt.days
    
    dim = d.dt.days_in_month
    df['days_to_eom'] = dim - df['day']
    for k in [1, 2, 3]:
        df[f'is_last{k}'] = (df['days_to_eom'] <= k - 1).astype(int)
        df[f'is_first{k}'] = (df['day'] <= k).astype(int)

    for k in range(1, 6):
        df[f'sin_y{k}'] = np.sin(TAU * k * df['doy'] / 365.25)
        df[f'cos_y{k}'] = np.cos(TAU * k * df['doy'] / 365.25)
    for k in range(1, 3):
        df[f'sin_w{k}'] = np.sin(TAU * k * df['dow'] / 7.0)
        df[f'cos_w{k}'] = np.cos(TAU * k * df['dow'] / 7.0)

    tet_lut = {y: pd.Timestamp(v) for y, v in TET_DATES.items()}
    df['tet_days_diff'] = d.apply(lambda dt: (dt - tet_lut.get(dt.year)).days if dt.year in tet_lut else 999)
    df['tet_in_7'] = (np.abs(df['tet_days_diff']) <= 7).astype(int)
    return df

# Build holdout features
X_all = build_features(df_sales["Date"])
feature_cols = [c for c in X_all.columns if c not in ["Date", "year"]]

train_mask = X_all['year'] < 2022
val_mask = X_all['year'] == 2022

val_dates = df_sales.loc[val_mask, 'Date']
actuals_2022 = df_sales.loc[val_mask, 'Revenue'].values
val_preds_raw = np.array([])
mape_score = 0.0

if val_mask.sum() > 0 and train_mask.sum() > 0:
    val_years = X_all.loc[train_mask, 'year'].values
    val_weights = np.full(train_mask.sum(), 0.10)
    val_weights[val_years >= 2020] = 1.0

    y_train = np.log1p(df_sales.loc[train_mask, 'Revenue'].values)
    dval_train = lgb.Dataset(X_all.loc[train_mask, feature_cols].values, label=y_train, weight=val_weights)
    
    lgb_params = {
        'objective': 'regression', 'metric': 'mae', 'learning_rate': 0.03,
        'num_leaves': 63, 'verbosity': -1, 'seed': 42
    }
    val_model = lgb.train(lgb_params, dval_train, num_boost_round=600)
    val_preds_log = val_model.predict(X_all.loc[val_mask, feature_cols].values)
    val_preds_raw = np.expm1(val_preds_log)
    mape_score = mean_absolute_percentage_error(actuals_2022, val_preds_raw) * 100
    print(f"2022 Holdout Validation MAPE: {mape_score:.2f}%")

# ==========================================
# 3. GENERATE AND SAVE PLOTS
# ==========================================
print("\n--- [3/3] Generating Visualizations in /plots ---")

# --- Plot 1: Full Historical Timeline vs. Out-of-Sample Forecast ---
fig, ax = plt.subplots(figsize=(15, 6), dpi=150)
ax.plot(df_sales["Date"], df_sales["Revenue"], label="Historical Actual Revenue", color="#1F3A5F", alpha=0.55, linewidth=1.2)
ax.plot(df_sub["Date"], df_sub["Revenue"], label="Predicted Revenue (2023-2024)", color="#E63946", linewidth=1.6)

# 30-Day Moving Average Trendline for Future Predictions
df_sub["MA30"] = df_sub["Revenue"].rolling(30, min_periods=1).mean()
ax.plot(df_sub["Date"], df_sub["MA30"], label="Forecast 30-Day Trend (MA30)", color="#F4A261", linewidth=2.2, linestyle="--")

ax.axvline(df_sub["Date"].min(), color="black", linestyle=":", alpha=0.8, label="Forecast Boundary (2023-01-01)")
ax.set_title("Retail Datathon 2026: End-to-End Revenue Trajectory & Projections", fontsize=13, fontweight="bold")
ax.set_xlabel("Date", fontsize=10)
ax.set_ylabel("Daily Revenue ($)", fontsize=10)
ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("${x:,.0f}"))
ax.grid(True, linestyle="--", alpha=0.4)
ax.legend(loc="upper left")
fig.tight_layout()

chart1_path = PLOTS_DIR / "1_full_forecast_timeline.png"
fig.savefig(chart1_path)
plt.close(fig)
print(f"  --> Saved: {chart1_path}")

# --- Plot 2: 2022 Model Accuracy (Holdout Validation) ---
if len(val_preds_raw) > 0:
    fig, ax = plt.subplots(figsize=(14, 5), dpi=150)
    ax.plot(val_dates, actuals_2022, label="2022 Actual Revenue", color="#2A9D8F", linewidth=1.8)
    ax.plot(val_dates, val_preds_raw, label=f"2022 LightGBM Preds (MAPE: {mape_score:.2f}%)", color="#E76F51", linestyle="--", linewidth=1.8)
    ax.set_title("Model Validation Check: 2022 Actuals vs. Predicted", fontsize=12, fontweight="bold")
    ax.set_xlabel("Date", fontsize=10)
    ax.set_ylabel("Daily Revenue ($)", fontsize=10)
    ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("${x:,.0f}"))
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="upper left")
    fig.tight_layout()

    chart2_path = PLOTS_DIR / "2_validation_holdout_2022.png"
    fig.savefig(chart2_path)
    plt.close(fig)
    print(f"  --> Saved: {chart2_path}")

# --- Plot 3: 2023-2024 Revenue vs. COGS Profitability Spread ---
fig, ax = plt.subplots(figsize=(14, 5), dpi=150)
ax.plot(df_sub["Date"], df_sub["Revenue"], label="Forecasted Revenue", color="#1D3557", linewidth=1.6)
ax.plot(df_sub["Date"], df_sub["COGS"], label="Forecasted COGS", color="#457B9D", linewidth=1.4, linestyle="--")
ax.fill_between(df_sub["Date"], df_sub["COGS"], df_sub["Revenue"], color="#A8DADC", alpha=0.45, label="Projected Gross Profit")

ax.set_title("Forecasted Revenue vs. COGS Margin Spread (2023 - 2024)", fontsize=12, fontweight="bold")
ax.set_xlabel("Date", fontsize=10)
ax.set_ylabel("Amount ($)", fontsize=10)
ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("${x:,.0f}"))
ax.grid(True, linestyle="--", alpha=0.4)
ax.legend(loc="upper left")
fig.tight_layout()

chart3_path = PLOTS_DIR / "3_margin_spread_forecast.png"
fig.savefig(chart3_path)
plt.close(fig)
print(f"  --> Saved: {chart3_path}")

print("\nVisualizations successfully saved to the 'plots/' directory.")