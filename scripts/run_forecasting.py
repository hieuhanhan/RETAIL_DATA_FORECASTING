from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
import lightgbm as lgb

# ==========================================
# 0. CORE CONSTANTS & LUNAR HOLIDAY LOOKUP
# ==========================================
TAU = 2 * np.pi
TET_DATES = {
    2011: "2011-02-03", 2012: "2012-01-23", 2013: "2013-02-10", 2014: "2014-01-31",
    2015: "2015-02-19", 2016: "2016-02-08", 2017: "2017-01-28", 2018: "2018-02-16",
    2019: "2019-02-05", 2020: "2020-01-25", 2021: "2021-02-12", 2022: "2022-02-01",
    2023: "2023-01-22", 2024: "2024-02-10", 2025: "2025-01-29", 2026: "2026-02-17"
}

def build_features(dates):
  """Generates calendar, seasonality, promo, and continuous trend features."""
  df = pd.DataFrame({'Date': pd.to_datetime(dates)})
  d = df['Date']

  # 1. Standard Calendar Dimensions
  df['year'] = d.dt.year
  df['month'] = d.dt.month
  df['day'] = d.dt.day
  df['dow'] = d.dt.dayofweek
  df['doy'] = d.dt.dayofyear
  df['quarter'] = d.dt.quarter
  df['is_weekend'] = (df['dow'] >= 5).astype(int)
  df['is_odd_year'] = (df['year'] % 2).astype(int)
  df['is_double_day_promo'] = (
      (df['month'].isin([9, 10, 11, 12])) & (df['month'] == df['day'])
  ).astype(int)
  df['is_mid_month_promo'] = (df['day'] == 15).astype(int)

  # 2. Continuous Trend Index (CRITICAL for tree models to learn growth over time)
  base_date = pd.Timestamp('2011-01-01')
  df['time_idx'] = (d - base_date).dt.days

  # 3. Structural Edge-of-Month Payday / Salary Triggers
  dim = d.dt.days_in_month
  df['days_to_eom'] = dim - df['day']
  for k in [1, 2, 3]:
    df[f'is_last{k}'] = (df['days_to_eom'] <= k - 1).astype(int)
    df[f'is_first{k}'] = (df['day'] <= k).astype(int)

  # 4. Fourier Transformations for Smooth Annual & Weekly Seasonality
  for k in range(1, 6):
    df[f'sin_y{k}'] = np.sin(TAU * k * df['doy'] / 365.25)
    df[f'cos_y{k}'] = np.cos(TAU * k * df['doy'] / 365.25)
  for k in range(1, 3):
    df[f'sin_w{k}'] = np.sin(TAU * k * df['dow'] / 7.0)
    df[f'cos_w{k}'] = np.cos(TAU * k * df['dow'] / 7.0)

  # 5. Safe Lunar Calendar Drift Calculator (Tet Holiday Proximity)
  tet_lut = {y: pd.Timestamp(v) for y, v in TET_DATES.items()}

  def calculate_tet_dist(dt):
    base_tet = tet_lut.get(dt.year)
    if base_tet is None:
      return 999
    return (dt - base_tet).days

  df['tet_days_diff'] = d.apply(calculate_tet_dist)
  # PHASE 1: The 40-Day Planning Window (Browsing & Early Buys)
  df['is_tet_planning'] = ((df['tet_days_diff'] >= -40) & (df['tet_days_diff'] <= -22)).astype(int)
  # PHASE 2: The Golden E-commerce Rush (The True Peak)
  df['is_tet_golden_rush'] = ((df['tet_days_diff'] >= -21) & (df['tet_days_diff'] <= -8)).astype(int)
  # PHASE 3: The Shipping Cutoff Crash (Shift to Offline)
  df['is_tet_shipping_cutoff'] = ((df['tet_days_diff'] >= -7) & (df['tet_days_diff'] <= -1)).astype(int)
  # PHASE 4: Post-Tet "Lì Xì" / Self-Rewarding Surge
  df['is_post_tet_surge'] = ((df['tet_days_diff'] >= 1) & (df['tet_days_diff'] <= 7)).astype(int)

  # 6. Deterministic Recurring Promotional Exogenous Signals
  df['is_spring_sale_window'] = (
      ((df['month'] == 3) & (df['day'] >= 15))
      | ((df['month'] == 4) & (df['day'] <= 15))
  ).astype(int)
  df['is_midyear_sale_window'] = (
      ((df['month'] == 6) & (df['day'] >= 20))
      | ((df['month'] == 7) & (df['day'] <= 20))
  ).astype(int)
  df['is_yearend_sale_window'] = (
      ((df['month'] == 11) & (df['day'] >= 15)) | (df['month'] == 12)
  ).astype(int)
  df['is_double_day_promo'] = (
      (df['month'].isin([9, 10, 11, 12])) & (df['month'] == df['day'])
  ).astype(int)

  return df

# ==========================================
# 1. LOAD DATA & BUILD FEATURE MATRICES
# ==========================================
print("--- [1/6] Loading historical data and building feature matrices ---")
project_root = Path(__file__).resolve().parent.parent
data_path = project_root / "data" / "sales.csv"
sales_df = pd.read_csv(data_path, parse_dates=["Date"])

X_train_df = build_features(sales_df["Date"])
y_train_rev = sales_df["Revenue"].values
y_train_cog = sales_df["COGS"].values

test_dates = pd.date_range("2023-01-01", "2024-07-01", freq="D")
X_test_df = build_features(test_dates)

drop_cols = ["Date", "year"]
feature_cols = [c for c in X_train_df.columns if c not in drop_cols]

X_train = X_train_df[feature_cols].values
X_test = X_test_df[feature_cols].values

# ==========================================
# 2. IN-CONSOLE DIAGNOSTIC: HOLDOUT VALIDATION
# ==========================================
print(

'--- [2/6] Running Temporal Holdout Validation Check (Train < 2022, Eval'

' 2022) ---')

val_mask = X_train_df['year'] == 2022
train_mask = X_train_df['year'] < 2022

if val_mask.sum() > 0 and train_mask.sum() > 0:
  val_years = X_train_df.loc[train_mask, 'year'].values
  val_weights = np.full(train_mask.sum(), 0.10)
  val_weights[val_years >= 2020] = 1.0 

  lgb_params_val = {
      'objective': 'regression',
      'metric': 'mae',
      'learning_rate': 0.03,
      'num_leaves': 63,
      'verbosity': -1,
      'seed': 42,
  }

  dval_train = lgb.Dataset(
      X_train[train_mask],
      label=y_train_rev[train_mask],
      weight=val_weights,
  )
  val_model = lgb.train(lgb_params_val, dval_train, num_boost_round=600)

  val_preds_log = val_model.predict(X_train[val_mask])
  val_preds_raw = val_model.predict(X_train[val_mask])
  actuals_2022 = sales_df.loc[val_mask, 'Revenue'].values

  mape_score = mean_absolute_percentage_error(actuals_2022, val_preds_raw) * 100
  mae_score = mean_absolute_error(actuals_2022, val_preds_raw)
  print(f' --> 2022 Holdout MAPE : {mape_score:.2f}%')
  print(f' --> 2022 Holdout MAE : ${mae_score:,.2f}')
else:
  print('  --> Skipping holdout check: insufficient historical years.')

# ==========================================
# 3. FAMILY MODEL 1: RIDGE REGRESSION ANCHOR
# ==========================================
print("--- [3/6] Training Family 1: Ridge parametric linear model ---")
mu, sigma = X_train.mean(axis=0), X_train.std(axis=0) + 1e-8
X_train_scaled = (X_train - mu) / sigma
X_test_scaled = (X_test - mu) / sigma

ridge_rev = Ridge(alpha=0.1, random_state=42).fit(X_train_scaled, y_train_rev)
ridge_cog = Ridge(alpha=0.1, random_state=42).fit(X_train_scaled, y_train_cog)

p_ridge_rev = ridge_rev.predict(X_test_scaled)
p_ridge_cog = ridge_cog.predict(X_test_scaled)

# ==========================================
# 4. FAMILY MODEL 2 & 3: LIGHTGBM + QUARTERLY SPECIALISTS
# ==========================================
print("--- [4/6] Training Family 2 & 3: Global LightGBM + Quarterly Specialists ---")
years_vector = X_train_df["year"].values
sample_weights = np.full(len(X_train), 0.20)
sample_weights[(years_vector >= 2017) & (years_vector <= 2019)] = 0.30   
sample_weights[(years_vector >= 2021) & (years_vector <= 2022)] = 1.00

lgb_params = {
      'objective': 'regression',
      'metric': 'rmse',
      'learning_rate': 0.03,
      'num_leaves': 63,
      'min_data_in_leaf': 7,  
      'verbosity': -1,
      'seed': 42,
  }

dtrain_rev = lgb.Dataset(X_train, label=y_train_rev, weight=sample_weights)
dtrain_cog = lgb.Dataset(X_train, label=y_train_cog, weight=sample_weights)

lgb_rev = lgb.train(lgb_params, dtrain_rev, num_boost_round=1200)
lgb_cog = lgb.train(lgb_params, dtrain_cog, num_boost_round=1200)

p_lgb_rev = lgb_rev.predict(X_test)
p_lgb_cog = lgb_cog.predict(X_test)

# Quarterly Specialists
quarter_test_vector = X_test_df["quarter"].values
quarter_train_vector = X_train_df["quarter"].values
spec_composed_rev = np.zeros(len(X_test))
spec_composed_cog = np.zeros(len(X_test))

for q in [1, 2, 3, 4]:
    q_weights = sample_weights.copy()
    q_weights[quarter_train_vector == q] *= 2.0
    
    dq_rev = lgb.Dataset(X_train, label=y_train_rev, weight=q_weights)
    dq_cog = lgb.Dataset(X_train, label=y_train_cog, weight=q_weights)
    
    m_spec_rev = lgb.train(lgb_params, dq_rev, num_boost_round=1000)
    m_spec_cog = lgb.train(lgb_params, dq_cog, num_boost_round=1000)
    
    test_mask = (quarter_test_vector == q)
    spec_composed_rev[test_mask] = m_spec_rev.predict(X_test[test_mask])
    spec_composed_cog[test_mask] = m_spec_cog.predict(X_test[test_mask])

# ==========================================
# 5. ENSEMBLING & CALIBRATION LAYER
# ==========================================
print("--- [5/6] Executing hierarchical ensembling and calibration ---")
ALPHA = 0.60
lgb_blend_rev = (ALPHA * spec_composed_rev) + ((1 - ALPHA) * p_lgb_rev)
lgb_blend_cog = (ALPHA * spec_composed_cog) + ((1 - ALPHA) * p_lgb_cog)

raw_rev = (0.80 * lgb_blend_rev) + (0.20 * p_ridge_rev)
raw_cog = (0.80 * lgb_blend_cog) + (0.20 * p_ridge_cog)

# Empirical scaling adjustment
CR, CC = 1.08, 1.08
final_rev = (raw_rev * CR).clip(min=0)
final_cog = (raw_cog * CC).clip(min=0)

# ==========================================
# 6. IN-CONSOLE SANITY CHECKS & EXPORT
# ==========================================
print("--- [6/6] Verifying Forecast Sanity & Generating Submission ---")

# A. Assertion checks
assert np.isnan(final_rev).sum() == 0, "ERROR: Forecast contains NaN values!"
assert (final_rev < 0).sum() == 0, "ERROR: Forecast contains negative revenue values!"

# B. Summary Table: Historical 2022 vs. Forecast 2023-2024
hist_mean = sales_df.loc[sales_df["Date"].dt.year == 2022, "Revenue"].mean()
hist_max = sales_df.loc[sales_df["Date"].dt.year == 2022, "Revenue"].max()

pred_mean = final_rev.mean()
pred_max = final_rev.max()

print("\n======================= MODEL SANITY SCORECARD =======================")
print(f"  Metric              |  2022 Actual (Recent)  |  2023-2024 Forecast")
print(f"----------------------|------------------------|---------------------")
print(f"  Daily Mean Revenue  |  ${hist_mean:19,.2f} |  ${pred_mean:17,.2f}")
print(f"  Daily Peak Revenue  |  ${hist_max:19,.2f} |  ${pred_max:17,.2f}")
print("======================================================================\n")

# C. Promo Signal Sanity Check
payday_mask = (X_test_df['day'] <= 3) | (X_test_df['days_to_eom'] <= 2)
if payday_mask.sum() > 0:
    promo_avg = final_rev[payday_mask].mean()
    base_avg = final_rev[~payday_mask].mean()
    lift_pct = ((promo_avg - base_avg) / base_avg) * 100
    print(f"  --> Payday Window Lift Check: {lift_pct:+.1f}% vs. mid-month baseline days.")

submission = pd.DataFrame({
    'Date': test_dates.strftime('%Y-%m-%d'),
    'Revenue': final_rev,
    'COGS': final_cog
})
output_path = project_root / "submission.csv"
submission.to_csv(output_path, index=False)
print(f"{output_path.name} successfully compiled and written to {output_path}!")
