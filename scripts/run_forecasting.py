import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
import lightgbm as lgb
from prophet import Prophet
import os

# Define core constants mapping back to historical validation insights
TAU = 2 * np.pi
TET_DATES = {2013: "2013-02-10", 2014: "2014-01-31", 2015: "2015-02-19", 2016: "2016-02-08",
             2017: "2017-01-28", 2018: "2018-02-16", 2019: "2019-02-05", 2020: "2020-01-25",
             2021: "2021-02-12", 2022: "2022-02-01", 2023: "2023-01-22", 2024: "2024-02-10"}

def build_features(dates):
    """Generates absolute calendar features accessible across any horizon."""
    df = pd.DataFrame({'Date': pd.to_datetime(dates)})
    d = df['Date']
    
    df['year'] = d.dt.year
    df['month'] = d.dt.month
    df['day'] = d.dt.day
    df['dow'] = d.dt.dayofweek
    df['doy'] = d.dt.dayofyear
    df['quarter'] = d.dt.quarter
    df['is_weekend'] = (df['dow'] >= 5).astype(int)
    df['is_odd_year'] = (df['year'] % 2).astype(int)
    
    # Structural Edge-of-month salary triggers
    dim = d.dt.days_in_month
    df['days_to_eom'] = dim - df['day']
    for k in [1, 2, 3]:
        df[f'is_last{k}'] = (df['days_to_eom'] <= k-1).astype(int)
        df[f'is_first{k}'] = (df['day'] <= k).astype(int)
        
    # Multi-frequency Fourier transformation vectors for seasonality
    for k in range(1, 6):
        df[f'sin_y{k}'] = np.sin(TAU * k * df['doy'] / 365.25)
        df[f'cos_y{k}'] = np.cos(TAU * k * df['doy'] / 365.25)
    for k in range(1, 3):
        df[f'sin_w{k}'] = np.sin(TAU * k * df['dow'] / 7.0)
        df[f'cos_w{k}'] = np.cos(TAU * k * df['dow'] / 7.0)

    # Lunar Calendar Drift Calculator
    tet_lut = {y: pd.Timestamp(v) for y, v in TET_DATES.items()}
    def calculate_tet_dist(dt):
        base_tet = tet_lut.get(dt.year)
        return (dt - base_tet).days
    df['tet_days_diff'] = d.apply(calculate_tet_dist)
    df['tet_in_7'] = (np.abs(df['tet_days_diff']) <= 7).astype(int)
    
    return df

# Initialize Datasets
sales_df = pd.read_csv("data/sales.csv", parse_dates=["Date"])
X_train_df = build_features(sales_df["Date"])
y_train_rev = np.log1p(sales_df["Revenue"].values) # Log stability compression
y_train_cog = np.log1p(sales_df["COGS"].values)

test_dates = pd.date_range("2023-01-01", "2024-07-01", freq="D")
X_test_df = build_features(test_dates)

drop_cols = ["Date", "year"]
feature_cols = [c for c in X_train_df.columns if c not in drop_cols]

X_train = X_train_df[feature_cols].values
X_test = X_test_df[feature_cols].values

# ==========================================
# 1. Family Model 1: Ridge Regression Anchor
# ==========================================
print("Training Family 1: Ridge parametric linear model...")
mu, sigma = X_train.mean(axis=0), X_train.std(axis=0) + 1e-8
X_train_scaled = (X_train - mu) / sigma
X_test_scaled = (X_test - mu) / sigma

ridge_rev = Ridge(alpha=3.0, random_state=42).fit(X_train_scaled, y_train_rev)
ridge_cog = Ridge(alpha=3.0, random_state=42).fit(X_train_scaled, y_train_cog)

p_ridge_rev = np.expm1(ridge_rev.predict(X_test_scaled))
p_ridge_cog = np.expm1(ridge_cog.predict(X_test_scaled))

# ==========================================
# 2. Family Model 2: Base LightGBM with Era Mask Weights
# ==========================================
print("Training Family 2: LightGBM with structural sample weighting...")
# Downweight noisy eras; prioritize stable baseline structure era (2014-2018)
years_vector = X_train_df["year"].values
sample_weights = np.full(len(X_train), 0.01)
sample_weights[(years_vector >= 2014) & (years_vector <= 2018)] = 1.0

lgb_params = {'objective': 'regression', 'metric': 'mae', 'learning_rate': 0.03, 'num_leaves': 63, 'verbosity': -1, 'seed': 42}
dtrain_rev = lgb.Dataset(X_train, label=y_train_rev, weight=sample_weights)
dtrain_cog = lgb.Dataset(X_train, label=y_train_cog, weight=sample_weights)

lgb_rev = lgb.train(lgb_params, dtrain_rev, num_boost_round=1200)
lgb_cog = lgb.train(lgb_params, dtrain_cog, num_boost_round=1200)

p_lgb_rev = np.expm1(lgb_rev.predict(X_test))
p_lgb_cog = np.expm1(lgb_cog.predict(X_test))

# ==========================================
# 3. Quarterly Specialists (8 Isolated LightGBM Instances)
# ==========================================
print("Training Quarterly Specialists to tackle localized variance volatility...")
quarter_test_vector = X_test_df["quarter"].values
spec_composed_rev = np.zeros(len(X_test))
spec_composed_cog = np.zeros(len(X_test))

quarter_train_vector = X_train_df["quarter"].values

for q in [1, 2, 3, 4]:
    # Inject a 2.0x weight amplification for records belonging to the target quarter
    q_weights = sample_weights.copy()
    q_weights[quarter_train_vector == q] *= 2.0
    
    dq_rev = lgb.Dataset(X_train, label=y_train_rev, weight=q_weights)
    dq_cog = lgb.Dataset(X_train, label=y_train_cog, weight=q_weights)
    
    m_spec_rev = lgb.train(lgb_params, dq_rev, num_boost_round=1000)
    m_spec_cog = lgb.train(lgb_params, dq_cog, num_boost_round=1000)
    
    test_mask = (quarter_test_vector == q)
    spec_composed_rev[test_mask] = np.expm1(m_spec_rev.predict(X_test[test_mask]))
    spec_composed_cog[test_mask] = np.expm1(m_spec_cog.predict(X_test[test_mask]))

# ==========================================
# Phase 5: Hierarchical Stacking & Level Calibration Blend
# ==========================================
print("Executing hierarchical ensembling and calibration layer...")
# Tier 1: Merge tree algorithms
ALPHA = 0.60
lgb_blend_rev = (ALPHA * spec_composed_rev) + ((1 - ALPHA) * p_lgb_rev)
lgb_blend_cog = (ALPHA * spec_composed_cog) + ((1 - ALPHA) * p_lgb_cog)

# Tier 2: Cross-family mathematical integration blend
raw_rev = (0.80 * lgb_blend_rev) + (0.20 * p_ridge_rev)
raw_cog = (0.80 * lgb_blend_cog) + (0.20 * p_ridge_cog)

# Tier 3: Hard Empirical Calibration Scaling to correct historic regime shift drift
CR, CC = 1.26, 1.32 # Fine-tuned competition scale multipliers
final_rev = raw_rev * CR
final_cog = raw_cog * CC

# Export submission payload file
submission = pd.DataFrame({
    'Date': test_dates.strftime('%Y-%m-%d'),
    'Revenue': final_rev,
    'COGS': final_cog
})
submission.to_csv("submission.csv", index=False)
print("🎉 Submission.csv successfully compiled and written to disk without system memory exhaustion!")