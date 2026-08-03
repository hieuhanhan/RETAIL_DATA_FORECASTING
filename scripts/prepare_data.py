import pandas as pd
import numpy as np
import os

# Define path constants
DATA_DIR = "data"
OUT_DIR = "tableau_data"
os.makedirs(OUT_DIR, exist_ok=True)

def load_table(file_name, date_cols=None):
    path = os.path.join(DATA_DIR, file_name)
    if not os.path.exists(path):
        print(f"Warning: {file_name} missing.")
        return None
    df = pd.read_csv(path, parse_dates=date_cols)
    print(f"Loaded {file_name}: {df.shape[0]:,} rows x {df.shape[1]} cols")
    return df

# ==========================================
# STEP 1: Load Raw Datasets with Date Types
# ==========================================
orders       = load_table("orders.csv", date_cols=["order_date"])
order_items  = load_table("order_items.csv")
payments     = load_table("payments.csv")
shipments    = load_table("shipments.csv", date_cols=["ship_date", "delivery_date"])
returns      = load_table("returns.csv", date_cols=["return_date"])

products     = load_table("products.csv")
inventory    = load_table("inventory.csv", date_cols=["snapshot_date"])
promotions   = load_table("promotions.csv", date_cols=["start_date", "end_date"])

customers    = load_table("customers.csv", date_cols=["signup_date"]) 
geography    = load_table("geography.csv")
reviews      = load_table("reviews.csv", date_cols=["review_date"])

sales             = load_table("sales.csv", date_cols=["Date"])
web_traffic       = load_table("web_traffic.csv", date_cols=["date"])
sample_submission = load_table("sample_submission.csv", date_cols=["Date"])

# ==========================================
# STEP 2: Product Dimension Enrichment (dim_products)
# ==========================================
dim_products = products.copy()

# 1. Calculate Core Financial & Margin Metrics (with zero-division safety)
dim_products["gross_margin_pct"] = (
    ((dim_products["price"] - dim_products["cogs"]) / dim_products["price"].replace(0, 1) * 100)
    .round(2)
)
dim_products["profit_per_unit"] = (dim_products["price"] - dim_products["cogs"]).round(2)

# 2. Aggregate Customer Reviews
avg_rating = reviews.groupby("product_id")["rating"].agg(["mean", "count"]).reset_index()
avg_rating.columns = ["product_id", "avg_rating", "review_count"]
avg_rating["avg_rating"] = avg_rating["avg_rating"].round(2)

# 3. Aggregate Return Behaviors
ret_stats = returns.groupby("product_id").agg(
    total_returns=("return_id", "count"),
    total_return_qty=("return_quantity", "sum"),
    total_refund=("refund_amount", "sum")
).reset_index()
ret_stats["total_refund"] = ret_stats["total_refund"].round(2)

# 4. Aggregate Sales Volume from order_items 
if "order_items" in globals() and order_items is not None and "product_id" in order_items.columns:
    qty_col = "quantity" if "quantity" in order_items.columns else "order_id"
    agg_func = "sum" if "quantity" in order_items.columns else "count"
    
    sales_stats = order_items.groupby("product_id").agg(
        total_units_sold=(qty_col, agg_func),
        total_orders=("order_id", "nunique")
    ).reset_index()
    dim_products = dim_products.merge(sales_stats, on="product_id", how="left")
    dim_products[["total_units_sold", "total_orders"]] = (
        dim_products[["total_units_sold", "total_orders"]].fillna(0).astype(int)
    )

# 5. Multi-table Left Joins
dim_products = dim_products.merge(avg_rating, on="product_id", how="left")
dim_products = dim_products.merge(ret_stats, on="product_id", how="left")

# 6. Handle Structural Nulls & Type Casting
int_cols = ["review_count", "total_returns", "total_return_qty"]
dim_products[int_cols] = dim_products[int_cols].fillna(0).astype(int)
dim_products["total_refund"] = dim_products["total_refund"].fillna(0.0).round(2)
dim_products["avg_rating"] = dim_products["avg_rating"].fillna(0.0).round(2)

# 7. Normalized Return Rate (if sales volume was joined)
if "total_units_sold" in dim_products.columns:
    dim_products["return_rate_pct"] = (
        (dim_products["total_return_qty"] / dim_products["total_units_sold"].replace(0, 1) * 100)
        .round(2)
    )

# 8. Save Enriched Dimension Table
out_path = os.path.join(OUT_DIR, "dim_products.csv")
dim_products.to_csv(out_path, index=False)
print(f"Saved enriched dim_products: {dim_products.shape[0]:,} rows x {dim_products.shape[1]} cols")

# ==========================================
# STEP 3: Central Fact Orders Enriched (fact_orders_enriched)
# ==========================================
print("--- Processing fact_orders_enriched ---")
# Keep lowest grain intact (line-item level)
fact_orders = order_items.copy()

# 1. Merge Core Order Metadata
fact_orders = fact_orders.merge(
    orders[["order_id", "order_date", "customer_id", "zip", "order_status", "payment_method", "device_type", "order_source"]],
    on="order_id", how="left"
)

# 2. Merge Product Economics
fact_orders = fact_orders.merge(
    products[["product_id", "category", "segment", "cogs"]],
    on="product_id", how="left"
)

# 3. Merge Order Payments (Aggregated by order_id to prevent multi-payment row explosion)
order_payments = payments.groupby("order_id")["payment_value"].sum().reset_index()
fact_orders = fact_orders.merge(order_payments, on="order_id", how="left")

# 4. Merge Shipping & Delivery Metrics from shipments.csv
if "shipments" in globals() and shipments is not None:
    ship_clean = shipments[["order_id", "ship_date", "delivery_date", "shipping_fee"]].drop_duplicates(subset="order_id")
    fact_orders = fact_orders.merge(ship_clean, on="order_id", how="left")
    fact_orders["delivery_days"] = (fact_orders["delivery_date"] - fact_orders["order_date"]).dt.days
    fact_orders["shipping_fee"] = fact_orders["shipping_fee"].fillna(0.0).round(2)

# 5. De-duplicate Geography
geo_clean = geography[["zip", "city", "region"]].drop_duplicates(subset="zip")
fact_orders = fact_orders.merge(geo_clean, on="zip", how="left")

# 6. Vectorized Financial Computed Fields
price_col = "unit_price" if "unit_price" in fact_orders.columns else "price"
fact_orders["line_revenue"] = (fact_orders["quantity"] * fact_orders[price_col]).round(2)
fact_orders["line_cost"] = (fact_orders["quantity"] * fact_orders["cogs"]).round(2)
fact_orders["line_gross_profit"] = (fact_orders["line_revenue"] - fact_orders["line_cost"]).round(2)
fact_orders["line_margin_pct"] = np.where(
    fact_orders["line_revenue"] > 0,
    (fact_orders["line_gross_profit"] / fact_orders["line_revenue"] * 100).round(2),
    0.0
)

# 7. Promo Flagging (Safe Check)
if "promo_id" in fact_orders.columns:
    fact_orders["has_promo"] = (~fact_orders["promo_id"].isna() & (fact_orders["promo_id"] != '')).astype(int)
else:
    fact_orders["has_promo"] = 0

# 8. Calendar Dimensions
fact_orders["order_year"] = fact_orders["order_date"].dt.year
fact_orders["order_month"] = fact_orders["order_date"].dt.month
fact_orders["order_quarter"] = fact_orders["order_date"].dt.quarter
fact_orders["order_ym"] = fact_orders["order_date"].dt.to_period("M").astype(str)

fact_orders.to_csv(os.path.join(OUT_DIR, "fact_orders_enriched.csv"), index=False)


# ==========================================
# STEP 4: Customer RFM Segmentation (dim_customers_rfm)
# ==========================================
print("--- Processing dim_customers_rfm (RFM Segmentation) ---")
ref_date = orders["order_date"].max() + pd.Timedelta(days=1)

# Cleanse cancelled records
valid_sales = orders[orders["order_status"] != "cancelled"].merge(order_payments, on="order_id", how="left")

rfm_agg = valid_sales.groupby("customer_id").agg(
    last_order_date=("order_date", "max"),
    frequency=("order_id", "nunique"),
    monetary=("payment_value", "sum")
).reset_index()

rfm_agg["recency_days"] = (ref_date - rfm_agg["last_order_date"]).dt.days

# Mathematical Quintiles
rfm_agg["R_score"] = pd.qcut(rfm_agg["recency_days"], q=5, labels=[5, 4, 3, 2, 1]).astype(int)
rfm_agg["F_score"] = pd.qcut(rfm_agg["frequency"].rank(method="first"), q=5, labels=[1, 2, 3, 4, 5]).astype(int)
rfm_agg["M_score"] = pd.qcut(rfm_agg["monetary"].rank(method="first"), q=5, labels=[1, 2, 3, 4, 5]).astype(int)

# Corrected Evaluation Order 
def rfm_labeler(row):
    r, f, m = row["R_score"], row["F_score"], row["M_score"]
    if r >= 4 and f >= 4: return "Champions"
    elif r >= 3 and f >= 3: return "Loyal Customers"
    elif r >= 4 and f <= 2: return "New Customers"
    elif r >= 3 and m >= 3: return "Potential Loyalists"
    elif r <= 2 and f >= 4 and m >= 4: return "Can't Lose Them" 
    elif r <= 2 and f >= 3: return "At Risk"
    elif r <= 2 and f <= 2: return "Lost"
    else: return "Need Attention"

rfm_agg["rfm_segment"] = rfm_agg.apply(rfm_labeler, axis=1)

dim_customers_rfm = customers.merge(
    rfm_agg[["customer_id", "recency_days", "frequency", "monetary", "rfm_segment"]],
    on="customer_id", how="left"
)
dim_customers_rfm["rfm_segment"] = dim_customers_rfm["rfm_segment"].fillna("Never Purchased")
dim_customers_rfm[["frequency", "monetary"]] = dim_customers_rfm[["frequency", "monetary"]].fillna(0.0)

dim_customers_rfm.to_csv(os.path.join(OUT_DIR, "dim_customers_rfm.csv"), index=False)


# ==========================================
# STEP 5: Time-Series Daily Base (fact_sales_daily)
# ==========================================
print("--- Processing fact_sales_daily ---")
fact_sales_daily = sales.copy()
fact_sales_daily.columns = ["date", "revenue", "cogs"]
fact_sales_daily["gross_profit"] = (fact_sales_daily["revenue"] - fact_sales_daily["cogs"]).round(2)
fact_sales_daily["gross_margin_pct"] = np.where(
    fact_sales_daily["revenue"] > 0,
    (fact_sales_daily["gross_profit"] / fact_sales_daily["revenue"] * 100).round(2),
    0.0
)

# 1. Add Calendar Features for Time-Series Modeling
fact_sales_daily["day_of_week"] = fact_sales_daily["date"].dt.dayofweek
fact_sales_daily["is_weekend"] = fact_sales_daily["day_of_week"].isin([5, 6]).astype(int)
fact_sales_daily["month"] = fact_sales_daily["date"].dt.month
fact_sales_daily["day_of_year"] = fact_sales_daily["date"].dt.dayofyear

# 2. Merge Daily Web Traffic (web_traffic.csv)
if "web_traffic" in globals() and web_traffic is not None:
    fact_sales_daily = fact_sales_daily.merge(web_traffic, on="date", how="left")

# 3. Rolling Trends (Sorted by Date)
fact_sales_daily = fact_sales_daily.sort_values("date").reset_index(drop=True)
fact_sales_daily["revenue_ma7"] = fact_sales_daily["revenue"].rolling(7, min_periods=1).mean().round(2)
fact_sales_daily["revenue_ma30"] = fact_sales_daily["revenue"].rolling(30, min_periods=1).mean().round(2)

fact_sales_daily.to_csv(os.path.join(OUT_DIR, "fact_sales_daily.csv"), index=False)


# ==========================================
# STEP 6: User Cohort Retention Matrix (agg_cohort_retention)
# ==========================================
print("--- Processing agg_cohort_retention ---")
# Track acquisition date
first_purchase = orders[orders["order_status"] != "cancelled"].groupby("customer_id")["order_date"].min().reset_index()
first_purchase.columns = ["customer_id", "cohort_date"]
first_purchase["cohort_month"] = first_purchase["cohort_date"].dt.to_period("M")

# Progression Index
cohort_lookup = orders[orders["order_status"] != "cancelled"][["order_id", "customer_id", "order_date"]].copy()
cohort_lookup["order_month"] = cohort_lookup["order_date"].dt.to_period("M")
cohort_lookup = cohort_lookup.merge(first_purchase[["customer_id", "cohort_month"]], on="customer_id")

# Removed illegal assignment inside arithmetic expression
cohort_lookup["period_number"] = (
    (cohort_lookup["order_month"].dt.year - cohort_lookup["cohort_month"].dt.year) * 12
    + (cohort_lookup["order_month"].dt.month - cohort_lookup["cohort_month"].dt.month)
)

# Removed duplicate chaining of .merge(sizes, on="cohort_month")
sizes = cohort_lookup.groupby("cohort_month")["customer_id"].nunique().reset_index().rename(columns={"customer_id": "cohort_size"})
matrix = cohort_lookup.groupby(["cohort_month", "period_number"])["customer_id"].nunique().reset_index().rename(columns={"customer_id": "active_customers"})
matrix = matrix.merge(sizes, on="cohort_month")  # Single merge only!
matrix["retention_rate"] = ((matrix["active_customers"] / matrix["cohort_size"]) * 100).round(2)

matrix[matrix["period_number"] <= 24].to_csv(os.path.join(OUT_DIR, "agg_cohort_retention.csv"), index=False)
print("\nPipeline complete!")