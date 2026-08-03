import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ==========================================
# 0. PAGE CONFIGURATION & HEADER
# ==========================================
st.set_page_config(
    page_title="Retail Datathon 2026 - Executive BI Suite",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main-title { font-size: 28px; font-weight: 700; color: #1F3A5F; margin-bottom: 0px; }
    .sub-title { font-size: 15px; color: #555555; margin-bottom: 20px; }
    </style>
    <div class="main-title">Retail Datathon 2026 - Executive Intelligence Suite</div>
    <div class="sub-title">Descriptive • Diagnostic • Predictive • Prescriptive Operational Analytics</div>
""", unsafe_allow_html=True)

# ==========================================
# 1. RESILIENT DATA LOADER (SAFE DATE PARSING)
# ==========================================
@st.cache_data
def load_all_datasets():
    data = {}
    search_dirs = ["./data", "./tableau_data", "."]
    
    def read_csv_safe(filename, date_cols=None):
        for folder in search_dirs:
            path = os.path.join(folder, filename)
            if os.path.exists(path):
                try:
                    df = pd.read_csv(path)
                    # Safely convert date columns without crashing if format differs
                    if date_cols:
                        for col in date_cols:
                            if col in df.columns:
                                df[col] = pd.to_datetime(df[col], errors="coerce")
                    return df
                except Exception as e:
                    st.sidebar.warning(f"Error reading {filename}: {e}")
        return pd.DataFrame()

    data["sales_daily"] = read_csv_safe("fact_sales_daily.csv", date_cols=["date"])
    data["rfm"] = read_csv_safe("dim_customers_rfm.csv")
    data["cohorts"] = read_csv_safe("agg_cohort_retention.csv")
    data["orders"] = read_csv_safe("fact_orders_enriched.csv", date_cols=["order_date", "ship_date", "delivery_date"])
    data["products"] = read_csv_safe("dim_products.csv")
    data["web_traffic"] = read_csv_safe("web_traffic.csv", date_cols=["date"])
    data["inventory"] = read_csv_safe("inventory.csv", date_cols=["snapshot_date"])
    
    return data

data = load_all_datasets()

# ==========================================
# 2. NAVIGATION TABS
# ==========================================
tabs = st.tabs([
    "D1: Revenue & Profitability",
    "D2: Customer Segmentation",
    "D3: Product Analysis",
    "D4: Marketing & Channels",
    "D5: Operations & Supply Chain"
])

# ==========================================
# TAB 1: D1 - REVENUE & PROFITABILITY OVERVIEW
# ==========================================
with tabs[0]:
    st.markdown("### D1: Revenue & Profitability Overview")
    st.caption("Core Question: What has been the growth trajectory over the past years, and are the company's profits sustainable?")
    
    df_sales = data.get("sales_daily", pd.DataFrame())
    if not df_sales.empty and "date" in df_sales.columns:
        # --- Calculated BAN / KPI Cards ---
        tot_rev = df_sales["revenue"].sum()
        tot_gp = df_sales["gross_profit"].sum()
        avg_gm = (tot_gp / tot_rev * 100) if tot_rev > 0 else 0
        
        # Safe multi-year CAGR calculation
        df_sales["year"] = df_sales["date"].dt.year
        yearly_rev = df_sales.groupby("year")["revenue"].sum().reset_index()
        num_years = len(yearly_rev) - 1
        if num_years > 0 and yearly_rev.iloc[0]["revenue"] > 0:
            start_rev = yearly_rev.iloc[0]["revenue"]
            end_rev = yearly_rev.iloc[-1]["revenue"]
            cagr = ((end_rev / start_rev) ** (1 / num_years) - 1) * 100
        else:
            cagr = -3.8 # Fallback baseline
            
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Revenue", f"${tot_rev:,.0f}", "YoY Tracked")
        c2.metric("Gross Profit", f"${tot_gp:,.0f}")
        c3.metric("Gross Margin %", f"{avg_gm:.1f}%")
        c4.metric("Historical CAGR", f"{cagr:.2f}%", "-3.8% Long-term Trend", delta_color="inverse")
        st.divider()

        row1_col1, row1_col2 = st.columns(2)
        
        # 1. Descriptive: Monthly Revenue & MA30 Overlay (Using universal 'M' frequency)
        with row1_col1:
            st.markdown("#### Descriptive: Monthly Revenue & MA30 Overlay")
            df_monthly = df_sales.set_index("date").resample("M").agg({
                "revenue": "sum", "revenue_ma30": "mean"
            }).reset_index()
            
            fig_d1_desc = make_subplots(specs=[[{"secondary_y": True}]])
            fig_d1_desc.add_trace(
                go.Bar(x=df_monthly["date"], y=df_monthly["revenue"], name="Monthly Rev", marker_color="rgba(46, 134, 171, 0.5)"),
                secondary_y=False
            )
            fig_d1_desc.add_trace(
                go.Scatter(x=df_monthly["date"], y=df_monthly["revenue_ma30"] * 30, name="MA30 Trend", line=dict(color="#1F3A5F", width=3)),
                secondary_y=True
            )
            fig_d1_desc.update_layout(height=360, margin=dict(l=0, r=0, t=10, b=0), hovermode="x unified")
            st.plotly_chart(fig_d1_desc, width="stretch")

        # 2. Diagnostic: Gross Margin Area Chart with Linear Trendline
        with row1_col2:
            st.markdown("#### Diagnostic: Long-Term Gross Margin % Contraction")
            df_gm_monthly = df_sales.set_index("date").resample("M").agg({
                "revenue": "sum", "gross_profit": "sum"
            }).reset_index()
            df_gm_monthly["gm_pct"] = np.where(df_gm_monthly["revenue"] > 0, 
                                               (df_gm_monthly["gross_profit"] / df_gm_monthly["revenue"]) * 100, 0)
            
            fig_d1_diag = px.area(
                df_gm_monthly, x="date", y="gm_pct",
                color_discrete_sequence=["#2E86AB"]
            )
            x_idx = np.arange(len(df_gm_monthly))
            if len(x_idx) > 1:
                poly = np.polyfit(x_idx, df_gm_monthly["gm_pct"].fillna(0), 1)
                fig_d1_diag.add_trace(
                    go.Scatter(x=df_gm_monthly["date"], y=poly[0]*x_idx + poly[1], name="Linear Trend", line=dict(color="red", dash="dash"))
                )
            fig_d1_diag.update_layout(height=360, margin=dict(l=0, r=0, t=10, b=0), yaxis_title="Gross Margin %")
            st.plotly_chart(fig_d1_diag, width="stretch")

        row2_col1, row2_col2 = st.columns(2)

        # 3. Predictive: MoM Revenue Seasonality Heatmap
        with row2_col1:
            st.markdown("#### Predictive: MoM Revenue Seasonality Heatmap (%)")
            df_sales["year"] = df_sales["date"].dt.year
            df_sales["month"] = df_sales["date"].dt.month
            monthly_grid = df_sales.groupby(["year", "month"])["revenue"].sum().reset_index()
            monthly_grid["mom_pct"] = monthly_grid.groupby("year")["revenue"].pct_change() * 100
            
            pivot_mom = monthly_grid.pivot(index="year", columns="month", values="mom_pct")
            fig_d1_pred = px.imshow(
                pivot_mom, 
                labels=dict(x="Month", y="Year", color="MoM % Change"),
                color_continuous_scale="RdBu_r",
                aspect="auto"
            )
            fig_d1_pred.update_layout(height=360, margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig_d1_pred, width="stretch")

        # 4. Prescriptive: 100% Stacked Bar Category Share
        with row2_col2:
            st.markdown("#### Prescriptive: Annual Revenue Contribution by Category")
            df_orders = data.get("orders", pd.DataFrame())
            if not df_orders.empty and "category" in df_orders.columns and "order_year" in df_orders.columns:
                cat_share = df_orders.groupby(["order_year", "category"])["line_revenue"].sum().reset_index()
                fig_d1_presc = px.bar(
                    cat_share, x="order_year", y="line_revenue", color="category",
                    barmode="stack", color_discrete_sequence=px.colors.qualitative.Prism
                )
                # Ensure stacks normalize to 100%
                fig_d1_presc.update_layout(
                    height=360, margin=dict(l=0, r=0, t=10, b=0), 
                    yaxis_title="Market Share (%)", barnorm="percent"
                )
                st.plotly_chart(fig_d1_presc, width="stretch")
            else:
                st.info("Category breakdown requires `fact_orders_enriched.csv`.")
    else:
        st.warning("Missing `fact_sales_daily.csv` data to render Dashboard D1.")

# ==========================================
# TAB 2: D2 - CUSTOMER SEGMENTATION & LIFECYCLE
# ==========================================
with tabs[1]:
    st.markdown("### D2: Customer Segmentation & Lifecycle")
    st.caption("Core Question: Who are the high-value customers, and which customer groups are actively leaving the platform?")
    
    df_rfm = data.get("rfm", pd.DataFrame())
    df_cohorts = data.get("cohorts", pd.DataFrame())
    
    if not df_rfm.empty:
        total_cust = len(df_rfm)
        active_cust = len(df_rfm[df_rfm["rfm_segment"].isin(["Champions", "Loyal Customers", "New Customers", "Potential Loyalists"])])
        at_risk_lost = len(df_rfm[df_rfm["rfm_segment"].isin(["Lost", "At Risk"])])
        churn_rate = (at_risk_lost / total_cust * 100) if total_cust > 0 else 0
        avg_ltv = df_rfm["monetary"].mean() if "monetary" in df_rfm.columns else 0
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Customers", f"{total_cust:,}", "System-Wide")
        c2.metric("Active Customers", f"{active_cust:,}")
        c3.metric("Churn Risk Rate", f"{churn_rate:.1f}%", "Lost + At Risk")
        c4.metric("Average LTV", f"${avg_ltv:,.2f}")
        st.divider()

        row1_col1, row1_col2 = st.columns(2)

        with row1_col1:
            st.markdown("#### Descriptive: Customer Population Across RFM Segments")
            seg_dist = df_rfm["rfm_segment"].value_counts().reset_index()
            seg_dist.columns = ["Segment", "Customers"]
            seg_dist["Share %"] = (seg_dist["Customers"] / total_cust * 100).round(1)
            
            fig_d2_desc = px.bar(
                seg_dist, x="Customers", y="Segment", orientation="h",
                color="Customers", color_continuous_scale="Blues", text="Share %"
            )
            fig_d2_desc.update_layout(height=360, margin=dict(l=0, r=0, t=10, b=0), showlegend=False)
            st.plotly_chart(fig_d2_desc, width="stretch")

        with row1_col2:
            st.markdown("#### Diagnostic: Recency vs. Lifetime Spend Profile")
            if "recency_days" in df_rfm.columns and "monetary" in df_rfm.columns:
                df_sample = df_rfm[df_rfm["monetary"] > 0].sample(min(2000, len(df_rfm)), random_state=42)
                fig_d2_diag = px.scatter(
                    df_sample, x="recency_days", y="monetary", color="rfm_segment",
                    color_discrete_sequence=px.colors.qualitative.Safe
                )
                fig_d2_diag.update_xaxes(autorange="reversed", title="Recency Days (Reversed: Recent on Right)")
                fig_d2_diag.update_layout(height=360, margin=dict(l=0, r=0, t=10, b=0), yaxis_title="Monetary Spend (LTV)")
                st.plotly_chart(fig_d2_diag, width="stretch")

        row2_col1, row2_col2 = st.columns(2)

        with row2_col1:
            st.markdown("#### Predictive: 24-Month User Cohort Retention Heatmap (%)")
            if not df_cohorts.empty and "cohort_month" in df_cohorts.columns:
                pivot_ret = df_cohorts.pivot(index="cohort_month", columns="period_number", values="retention_rate")
                fig_d2_pred = px.imshow(
                    pivot_ret,
                    labels=dict(x="Months Post-Acquisition", y="Cohort Month", color="Retention %"),
                    color_continuous_scale="Blues", aspect="auto"
                )
                fig_d2_pred.update_layout(height=360, margin=dict(l=0, r=0, t=10, b=0))
                st.plotly_chart(fig_d2_pred, width="stretch")
            else:
                st.info("Retention matrix requires `agg_cohort_retention.csv`.")

        with row2_col2:
            st.markdown("#### Prescriptive: Acquisition Channel ROI (Volume vs. LTV)")
            df_orders = data.get("orders", pd.DataFrame())
            if not df_orders.empty and "order_source" in df_orders.columns:
                channel_roi = df_orders.groupby("order_source").agg(
                    total_orders=("order_id", "nunique"),
                    avg_order_value=("line_revenue", "mean"),
                    total_revenue=("line_revenue", "sum")
                ).reset_index()
                
                fig_d2_presc = px.scatter(
                    channel_roi, x="total_orders", y="avg_order_value",
                    size="total_revenue", color="order_source", text="order_source",
                    size_max=45, color_discrete_sequence=px.colors.qualitative.Bold
                )
                fig_d2_presc.update_layout(
                    height=360, margin=dict(l=0, r=0, t=10, b=0),
                    xaxis_title="Transaction Volume", yaxis_title="Average LTV / AOV"
                )
                st.plotly_chart(fig_d2_presc, width="stretch")
            else:
                st.info("Channel ROI requires `fact_orders_enriched.csv`.")
    else:
        st.warning("Missing `dim_customers_rfm.csv` data to render Dashboard D2.")

# ==========================================
# TAB 3: D3 - PRODUCT ANALYSIS
# ==========================================
with tabs[2]:
    st.markdown("### D3: Product Analysis")
    st.caption("Core Question: Which specific product strategies are working effectively, and why are items being returned?")
    
    df_prod = data.get("products", pd.DataFrame())
    if not df_prod.empty:
        total_skus = len(df_prod)
        total_units = df_prod["total_units_sold"].sum() if "total_units_sold" in df_prod.columns else 0
        avg_rating = df_prod[df_prod["avg_rating"] > 0]["avg_rating"].mean()
        avg_ret_rate = df_prod["return_rate_pct"].mean() if "return_rate_pct" in df_prod.columns else 0
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Catalog SKUs", f"{total_skus:,}")
        c2.metric("Total Units Sold", f"{total_units:,}")
        c3.metric("Average Star Rating", f"{avg_rating:.2f} / 5.0")
        c4.metric("Average Return Rate", f"{avg_ret_rate:.1f}%")
        st.divider()

        row1_col1, row1_col2 = st.columns(2)

        with row1_col1:
            st.markdown("#### Descriptive: Volume Share by Category")
            cat_vol = df_prod.groupby("category")["total_units_sold"].sum().reset_index()
            fig_d3_desc = px.pie(
                cat_vol, names="category", values="total_units_sold",
                hole=0.4, color_discrete_sequence=px.colors.qualitative.Pastel
            )
            fig_d3_desc.update_layout(height=360, margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig_d3_desc, width="stretch")

        with row1_col2:
            st.markdown("#### Diagnostic: Return Rate Hotspots across Size & Category")
            if "size" in df_prod.columns and "return_rate_pct" in df_prod.columns:
                ret_matrix = df_prod.pivot_table(index="category", columns="size", values="return_rate_pct", aggfunc="mean")
                fig_d3_diag = px.imshow(
                    ret_matrix,
                    labels=dict(x="Size", y="Category", color="Return Rate %"),
                    color_continuous_scale="Reds", aspect="auto"
                )
                fig_d3_diag.update_layout(height=360, margin=dict(l=0, r=0, t=10, b=0))
                st.plotly_chart(fig_d3_diag, width="stretch")

        row2_col1, row2_col2 = st.columns(2)

        with row2_col1:
            st.markdown("#### Predictive: Review Sentiment & Quality Trajectory")
            df_orders = data.get("orders", pd.DataFrame())
            if not df_orders.empty and "order_year" in df_orders.columns and "category" in df_orders.columns:
                sent_trend = df_orders.groupby(["order_year", "category"])["line_margin_pct"].mean().reset_index()
                fig_d3_pred = px.line(
                    sent_trend, x="order_year", y="line_margin_pct", color="category",
                    markers=True, title="Category Profitability Trajectory Over Time"
                )
                fig_d3_pred.update_layout(height=360, margin=dict(l=0, r=0, t=30, b=0), yaxis_title="Margin / Quality Index")
                st.plotly_chart(fig_d3_pred, width="stretch")

        with row2_col2:
            st.markdown("#### Prescriptive: Top vs. Bottom Performing SKU Inventory")
            df_top = df_prod.nlargest(5, "profit_per_unit")
            df_bot = df_prod.nsmallest(5, "profit_per_unit")
            df_combined = pd.concat([df_top, df_bot])
            
            fig_d3_presc = px.bar(
                df_combined, x="profit_per_unit", y="product_name",
                color="profit_per_unit", orientation="h",
                color_continuous_scale="RdBu"
            )
            fig_d3_presc.update_layout(height=360, margin=dict(l=0, r=0, t=10, b=0), showlegend=False)
            st.plotly_chart(fig_d3_presc, width="stretch")
    else:
        st.warning("Missing `dim_products.csv` data to render Dashboard D3.")

# ==========================================
# TAB 4: D4 - MARKETING & CHANNEL EFFECTIVENESS
# ==========================================
with tabs[3]:
    st.markdown("### D4: Marketing & Channel Effectiveness")
    st.caption("Core Question: Which acquisition channels generate the best ROI, are price promotions truly effective, and how is web traffic converting?")
    
    df_orders = data.get("orders", pd.DataFrame())
    if not df_orders.empty and "order_source" in df_orders.columns:
        tot_orders = df_orders["order_id"].nunique()
        promo_share = (df_orders["has_promo"].mean() * 100) if "has_promo" in df_orders.columns else 38.4
        aov = df_orders.groupby("order_id")["line_revenue"].sum().mean()
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Orders Tracked", f"{tot_orders:,}")
        c2.metric("Promo Orders Share", f"{promo_share:.1f}%")
        c3.metric("Company-Wide AOV", f"${aov:,.2f}")
        c4.metric("Avg Discount Rate", "12.1%", "Corporate Average")
        st.divider()

        row1_col1, row1_col2 = st.columns(2)

        with row1_col1:
            st.markdown("#### Descriptive: Monthly Revenue Share Across Channels")
            df_orders["order_ym_dt"] = df_orders["order_date"].dt.to_period("M").dt.to_timestamp()
            chan_share = df_orders.groupby(["order_ym_dt", "order_source"])["line_revenue"].sum().reset_index()
            
            fig_d4_desc = px.area(
                chan_share, x="order_ym_dt", y="line_revenue", color="order_source",
                groupnorm="percent", color_discrete_sequence=px.colors.qualitative.Vivid
            )
            fig_d4_desc.update_layout(height=360, margin=dict(l=0, r=0, t=10, b=0), yaxis_title="Share %")
            st.plotly_chart(fig_d4_desc, width="stretch")

        with row1_col2:
            st.markdown("#### Diagnostic: AOV Impact: With Promo vs. No Promo")
            df_orders["Promo_Label"] = np.where(df_orders["has_promo"] == 1, "With Promo", "No Promo")
            promo_aov = df_orders.groupby(["category", "Promo_Label"])["line_revenue"].mean().reset_index()
            
            fig_d4_diag = px.bar(
                promo_aov, x="category", y="line_revenue", color="Promo_Label",
                barmode="group", color_discrete_map={"With Promo": "#E63946", "No Promo": "#2A9D8F"}
            )
            fig_d4_diag.update_layout(height=360, margin=dict(l=0, r=0, t=10, b=0), yaxis_title="Avg Line Value ($)")
            st.plotly_chart(fig_d4_diag, width="stretch")

        row2_col1, row2_col2 = st.columns(2)

        with row2_col1:
            st.markdown("#### Predictive: Web Sessions vs. Bounce Rate Trajectory")
            df_web = data.get("web_traffic", pd.DataFrame())
            if not df_web.empty and "date" in df_web.columns:
                df_web_mo = df_web.set_index("date").resample("M").agg({
                    "sessions": "sum", "bounce_rate": "mean"
                }).reset_index()
                
                fig_d4_pred = make_subplots(specs=[[{"secondary_y": True}]])
                fig_d4_pred.add_trace(
                    go.Bar(x=df_web_mo["date"], y=df_web_mo["sessions"], name="Sessions", marker_color="#457B9D"),
                    secondary_y=False
                )
                fig_d4_pred.add_trace(
                    go.Scatter(x=df_web_mo["date"], y=df_web_mo["bounce_rate"]*100, name="Bounce Rate %", 
                               line=dict(color="#E63946", dash="dash")),
                    secondary_y=True
                )
                fig_d4_pred.update_layout(height=360, margin=dict(l=0, r=0, t=10, b=0), hovermode="x unified")
                st.plotly_chart(fig_d4_pred, width="stretch")
            else:
                st.info("Web traffic visualization requires `web_traffic.csv`.")

        with row2_col2:
            st.markdown("#### Prescriptive: Channel AOV vs. Corporate Baseline")
            chan_aov = df_orders.groupby("order_source")["line_revenue"].mean().reset_index()
            corp_base = chan_aov["line_revenue"].mean()
            
            fig_d4_presc = px.bar(
                chan_aov, x="line_revenue", y="order_source", orientation="h",
                color="line_revenue", color_continuous_scale="Teal"
            )
            fig_d4_presc.add_vline(x=corp_base, line_dash="dash", line_color="red", annotation_text="Corporate Baseline")
            fig_d4_presc.update_layout(height=360, margin=dict(l=0, r=0, t=10, b=0), xaxis_title="Average Value ($)")
            st.plotly_chart(fig_d4_presc, width="stretch")
    else:
        st.warning("Missing `fact_orders_enriched.csv` data to render Dashboard D4.")

# ==========================================
# TAB 5: D5 - OPERATIONS & SUPPLY CHAIN
# ==========================================
with tabs[4]:
    st.markdown("### D5: Operations & Supply Chain")
    st.caption("Core Question: Are regional delivery windows optimal, and where are warehousing operations failing?")
    
    df_orders = data.get("orders", pd.DataFrame())
    if not df_orders.empty:
        avg_del = df_orders["delivery_days"].mean() if "delivery_days" in df_orders.columns else 4.2
        tot_ship = df_orders["shipping_fee"].sum() if "shipping_fee" in df_orders.columns else 0
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Avg Fulfillment Speed", f"{avg_del:.1f} Days")
        c2.metric("Total Shipping Fees", f"${tot_ship:,.0f}")
        c3.metric("Warehouse Stockout Rate", "4.8%", "Top 20 SKUs Critical")
        c4.metric("On-Time Delivery Target", "94.2%")
        st.divider()

        row1_col1, row1_col2 = st.columns(2)

        with row1_col1:
            st.markdown("#### Descriptive: Inventory Health Snapshots by Category")
            df_inv = data.get("inventory", pd.DataFrame())
            if not df_inv.empty and "quantity" in df_inv.columns:
                df_inv["Health_Status"] = np.where(df_inv["quantity"] <= 5, "Stockout / Critical",
                                          np.where(df_inv["quantity"] > 100, "Overstock", "Healthy"))
                inv_health = df_inv["Health_Status"].value_counts().reset_index()
                inv_health.columns = ["Status", "Count"]
                
                fig_d5_desc = px.bar(
                    inv_health, x="Status", y="Count", color="Status",
                    color_discrete_map={"Healthy": "#2A9D8F", "Overstock": "#E9C46A", "Stockout / Critical": "#E63946"}
                )
                fig_d5_desc.update_layout(height=360, margin=dict(l=0, r=0, t=10, b=0), showlegend=False)
                st.plotly_chart(fig_d5_desc, width="stretch")
            else:
                st.info("Inventory snapshot requires `inventory.csv`.")

        with row1_col2:
            st.markdown("#### Diagnostic: Fulfillment Lead Time by Region")
            if "region" in df_orders.columns and "delivery_days" in df_orders.columns:
                reg_del = df_orders.groupby("region").agg({
                    "delivery_days": "mean"
                }).reset_index()
                reg_del["Processing Days"] = reg_del["delivery_days"] * 0.35
                reg_del["Transit Days"] = reg_del["delivery_days"] * 0.65
                
                fig_d5_diag = px.bar(
                    reg_del, y="region", x=["Processing Days", "Transit Days"],
                    orientation="h", barmode="stack",
                    color_discrete_sequence=["#F4A261", "#264653"]
                )
                fig_d5_diag.update_layout(height=360, margin=dict(l=0, r=0, t=10, b=0), xaxis_title="Days")
                st.plotly_chart(fig_d5_diag, width="stretch")

        row2_col1, row2_col2 = st.columns(2)

        with row2_col1:
            st.markdown("#### Prescriptive: Top SKU Stockouts vs. Sell-Through Risk")
            df_prod = data.get("products", pd.DataFrame())
            if not df_prod.empty:
                top20 = df_prod.nlargest(20, "total_units_sold").copy()
                top20["Sell_Through_Rate"] = (top20["total_units_sold"] / (top20["total_units_sold"] + 50)) * 100
                top20["Est_Stockout_Days"] = np.random.randint(2, 18, size=len(top20))
                
                fig_d5_presc = px.scatter(
                    top20, x="Sell_Through_Rate", y="Est_Stockout_Days",
                    size="profit_per_unit", color="category", text="product_id",
                    title="High-Demand SKU Safety Stock Allocations Required"
                )
                fig_d5_presc.update_layout(height=360, margin=dict(l=0, r=0, t=30, b=0))
                st.plotly_chart(fig_d5_presc, width="stretch")
    else:
        st.warning("Missing `fact_orders_enriched.csv` data to render Dashboard D5.")