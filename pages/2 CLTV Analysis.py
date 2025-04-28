# Scripts/pages/cltv_page.py
import pandas as pd
import numpy as np
import plotly.express as px
from analysis.rfm import run_rfm_analysis
from analysis.cltv import run_cltv_analysis, enrich_cltv_with_segments, model_cltv_lifetimes

# --- Page Config ---
import streamlit as st
st.set_page_config(page_title="Customer Lifetime Value (CLTV)", layout="wide")
st.title("💸 Customer Lifetime Value (CLTV) Analysis")

# --- Caching heavy processing functions ---
@st.cache_data
def load_full_orders():
    from analysis.Preprocessing import full_orders
    return full_orders.dropna(subset=['customer_city', 'customer_state', 'product_category'])

@st.cache_data
def get_rfm(full_orders):
    return run_rfm_analysis(full_orders)

@st.cache_data
def get_cltv(full_orders):
    return run_cltv_analysis(full_orders)

@st.cache_data
def get_enriched_cltv(cltv_df):
    return enrich_cltv_with_segments(cltv_df)

@st.cache_data
def get_lifetimes_model(full_orders):
    return model_cltv_lifetimes(full_orders)

@st.cache_data
def merge_rfm_cltv(rfm_df, cltv_df):
    return pd.merge(
        rfm_df,
        cltv_df[["customer_unique_id", "better_cltv", "cltv_normalized", "CLTV_new_Segment"]],
        on="customer_unique_id",
        how="inner"
    )

# --- Load data ---
with st.spinner("Loading and processing data..."):
    full_orders = load_full_orders()
    rfm_df = get_rfm(full_orders)
    cltv_raw = get_cltv(full_orders)
    cltv_df = get_enriched_cltv(cltv_raw)
    summary_df = get_lifetimes_model(full_orders)
    rfm_cltv_df = merge_rfm_cltv(rfm_df, cltv_df)

# --- 1. Introduction Section ---
st.markdown("""
Customer Lifetime Value (CLTV) helps estimate the total value a customer brings to your business over their entire relationship.

This page models future revenue using **BG/NBD + Gamma-Gamma** models.
""")

with st.expander("ℹ️ What is BG/NBD + Gamma-Gamma?"):
    st.markdown("""
- **BG/NBD (Beta-Geometric/Negative Binomial Distribution)** predicts the number of future transactions.
- **Gamma-Gamma** predicts the average revenue per transaction.
Together, they allow us to estimate a customer's **total future value**.

These models assume that purchase and monetary patterns follow specific probability distributions.
""")

# --- 2. Model Performance Summary ---
st.subheader("📈 Model Performance")

with st.expander("🔍 See Model Assumptions and Confidence"):
    st.markdown("""
- Models assume customer dropout and spending patterns are stable over time.
- Prediction confidence is approximate; unexpected changes (economic, business, etc.) may cause deviations.
- Evaluation metrics (example placeholders):
    - **Mean Absolute Error (MAE):** ~\$10
    - **Root Mean Squared Error (RMSE):** ~\$14
""")

# --- 3. CLTV Segments Explanation ---
st.subheader("🎯 CLTV Segments Defined")

with st.expander("🧩 How CLTV Segments Were Created"):
    st.markdown("""
Customers were segmented based on normalized CLTV scores:

- **Champions** (Top 20%)
- **High Value** (20%-50%)
- **Medium Value** (50%-80%)
- **Low Value** (Bottom 20%)

Segments help prioritize marketing and retention efforts.
""")

st.dataframe(cltv_df.head(10))

# --- 4. Past vs Predicted Revenue ---
st.subheader("💵 Revenue Comparison: Past vs Predicted")

full_orders['order_purchase_timestamp'] = pd.to_datetime(full_orders['order_purchase_timestamp'])
full_orders['order_purchase_quarter'] = full_orders['order_purchase_timestamp'].dt.to_period('Q')

past_revenue = full_orders.groupby('customer_unique_id').agg(
    past_revenue=('payment_value', 'sum')
).reset_index()

summary_revenue = summary_df.merge(past_revenue, on='customer_unique_id', how='left').fillna(0)

col1, col2 = st.columns(2)

with col1:
    fig1 = px.bar(
        summary_revenue.groupby('cltv_segment')['past_revenue'].sum().reset_index(),
        x="cltv_segment",
        y="past_revenue",
        title="Past 12M Revenue by Segment",
        text_auto=".2s",
        color="cltv_segment"
    )
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    fig2 = px.bar(
        summary_revenue.groupby('cltv_segment')['predicted_cltv'].sum().reset_index(),
        x="cltv_segment",
        y="predicted_cltv",
        title="Predicted 12M Revenue by Segment",
        text_auto=".2s",
        color="cltv_segment"
    )
    st.plotly_chart(fig2, use_container_width=True)

# --- 5. Cohort Analysis ---
st.subheader("👥 Customer Cohort Analysis")

cohort_df = full_orders.copy()
cohort_df['first_purchase_quarter'] = cohort_df.groupby('customer_unique_id')['order_purchase_quarter'].transform('min')

cohort_summary = cohort_df.groupby('first_purchase_quarter').agg(
    customer_count=('customer_unique_id', 'nunique'),
    total_revenue=('payment_value', 'sum')
).reset_index()

fig3 = px.line(
    cohort_summary,
    x="first_purchase_quarter",
    y=["customer_count", "total_revenue"],
    markers=True,
    title="Cohort Analysis: New Customers and Revenue per Quarter"
)
st.plotly_chart(fig3, use_container_width=True)

# --- 6. Visualizations ---
st.subheader("📊 CLTV Distributions and Relationships")

if 'log_applied' not in st.session_state:
    st.session_state.log_applied = False

log_toggle = st.toggle("Apply Log Transformation", value=st.session_state.log_applied)
st.session_state.log_applied = log_toggle

cltv_pd = rfm_cltv_df[["cltv_normalized", "CLTV_new_Segment"]].copy()

if st.session_state.log_applied:
    cltv_pd['cltv_transformed'] = np.log1p(cltv_pd['cltv_normalized'] * 1000)
    title_suffix = "(Log Scale)"
else:
    cltv_pd['cltv_transformed'] = cltv_pd['cltv_normalized']
    title_suffix = "(Raw Scale)"

with st.expander("🌍 CLTV Distribution Histogram"):
    fig = px.histogram(
        cltv_pd,
        x="cltv_normalized",
        color="CLTV_new_Segment",
        nbins=30,
        title=f"CLTV Distribution by Segment {title_suffix}",
        labels={"cltv_normalized": "Normalized CLTV"},
        barmode="overlay",
        opacity=0.7
    )
    st.plotly_chart(fig, use_container_width=True)

with st.expander("📦 Box Plot of CLTV"):
    fig = px.box(
        cltv_pd,
        x="CLTV_new_Segment",
        y="cltv_transformed",
        color="CLTV_new_Segment",
        title=f"Boxplot: CLTV Distribution by Segment {title_suffix}"
    )
    st.plotly_chart(fig, use_container_width=True)

with st.expander("🧬 Scatterplots"):
    scatter_df = rfm_cltv_df.copy()
    scatter_df['RFM_Score'] = scatter_df['recency_score'] + scatter_df['frequency_score'] + scatter_df['monetary_score']

    fig4 = px.scatter(
        scatter_df,
        x="frequency_score",
        y="monetary_score",
        color="CLTV_new_Segment",
        title="Frequency vs Monetary Value by CLTV Segment",
        size_max=10
    )
    st.plotly_chart(fig4, use_container_width=True)

    fig5 = px.scatter(
        scatter_df,
        x="RFM_Score",
        y="cltv_normalized",
        color="CLTV_new_Segment",
        title="RFM Score vs CLTV Normalized",
        size_max=10
    )
    st.plotly_chart(fig5, use_container_width=True)

# --- 7. User Actions by Segment ---
st.subheader("🚀 Actions Suggested by Customer Segment")

st.markdown("""
- **Champions:** Exclusive offers, loyalty programs
- **High Value:** Upselling/cross-selling
- **Medium Value:** Targeted promotions
- **Low Value:** Win-back campaigns, reactivation
""")

# --- 8. Table Download ---
st.subheader("📥 Download CLTV Data")

csv = cltv_df.to_csv(index=False).encode('utf-8')
st.download_button("Download CLTV Data as CSV", data=csv, file_name="cltv_analysis.csv", mime='text/csv')

# --- 9. Filtering Section ---
st.subheader("🔍 Optional: Filter Customers")

with st.expander("📂 Filter Options"):
    state_filter = st.multiselect("Select States", full_orders['customer_state'].unique())
    city_filter = st.multiselect("Select Cities", full_orders['customer_city'].unique())
    product_filter = st.multiselect("Select Product Categories", full_orders['product_category'].unique())

    if state_filter or city_filter or product_filter:
        filtered_orders = full_orders[
            (full_orders['customer_state'].isin(state_filter) if state_filter else True) &
            (full_orders['customer_city'].isin(city_filter) if city_filter else True) &
            (full_orders['product_category'].isin(product_filter) if product_filter else True)
        ]
        st.write(f"Filtered Orders: {len(filtered_orders)} transactions")
        st.dataframe(filtered_orders.head(10))

