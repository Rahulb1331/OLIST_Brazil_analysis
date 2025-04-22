# Scripts/pages/cltv_page.py
import streamlit as st
import pandas as pd
import numpy as np
from analysis.rfm import run_rfm_analysis
from analysis.cltv import run_cltv_analysis, enrich_cltv_with_segments, model_cltv_lifetimes
import plotly.express as px

@st.cache_data
def load_data():
    from analysis.Preprocessing import full_orders
    return full_orders

full_orders = load_data()

st.title("💸 Customer Lifetime Value (CLTV) Analysis")

# Load Data
@st.cache_data
orders_df = full_orders
rfm_df = run_rfm_analysis(orders_df)

# Initialize session state
if 'log_applied' not in st.session_state:
    st.session_state.log_applied = False

# Run CLTV analysis
@st.cache_data
cltv_df = run_cltv_analysis(orders_df)
@st.cache_data
cltv_df = enrich_cltv_with_segments(cltv_df)

# Join with RFM
@st.cache_data
rfm_cltv_df = pd.merge(
    rfm_df,
    cltv_df[["customer_unique_id", "better_cltv", "cltv_normalized", "CLTV_new_Segment"]],
    on="customer_unique_id",
    how="inner"
)

st.subheader("🔍 CLTV Segmentation")
st.dataframe(cltv_df.head(10))

# Visualize CLTV Distribution
cltv_pd = rfm_cltv_df[["cltv_normalized", "CLTV_new_Segment"]]

# Log transformation button in main area
if not st.session_state.log_applied:
    if st.button("Apply Log Transformation"):
        st.session_state.log_applied = True
        st.success("Log transformation applied.")

# Apply transformation
if st.session_state.log_applied:
    cltv_pd['cltv_transformed'] = np.log1p(cltv_pd['cltv_normalized'] * 1000)
    title_suffix = "(Log Scale)"
else:
    cltv_pd['cltv_transformed'] = cltv_pd['cltv_normalized']
    title_suffix = "(Raw Scale)"


fig = px.histogram(
    cltv_pd,
    x="cltv_normalized",
    color="CLTV_new_Segment",
    nbins=30,
    title="CLTV Distribution by Segment",
    labels={"cltv_normalized": "Normalized CLTV"},
    barmode="overlay",
    opacity=0.7
)
st.plotly_chart(fig)

import plotly.express as px


fig = px.box(
    cltv_pd,
    x="CLTV_new_Segment",
    y="cltv_transformed",
    color="CLTV_new_Segment",
    title="📦 CLTV Distribution by Segment",
    labels={"normalized_cltv": "Normalized CLTV", "CLTV_new_Segment": "CLTV Segment"},
    points="all",  # Show all individual points (optional)
    template="plotly_white"
)
st.plotly_chart(fig, use_container_width=True)

fig = px.violin(
    cltv_pd,
    x="CLTV_new_Segment",
    y="cltv_transformed",
    color="CLTV_new_Segment",
    box=True,  # show box inside
    points="all",
    title="📦 CLTV Distribution by Segment (Violin Plot)",
    labels={"normalized_cltv": "Normalized CLTV", "CLTV_new_Segment": "CLTV Segment"},
    template="plotly_white"
)
st.plotly_chart(fig, use_container_width=True)

# Lifetimes Modeling
st.subheader("🧪 BG/NBD + Gamma-Gamma CLTV Modeling")
@st.cache_data
summary_df = model_cltv_lifetimes(orders_df)
st.write("Top Customers by Predicted CLTV")
st.dataframe(summary_df[["customer_unique_id", "predicted_cltv"]].sort_values(by="predicted_cltv", ascending=False).head(10))

# Revenue Visualization
fig_rev = px.bar(
    summary_df.groupby("cltv_segment")["predicted_cltv"].sum().reset_index(),
    x="cltv_segment",
    y="predicted_cltv",
    title="Revenue Forecast by CLTV Segment (12 months)",
    text_auto=".2s",
    color="cltv_segment"
)
st.plotly_chart(fig_rev)
