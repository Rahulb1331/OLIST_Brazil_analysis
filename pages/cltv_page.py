# Scripts/pages/cltv_page.py
import streamlit as st
import pandas as pd
import numpy as np
from analysis.rfm import run_rfm_analysis
from analysis.cltv import run_cltv_analysis, enrich_cltv_with_segments, model_cltv_lifetimes
import plotly.express as px

st.title("💸 Customer Lifetime Value (CLTV) Analysis")

# --- Caching heavy processing functions ---
@st.cache_data
def load_full_orders():
    from analysis.Preprocessing import full_orders
    return full_orders

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

# --- Load data ---
full_orders = load_full_orders()
rfm_df = get_rfm(full_orders)
cltv_raw = get_cltv(full_orders)
cltv_df = get_enriched_cltv(cltv_raw)
summary_df = get_lifetimes_model(full_orders)

# --- Merge CLTV & RFM for Visualization ---
@st.cache_data
def merge_rfm_cltv(rfm_df, cltv_df):
    return pd.merge(
        rfm_df,
        cltv_df[["customer_unique_id", "better_cltv", "cltv_normalized", "CLTV_new_Segment"]],
        on="customer_unique_id",
        how="inner"
    )

rfm_cltv_df = merge_rfm_cltv(rfm_df, cltv_df)

# --- Display Initial Segmentation ---
st.subheader("🔍 CLTV Segmentation")
st.dataframe(cltv_df.head(10))

# --- Log Transformation Toggle ---
if 'log_applied' not in st.session_state:
    st.session_state.log_applied = False

#if not st.session_state.log_applied:
    #if st.button("Apply Log Transformation"):
        #st.session_state.log_applied = True
        #st.success("Log transformation applied.")

log_toggle = st.toggle("Apply Log Transformation", value=st.session_state.log_applied)
st.session_state.log_applied = log_toggle

# --- Visualization Section ---
cltv_pd = rfm_cltv_df[["cltv_normalized", "CLTV_new_Segment"]].copy()
#cltv_pd['cltv_transformed'] = np.log1p(cltv_pd['cltv_normalized'] * 1000) if st.session_state.log_applied else cltv_pd['cltv_normalized']
#title_suffix = "(Log Scale)" if st.session_state.log_applied else "(Raw Scale)"

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

with st.expander("🌆 CLTV Distribution Box Plot"):
    fig = px.box(
        cltv_pd,
        x="CLTV_new_Segment",
        y="cltv_transformed",
        color="CLTV_new_Segment",
        title=f"📦 CLTV Distribution by Segment {title_suffix}",
        labels={"cltv_transformed": "CLTV Value", "CLTV_new_Segment": "CLTV Segment"},
        template="plotly_white"
    )
    st.plotly_chart(fig, use_container_width=True)

with st.expander("🌋 CLTV Distribution Violin Plot"):
    fig = px.violin(
        cltv_pd,
        x="CLTV_new_Segment",
        y="cltv_transformed",
        color="CLTV_new_Segment",
        box=True,
        points="outliers",
        title=f"📦 CLTV Distribution by Segment (Violin Plot) {title_suffix}",
        template="plotly_white"
    )
    st.plotly_chart(fig, use_container_width=True)

# --- Lifetimes Modeling ---
st.subheader("🧪 BG/NBD + Gamma-Gamma CLTV Modeling")
st.dataframe(summary_df[["customer_unique_id", "predicted_cltv"]].sort_values(by="predicted_cltv", ascending=False).head(10))

# --- Revenue Forecast ---
fig_rev = px.bar(
    summary_df.groupby("cltv_segment")["predicted_cltv"].sum().reset_index(),
    x="cltv_segment",
    y="predicted_cltv",
    title="Revenue Forecast by CLTV Segment (12 months)",
    text_auto=".2s",
    color="cltv_segment"
)
st.plotly_chart(fig_rev, use_container_width=True)
