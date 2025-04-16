# Scripts/pages/cltv_page.py

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from Scripts.config import setup_environment
setup_environment()

import streamlit as st
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit, when, min as spark_min, max as spark_max
from analysis.Preprocessing import full_orders
from analysis.rfm import run_rfm_analysis
from analysis.cltv import run_cltv_analysis, enrich_cltv_with_segments, model_cltv_lifetimes
import plotly.express as px

# Setup Spark session
venv_python_path = sys.executable
spark = SparkSession.builder \
    .appName("CLTV Streamlit") \
    .config("spark.sql.shuffle.partitions", "50") \
    .config("spark.executorEnv.PYSPARK_PYTHON", venv_python_path) \
    .config("spark.driverEnv.PYSPARK_PYTHON", venv_python_path) \
    .config("spark.pyspark.python", venv_python_path) \
    .config("spark.pyspark.driver.python", venv_python_path) \
    .getOrCreate()

st.title("💸 Customer Lifetime Value (CLTV) Analysis")

# Load Data
orders_df = full_orders
rfm_df = run_rfm_analysis(orders_df)

# Run CLTV analysis
cltv_df = run_cltv_analysis(orders_df)
cltv_df = enrich_cltv_with_segments(cltv_df)

# Join with RFM
rfm_cltv_df = rfm_df.join(
    cltv_df.select("customer_unique_id", "better_cltv", "cltv_normalized", "CLTV_new_Segment"),
    on="customer_unique_id",
    how="inner"
)

st.subheader("🔍 CLTV Segmentation")
st.dataframe(cltv_df.toPandas().head(10))

# Visualize CLTV Distribution
cltv_pd = rfm_cltv_df.select("cltv_normalized", "CLTV_new_Segment").toPandas()

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

# Lifetimes Modeling
st.subheader("🧪 BG/NBD + Gamma-Gamma CLTV Modeling")
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
