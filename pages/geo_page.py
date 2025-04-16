import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from Scripts.config import setup_environment
setup_environment()

import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from analysis.Preprocessing import full_orders, geolocation
from analysis.mba import summary_spark
from analysis.Others import customer_features
from pyspark.sql import functions as F

st.set_page_config(page_title="Geolocation & CLTV Dashboard", layout="wide")

st.title("📍 Geolocation Insights & Customer Segmentation")

# Merge with CLTV
st.header("1. CLTV by State and City")
cltv_geo_df = full_orders.join(
    summary_spark.select("customer_unique_id", "predicted_cltv"),
    on="customer_unique_id", how="inner"
)

top_states = cltv_geo_df.groupBy("customer_state") \
    .agg(F.sum("predicted_cltv").alias("total_cltv"),
         F.countDistinct("customer_unique_id").alias("unique_customers")) \
    .orderBy(F.desc("total_cltv"))

top_states_pd = top_states.toPandas()

fig1 = px.bar(
    top_states_pd.head(10), x="customer_state", y="total_cltv",
    text="unique_customers", title="Top 10 States by Total CLTV",
    labels={"customer_state": "State", "total_cltv": "Total CLTV"},
    color="total_cltv"
)
fig1.update_traces(textposition="outside")
fig1.update_layout(template="plotly_white")
st.plotly_chart(fig1, use_container_width=True)

# Revenue by region
st.header("2. Revenue by State")

revenue_by_state = full_orders.groupBy("customer_state").agg(
    F.sum("payment_value").alias("total_revenue"),
    F.countDistinct("order_id").alias("total_orders")
).orderBy(F.desc("total_revenue"))

revenue_by_state_pd = revenue_by_state.toPandas()

fig2 = px.bar(
    revenue_by_state_pd.head(10),
    x="customer_state", y="total_revenue", text="total_orders",
    title="Top States by Revenue",
    labels={"customer_state": "State", "total_revenue": "Revenue (BRL)"},
    color="total_revenue"
)
fig2.update_traces(textposition="outside")
fig2.update_layout(template="plotly_white")
st.plotly_chart(fig2, use_container_width=True)

# Geo Bubble Map
st.header("3. Geo Revenue Bubble Map")

geo_coords = geolocation.groupBy("geolocation_zip_code_prefix") \
    .agg(
        F.avg("geolocation_lat").alias("lat"),
        F.avg("geolocation_lng").alias("lon"),
        F.first("geolocation_city").alias("city"),
        F.first("geolocation_state").alias("state")
    )

orders_by_zip = full_orders.groupBy("customer_zip_code_prefix") \
    .agg(
        F.sum("payment_value").alias("total_revenue"),
        F.count("order_id").alias("total_orders")
    )

geo_insights = orders_by_zip.join(
    geo_coords,
    orders_by_zip.customer_zip_code_prefix == geo_coords.geolocation_zip_code_prefix,
    how="inner"
).select("customer_zip_code_prefix", "lat", "lon", "city", "state", "total_revenue", "total_orders")

geo_pd = geo_insights.toPandas()

fig3 = px.scatter_geo(
    geo_pd, lat="lat", lon="lon", size="total_revenue",
    color="total_revenue", hover_name="city",
    scope="south america", title="Revenue by City (Geo Bubble Map)",
    projection="natural earth"
)
fig3.update_layout(template="plotly_white")
st.plotly_chart(fig3, use_container_width=True)

# Geo Clustering
st.header("4. Geo Segmentation (KMeans Clustering)")

geo_features = geo_pd[["lat", "lon", "total_revenue"]].copy()
scaler = StandardScaler()
scaled_features = scaler.fit_transform(geo_features)

kmeans_geo = KMeans(n_clusters=5, random_state=42)
geo_pd["cluster"] = kmeans_geo.fit_predict(scaled_features)

fig4 = px.scatter_geo(
    geo_pd, lat="lat", lon="lon", color="cluster",
    hover_name="city", size="total_revenue",
    scope="south america", title="Geo Segmentation using KMeans",
    projection="natural earth"
)
fig4.update_layout(template="plotly_white")
st.plotly_chart(fig4, use_container_width=True)

# Customer Segmentation
st.header("5. Customer Behavioral Clustering (KMeans + PCA)")

cluster_df = customer_features.toPandas()
cluster_df['first_purchase'] = pd.to_datetime(cluster_df['first_purchase'])
cluster_df['last_purchase'] = pd.to_datetime(cluster_df['last_purchase'])

max_date = cluster_df['last_purchase'].max()

cluster_df["recency_days"] = (max_date - cluster_df["last_purchase"]).dt.days
cluster_df["purchase_span_days"] = (cluster_df["last_purchase"] - cluster_df["first_purchase"]).dt.days
cluster_df["avg_order_value"] = cluster_df["total_revenue"] / cluster_df["num_orders"]

cluster_df = cluster_df.drop(columns=["first_purchase", "last_purchase", "customer_unique_id"])

features = cluster_df[["num_orders", "recency_days", "avg_order_value", "purchase_span_days"]]
scaled = StandardScaler().fit_transform(features)

kmeans = KMeans(n_clusters=4, random_state=42)
cluster_df["cluster"] = kmeans.fit_predict(scaled)

pca = PCA(n_components=2)
pca_result = pca.fit_transform(scaled)
cluster_df["pca1"] = pca_result[:, 0]
cluster_df["pca2"] = pca_result[:, 1]

fig5 = px.scatter(
    cluster_df, x="pca1", y="pca2", color="cluster",
    title="Customer Segments (PCA Projection)",
    hover_data=["num_orders", "avg_order_value", "recency_days"]
)
fig5.update_layout(template="plotly_white")
st.plotly_chart(fig5, use_container_width=True)
