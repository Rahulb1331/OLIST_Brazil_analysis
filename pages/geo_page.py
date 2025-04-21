import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from analysis.Preprocessing import full_orders, geolocation
from analysis.cltv import summary
from analysis.Others import customer_features

st.set_page_config(page_title="Geolocation & CLTV Dashboard", layout="wide")

st.title("📍 Geolocation Insights & Customer Segmentation")

# Merge with CLTV
st.header("1. CLTV by State and City")
cltv_geo_df = pd.merge(
    full_orders,
    summary[["customer_unique_id", "predicted_cltv"]],
    on="customer_unique_id",
    how="inner"
)

top_states_pd = cltv_geo_df.groupby("customer_state").agg(
    total_cltv=("predicted_cltv", "sum"),
    unique_customers=("customer_unique_id", pd.Series.nunique)
).sort_values("total_cltv", ascending=False).reset_index()


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

revenue_by_state_pd = full_orders.groupby("customer_state").agg(
    total_revenue=("payment_value", "sum"),
    total_orders=("order_id", pd.Series.nunique)
).sort_values("total_revenue", ascending=False).reset_index()

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

grouped_geo = geolocation.groupby("geolocation_zip_code_prefix").agg(
    lat=("geolocation_lat", "mean"),
    lon=("geolocation_lng", "mean"),
    city=("geolocation_city", "first"),
    state=("geolocation_state", "first")
).reset_index()

orders_by_zip = full_orders.groupby("customer_zip_code_prefix").agg(
    total_revenue=("payment_value", "sum"),
    total_orders=("order_id", "count")
).reset_index()

geo_pd = pd.merge(
    orders_by_zip,
    grouped_geo,
    left_on="customer_zip_code_prefix",
    right_on="geolocation_zip_code_prefix",
    how="inner"
)

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

cluster_df = customer_features.copy()
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

# Summary statistics for each cluster
cluster_summary = cluster_df.groupby("cluster")[["num_orders", "recency_days", "avg_order_value", "purchase_span_days"]].mean().round(2)
st.dataframe(cluster_summary)

cluster_labels = {
    0: "Loyal High Spenders",
    1: "Occasional Buyers",
    2: "Inactive Low Spenders",
    3: "New Customers"
}
cluster_df["segment"] = cluster_df["cluster"].map(cluster_labels)

fig5 = px.scatter(
    cluster_df, x="pca1", y="pca2", color="segment",
    title="Customer Segments (PCA Projection)",
    hover_data=["num_orders", "avg_order_value", "recency_days"]
)
fig5.update_layout(template="plotly_white")
st.plotly_chart(fig5, use_container_width=True)
