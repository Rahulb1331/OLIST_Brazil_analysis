import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

st.set_page_config(page_title="Geolocation & CLTV Dashboard", layout="wide")
st.title("📍 Geolocation Insights & Customer Segmentation")

# --- Caching heavy data loads ---
@st.cache_data
def load_data():
    from analysis.Preprocessing import full_orders, geolocation
    from analysis.cltv import summary
    from analysis.Others import customer_features
    return full_orders, geolocation, summary, customer_features

full_orders, geolocation, summary, customer_features = load_data()

# --- Cached processing steps ---
@st.cache_data
def prepare_cltv_geo_df(full_orders, summary):
    return pd.merge(
        full_orders,
        summary[["customer_unique_id", "predicted_cltv"]],
        on="customer_unique_id",
        how="inner"
    )

@st.cache_data
def get_geo_bubble_data(full_orders, geolocation):
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

    return pd.merge(
        orders_by_zip,
        grouped_geo,
        left_on="customer_zip_code_prefix",
        right_on="geolocation_zip_code_prefix",
        how="inner"
    )

@st.cache_data
def run_geo_clustering(geo_pd):
    geo_features = geo_pd[["lat", "lon", "total_revenue"]]
    scaled_features = StandardScaler().fit_transform(geo_features)

    kmeans_geo = KMeans(n_clusters=5, random_state=42)
    geo_pd["cluster"] = kmeans_geo.fit_predict(scaled_features)
    return geo_pd

@st.cache_data
def run_customer_segmentation(customer_features):
    df = customer_features.copy()
    df['first_purchase'] = pd.to_datetime(df['first_purchase'])
    df['last_purchase'] = pd.to_datetime(df['last_purchase'])

    max_date = df['last_purchase'].max()
    df["recency_days"] = (max_date - df["last_purchase"]).dt.days
    df["purchase_span_days"] = (df["last_purchase"] - df["first_purchase"]).dt.days
    df["avg_order_value"] = df["total_revenue"] / df["num_orders"]

    df = df.drop(columns=["first_purchase", "last_purchase", "customer_unique_id"])
    features = df[["num_orders", "recency_days", "avg_order_value", "purchase_span_days"]]
    scaled = StandardScaler().fit_transform(features)

    kmeans = KMeans(n_clusters=4, random_state=42)
    df["cluster"] = kmeans.fit_predict(scaled)

    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(scaled)
    df["pca1"] = pca_result[:, 0]
    df["pca2"] = pca_result[:, 1]

    labels = {0: "Loyal High Spenders", 1: "Occasional Buyers", 2: "Inactive Low Spenders", 3: "New Customers"}
    df["segment"] = df["cluster"].map(labels)
    return df, df.groupby("cluster")[["num_orders", "recency_days", "avg_order_value", "purchase_span_days"]].mean().round(2)

# --- Section 1: CLTV by State ---
with st.expander("📦 1. CLTV by State and City", expanded=False):
    cltv_geo_df = prepare_cltv_geo_df(full_orders, summary)

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

# --- Section 2: Revenue by State ---
with st.expander("💰 2. Revenue by State", expanded=False):
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

# --- Section 3: Geo Bubble Map ---
with st.expander("🗺️ 3. Geo Revenue Bubble Map", expanded=False):
    geo_pd = get_geo_bubble_data(full_orders, geolocation)

    fig3 = px.scatter_geo(
        geo_pd, lat="lat", lon="lon", size="total_revenue",
        color="total_revenue", hover_name="city",
        scope="south america", title="Revenue by City (Geo Bubble Map)",
        projection="natural earth"
    )
    fig3.update_layout(template="plotly_white")
    st.plotly_chart(fig3, use_container_width=True)

# --- Section 4: Geo Clustering ---
with st.expander("🧭 4. Geo Segmentation (KMeans Clustering)", expanded=False):
    geo_clustered = run_geo_clustering(geo_pd)

    fig4 = px.scatter_geo(
        geo_clustered, lat="lat", lon="lon", color="cluster",
        hover_name="city", size="total_revenue",
        scope="south america", title="Geo Segmentation using KMeans",
        projection="natural earth"
    )
    fig4.update_layout(template="plotly_white")
    st.plotly_chart(fig4, use_container_width=True)

# --- Section 5: Customer Behavioral Clustering ---
with st.expander("🧠 5. Customer Behavioral Clustering (KMeans + PCA)", expanded=False):
    cluster_df, cluster_summary = run_customer_segmentation(customer_features)

    st.dataframe(cluster_summary)

    fig5 = px.scatter(
        cluster_df, x="pca1", y="pca2", color="segment",
        title="Customer Segments (PCA Projection)",
        hover_data=["num_orders", "avg_order_value", "recency_days"]
    )
    fig5.update_layout(template="plotly_white")
    st.plotly_chart(fig5, use_container_width=True)
