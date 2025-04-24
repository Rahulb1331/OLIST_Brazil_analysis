import streamlit as st
import pandas as pd
import pydeck as pdk
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from datetime import datetime

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
    full_orders['order_purchase_timestamp'] = pd.to_datetime(full_orders['order_purchase_timestamp'])
    full_orders['year_month'] = full_orders['order_purchase_timestamp'].dt.to_period('M').astype(str)

    grouped_geo = geolocation.groupby("geolocation_zip_code_prefix").agg(
        lat=("geolocation_lat", "mean"),
        lon=("geolocation_lng", "mean"),
        city=("geolocation_city", "first"),
        state=("geolocation_state", "first")
    ).reset_index()

    orders_by_zip_monthly = full_orders.groupby(["customer_zip_code_prefix", "year_month"]).agg(
        total_revenue=("payment_value", "sum"),
        total_orders=("order_id", "count"),
        state=("customer_state", "first"),
        city=("customer_city", "first")
    ).reset_index()

    merged = pd.merge(
        orders_by_zip_monthly,
        grouped_geo,
        left_on="customer_zip_code_prefix",
        right_on="geolocation_zip_code_prefix",
        how="inner"
    )
    return merged

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

    st.bar_chart(top_states_pd.set_index("customer_state")["total_cltv"])

# --- Section 2: Monthly Sales/Orders Map with Pydeck ---
with st.expander("🌍 2. Monthly Revenue/Orders Map", expanded=True):
    geo_pd = get_geo_bubble_data(full_orders, geolocation)
    geo_pd['year_month'] = pd.to_datetime(geo_pd['year_month'])

    metric = st.selectbox("Select Metric", ["total_revenue", "total_orders"], index=0)

    min_month = geo_pd['year_month'].min().date()
    max_month = geo_pd['year_month'].max().date()
    start, end = st.slider("Select Month Range", min_value=min_month, max_value=max_month, value=(min_month, max_month), format="MMM YYYY")

    filtered = geo_pd[(geo_pd['year_month'] >= pd.to_datetime(start)) & (geo_pd['year_month'] <= pd.to_datetime(end))]
    filtered = filtered.rename(columns={"state_x": "state", "city_x": "city"})
    filtered = filtered.drop(columns=["city_y", "state_y"])
    state_agg = filtered.groupby(["state", "lat", "lon"]).agg({metric: "sum"}).reset_index()

    st.pydeck_chart(pdk.Deck(
        initial_view_state=pdk.ViewState(
            latitude=-14.2350, longitude=-51.9253, zoom=4.5, pitch=40
        ),
        layers=[
        pdk.Layer(
            "ScatterplotLayer",
            data=state_agg,
            get_position='[lon, lat]',
            get_radius=f"{metric} / 10",  # Adjust scaling as needed
            get_fill_color="[255, 140, 0, 180]",
            pickable=True,
            radius_scale=20,
            tooltip=True
            )
        ],
        tooltip={
            "html": """
                <b>State:</b> {state}<br/>
                <b>Total Value:</b> {""" + metric + """}<br/>
                <b>Lat:</b> {lat}<br/>
                <b>Lon:</b> {lon}
            """,
            "style": {
                "backgroundColor": "rgba(0, 0, 0, 0.8)",
                "color": "white",
                "fontSize": "13px",
                "padding": "8px",
                "borderRadius": "5px"
            }
        }
    ))
       

    top_cities = filtered.groupby("city")[metric].sum().sort_values(ascending=False).head(10).reset_index()
    st.subheader("Top Contributing Cities")
    st.dataframe(top_cities)

# --- Section 3: Geo Clustering ---
with st.expander("🧭 3. Geo Segmentation (KMeans Clustering)", expanded=False):
    sta_agg = filtered.groupby(["state", "lat", "lon"]).agg({"total_revenue": "sum"}).reset_index()
    geo_clustered = run_geo_clustering(sta_agg)

    st.map(geo_clustered, latitude="lat", longitude="lon")

# --- Section 4: Customer Behavioral Clustering ---
with st.expander("🧠 4. Customer Behavioral Clustering (KMeans + PCA)", expanded=False):
    cluster_df, cluster_summary = run_customer_segmentation(customer_features)

    st.dataframe(cluster_summary)

    st.scatter_chart(cluster_df, x="pca1", y="pca2", color="segment")
