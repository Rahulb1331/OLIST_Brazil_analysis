import streamlit as st
import pandas as pd
import pydeck as pdk
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
        total_orders=("order_id", "count"),
        first_order_date=("order_purchase_timestamp", "min"),
        last_order_date=("order_purchase_timestamp", "max")
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

# --- Section 3: Pydeck Hex Grid & Daily Revenue Bar Plot ---
with st.expander("🗺️ 3. Interactive Geo Revenue View", expanded=True):
    geo_pd = get_geo_bubble_data(full_orders, geolocation)
    geo_pd['order_day'] = pd.to_datetime(full_orders['order_purchase_timestamp']).dt.date

    min_date = geo_pd['first_order_date'].min().date()
    max_date = geo_pd['last_order_date'].max().date()

    start_date, end_date = st.slider("Select order date range", min_value=min_date, max_value=max_date,
                                      value=(min_date, max_date), format="YYYY-MM-DD")

    filtered_geo = geo_pd[(geo_pd['first_order_date'].dt.date >= start_date) & (geo_pd['last_order_date'].dt.date <= end_date)]

    st.pydeck_chart(pdk.Deck(
        map_style="mapbox://styles/mapbox/light-v9",
        initial_view_state=pdk.ViewState(
            latitude=-14.2350,
            longitude=-51.9253,
            zoom=3.5,
            pitch=40,
        ),
        layers=[
            pdk.Layer(
                "HexagonLayer",
                data=filtered_geo,
                get_position='[lon, lat]',
                radius=10000,
                elevation_scale=100,
                elevation_range=[0, 3000],
                pickable=True,
                extruded=True,
            )
        ],
    ))

    st.markdown("### 📊 Daily Sales in Selected Range")
    date_filtered_orders = full_orders[(full_orders['order_purchase_timestamp'].dt.date >= start_date) &
                                       (full_orders['order_purchase_timestamp'].dt.date <= end_date)]

    daily_sales = date_filtered_orders.groupby(date_filtered_orders['order_purchase_timestamp'].dt.date).agg(
        total_revenue=('payment_value', 'sum'),
        total_orders=('order_id', 'count')
    ).reset_index().rename(columns={'order_purchase_timestamp': 'date'})

    fig_daily = px.bar(
        daily_sales, x='date', y='total_revenue',
        labels={'total_revenue': 'Revenue (BRL)', 'date': 'Date'},
        title='Total Revenue per Day'
    )
    fig_daily.update_layout(template="plotly_white")
    st.plotly_chart(fig_daily, use_container_width=True)

# --- Section 5: Customer Behavioral Clustering ---
with st.expander("🧠 4. Customer Behavioral Clustering (KMeans + PCA)", expanded=False):
    cluster_df, cluster_summary = run_customer_segmentation(customer_features)

    st.dataframe(cluster_summary)

    fig5 = px.scatter(
        cluster_df, x="pca1", y="pca2", color="segment",
        title="Customer Segments (PCA Projection)",
        hover_data=["num_orders", "avg_order_value", "recency_days"]
    )
    fig5.update_layout(template="plotly_white")
    st.plotly_chart(fig5, use_container_width=True)
