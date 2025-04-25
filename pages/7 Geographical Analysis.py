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
    from analysis.cltv import summary, cltv_df
    from analysis.Others import customer_features
    return full_orders, geolocation, summary, cltv_df, customer_features

full_orders, geolocation, summary, cltv_df, customer_features = load_data()

cltv_df = cltv_df.dropna()

# --- Cached processing steps ---
@st.cache_data
def prepare_cltv_geo_df(full_orders, cltv_df):
    return pd.merge(
        full_orders,
        cltv_df[["customer_unique_id", "better_cltv"]],
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
    # Calculating the summary
    summary = df.groupby("cluster")[["num_orders", "recency_days", "avg_order_value", "purchase_span_days"]].mean().round(2)
    # Step 2: Assign human-readable segment labels based on patterns
    # Compute averages for use in thresholding
    mean_orders = summary["num_orders"].mean()
    mean_recency = summary["recency_days"].mean()
    mean_order_value = summary["avg_order_value"].mean()
    mean_span = summary["purchase_span_days"].mean()

    segment_map = {}

    for idx, row in summary.iterrows():
        if (
            row["num_orders"] > mean_orders and
            row["avg_order_value"] > mean_order_value and
            row["purchase_span_days"] > mean_span and
            row["recency_days"] < mean_recency
        ):
            segment_map[idx] = "Loyal Customers"
        
        elif row["avg_order_value"] > mean_order_value * 2:
            segment_map[idx] = "High Value Buyers"
        
        elif (
            row["num_orders"] <= 2 and 
            row["purchase_span_days"] < 5 and 
            row["recency_days"] < mean_recency
        ):
            segment_map[idx] = "New Customers"
        
        elif row["recency_days"] > mean_recency * 1.2:
            segment_map[idx] = "Inactive Customers"
        
        else:
            segment_map[idx] = "Occasional Buyers"
   
    # Step 3: Apply labels
    df["segment"] = df["cluster"].map(segment_map)
    summary["segment"] = summary.index.map(segment_map)
    summary = summary.reset_index()[["cluster", "segment", "num_orders", "recency_days", "avg_order_value", "purchase_span_days"]]

    return df, summary   

# --- Section 1: CLTV by State and City ---
with st.expander("📦 1. CLTV by State and City", expanded=False):
    cltv_geo_df = prepare_cltv_geo_df(full_orders, cltv_df)
    group_choice = st.selectbox("Group by", ["State", "City"])
    top_n = st.slider("Select Top N", min_value=5, max_value=30, value=10)

    if group_choice == "State":
        top_df = cltv_geo_df.groupby("customer_state").agg(
            total_cltv=("better_cltv", "sum"),
            unique_customers=("customer_unique_id", pd.Series.nunique)
        ).sort_values("total_cltv", ascending=False).head(top_n).reset_index()

        st.subheader(f"Top {top_n} States by CLTV")
        st.bar_chart(top_df.set_index("customer_state")["total_cltv"])

    else:
        top_df = cltv_geo_df.groupby("customer_city").agg(
            total_cltv=("predicted_cltv", "sum"),
            unique_customers=("customer_unique_id", pd.Series.nunique)
        ).sort_values("total_cltv", ascending=False).head(top_n).reset_index()

        st.subheader(f"Top {top_n} Cities by CLTV")
        st.bar_chart(top_df.set_index("customer_city")["total_cltv"])
    if st.checkbox("Show Insights", key="unique_key_1"):
        st.info("""
        This chart ranks states or cities based on their **total Customer Lifetime Value (CLTV)**.  
    
        - **CLTV** estimates how much revenue a customer will generate over their relationship with the business.
        - By aggregating CLTV across regions, we are identifying the **high-value markets**.
        - This can help us do **targeted marketing**, **logistics planning**, or **inventory decisions**.
    
        **Insights**:
        - States/cities with high CLTV might have stronger customer engagement, loyalty, or purchasing power.
        - A lower number of unique customers but high CLTV indicates **fewer but very valuable customers**.
        """)

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
    state_agg = filtered.groupby(["state", "city", "lat", "lon"]).agg({metric: "sum"}).reset_index()
    if metric == "total_orders":
        # Scale orders to a reasonable visual range
        state_agg["scaled_metric"] = state_agg[metric].apply(lambda x: (x / 18735) * 40000 + 1000)
    else:
        # Use raw values for total_revenue, or slight scaling if needed
        state_agg["scaled_metric"] = state_agg[metric] / 10
    anomal = ["porto trombetas", "ibiajara", "vila dos cabanos", "pau d'arco", "santana do sobrado", "santo antonio do canaa"]

    # Filter out rows where the city column matches any city in the anomal list
    state_agg = state_agg[~state_agg["city"].isin(anomal)]

    st.pydeck_chart(pdk.Deck(
        initial_view_state=pdk.ViewState(
            latitude=-14.2350, longitude=-51.9253, zoom=4.5, pitch=40
        ),
        layers=[
        pdk.Layer(
            "ScatterplotLayer",
            data=state_agg,
            get_position='[lon, lat]',
            get_radius='scaled_metric', #f"{metric} / 10",   Adjust scaling as needed
            get_fill_color="[255, 140, 0, 180]",
            pickable=True,
            radius_scale=20,
            tooltip=True
            )
        ],
        tooltip={
            "html": """
                <b>State:</b> {state}<br/>
                <b>City:</b> {city}<br/>
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

    if st.checkbox("Show Insights", key="unique_key_2"):
        st.info(f"""
        The map above visualizes **{metric.replace('_', ' ').title()}** across Brazilian cities over time.

        - **Bubble size** reflects the magnitude of {metric.replace('_', ' ')}.
        - And we can explore changes over months using the slider above.
    
        **Use Cases**:
        - Spotting **sales surges** or **drops** over time.
        - Identifying the **regional demand trends** for strategic planning.

        **Insights**:
        - Larger bubbles in regions like São Paulo, Rio de Janeiro, or Belo Horizonte indicate economic hubs.
        - A larger radius bubble can be seen for Rio de Janeiro, even though it is the second most contributing city after São Paulo, this is because the grouping is done on the city name to identify the city-wise total {metric.replace('_', ' ')} and for Rio de Janeiro the lat, lon values are the same, while for São Paulo the different transactions have similar but varying lat, lon values showing that there is more granularity in capturing the locations for the transactions made from São Paulo.  
        - In the initial few months the total revenue was in the lower range all across Brazil but around March and April 2017 the revenue picked up, not only in the major cities like São Paulo, Rio de Janeiro, or Belo Horizonte, but also across the other cities in Brazil. This might suggest that either the OList operations picked up pace during these months, or that the transactions for earlier periods are not as extensively available as in this period.
        - Seasonal fluctuations can be observed in the major cities, especially during the Christmas and New Year period from Dec 2017 to Jan 2018, useful for promotions in cities like Mogi das Cruzes, to drive revenue during the off-season period as well.
        """)


# --- Section 3: Geo Clustering ---
with st.expander("🧭 3. Geo Segmentation (KMeans Clustering)", expanded=False):
    sta_agg = filtered.groupby(["state", "city", "lat", "lon"]).agg({"total_revenue": "sum"}).reset_index()
    anomal = ["porto trombetas", "ibiajara", "vila dos cabanos", "pau d'arco", "santana do sobrado", "santo antonio do canaa"]

    # Filter out rows where the city column matches any city in the anomal list
    sta_agg = sta_agg[~sta_agg["city"].isin(anomal)]
    
    geo_clustered = run_geo_clustering(sta_agg)

    # Assign a color to each cluster
    cluster_colors = {
        0: [255, 99, 132],   # Red
        1: [54, 162, 235],   # Blue
        2: [255, 206, 86],   # Yellow
        3: [75, 192, 192],   # Teal
        4: [153, 102, 255],  # Purple
    }
    geo_clustered["color"] = geo_clustered["cluster"].map(cluster_colors)

    # NEW: Let user filter which clusters to show
    unique_clusters = sorted(geo_clustered["cluster"].unique())
    selected_clusters = st.multiselect(
        "Select clusters to display:",
        options=unique_clusters,
        default=unique_clusters,
        format_func=lambda x: f"Cluster {x}"
    )

    # Filter data based on selection
    visible_data = geo_clustered[geo_clustered["cluster"].isin(selected_clusters)]
    
    st.pydeck_chart(pdk.Deck(
        initial_view_state=pdk.ViewState(
            latitude=-14.2350,
            longitude=-51.9253,
            zoom=4.5,
            pitch=40,
        ),
        layers=[
            pdk.Layer(
                "ScatterplotLayer",
                data=visible_data,
                get_position='[lon, lat]',
                get_fill_color="color",
                get_radius=30000,
                pickable=True
            )
        ],
        tooltip={
            "html": """
                <b>City:</b> {city}<br/>
                <b>State:</b> {state}<br/>
                <b>Total Revenue:</b> {total_revenue}<br/>
                <b>Cluster:</b> {cluster}
            """,
            "style": {
                "backgroundColor": "black",
                "color": "white",
                "fontSize": "13px",
                "padding": "8px",
            },
        }
    ))
    
    show_insights = st.checkbox("Show Insights", key="unique_key_3")
    if show_insights:
        st.info("""
        The map above shows geospatial clusters of cities based on three features:
        - **Latitude & Longitude** (location)
        - **Total Revenue** (economic activity)

        Using KMeans clustering, the cities were grouped into 5 distinct clusters based on the **proximity and revenue behavior**.  
        Each color on the map represents a unique cluster, helping to identify regional similarities or differences in performance.

        **Interpretations**:
        - A cluster of high-revenue cities is represented by cluster indexed 1 (yellow color) and represents economic hotspots.
        - Geographically close cities may still belong to different clusters if their revenues differ significantly. 
        """)
        @st.cache_data
        def explain_geo_clusters(geo_clustered):
            return geo_clustered.groupby("cluster")[["lat", "lon", "total_revenue"]].mean().round(2)

        st.subheader("🧩 Cluster Averages Overview")
        st.dataframe(explain_geo_clusters(geo_clustered))


# --- Section 4: Monthly Revenue Trend by State ---

# --- Section 4: 📈 Monthly Revenue Time-Series by State ---
@st.cache_data
def get_monthly_revenue_by_state(full_orders):
    full_orders['order_purchase_timestamp'] = pd.to_datetime(full_orders['order_purchase_timestamp'])
    full_orders['year_month'] = full_orders['order_purchase_timestamp'].dt.to_period('M').astype(str)
    return full_orders.groupby(['customer_state', 'year_month'])['payment_value'].sum().reset_index()

with st.expander("📈 4. Monthly Revenue Time-Series by State", expanded=False):
    ts_data = get_monthly_revenue_by_state(full_orders)

    selected_states = st.multiselect("Select states to view", sorted(ts_data["customer_state"].unique()), default=["SP", "RJ", "MG"])
    filtered_ts = ts_data[ts_data["customer_state"].isin(selected_states)]

    pivot_df = filtered_ts.pivot(index="year_month", columns="customer_state", values="payment_value").fillna(0)

    st.line_chart(pivot_df)

    if st.checkbox("Show Insights", key="unique_key_4"):
        st.info("""
        This chart displays **monthly revenue trends by state**, offering a temporal perspective beyond static totals.

        **Use Cases**:
        - Understand **seasonal fluctuations**
        - Monitor **growth vs stagnation** over time
        - Help with **regional marketing timing**

        **Insights**:
        - Consistent upward trends may signal strong market penetration.
        - States with flat or declining lines warrant attention or intervention.
        """)

# --- Section 5: 🧭 Top Customer Segments per State ---
# --- Section 5: 🧭 Top Customer Segments per State ---
# --- Section 5: 🧭 Top Customer Segments per State ---
@st.cache_data
def get_top_segments_by_state(cltv_df, full_orders):
    # Merge to get state and segment info
    merged_df = pd.merge(
        cltv_df[["customer_unique_id", "CLTV_new_Segment"]],
        full_orders[["customer_unique_id", "customer_state"]],
        on="customer_unique_id",
        how="inner"
    )
    
    # Group to get segment counts per state
    segment_counts = (
        merged_df.groupby(["customer_state", "CLTV_new_Segment"])
        .size()
        .reset_index(name="count")
        .rename(columns={"CLTV_new_Segment": "segment"})
    )
    return segment_counts

with st.expander("🧭 5. Top Customer Segments per State", expanded=False):
    
    seg_data = get_top_segments_by_state(cltv_df, full_orders)

    st.dataframe(seg_data)
    selected_states_seg = st.multiselect("Select states", sorted(seg_data["customer_state"].unique()), default=["SP", "RJ"])
    filtered_seg = seg_data[seg_data["customer_state"].isin(selected_states_seg)]

    st.bar_chart(
        filtered_seg.pivot(index='segment', columns='customer_state', values='count').fillna(0)
    )

    if st.checkbox("Show Insights", key="unique_key_5"):
        st.info("""
        This view helps understand **which types of customers dominate in each state**.

        **Use Cases**:
        - Tailor promotions by customer behavior in a state
        - Adjust acquisition vs retention strategies regionally

        **Insights**:
        - States with more “Loyal Customers” = retention opportunity.
        - States with many “Inactive Customers” = reactivation opportunity.
        """)



# --- Section 6: 📊 CLTV vs Revenue + 🚨 Drop Detection ---
@st.cache_data
def cltv_vs_revenue(full_orders, summary):
    df = pd.merge(full_orders, summary[["customer_unique_id", "predicted_cltv"]], on="customer_unique_id", how="inner")
    df["order_purchase_timestamp"] = pd.to_datetime(df["order_purchase_timestamp"])
    df["year_month"] = df["order_purchase_timestamp"].dt.to_period("M").astype(str)

    revenue_monthly = df.groupby(["customer_state", "year_month"]).agg(
        revenue=("payment_value", "sum"),
        cltv=("predicted_cltv", "sum")
    ).reset_index()

    revenue_monthly["prev_revenue"] = revenue_monthly.groupby("customer_state")["revenue"].shift(1)
    revenue_monthly["pct_change"] = (revenue_monthly["revenue"] - revenue_monthly["prev_revenue"]) / revenue_monthly["prev_revenue"] * 100

    revenue_monthly["drop_flag"] = revenue_monthly["pct_change"] < -20  # drop threshold
    return revenue_monthly

with st.expander("📊 6. CLTV vs Revenue by State + 🚨 Drop Detection", expanded=False):
    comparison_df = cltv_vs_revenue(full_orders, summary)

    selected_state_cmp = st.selectbox("Select a state", sorted(comparison_df["customer_state"].unique()), index=0)
    cmp_data = comparison_df[comparison_df["customer_state"] == selected_state_cmp]

    st.line_chart(cmp_data.set_index("year_month")[["revenue", "cltv"]])

    flagged = cmp_data[cmp_data["drop_flag"] == True]
    if not flagged.empty:
        st.warning(f"🚨 Revenue drop detected in {len(flagged)} month(s):")
        st.dataframe(flagged[["year_month", "revenue", "pct_change"]])
    else:
        st.success("✅ No major revenue drops detected for this state.")

    if st.checkbox("Show Insights", key="unique_key_6"):
        st.info("""
        This dual analysis tracks **revenue vs CLTV over time** and **automatically flags major drops**.

        **Use Cases**:
        - Early detection of revenue slowdowns despite high CLTV
        - Identify **regions needing immediate business attention**

        **Insights**:
        - High CLTV + Dropping Revenue = Potential churn or operational issue.
        - Flat CLTV + Revenue Drop = Market disengagement or economic shift.
        """)
