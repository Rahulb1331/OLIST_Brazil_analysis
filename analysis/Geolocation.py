from analysis.Preprocessing import full_orders, geolocation
from analysis.mba import summary_spark
from analysis.Others import customer_features
from pyspark.sql.functions import col, countDistinct, count, sum, avg, when, lit, first
from pyspark.sql import functions as F
import plotly.express as px
import pandas as pd

# Merging full_orders with CLTV summary (summary_spark) if not already merged
cltv_geo_df = full_orders.join(
    summary_spark.select("customer_unique_id", "predicted_cltv"),
    on="customer_unique_id",
    how="inner"
)

# Grouping by state and city
top_states = cltv_geo_df.groupBy("customer_state") \
    .agg(F.sum("predicted_cltv").alias("total_cltv"),
         F.countDistinct("customer_unique_id").alias("unique_customers")) \
    .orderBy(F.desc("total_cltv"))

top_cities = cltv_geo_df.groupBy("customer_city") \
    .agg(F.sum("predicted_cltv").alias("total_cltv"),
         F.countDistinct("customer_unique_id").alias("unique_customers")) \
    .orderBy(F.desc("total_cltv"))

top_states_pd = top_states.toPandas()
top_cities_pd = top_cities.toPandas()


# Top 10 states by CLTV
fig = px.bar(
    top_states_pd.head(10),
    x="customer_state",
    y="total_cltv",
    text="unique_customers",
    title="Top States by Total CLTV",
    labels={"customer_state": "State", "total_cltv": "Total CLTV"},
    color="total_cltv"
)
fig.update_traces(textposition="outside")
fig.update_layout(template="plotly_white")
fig.show()



# Revenue by state
revenue_by_state = full_orders.groupBy("customer_state").agg(
    F.sum("payment_value").alias("total_revenue"),
    F.countDistinct("order_id").alias("total_orders")
).orderBy(F.desc("total_revenue"))

# Revenue by city
revenue_by_city = full_orders.groupBy("customer_city").agg(
    F.sum("payment_value").alias("total_revenue"),
    F.countDistinct("order_id").alias("total_orders")
).orderBy(F.desc("total_revenue"))

revenue_by_state_pd = revenue_by_state.toPandas()
revenue_by_city_pd = revenue_by_city.toPandas()


fig = px.bar(
    revenue_by_state_pd.head(10),
    x="customer_state",
    y="total_revenue",
    text="total_orders",
    title="Top States by Revenue",
    labels={"customer_state": "State", "total_revenue": "Revenue (BRL)"},
    color="total_revenue"
)

fig.update_traces(textposition="outside")
fig.update_layout(template="plotly_white")
fig.show()

# Prepare average lat/lng per zip
geo_coords = geolocation.groupBy("geolocation_zip_code_prefix") \
    .agg(
        avg("geolocation_lat").alias("lat"),
        avg("geolocation_lng").alias("lon"),
        first("geolocation_city").alias("city"),
        first("geolocation_state").alias("state")
    )

#Revenue + Orders by Zip
from pyspark.sql.functions import sum as _sum

orders_by_zip = full_orders.groupBy("customer_zip_code_prefix") \
    .agg(
        _sum("payment_value").alias("total_revenue"),
        count("order_id").alias("total_orders")
    )

#Join with geolocation
geo_insights = orders_by_zip.join(
    geo_coords,
    orders_by_zip.customer_zip_code_prefix == geo_coords.geolocation_zip_code_prefix,
    how="inner"
).select("customer_zip_code_prefix", "lat", "lon", "city", "state", "total_revenue", "total_orders")

geo_pd = geo_insights.toPandas()

fig = px.scatter_geo(
    geo_pd,
    lat="lat",
    lon="lon",
    size="total_revenue",  # Or "total_orders"
    color="total_revenue",
    hover_name="city",
    scope="south america",
    title="Revenue by City (Geo Bubble Map)",
    projection="natural earth"
)

fig.update_layout(template="plotly_white")
fig.show()


# Geo Segmentation with KMeans Clustering
# We’ll cluster based on location (lat/lon) and total revenue to segment cities/zips into business clusters.

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

features = geo_pd[["lat", "lon", "total_revenue"]].copy()

scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# KMeans Clustering
kmeans = KMeans(n_clusters=5, random_state=42)
geo_pd["cluster"] = kmeans.fit_predict(scaled_features)

fig = px.scatter_geo(
    geo_pd,
    lat="lat",
    lon="lon",
    color="cluster",
    hover_name="city",
    size="total_revenue",
    scope="south america",
    title="Geo Segmentation using KMeans Clustering",
    projection="natural earth"
)
fig.update_layout(template="plotly_white")
fig.show()

# Convert to Pandas
cluster_df = customer_features.toPandas()

# Convert datetime columns
cluster_df['first_purchase'] = pd.to_datetime(cluster_df['first_purchase'])
cluster_df['last_purchase'] = pd.to_datetime(cluster_df['last_purchase'])

# Create new features
from datetime import datetime

max_date = cluster_df['last_purchase'].max()

cluster_df["recency_days"] = (max_date - cluster_df["last_purchase"]).dt.days
cluster_df["purchase_span_days"] = (cluster_df["last_purchase"] - cluster_df["first_purchase"]).dt.days
cluster_df["avg_order_value"] = cluster_df["total_revenue"] / cluster_df["num_orders"]

# Drop unnecessary cols
cluster_df = cluster_df.drop(columns=["first_purchase", "last_purchase", "customer_unique_id"])


# Select only numeric clustering features
features = cluster_df[["num_orders", "recency_days", "avg_order_value", "purchase_span_days"]]

# Standardize
scaler = StandardScaler()
scaled = scaler.fit_transform(features)

# KMeans Clustering
kmeans = KMeans(n_clusters=4, random_state=42)
cluster_df["cluster"] = kmeans.fit_predict(scaled)


from sklearn.decomposition import PCA

pca = PCA(n_components=2)
pca_result = pca.fit_transform(scaled)

cluster_df["pca1"] = pca_result[:, 0]
cluster_df["pca2"] = pca_result[:, 1]

fig = px.scatter(
    cluster_df, x="pca1", y="pca2", color="cluster",
    title="Customer Segments (PCA Projection)",
    hover_data=["num_orders", "avg_order_value", "recency_days"]
)
fig.show()
