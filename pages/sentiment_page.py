import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from Scripts.config import setup_environment

setup_environment()
import streamlit as st
from pyspark.sql.functions import col, count, avg, datediff
from analysis.Preprocessing import full_orders, order_reviews
import plotly.express as px

# Setup Streamlit
st.set_page_config(page_title="Review Sentiment Analysis", layout="wide")
st.title("📊 Review Sentiment Dashboard")

# Join data
orders_with_reviews = full_orders.join(order_reviews, on="order_id", how="inner")
orders_with_reviews = orders_with_reviews.filter(col("seller_id").isNotNull())

# Tabs for exploration
tab1, tab2, tab3, tab4 = st.tabs([
    "⭐ Top & Worst Rated",
    "⏱️ Delivery vs Review",
    "🚚 Freight vs Review",
    "📊 Raw Data Snapshots"
])

with tab1:
    st.header("📦 Product & Seller Ratings")

    top_sellers = orders_with_reviews.groupBy("seller_id").agg(
        count("review_score").alias("num_reviews"),
        avg("review_score").alias("avg_rating")
    ).orderBy(col("avg_rating").desc(), col("num_reviews").desc()).limit(10)

    top_products = orders_with_reviews.groupBy("product_category").agg(
        count("review_score").alias("num_reviews"),
        avg("review_score").alias("avg_rating")
    ).orderBy(col("avg_rating").desc(), col("num_reviews").desc()).limit(10)

    worst_sellers = orders_with_reviews.groupBy("seller_id").agg(
        count("review_score").alias("num_reviews"),
        avg("review_score").alias("avg_rating")
    ).filter(col("num_reviews") >= 10).orderBy(col("avg_rating").asc()).limit(10)

    worst_products = orders_with_reviews.groupBy("product_category").agg(
        count("review_score").alias("num_reviews"),
        avg("review_score").alias("avg_rating")
    ).filter(col("num_reviews") >= 10).orderBy(col("avg_rating").asc()).limit(10)

    # Display plots
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Top 10 Products by Rating")
        top_pd = top_products.toPandas()
        st.plotly_chart(px.bar(top_pd, x="product_category", y="avg_rating", title="Top Products"))

    with col2:
        st.subheader("Worst 10 Products by Rating")
        worst_pd = worst_products.toPandas()
        st.plotly_chart(px.bar(worst_pd, x="product_category", y="avg_rating", title="Worst Products"))

    col3, col4 = st.columns(2)
    with col3:
        st.subheader("Top 10 Sellers by Rating")
        top_seller_pd = top_sellers.toPandas()
        st.plotly_chart(px.bar(top_seller_pd, x="seller_id", y="avg_rating", title="Top Sellers"))

    with col4:
        st.subheader("Worst 10 Sellers by Rating")
        worst_seller_pd = worst_sellers.toPandas()
        st.plotly_chart(px.bar(worst_seller_pd, x="seller_id", y="avg_rating", title="Worst Sellers"))

with tab2:
    st.header("⏳ Delivery Time Impact on Reviews")

    orders_with_reviews = orders_with_reviews.withColumn(
        "delivery_days", datediff("order_delivered_customer_date", "order_purchase_timestamp"))

    delivery_analysis = orders_with_reviews.groupBy("review_score").agg(
        avg("delivery_days").alias("avg_delivery_days"),
        count("*").alias("num_orders")
    ).orderBy("review_score")

    delivery_pd = delivery_analysis.toPandas()
    fig = px.bar(delivery_pd, x="review_score", y="avg_delivery_days",
                 text="avg_delivery_days", title="Delivery Days vs Review Score",
                 labels={"avg_delivery_days": "Avg Delivery Days", "review_score": "Review Score"})
    fig.update_traces(marker_color="tomato", texttemplate='%{text:.1f}')
    st.plotly_chart(fig)

    # Delay-based Worst Sellers
    st.subheader("Worst Sellers with Longest Delivery")
    seller_delay = orders_with_reviews.groupBy("seller_id").agg(
        count("*").alias("num_orders"),
        avg("review_score").alias("avg_rating"),
        avg("delivery_days").alias("avg_delivery_days")
    ).filter((col("num_orders") >= 10) & (col("avg_rating") <= 2)) \
     .orderBy(col("avg_delivery_days").desc()).limit(10)

    st.dataframe(seller_delay.toPandas())

with tab3:
    st.header("🚚 Freight Charges vs Review Score")

    freight_reviews = orders_with_reviews.groupBy("seller_id").agg(
        count("*").alias("num_reviews"),
        avg("review_score").alias("avg_rating"),
        avg("freight_value").alias("avg_freight")
    ).filter(col("num_reviews") > 10).orderBy("avg_rating")

    freight_pd = freight_reviews.toPandas()

    fig = px.scatter(freight_pd, x="avg_freight", y="avg_rating", size="num_reviews",
                     hover_name="seller_id", title="Freight vs Average Rating",
                     labels={"avg_freight": "Average Freight", "avg_rating": "Average Rating"})
    fig.update_traces(marker=dict(opacity=0.7))
    st.plotly_chart(fig)

with tab4:
    st.header("📂 Raw Data Snapshots")
    st.write("Preview of `orders_with_reviews`")
    st.dataframe(orders_with_reviews.limit(10).toPandas())
