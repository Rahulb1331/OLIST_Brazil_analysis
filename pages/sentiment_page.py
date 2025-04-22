import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# Setup Streamlit
st.set_page_config(page_title="Review Sentiment Analysis", layout="wide")
st.title("📊 Review Sentiment Dashboard")
@st.cache_data
def load_data():
    from analysis.Preprocessing import full_orders, order_reviews
    return full_orders, order_reviews
full_orders, order_reviews = load_data()

# Join data
orders_with_reviews = pd.merge(full_orders, order_reviews, on="order_id", how="inner")
orders_with_reviews = orders_with_reviews[orders_with_reviews["seller_id"].notna()]

# Tabs for exploration
tab1, tab2, tab3, tab4 = st.tabs([
    "⭐ Top & Worst Rated",
    "⏱️ Delivery vs Review",
    "🚚 Freight vs Review",
    "📊 Raw Data Snapshots"
])

with tab1:
    st.header("📦 Product & Seller Ratings")

    top_sellers = orders_with_reviews.groupby("seller_id").agg(
        num_reviews=pd.NamedAgg(column="review_score", aggfunc="count"),
        avg_rating=pd.NamedAgg(column="review_score", aggfunc="mean")
    ).sort_values(by=["avg_rating", "num_reviews"], ascending=[False, False]).head(10)

    top_products = orders_with_reviews.groupby("product_category").agg(
        num_reviews=pd.NamedAgg(column="review_score", aggfunc="count"),
        avg_rating=pd.NamedAgg(column="review_score", aggfunc="mean")
    ).sort_values(by=["avg_rating", "num_reviews"], ascending=[False, False]).head(10)

    worst_sellers = orders_with_reviews.groupby("seller_id").agg(
        num_reviews=pd.NamedAgg(column="review_score", aggfunc="count"),
        avg_rating=pd.NamedAgg(column="review_score", aggfunc="mean")
    )
    worst_sellers = worst_sellers[worst_sellers["num_reviews"] >= 10]
    worst_sellers = worst_sellers.sort_values("avg_rating").head(10)

    worst_products = orders_with_reviews.groupby("product_category").agg(
        num_reviews=pd.NamedAgg(column="review_score", aggfunc="count"),
        avg_rating=pd.NamedAgg(column="review_score", aggfunc="mean")
    )
    worst_products = worst_products[worst_products["num_reviews"] >= 10]
    worst_products = worst_products.sort_values("avg_rating").head(10)

    # Display plots
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Top 10 Products by Rating")
        st.plotly_chart(px.bar(top_products.reset_index(), x="product_category", y="avg_rating", title="Top Products"))

    with col2:
        st.subheader("Worst 10 Products by Rating")
        st.plotly_chart(px.bar(worst_products.reset_index(), x="product_category", y="avg_rating", title="Worst Products"))

    col3, col4 = st.columns(2)
    with col3:
        st.subheader("Top 10 Sellers by Rating")
        st.plotly_chart(px.bar(top_sellers.reset_index(), x="seller_id", y="avg_rating", title="Top Sellers"))

    with col4:
        st.subheader("Worst 10 Sellers by Rating")
        st.plotly_chart(px.bar(worst_sellers.reset_index(), x="seller_id", y="avg_rating", title="Worst Sellers"))

with tab2:
    st.header("⏳ Delivery Time Impact on Reviews")

    orders_with_reviews["delivery_days"] = (pd.to_datetime(orders_with_reviews["order_delivered_customer_date"]) - 
                                              pd.to_datetime(orders_with_reviews["order_purchase_timestamp"]))
    orders_with_reviews["delivery_days"] = orders_with_reviews["delivery_days"].dt.days
    
    delivery_analysis = orders_with_reviews.groupby("review_score").agg(
        avg_delivery_days=pd.NamedAgg(column="delivery_days", aggfunc="mean"),
        num_orders=pd.NamedAgg(column="order_id", aggfunc="count")
    ).reset_index()

    fig = px.bar(delivery_analysis, x="review_score", y="avg_delivery_days",
                 text="avg_delivery_days", title="Delivery Days vs Review Score",
                 labels={"avg_delivery_days": "Avg Delivery Days", "review_score": "Review Score"})
    fig.update_traces(marker_color="tomato", texttemplate='%{text:.1f}')
    st.plotly_chart(fig)

    # Delay-based Worst Sellers
    st.subheader("Worst Sellers with Longest Delivery")
    seller_delay = orders_with_reviews.groupby("seller_id").agg(
        num_orders=pd.NamedAgg(column="order_id", aggfunc="count"),
        avg_rating=pd.NamedAgg(column="review_score", aggfunc="mean"),
        avg_delivery_days=pd.NamedAgg(column="delivery_days", aggfunc="mean")
    ).reset_index()

    seller_delay = seller_delay[(seller_delay["num_orders"] >= 10) & (seller_delay["avg_rating"] <= 2)]
    seller_delay = seller_delay.sort_values("avg_delivery_days", ascending=False).head(10)

    st.dataframe(seller_delay)
    
with tab3:
    st.header("🚚 Freight Charges vs Review Score")

    freight_reviews = orders_with_reviews.groupby("seller_id").agg(
        num_reviews=pd.NamedAgg(column="order_id", aggfunc="count"),
        avg_rating=pd.NamedAgg(column="review_score", aggfunc="mean"),
        avg_freight=pd.NamedAgg(column="freight_value", aggfunc="mean")
    ).reset_index()
    freight_reviews = freight_reviews[freight_reviews["num_reviews"] > 10]

    fig = px.scatter(freight_reviews, x="avg_freight", y="avg_rating", size="num_reviews",
                     hover_name="seller_id", title="Freight vs Average Rating",
                     labels={"avg_freight": "Average Freight", "avg_rating": "Average Rating"})
    fig.update_traces(marker=dict(opacity=0.7))
    st.plotly_chart(fig)

with tab4:
    st.header("📂 Raw Data Snapshots")
    st.write("Preview of `orders_with_reviews`")
    st.dataframe(orders_with_reviews.head(10))
