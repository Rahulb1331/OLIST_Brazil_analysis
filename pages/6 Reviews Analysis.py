import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

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

# Compute helpful metrics
orders_with_reviews["delivery_days"] = (
    pd.to_datetime(orders_with_reviews["order_delivered_customer_date"]) -
    pd.to_datetime(orders_with_reviews["order_purchase_timestamp"])
).dt.days
orders_with_reviews = orders_with_reviews[orders_with_reviews["delivery_days"] < 100]  # Outlier filter
orders_with_reviews["low_score"] = orders_with_reviews["review_score"] < 3

# Tabs for exploration
tab1, tab2, tab3, tab4 = st.tabs([
    "⭐ Top & Worst Rated",
    "⏱️ Delivery vs Review",
    "🚚 Freight vs Review",
    "📊 Raw Data Snapshots"
])

with tab1:
    st.header("📦 Product & Seller Ratings")

    rating_stats = orders_with_reviews.groupby("seller_id").agg(
        num_reviews=("review_score", "count"),
        avg_rating=("review_score", "mean"),
        rating_std=("review_score", "std"),
        pct_low_scores=("low_score", "mean")
    ).reset_index()
    rating_stats["pct_low_scores"] *= 100

    top_sellers = rating_stats.sort_values(by=["avg_rating", "num_reviews"], ascending=[False, False]).head(10)
    worst_sellers = rating_stats[rating_stats["num_reviews"] >= 10].sort_values("avg_rating").head(10)

    prod_stats = orders_with_reviews.groupby("product_category").agg(
        num_reviews=("review_score", "count"),
        avg_rating=("review_score", "mean"),
        rating_std=("review_score", "std"),
        pct_low_scores=("low_score", "mean")
    ).reset_index()
    prod_stats["pct_low_scores"] *= 100

    top_products = prod_stats.sort_values(by=["avg_rating", "num_reviews"], ascending=[False, False]).head(10)
    worst_products = prod_stats[prod_stats["num_reviews"] >= 10].sort_values("avg_rating").head(10)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Top 10 Products by Rating")
        st.plotly_chart(px.bar(top_products, x="product_category", y="avg_rating", error_y="rating_std", title="Top Products"))

    with col2:
        st.subheader("Worst 10 Products by Rating")
        st.plotly_chart(px.bar(worst_products, x="product_category", y="avg_rating", error_y="rating_std", title="Worst Products"))

    col3, col4 = st.columns(2)
    with col3:
        st.subheader("Top 10 Sellers by Rating")
        st.plotly_chart(px.bar(top_sellers, x="seller_id", y="avg_rating", error_y="rating_std", title="Top Sellers"))

    with col4:
        st.subheader("Worst 10 Sellers by Rating")
        st.plotly_chart(px.bar(worst_sellers, x="seller_id", y="avg_rating", error_y="rating_std", title="Worst Sellers"))

    st.caption("Note: Error bars represent rating standard deviation. % of low scores available in detail view.")

    with st.expander("🔍 View Detailed Seller & Product Rating Stats"):
        st.subheader("Detailed Seller Stats")
        st.dataframe(rating_stats.sort_values("avg_rating", ascending=False).reset_index(drop=True))

        st.subheader("Detailed Product Stats")
        st.dataframe(prod_stats.sort_values("avg_rating", ascending=False).reset_index(drop=True))

    
    if st.checkbox("Show Insights for Ratings Tab"):
        st.info("""
        🔍 **Analysis Explanation**:
        - This tab identifies the top and worst performing products and sellers based on customer review scores.
        - Grouping and aggregations calculate average ratings, standard deviations, and the percentage of low review scores.
        - Sellers/products with more than 10 reviews are emphasized to reduce statistical bias from low-sample anomalies.

        💡 **Key Insights**:
        - Top performers have both high average scores and low variation in ratings.
        - A high percentage of low reviews can flag customer dissatisfaction even if the average rating appears reasonable.
        - Standard deviation reveals consistency—products/sellers with wild swings in reviews may lack reliability.

        ✅ **Recommendations**:
        - Prioritize high-rated and consistent performers for promotions or visibility.
        - Investigate sellers/products with high standard deviation or low-score percentage for quality control.
        - Consider minimum review thresholds before making business decisions based on rating data.
        """)


with tab2:
    st.header("⏳ Delivery Time Impact on Reviews")

    delivery_analysis = orders_with_reviews.groupby("review_score").agg(
        avg_delivery_days=("delivery_days", "mean"),
        num_orders=("order_id", "count")
    ).reset_index()

    fig = px.bar(delivery_analysis, x="review_score", y="avg_delivery_days",
                 text="avg_delivery_days", title="Delivery Days vs Review Score",
                 labels={"avg_delivery_days": "Avg Delivery Days", "review_score": "Review Score"})
    fig.update_traces(marker_color="tomato", texttemplate='%{text:.1f}')
    st.plotly_chart(fig)

    st.subheader("Worst Sellers with Longest Delivery")
    seller_delay = orders_with_reviews.groupby("seller_id").agg(
        num_orders=("order_id", "count"),
        avg_rating=("review_score", "mean"),
        avg_delivery_days=("delivery_days", "mean")
    ).reset_index()

    seller_delay = seller_delay[(seller_delay["num_orders"] >= 10) & (seller_delay["avg_rating"] <= 2)]
    seller_delay = seller_delay.sort_values("avg_delivery_days", ascending=False).head(10)
    st.dataframe(seller_delay)

    st.subheader("Late vs On-Time Deliveries Impact")
    orders_with_reviews["delivery_estimate"] = pd.to_datetime(orders_with_reviews["order_estimated_delivery_date"])
    orders_with_reviews["actual_delivery"] = pd.to_datetime(orders_with_reviews["order_delivered_customer_date"])
    orders_with_reviews["delayed"] = orders_with_reviews["actual_delivery"] > orders_with_reviews["delivery_estimate"]

    delay_impact = orders_with_reviews.groupby("delayed").agg(
        avg_rating=("review_score", "mean"),
        count=("review_score", "count")
    ).reset_index()
    delay_impact["delayed"] = delay_impact["delayed"].map({True: "Late", False: "On-Time"})

    st.plotly_chart(px.bar(delay_impact, x="delayed", y="avg_rating", title="Avg Review by Delivery Timeliness"))

    if st.checkbox("Show Insights for Delivery Tab"):
        st.info("""
        🔍 **Analysis Explanation**:
        - This tab explores how delivery duration and punctuality affect customer review scores.
        - It compares average delivery days by review score and examines review impact from late vs on-time deliveries.
        - Sellers with many low-rated, slow deliveries are highlighted for deeper evaluation.

        💡 **Key Insights**:
        - Longer delivery times are associated with lower review scores, even when the product is satisfactory.
        - On-time deliveries yield higher ratings; lateness consistently reduces satisfaction.
        - Some sellers suffer reputational damage due to systemic delivery delays.

        ✅ **Recommendations**:
        - Improve logistics to reduce actual delivery time and minimize delays.
        - Communicate realistic delivery estimates to better align with customer expectations.
        - Focus on sellers with long delivery delays and poor ratings for corrective action.
        """)

with tab3:
    st.header("🚚 Freight Charges vs Review Score")

    freight_reviews = orders_with_reviews.groupby(["seller_id", "product_category"]).agg(
        num_reviews=("order_id", "count"),
        avg_rating=("review_score", "mean"),
        avg_freight=("freight_value", "mean")
    ).reset_index()

    freight_reviews = freight_reviews[freight_reviews["num_reviews"] > 10]

    fig = px.scatter(freight_reviews, x="avg_freight", y="avg_rating", size="num_reviews",
                     hover_data=["seller_id", "product_category"],
                     title="Freight vs Rating by Seller and Product Category",
                     labels={"avg_freight": "Average Freight", "avg_rating": "Average Rating"})
    fig.update_traces(marker=dict(opacity=0.6))
    st.plotly_chart(fig)

    if st.checkbox("Show Insights for Freight Tab"):
        st.info("""
        🔍 **Analysis Explanation**:
        - This tab investigates the relationship between freight costs and review scores.
        - It aggregates ratings and average freight values at the seller and product category level.
        - A bubble chart highlights potential overcharges or acceptable premium delivery segments.

        💡 **Key Insights**:
        - Freight cost alone does not guarantee lower ratings; value perception matters.
        - Certain categories with high freight (e.g., heavy or premium items) still achieve strong reviews.
        - Sellers with high freight and low ratings may be overcharging or underdelivering.

        ✅ **Recommendations**:
        - Audit categories and sellers with disproportionately high freight and low ratings.
        - Consider subsidizing freight for budget-conscious categories to boost satisfaction.
        - Promote sellers who maintain good ratings despite premium freight—indicating added value or service.
        """)


with tab4:
    st.header("📂 Raw Data Snapshots")
    st.write("Preview of `orders_with_reviews`")
    st.dataframe(orders_with_reviews.head(10))
