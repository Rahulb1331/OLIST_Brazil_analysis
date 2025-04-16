# Assuming full_orders and order_reviews are both Spark DataFrames
from pyspark.sql.functions import col, count, avg
from Preprocessing import full_orders, order_reviews, order_items

# Join
orders_with_reviews = full_orders.join(order_reviews, on="order_id", how="inner")
orders_with_reviews = orders_with_reviews.filter(col("seller_id").isNotNull())

top_sellers = (
    orders_with_reviews
    .groupBy("seller_id")
    .agg(
        count("review_score").alias("num_reviews"),
        avg("review_score").alias("avg_rating")
    )
    .orderBy(col("avg_rating").desc(), col("num_reviews").desc())
)

top_sellers.show(10)

top_products = (
    orders_with_reviews
    .groupBy("product_category")
    .agg(
        count("review_score").alias("num_reviews"),
        avg("review_score").alias("avg_rating")
    )
    .orderBy(col("avg_rating").desc(), col("num_reviews").desc())
)

top_products.show(10)

# Worst Reviewed ones
worst_sellers = (
    orders_with_reviews
    .groupBy("seller_id")
    .agg(
        count("review_score").alias("num_reviews"),
        avg("review_score").alias("avg_rating")
    )
    .filter(col("num_reviews") >= 10)  # optional: filter for reliability
    .orderBy(col("avg_rating").asc(), col("num_reviews").desc())
)

worst_sellers.show(10)

worst_products = (
    orders_with_reviews
    .groupBy("product_category")
    .agg(
        count("review_score").alias("num_reviews"),
        avg("review_score").alias("avg_rating")
    )
    .filter(col("num_reviews") >= 10)
    .orderBy(col("avg_rating").asc(), col("num_reviews").desc())
)

worst_products.show(10)

import plotly.express as px

# Convert to pandas
top_products_pd = top_products.orderBy("avg_rating", ascending=False).limit(10).toPandas()

fig = px.bar(top_products_pd,
             x="product_category",
             y="avg_rating",
             title="Top 10 Reviewed Products",
             labels={"product_category_name": "Product Category", "avg_rating": "Average Rating"})

fig.show()

# Trying to find the reason for low review scores, looking whether it is due to the delayed delivery
from pyspark.sql.functions import datediff

orders_with_reviews = orders_with_reviews.withColumn(
    "delivery_days", datediff("order_delivered_customer_date", "order_purchase_timestamp")
)

# Correlating with the review score:
delivery_analysis = orders_with_reviews.groupBy("review_score").agg(
    avg("delivery_days").alias("avg_delivery_days"),
    count("*").alias("num_orders")
).orderBy("review_score")

delivery_analysis.show()

#Worst reviewed sellers with long deliveries
from pyspark.sql.functions import round

worst_seller_delays = orders_with_reviews.groupBy("seller_id") \
    .agg(
        count("*").alias("num_orders"),
        avg("review_score").alias("avg_rating"),
        avg("delivery_days").alias("avg_delivery_days")
    ) \
    .filter((col("num_orders") >= 10) & (col("avg_rating") <= 2)) \
    .orderBy(col("avg_delivery_days").desc())

worst_seller_delays.show()

#Same for product categories:
worst_product_delays = orders_with_reviews.groupBy("product_category") \
    .agg(
        count("*").alias("num_orders"),
        avg("review_score").alias("avg_rating"),
        avg("delivery_days").alias("avg_delivery_days")
    ) \
    .filter((col("num_orders") >= 10) & (col("avg_rating") <= 2)) \
    .orderBy(col("avg_delivery_days").desc())

worst_product_delays.show()

import plotly.express as px

# Convert to pandas for plotting
delivery_analysis_pd = delivery_analysis.toPandas()

fig = px.bar(
    delivery_analysis_pd,
    x="review_score",
    y="avg_delivery_days",
    title="Average Delivery Days vs Review Score",
    labels={
        "review_score": "Review Score",
        "avg_delivery_days": "Average Delivery Time (days)"
    },
    text="avg_delivery_days"
)

fig.update_traces(marker_color='tomato', texttemplate='%{text:.2f}', textposition='outside')
fig.update_layout(template="plotly_white", height=500)
fig.show()


# Adding the freight column as well to see if the freight value (the delivery charges) are a reason for the negative reviews
orders_with_reviews.printSchema()

# Freight vs Review Score

freight_vs_reviews = orders_with_reviews.groupBy("seller_id", "product_id").agg(
    avg("review_score").alias("avg_review"),
    avg("freight_value").alias("avg_freight")
).orderBy("avg_review")

worst_sellers = orders_with_reviews.groupBy("seller_id").agg(
    count("*").alias("num_reviews"),
    avg("review_score").alias("avg_rating"),
    avg("freight_value").alias("avg_freight")
).filter(col("num_reviews") > 10).orderBy("avg_rating")

import plotly.express as px

# Convert Spark DataFrame to Pandas
worst_sellers_pd = worst_sellers.toPandas()

fig = px.scatter(
    worst_sellers_pd,
    x="avg_freight",
    y="avg_rating",
    size="num_reviews",
    hover_name="seller_id",
    title="Freight Value vs Average Review Rating (Sellers)",
    labels={
        "avg_freight": "Average Freight Value",
        "avg_rating": "Average Review Rating"
    }
)

fig.update_traces(marker=dict(opacity=0.7))
fig.update_layout(template="plotly_white", height=600)
fig.show()
