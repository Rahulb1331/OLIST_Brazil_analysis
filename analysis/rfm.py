# analysis/rfm.py

from analysis.Preprocessing import full_orders
from datetime import timedelta
from pyspark.sql.functions import col, max as spark_max, count, sum as spark_sum, datediff, to_date, lit, udf, when
from pyspark.sql.types import IntegerType
from pyspark.sql.functions import concat_ws
from pyspark.sql import functions as F
from pyspark.sql.window import Window
import plotly.express as px



def run_rfm_analysis(order_customer_df):
    # Convert timestamp to date
    order_customer_df = order_customer_df.withColumn("order_purchase_date", to_date("order_purchase_timestamp"))

    # Reference date
    max_date = order_customer_df.agg(spark_max("order_purchase_date")).collect()[0][0]
    reference_date = max_date + timedelta(days=1)

    # RFM Calculation
    rfm_df = order_customer_df.groupBy("customer_unique_id").agg(
        datediff(lit(reference_date), spark_max("order_purchase_date")).alias("Recency"),
        count("order_id").alias("Frequency"),
        spark_sum("payment_value").alias("Monetary")
    )

    # Quantile scoring
    quantiles = rfm_df.approxQuantile(["Recency", "Frequency", "Monetary"], [0.25, 0.5, 0.75], 0.01)
    r_q, f_q, m_q = quantiles

    def r_score(r): return 4 if r <= r_q[0] else 3 if r <= r_q[1] else 2 if r <= r_q[2] else 1

    def fm_score(x, q):
        if x is None:
            return 1
        return 1 if x <= q[0] else 2 if x <= q[1] else 3 if x <= q[2] else 4

    r_score_udf = udf(r_score, IntegerType())
    f_score_udf = udf(lambda x: fm_score(x, f_q), IntegerType())
    m_score_udf = udf(lambda x: fm_score(x, m_q), IntegerType())

    rfm_scored = rfm_df.withColumn("R", r_score_udf("Recency")) \
        .withColumn("F", f_score_udf("Frequency")) \
        .withColumn("M", m_score_udf("Monetary")) \
        .withColumn("RFM_Score", concat_ws("", col("R"), col("F"), col("M")))

    return rfm_scored


#Preparing the dataset on which we will run the rfm analysis

rfm_df = run_rfm_analysis(full_orders)

#segmenting the customers into high value, medium value and low value based on their rfm score
rfm_df = rfm_df.withColumn(
    "CustomerGroup",
    when(col("RFM_Score") >= 444, "High-value")
    .when((col("RFM_Score") >= 222) & (col("RFM_Score") < 444), "Medium-value")
    .otherwise("Low-value")
)

rfm_df.select("customer_unique_id", "R", "F", "M", "RFM_Score", "CustomerGroup").show(10, truncate=False)


#Summary statistics per segment
rfm_summary = rfm_df.groupBy("CustomerGroup").agg(
    F.count("*").alias("CustomerCount"),
    F.round(F.avg("Recency"), 2).alias("AvgRecency"),
    F.round(F.avg("Frequency"), 2).alias("AvgFrequency"),
    F.round(F.avg("Monetary"), 2).alias("AvgMonetary")
)
rfm_summary.show()

# Countplot for the different customer segments

# Distribution of customer segments
rfm_pandas = rfm_df.toPandas()

fig = px.bar(
    rfm_pandas,
    x="CustomerGroup",
    title="Customer Segments Distribution",
    labels={"CustomerGroup": "Customer Group", "count": "Count"},
    color="CustomerGroup",
    color_discrete_sequence=px.colors.qualitative.Set2
)
fig.show()


# Advanced segmentation to show more granularity
rfm_df = rfm_df.withColumn("R_Quartile", F.ntile(4).over(Window.orderBy("Recency")))
rfm_df = rfm_df.withColumn("F_Quartile", F.ntile(4).over(Window.orderBy(F.desc("Frequency"))))
rfm_df = rfm_df.withColumn("M_Quartile", F.ntile(4).over(Window.orderBy(F.desc("Monetary"))))

#Behavioral tagging
rfm_df = rfm_df.withColumn(
    "BehaviorSegment",
    when((col("R") == 4) & (col("F") == 4) & (col("M") == 4), "Champions")
    .when((col("R") >= 3) & (col("F") >= 3), "Loyal Customers")
    .when((col("R") == 4), "Recent Customers")
    .when((col("F") == 4), "Frequent Buyers")
    .when((col("M") == 4), "Big Spenders")
    .otherwise("Others")
)
rfm_df.groupBy("BehaviorSegment").count().show()


#Heatmap of R, F, M scores
# Convert Spark to Pandas
rfm_pd = rfm_df.select("R", "F", "M").toPandas()

# Create pivot table: counts of customers in each R-F-M score combo
heatmap_data = rfm_pd.groupby(["R", "F", "M"]).size().reset_index(name='count')

# Pivot for heatmap ( R vs F)
rf_heatmap = heatmap_data.pivot_table(index="R", columns="F", values="count", fill_value=0)

# Get axis labels correctly
x_labels = rf_heatmap.columns.tolist()        # Frequency scores (columns)
y_labels = rf_heatmap.index.tolist()          # Recency scores (index/rows)

# Confirm dimensions match
print("x:", len(x_labels), "columns in heatmap")
print(x_labels)
print("y:", len(y_labels), "rows in heatmap")
print(y_labels)
print("heatmap values shape:", rf_heatmap.values.shape)

# Create heatmap using Plotly
fig = px.imshow(
    rf_heatmap.values,
    labels=dict(x="Frequency Score", y="Recency Score", color="Customer Count"),
    x=x_labels,
    y=y_labels,
    color_continuous_scale="YlGnBu",
    text_auto=True
)

fig.update_layout(
    title="Customer Distribution by Recency and Frequency Scores",
    xaxis_title="Frequency Score",
    yaxis_title="Recency Score",
    template="plotly_white",
)

# Show the interactive heatmap
fig.show()

# R vs M
# Create pivot table: counts of customers in each R-M score combo
heatmap_data = rfm_pd.groupby(["R", "M"]).size().reset_index(name='count')

# Pivot for heatmap (R vs M)
rm_heatmap = heatmap_data.pivot(index="R", columns="M", values="count").fillna(0)

# Get axis labels correctly
x_labels = rm_heatmap.columns.tolist()        # Frequency scores (columns)
y_labels = rm_heatmap.index.tolist()          # Recency scores (index/rows)

# Confirm dimensions match
print("x:", len(x_labels), "columns in heatmap")
print(x_labels)
print("y:", len(y_labels), "rows in heatmap")
print(y_labels)
print("heatmap values shape:", rf_heatmap.values.shape)

# Create heatmap using Plotly
fig = px.imshow(
    rm_heatmap.values,
    labels=dict(x="Monetary Score", y="Recency Score", color="Count"),
    x=x_labels,  # Monetary scores
    y=y_labels,  # Recency scores
    color_continuous_scale="YlGnBu",
    text_auto=True
)

# Customize layout
fig.update_layout(
    title="Customer Distribution by Recency and Monetary Scores",
    xaxis_title="Monetary Score",
    yaxis_title="Recency Score",
    template="plotly_white",
)

# Show the interactive heatmap
fig.show()

# M vs F
# Create pivot table: counts of customers in each M-F score combo
heatmap_data = rfm_pd.groupby(["M", "F"]).size().reset_index(name='count')

# Pivot for heatmap (M vs F)
mf_heatmap = heatmap_data.pivot(index="M", columns="F", values="count").fillna(0)

# Get axis labels correctly
x_labels = mf_heatmap.columns.tolist()        # Frequency scores (columns)
y_labels = mf_heatmap.index.tolist()          # Recency scores (index/rows)

# Confirm dimensions match
print("x:", len(x_labels), "columns in heatmap")
print(x_labels)
print("y:", len(y_labels), "rows in heatmap")
print(y_labels)
print("heatmap values shape:", rf_heatmap.values.shape)

# Create heatmap using Plotly
fig = px.imshow(
    mf_heatmap.values,
    labels=dict(x="Frequency Score", y="Monetary Score", color="Count"),
    x=x_labels,  # Frequency scores
    y=y_labels,  # Monetary scores
    color_continuous_scale="YlGnBu",
    text_auto=True
)

# Customize layout
fig.update_layout(
    title="Customer Distribution by Monetary and Frequency Scores",
    xaxis_title="Frequency Score",
    yaxis_title="Monetary Score",
    template="plotly_white",
)

# Show the interactive heatmap
fig.show()


# Link segments to products
# Join RFM segments back to full_orders
orders_with_rfm = full_orders.join(rfm_df.select("customer_unique_id", "CustomerGroup"), on="customer_unique_id", how="left")

# Analyzing top products per segments

top_products_by_segment = orders_with_rfm.groupBy("CustomerGroup", "product_category").agg(
    count("*").alias("purchase_count")
).orderBy("CustomerGroup", "purchase_count", ascending=False)

top_products_by_segment.show(10)

# Join RFM segments to full orders to explore their product preferences
rfm_orders = full_orders.join(rfm_df.select("customer_unique_id", "CustomerGroup"), on="customer_unique_id", how="inner")

# Now group to see top product categories per customer group
product_pref = (
    rfm_orders.groupBy("CustomerGroup", "product_category")
    .count()
    .orderBy("count", ascending=False)
    .toPandas()
)

fig_products = px.bar(
    product_pref,
    x="product_category",
    y="count",
    color="CustomerGroup",
    barmode="group",
    title="Top Product Categories by Customer Group",
    template="plotly_white"
)
fig_products.update_layout(xaxis_tickangle=-45)
fig_products.show()

