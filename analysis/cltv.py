from pyspark.sql.functions import col, countDistinct, count, sum, avg, when, lit
from pyspark.sql.window import Window
from pyspark.sql.functions import min as spark_min, max as spark_max
from pyspark.sql import functions as F
from analysis.Preprocessing import full_orders
from analysis.rfm import rfm_df
import plotly.express as px
import pandas as pd

def run_cltv_analysis(full_orders_df):
    # Step 1: Aggregate order stats by customer
    customer_metrics = full_orders_df.groupBy("customer_unique_id").agg(
        countDistinct("order_id").alias("total_orders"),
        sum("payment_value").alias("total_payment"),
        avg("payment_value").alias("avg_order_value")
    )

    # Step 2: Calculate global purchase frequency
    total_orders = full_orders_df.select("order_id").distinct().count()
    total_customers = full_orders_df.select("customer_unique_id").distinct().count()
    purchase_frequency = total_orders / total_customers

    # Step 3: Calculate CLTV
    cltv_df = customer_metrics.withColumn(
        "purchase_frequency", col("total_orders") / total_customers
    ).withColumn(
        "cltv", col("avg_order_value") * col("purchase_frequency")
    )

    return cltv_df.select("customer_unique_id", "total_orders", "avg_order_value", "purchase_frequency", "cltv")

cltv_df = run_cltv_analysis(full_orders)
cltv_df.show(10, truncate=False)

# Normalizing the CLTV
# Calculate min and max CLTV
min_cltv = cltv_df.agg(spark_min("cltv")).first()[0]
max_cltv = cltv_df.agg(spark_max("cltv")).first()[0]

# Avoid division by zero
range_cltv = max_cltv - min_cltv if max_cltv != min_cltv else 1.0

# Add normalized CLTV column
cltv_df = cltv_df.withColumn(
    "normalized_cltv",
    (col("cltv") - lit(min_cltv)) / lit(range_cltv)
)

# Segmenting the customers based on the normalized cltv
cltv_df = cltv_df.withColumn(
    "CLTV_Segment",
    when(col("normalized_cltv") >= 0.66, "High CLTV")
    .when(col("normalized_cltv") >= 0.33, "Medium CLTV")
    .otherwise("Low CLTV")
)

cltv_df.select(
    "customer_unique_id",
    "cltv",
    "normalized_cltv",
    "CLTV_Segment"
).show(10, truncate=False)


#Join CLTV Data with RFM
# Join on customer ID
rfm_cltv_df = rfm_df.join(cltv_df.select("customer_unique_id", "cltv", "normalized_cltv", "CLTV_Segment"), on="customer_unique_id", how="inner")


# Using the new formula
# Average order value already computed per customer
# Purchase frequency = total orders / total unique customers (global freq)
# Assume lifespan in months (tune this!)

total_customers = cltv_df.select("customer_unique_id").distinct().count()
global_purchase_frequency = cltv_df.agg(F.sum("total_orders")).first()[0] / total_customers
lifespan_months = 12

cltv_df = cltv_df.withColumn(
    "better_cltv",
    col("avg_order_value") * F.lit(global_purchase_frequency) * F.lit(lifespan_months)
)

windowSpec = Window.orderBy(F.lit(1))  # dummy window for global agg
min_val = cltv_df.agg(F.min("better_cltv")).first()[0]
max_val = cltv_df.agg(F.max("better_cltv")).first()[0]
range_val = max_val - min_val

cltv_df = cltv_df.withColumn(
    "cltv_normalized",
    ((col("better_cltv") - F.lit(min_val)) / F.lit(range_val))
)

# Get quantile breakpoints
quantiles = cltv_df.approxQuantile("better_cltv", [0.33, 0.66], 0.01)
q1, q2 = quantiles

# Segment
cltv_df = cltv_df.withColumn(
    "CLTV_new_Segment",
    when(col("better_cltv") >= q2, "High CLTV")
    .when(col("better_cltv") >= q1, "Medium CLTV")
    .otherwise("Low CLTV")
)


#Join CLTV Data with RFM
# Join on customer ID
rfm_cltv_df = rfm_df.join(cltv_df.select("customer_unique_id", "better_cltv", "cltv_normalized", "CLTV_new_Segment"), on="customer_unique_id", how="inner")

# Count Customers in Each CLTV Segment
segment_counts = rfm_cltv_df.groupBy("CLTV_new_Segment").agg(F.count("*").alias("CustomerCount"))
segment_counts.show()

#Cross Tab CLTV vs RFM Segments
#This gives a matrix-style overview of how customer segments overlap:
rfm_cltv_df.crosstab("CLTV_new_Segment", "CustomerGroup").show()

#Plotly: Visualize CLTV Distribution
cltv_pd = rfm_cltv_df.select("cltv_normalized", "CLTV_new_Segment").toPandas()

#Histogram by segment

fig = px.histogram(
    cltv_pd,
    x="cltv_normalized",
    color="CLTV_new_Segment",
    nbins=30,
    title="CLTV Distribution by Segment",
    labels={"cltv_normalized": "Normalized CLTV"},
    barmode="overlay",
    opacity=0.7
)

fig.update_layout(template="plotly_white")
fig.show()


# Preparing the dataset for the modeling
# Select necessary columns
orders_filtered = full_orders.select(
    "customer_unique_id",
    "order_id",
    "order_purchase_timestamp",
    "payment_value"
)

# Convert to Pandas
orders_pd = orders_filtered.toPandas()


# Reference date (end of observation period)
max_date = orders_pd["order_purchase_timestamp"].max()

# Create summary DataFrame
summary = orders_pd.groupby("customer_unique_id").agg(
    frequency=("order_id", lambda x: x.nunique() - 1),
    recency=("order_purchase_timestamp", lambda x: (x.max() - x.min()).days),
    T=("order_purchase_timestamp", lambda x: (max_date - x.min()).days),
    monetary_value=("payment_value", "mean")
).reset_index()

# Filter: Keep only repeat customers
summary = summary[summary["frequency"] > 0]
summary = summary.dropna()

import matplotlib.pyplot as plt
import seaborn as sns

# Visualize distribution
sns.histplot(summary["recency"], bins=50, kde=True)
plt.title("Recency Distribution")
plt.show()

sns.histplot(summary["T"], bins=50, kde=True)
plt.title("T Distribution")
plt.show()

# Keep only customers with frequency ≥ 2
summary = summary[summary["frequency"] >= 2]

# Cap monetary_value at 99th percentile to remove extreme outliers
upper_cap = summary["monetary_value"].quantile(0.99)
summary = summary[summary["monetary_value"] <= upper_cap]

# Final sanity check: remove any remaining invalid rows
summary = summary.dropna()
summary = summary[summary["monetary_value"] > 0]

# Clipping the upper value of recency and T to 365 (1 year)
summary["recency"] = summary["recency"].clip(upper=365)
summary["T"] = summary["T"].clip(upper=365)

print("Min values:\n", summary[["frequency", "recency", "T", "monetary_value"]].min())
print("Max values:\n", summary[["frequency", "recency", "T", "monetary_value"]].max())
print("Any NaNs?\n", summary.isnull().sum())


# Fitting BG/NBD model (predicts number of transactions)

# Initialize and fit model
from lifetimes import ParetoNBDFitter

pnbd = ParetoNBDFitter(penalizer_coef=1.0)
pnbd.fit(
    frequency=summary["frequency"],
    recency=summary["recency"],
    T=summary["T"]
)



#Fitting Gamma-Gamma model (predicts monetary value)
from lifetimes import GammaGammaFitter

# Initialize and fit
ggf = GammaGammaFitter(penalizer_coef=0.1) # Higher regularization coefficient earlier 0.01
ggf.fit(
    frequency=summary["frequency"],
    monetary_value=summary["monetary_value"]
)

# Predict Customer Lifetime Value
# Predict expected number of purchases in 12 months (12 * 4 = 48 weeks)
summary["predicted_purchases"] = pnbd.conditional_expected_number_of_purchases_up_to_time(
    48,  # weeks
    summary["frequency"],
    summary["recency"],
    summary["T"]
)

# Predict average monetary value
summary["predicted_avg_value"] = ggf.conditional_expected_average_profit(
    summary["frequency"],
    summary["monetary_value"]
)

# Calculate CLTV
summary["predicted_purchases"] = summary["predicted_purchases"].clip(lower=0)
summary["predicted_cltv"] = summary["predicted_purchases"] * summary["predicted_avg_value"]

print(summary.head())
print(summary[summary["predicted_cltv"] < 0])  # Check for anomalies


# Segmenting the customers by the predicted cltv
summary["cltv_segment"] = pd.qcut(summary["predicted_cltv"], q=4, labels=["Low", "Mid", "High", "Very High"])

# Top Customers
top_customers = summary.sort_values(by="predicted_cltv", ascending=False).head(10)
print(top_customers[["customer_unique_id", "predicted_cltv"]])

print("Total customers in summary: ", summary.count())


#Visualizing the Distribution
fig = px.histogram(
    summary,
    x="predicted_cltv",
    nbins=50,
    title="Distribution of Predicted Customer Lifetime Value",
    labels={"predicted_cltv": "Predicted CLTV"},
    template="plotly_white"
)

fig.update_layout(
    bargap=0.1,
    xaxis_title="Predicted CLTV",
    yaxis_title="Number of Customers"
)

fig.show()

# Segment-wise CLTV Distribution
fig_segment = px.histogram(
    summary,
    x="predicted_cltv",
    color="cltv_segment",
    nbins=50,
    barmode="overlay",
    title="CLTV Distribution by Segment",
    labels={"predicted_cltv": "Predicted CLTV", "cltv_segment": "CLTV Segment"},
    template="plotly_white"
)

fig_segment.update_layout(
    bargap=0.1,
    xaxis_title="Predicted CLTV",
    yaxis_title="Customer Count"
)

fig_segment.show()

# Top N Customers with Tooltips
top_n = summary.nlargest(20, "predicted_cltv").copy()

fig_top = px.scatter(
    top_n,
    x="frequency",
    y="predicted_cltv",
    color="cltv_segment",
    size="predicted_cltv",
    hover_data={
        "customer_unique_id": True,
        "predicted_cltv": True,
        "frequency": True,
        "monetary_value": True,
        "recency": True,
        "T": True
    },
    title="Top 20 Customers by Predicted CLTV",
    labels={"frequency": "Frequency", "predicted_cltv": "Predicted CLTV"},
    template="plotly_white"
)

fig_top.update_traces(marker=dict(opacity=0.7, line=dict(width=1, color='DarkSlateGrey')))
fig_top.show()


# REVENUE ANALYSIS

#Total Forecasted Revenue
total_cltv = summary["predicted_cltv"].sum()
print(f"💰 Total Forecasted Revenue (12 months): ${total_cltv:,.2f}")

# Revenue by CLTV Segment
revenue_by_segment = summary.groupby("cltv_segment")["predicted_cltv"].sum().sort_values()
print(revenue_by_segment)

# Revenue by Segment

fig = px.bar(
    revenue_by_segment.reset_index(),
    x="cltv_segment",
    y="predicted_cltv",
    title="Revenue Forecast by CLTV Segment (12 months)",
    text_auto=".2s",
    color="cltv_segment"
)
fig.show()

# Revenue Distribution Histogram

fig = px.histogram(
    summary,
    x="predicted_cltv",
    nbins=30,
    title="Distribution of Predicted CLTV (12 months)",
    color="cltv_segment",
    marginal="box"
)
fig.show()


def enrich_cltv_with_segments(cltv_df):
    min_cltv = cltv_df.agg(spark_min("cltv")).first()[0]
    max_cltv = cltv_df.agg(spark_max("cltv")).first()[0]
    range_cltv = max_cltv - min_cltv if max_cltv != min_cltv else 1.0

    cltv_df = cltv_df.withColumn(
        "normalized_cltv",
        (col("cltv") - lit(min_cltv)) / lit(range_cltv)
    )

    cltv_df = cltv_df.withColumn(
        "CLTV_Segment",
        when(col("normalized_cltv") >= 0.66, "High CLTV")
        .when(col("normalized_cltv") >= 0.33, "Medium CLTV")
        .otherwise("Low CLTV")
    )

    # Advanced CLTV
    global_purchase_frequency = cltv_df.agg(F.sum("total_orders")).first()[0] / cltv_df.count()
    lifespan_months = 12

    cltv_df = cltv_df.withColumn(
        "better_cltv",
        col("avg_order_value") * F.lit(global_purchase_frequency) * F.lit(lifespan_months)
    )

    q1, q2 = cltv_df.approxQuantile("better_cltv", [0.33, 0.66], 0.01)
    cltv_df = cltv_df.withColumn(
        "cltv_normalized",
        (col("better_cltv") - spark_min("better_cltv").over(Window.orderBy())).cast("double")
    ).withColumn(
        "CLTV_new_Segment",
        when(col("better_cltv") >= q2, "High CLTV")
        .when(col("better_cltv") >= q1, "Medium CLTV")
        .otherwise("Low CLTV")
    )

    return cltv_df


def model_cltv_lifetimes(df):
    import pandas as pd
    from lifetimes import ParetoNBDFitter, GammaGammaFitter

    orders_pd = df.select(
        "customer_unique_id", "order_id", "order_purchase_timestamp", "payment_value"
    ).toPandas()

    orders_pd["order_purchase_timestamp"] = pd.to_datetime(orders_pd["order_purchase_timestamp"])
    max_date = orders_pd["order_purchase_timestamp"].max()

    summary = orders_pd.groupby("customer_unique_id").agg(
        frequency=("order_id", lambda x: x.nunique() - 1),
        recency=("order_purchase_timestamp", lambda x: (x.max() - x.min()).days),
        T=("order_purchase_timestamp", lambda x: (max_date - x.min()).days),
        monetary_value=("payment_value", "mean")
    ).reset_index()

    summary = summary[summary["frequency"] > 0].dropna()
    summary = summary[summary["frequency"] >= 2]
    upper_cap = summary["monetary_value"].quantile(0.99)
    summary = summary[summary["monetary_value"] <= upper_cap]
    summary["recency"] = summary["recency"].clip(upper=365)
    summary["T"] = summary["T"].clip(upper=365)

    pnbd = ParetoNBDFitter(penalizer_coef=1.0)
    pnbd.fit(summary["frequency"], summary["recency"], summary["T"])

    ggf = GammaGammaFitter(penalizer_coef=0.1)
    ggf.fit(summary["frequency"], summary["monetary_value"])

    summary["predicted_purchases"] = pnbd.conditional_expected_number_of_purchases_up_to_time(
        48, summary["frequency"], summary["recency"], summary["T"]
    ).clip(lower=0)

    summary["predicted_avg_value"] = ggf.conditional_expected_average_profit(
        summary["frequency"], summary["monetary_value"]
    )

    summary["predicted_cltv"] = summary["predicted_purchases"] * summary["predicted_avg_value"]
    summary["cltv_segment"] = pd.qcut(summary["predicted_cltv"], q=4, labels=["Low", "Mid", "High", "Very High"])

    return summary