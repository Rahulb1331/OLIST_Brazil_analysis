# At the top of Scripts/pages/rfm_page.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from Scripts.config import setup_environment
setup_environment()

import streamlit as st
from datetime import timedelta
from pyspark.sql.functions import col, max as spark_max, count, sum as spark_sum, datediff, to_date, lit, udf, when
from pyspark.sql.types import IntegerType
from pyspark.sql.functions import concat_ws
from pyspark.sql import functions as F
from pyspark.sql.window import Window
import plotly.express as px

from analysis.Preprocessing import full_orders

st.title("🧮 RFM Analysis - Customer Segmentation")

# --- RFM Calculation Function ---
def run_rfm_analysis(order_customer_df):
    order_customer_df = order_customer_df.withColumn("order_purchase_date", to_date("order_purchase_timestamp"))
    max_date = order_customer_df.agg(spark_max("order_purchase_date")).collect()[0][0]
    reference_date = max_date + timedelta(days=1)

    rfm_df = order_customer_df.groupBy("customer_unique_id").agg(
        datediff(lit(reference_date), spark_max("order_purchase_date")).alias("Recency"),
        count("order_id").alias("Frequency"),
        spark_sum("payment_value").alias("Monetary")
    )

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

# --- Main RFM Execution ---
rfm_df = run_rfm_analysis(full_orders)

rfm_df = rfm_df.withColumn(
    "CustomerGroup",
    when(col("RFM_Score") >= 444, "High-value")
    .when((col("RFM_Score") >= 222) & (col("RFM_Score") < 444), "Medium-value")
    .otherwise("Low-value")
)

rfm_summary = rfm_df.groupBy("CustomerGroup").agg(
    F.count("*").alias("CustomerCount"),
    F.round(F.avg("Recency"), 2).alias("AvgRecency"),
    F.round(F.avg("Frequency"), 2).alias("AvgFrequency"),
    F.round(F.avg("Monetary"), 2).alias("AvgMonetary")
)

st.subheader("📊 RFM Segment Summary")
st.dataframe(rfm_summary.toPandas())

# --- Distribution of Customer Segments ---
rfm_pandas = rfm_df.toPandas()
fig1 = px.bar(
    rfm_pandas,
    x="CustomerGroup",
    title="Customer Segments Distribution",
    labels={"CustomerGroup": "Customer Group", "count": "Count"},
    color="CustomerGroup",
    color_discrete_sequence=px.colors.qualitative.Set2
)
st.plotly_chart(fig1)

# --- Advanced Tagging ---
rfm_df = rfm_df.withColumn("R_Quartile", F.ntile(4).over(Window.orderBy("Recency")))
rfm_df = rfm_df.withColumn("F_Quartile", F.ntile(4).over(Window.orderBy(F.desc("Frequency"))))
rfm_df = rfm_df.withColumn("M_Quartile", F.ntile(4).over(Window.orderBy(F.desc("Monetary"))))

rfm_df = rfm_df.withColumn(
    "BehaviorSegment",
    when((col("R") == 4) & (col("F") == 4) & (col("M") == 4), "Champions")
    .when((col("R") >= 3) & (col("F") >= 3), "Loyal Customers")
    .when((col("R") == 4), "Recent Customers")
    .when((col("F") == 4), "Frequent Buyers")
    .when((col("M") == 4), "Big Spenders")
    .otherwise("Others")
)

st.subheader("🧠 Behavioral Segments")
st.dataframe(rfm_df.groupBy("BehaviorSegment").count().toPandas())

# --- Heatmaps ---
st.subheader("🔥 RFM Heatmaps")

rfm_pd = rfm_df.select("R", "F", "M").toPandas()

def plot_heatmap(data, index, columns, title, xlab, ylab):
    heatmap_data = data.groupby([index, columns]).size().reset_index(name='count')
    pivot = heatmap_data.pivot(index=index, columns=columns, values='count').fillna(0)
    fig = px.imshow(
        pivot.values,
        labels=dict(x=xlab, y=ylab, color="Count"),
        x=pivot.columns.tolist(),
        y=pivot.index.tolist(),
        color_continuous_scale="YlGnBu",
        text_auto=True
    )
    fig.update_layout(title=title)
    return fig

st.plotly_chart(plot_heatmap(rfm_pd, "R", "F", "Recency vs Frequency", "Frequency Score", "Recency Score"))
st.plotly_chart(plot_heatmap(rfm_pd, "R", "M", "Recency vs Monetary", "Monetary Score", "Recency Score"))
st.plotly_chart(plot_heatmap(rfm_pd, "M", "F", "Monetary vs Frequency", "Frequency Score", "Monetary Score"))

# --- Product Preferences ---
st.subheader("🛍️ Top Products by Customer Group")
rfm_orders = full_orders.join(rfm_df.select("customer_unique_id", "CustomerGroup"), on="customer_unique_id", how="inner")
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
st.plotly_chart(fig_products)
