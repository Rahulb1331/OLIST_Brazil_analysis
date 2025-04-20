# analysis/rfm.py

from analysis.Preprocessing import full_orders
from datetime import timedelta
import pandas as pd
import numpy as np
import plotly.express as px



def run_rfm_analysis(order_customer_df):
    # Convert timestamp to date
    order_customer_df["order_purchase_date"] = pd.to_datetime(order_customer_df["order_purchase_timestamp"])

    # Reference date
    reference_date = order_customer_df["order_purchase_date"].max() + timedelta(days=1)

    # RFM Calculation
    rfm_df = order_customer_df.groupby("customer_unique_id").agg({
        "order_purchase_date": lambda x: (reference_date - x.max()).days,
        "order_id": "count",
        "payment_value": "sum"
    }).reset_index()

    rfm_df.columns = ["customer_unique_id", "Recency", "Frequency", "Monetary"]


    # Quantile scoring
    r_q = rfm_df["Recency"].quantile([0.25, 0.5, 0.75]).values
    f_q = rfm_df["Frequency"].quantile([0.25, 0.5, 0.75]).values
    m_q = rfm_df["Monetary"].quantile([0.25, 0.5, 0.75]).values

    def r_score(r):
        return 4 if r <= r_q[0] else 3 if r <= r_q[1] else 2 if r <= r_q[2] else 1

    def fm_score(x, q):
        if x is None:
            return 1
        return 1 if x <= q[0] else 2 if x <= q[1] else 3 if x <= q[2] else 4

    rfm_df["R"] = rfm_df["Recency"].apply(r_score)
    rfm_df["F"] = rfm_df["Frequency"].apply(lambda x: fm_score(x, f_q))
    rfm_df["M"] = rfm_df["Monetary"].apply(lambda x: fm_score(x, m_q))

    rfm_df["RFM_Score"] = rfm_df[["R", "F", "M"]].astype(str).agg("".join, axis=1)

    return rfm_df


#Preparing the dataset on which we will run the rfm analysis

rfm_df = run_rfm_analysis(full_orders)

#segmenting the customers into high value, medium value and low value based on their rfm score
rfm_df["CustomerGroup"] = np.where(
    rfm_df["RFM_Score"].astype(int) >= 444, "High-value",
    np.where(rfm_df["RFM_Score"].astype(int) >= 222, "Medium-value", "Low-value")
)


#Summary statistics per segment
rfm_summary = rfm_df.groupby("CustomerGroup").agg({
    "customer_unique_id": "count",
    "Recency": "mean",
    "Frequency": "mean",
    "Monetary": "mean"
}).rename(columns={"customer_unique_id": "CustomerCount"}).round(2)

print(rfm_summary)

# Countplot for the different customer segments

# Distribution of customer segments

fig = px.bar(
    rfm_df,
    x="CustomerGroup",
    title="Customer Segments Distribution",
    labels={"CustomerGroup": "Customer Group", "count": "Count"},
    color="CustomerGroup",
    color_discrete_sequence=px.colors.qualitative.Set2
)
fig.show()

# Add quartile segments
rfm_df["R_Quartile"] = pd.qcut(rfm_df["Recency"], 4, labels=False) + 1
rfm_df["F_Quartile"] = pd.qcut(rfm_df["Frequency"].rank(method="first", ascending=False), 4, labels=False) + 1
rfm_df["M_Quartile"] = pd.qcut(rfm_df["Monetary"].rank(method="first", ascending=False), 4, labels=False) + 1

# Behavioral tags
rfm_df["BehaviorSegment"] = "Others"
rfm_df.loc[(rfm_df.R == 4) & (rfm_df.F == 4) & (rfm_df.M == 4), "BehaviorSegment"] = "Champions"
rfm_df.loc[(rfm_df.R >= 3) & (rfm_df.F >= 3), "BehaviorSegment"] = "Loyal Customers"
rfm_df.loc[(rfm_df.R == 4), "BehaviorSegment"] = "Recent Customers"
rfm_df.loc[(rfm_df.F == 4), "BehaviorSegment"] = "Frequent Buyers"
rfm_df.loc[(rfm_df.M == 4), "BehaviorSegment"] = "Big Spenders"

print(rfm_df["BehaviorSegment"].value_counts())


#Heatmap of R, F, M scores
for (x, y) in [("R", "F"), ("R", "M"), ("M", "F")]:
    heatmap_data = rfm_df.groupby([x, y]).size().reset_index(name='count')
    heatmap_pivot = heatmap_data.pivot(index=x, columns=y, values='count').fillna(0)

    fig = px.imshow(
        heatmap_pivot.values,
        labels=dict(x=y + " Score", y=x + " Score", color="Customer Count"),
        x=heatmap_pivot.columns.tolist(),
        y=heatmap_pivot.index.tolist(),
        color_continuous_scale="YlGnBu",
        text_auto=True,
        title=f"Customer Distribution by {x} and {y} Scores"
    )
    fig.update_layout(template="plotly_white")
    fig.show()


# Link segments to products
# Join RFM segments back to full_orders
orders_with_rfm = full_orders.merge(rfm_df[["customer_unique_id", "CustomerGroup"]], on="customer_unique_id", how="left")

# Analyzing top products per segments
top_products_by_segment = orders_with_rfm.groupby(["CustomerGroup", "product_category"]).size().reset_index(name="purchase_count")
top_products_by_segment = top_products_by_segment.sort_values(by=["CustomerGroup", "purchase_count"], ascending=[True, False])

print(top_products_by_segment.head(10))

# Join RFM segments to full orders to explore their product preferences
rfm_orders = full_orders.join(rfm_df.select("customer_unique_id", "CustomerGroup"), on="customer_unique_id", how="inner")

# Now group to see top product categories per customer group
fig_products = px.bar(
    top_products_by_segment,
    x="product_category",
    y="purchase_count",
    color="CustomerGroup",
    barmode="group",
    title="Top Product Categories by Customer Group",
    template="plotly_white"
)
fig_products.update_layout(xaxis_tickangle=-45)
fig_products.show()

