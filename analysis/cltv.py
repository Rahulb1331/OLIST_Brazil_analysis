import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st
@st.cache_data
def load_data():
    from analysis.Preprocessing import full_orders
    from analysis.rfm import rfm_df
    return full_orders, rfm_df
full_orders, rfm_df = load_data()

@st.cache_data
def run_cltv_analysis(full_orders_df):
    # Step 1: Aggregate order stats by customer
    customer_metrics = full_orders_df.groupby("customer_unique_id").agg(
        total_orders=("order_id", "nunique"),
        total_payment=("payment_value", "sum"),
        avg_order_value=("payment_value", "mean")
    ).reset_index()

    # Step 2: Calculate global purchase frequency
    total_orders = full_orders_df["order_id"].nunique()
    total_customers = full_orders_df["customer_unique_id"].nunique()
    purchase_frequency = total_orders / total_customers

    # Step 3: Calculate CLTV
    customer_metrics["purchase_frequency"] = customer_metrics["total_orders"] / total_customers
    customer_metrics["cltv"] = customer_metrics["avg_order_value"] * customer_metrics["purchase_frequency"]

    return customer_metrics[["customer_unique_id", "total_orders", "avg_order_value", "purchase_frequency", "cltv"]]

cltv_df = run_cltv_analysis(full_orders)

# Normalizing the CLTV
# Calculate min and max CLTV
min_cltv = cltv_df["cltv"].min()
max_cltv = cltv_df["cltv"].max()

# Avoid division by zero
range_cltv = max_cltv - min_cltv if max_cltv != min_cltv else 1.0

# Add normalized CLTV column
cltv_df["normalized_cltv"] = (cltv_df["cltv"] - min_cltv) / range_cltv

# Segmenting the customers based on the normalized cltv
cltv_df["CLTV_Segment"] = pd.cut(
    cltv_df["normalized_cltv"],
    bins=[-np.inf, 0.20, 0.80, np.inf],
    labels=["Low CLTV", "Medium CLTV", "High CLTV"]
)

#Join CLTV Data with RFM
# Join on customer ID
rfm_cltv_df = pd.merge(rfm_df, cltv_df[["customer_unique_id", "cltv", "normalized_cltv", "CLTV_Segment"]], on="customer_unique_id", how="inner")


# Using the new formula
# Average order value already computed per customer
# Purchase frequency = total orders / total unique customers (global freq)
# Assume lifespan in months (taken 12 months)

total_customers = cltv_df["customer_unique_id"].nunique()
global_purchase_frequency = cltv_df["total_orders"].sum() / total_customers
lifespan_months = 12
cltv_df["better_cltv"] = cltv_df["avg_order_value"] * global_purchase_frequency * lifespan_months

min_val = cltv_df["better_cltv"].min()
max_val = cltv_df["better_cltv"].max()
range_val = max_val - min_val if max_val != min_val else 1.0
cltv_df["cltv_normalized"] = (cltv_df["better_cltv"] - min_val) / range_val


# Get quantile breakpoints
q1 = cltv_df["better_cltv"].quantile(0.20)
q2 = cltv_df["better_cltv"].quantile(0.80)

# Segment
cltv_df["CLTV_new_Segment"] = pd.cut(
    cltv_df["better_cltv"],
    bins=[-np.inf, q1, q2, np.inf],
    labels=["Low CLTV", "Medium CLTV", "High CLTV"]
)

#Join CLTV Data with RFM
# Join on customer ID
rfm_cltv_df = pd.merge(rfm_df, cltv_df[["customer_unique_id", "better_cltv", "cltv_normalized", "CLTV_new_Segment"]], on="customer_unique_id", how="inner")

# Count Customers in Each CLTV Segment
segment_counts = rfm_cltv_df["CLTV_new_Segment"].value_counts().reset_index(name="CustomerCount")
print(segment_counts)

#Cross Tab CLTV vs RFM Segments
#This gives a matrix-style overview of how customer segments overlap:
cross_tab = pd.crosstab(rfm_cltv_df["CLTV_new_Segment"], rfm_cltv_df["CustomerGroup"])
print(cross_tab)
#Plotly: Visualize CLTV Distribution
cltv_pd = rfm_cltv_df[["cltv_normalized", "CLTV_new_Segment"]]

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
orders_pd = full_orders[["customer_unique_id", "order_id", "order_purchase_timestamp", "payment_value"]].copy()
orders_pd["order_purchase_timestamp"] = pd.to_datetime(orders_pd["order_purchase_timestamp"])
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

sns.histplot(summary["recency"], bins=50, kde=True)
plt.title("Recency Distribution")
plt.show()

sns.histplot(summary["T"], bins=50, kde=True)
plt.title("T Distribution")
plt.show()

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
).clip(lower=0)

# Predict average monetary value
summary["predicted_avg_value"] = ggf.conditional_expected_average_profit(
    summary["frequency"],
    summary["monetary_value"]
)

# Calculate CLTV
summary["predicted_cltv"] = summary["predicted_purchases"] * summary["predicted_avg_value"]

print(summary.head())
print(summary[summary["predicted_cltv"] < 0])  # Check for anomalies


# Segmenting the customers by the predicted cltv
summary["cltv_segment"] = pd.qcut(summary["predicted_cltv"], q=3, labels=["Low", "Mid", "High"])

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

@st.cache_data
def enrich_cltv_with_segments(cltv_df):
    min_cltv = cltv_df["cltv"].min()
    max_cltv = cltv_df["cltv"].max()
    range_cltv = max_cltv - min_cltv if max_cltv != min_cltv else 1.0

    cltv_df["normalized_cltv"] = (cltv_df["cltv"] - min_cltv) / range_cltv

    cltv_df["CLTV_Segment"] = pd.cut(cltv_df["normalized_cltv"], bins=[-np.inf, 0.20, 0.80, np.inf], labels=["Low CLTV", "Medium CLTV", "High CLTV"])

    # Advanced CLTV
    global_purchase_frequency = cltv_df["total_orders"].sum() / cltv_df.shape[0]
    lifespan_months = 12

    cltv_df["better_cltv"] = cltv_df["avg_order_value"] * global_purchase_frequency * lifespan_months


    q1 = cltv_df["better_cltv"].quantile(0.20)
    q2 = cltv_df["better_cltv"].quantile(0.80)
    cltv_df["cltv_normalized"] = (cltv_df["better_cltv"] - cltv_df["better_cltv"].min()) / (cltv_df["better_cltv"].max() - cltv_df["better_cltv"].min())
    cltv_df["CLTV_new_Segment"] = pd.cut(cltv_df["better_cltv"], bins=[-np.inf, q1, q2, np.inf], labels=["Low CLTV", "Medium CLTV", "High CLTV"])


    return cltv_df

@st.cache_data
def model_cltv_lifetimes(df):
    
    orders_pd = df[["customer_unique_id", "order_id", "order_purchase_timestamp", "payment_value"]].copy()

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

    # with a 1 year horizon
    summary["predicted_purchases"] = pnbd.conditional_expected_number_of_purchases_up_to_time(
        365, summary["frequency"], summary["recency"], summary["T"]
    ).clip(lower=0)

    summary["predicted_avg_value"] = ggf.conditional_expected_average_profit(
        summary["frequency"], summary["monetary_value"]
    )

    summary["predicted_cltv"] = summary["predicted_purchases"] * summary["predicted_avg_value"]
    #summary["cltv_segment"] = pd.qcut(summary["predicted_cltv"], q=3, labels=["Low", "Mid", "High"])
    # Sort by predicted CLTV descending
    summary = summary.sort_values("predicted_cltv", ascending=False).reset_index(drop=True)

    # Calculate cumulative CLTV and percentage
    summary["cum_cltv"] = summary["predicted_cltv"].cumsum()
    summary["cum_cltv_perc"] = summary["cum_cltv"] / summary["predicted_cltv"].sum()

    # Assign Pareto segments
    summary["cltv_segment"] = "Low"
    summary.loc[summary["cum_cltv_perc"] <= 0.80, "cltv_segment"] = "High"
    summary.loc[(summary["cum_cltv_perc"] > 0.80) & (summary["cum_cltv_perc"] <= 0.95), "cltv_segment"] = "Mid"
    
    return summary

