import streamlit as st
from datetime import timedelta
import pandas as pd
import plotly.express as px

from analysis.Preprocessing import full_orders

st.title("🧮 RFM Analysis - Customer Segmentation")

# --- RFM Calculation Function ---
def run_rfm_analysis(df):
    df['order_purchase_date'] = pd.to_datetime(df['order_purchase_timestamp']).dt.date
    reference_date = df['order_purchase_date'].max() + timedelta(days=1)

    rfm = df.groupby('customer_unique_id').agg({
        'order_purchase_date': lambda x: (reference_date - x.max()).days,
        'order_id': 'count',
        'payment_value': 'sum'
    }).reset_index()
    rfm.columns = ['customer_unique_id', 'Recency', 'Frequency', 'Monetary']

    r_q = rfm['Recency'].quantile([0.25, 0.5, 0.75]).values
    f_q = rfm['Frequency'].quantile([0.25, 0.5, 0.75]).values
    m_q = rfm['Monetary'].quantile([0.25, 0.5, 0.75]).values

    def r_score(r): return 4 if r <= r_q[0] else 3 if r <= r_q[1] else 2 if r <= r_q[2] else 1
    def fm_score(x, q): return 1 if x <= q[0] else 2 if x <= q[1] else 3 if x <= q[2] else 4

    rfm['R'] = rfm['Recency'].apply(r_score)
    rfm['F'] = rfm['Frequency'].apply(lambda x: fm_score(x, f_q))
    rfm['M'] = rfm['Monetary'].apply(lambda x: fm_score(x, m_q))
    rfm['RFM_Score'] = rfm['R'].astype(str) + rfm['F'].astype(str) + rfm['M'].astype(str)

    return rfm

# --- Main RFM Execution ---
rfm_df = run_rfm_analysis(full_orders)

rfm_df['CustomerGroup'] = rfm_df['RFM_Score'].apply(
    lambda x: 'High-value' if int(x) >= 444 else ('Medium-value' if int(x) >= 222 else 'Low-value')
)

rfm_summary = rfm_df.groupby("CustomerGroup").agg({
    "Recency": "mean",
    "Frequency": "mean",
    "Monetary": "mean",
    "customer_unique_id": "count"
}).rename(columns={"customer_unique_id": "CustomerCount"}).round(2).reset_index()

st.subheader("📊 RFM Segment Summary")
st.dataframe(rfm_summary)

# --- Distribution of Customer Segments ---
fig1 = px.bar(
    rfm_df,
    x="CustomerGroup",
    title="Customer Segments Distribution",
    labels={"CustomerGroup": "Customer Group", "count": "Count"},
    color="CustomerGroup",
    color_discrete_sequence=px.colors.qualitative.Set2
)
st.plotly_chart(fig1)

# --- Advanced Tagging ---
rfm_df['BehaviorSegment'] = rfm_df.apply(lambda row:
    "Champions" if row['R'] == 4 and row['F'] == 4 and row['M'] == 4 else
    "Loyal Customers" if row['R'] >= 3 and row['F'] >= 3 else
    "Recent Customers" if row['R'] == 4 else
    "Frequent Buyers" if row['F'] == 4 else
    "Big Spenders" if row['M'] == 4 else
    "Others", axis=1)

st.subheader("🧠 Behavioral Segments")
st.dataframe(rfm_df.groupby("BehaviorSegment").size().reset_index(name='count'))

# --- Heatmaps ---
st.subheader("🔥 RFM Heatmaps")

rfm_pd = rfm_df[['R', 'F', 'M']]

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
rfm_orders = full_orders.merge(rfm_df[['customer_unique_id', 'CustomerGroup']], on='customer_unique_id', how='inner')
product_pref = (
    rfm_orders.groupby(["CustomerGroup", "product_category"]).size().reset_index(name='count')
    .sort_values("count", ascending=False)
)

# Dropdown to select group
available_groups = sorted(product_pref['CustomerGroup'].unique())
selected_group = st.selectbox("Select Customer Group", available_groups)

# Filter data based on selection
filtered_pref = product_pref[product_pref["CustomerGroup"] == selected_group]

# Plot
fig_products = px.bar(
    filtered_pref,
    x="product_category",
    y="count",
    color="product_category",
    title=f"Top Product Categories - {selected_group} Customers",
    template="plotly_white"
)
fig_products.update_layout(xaxis_tickangle=-45, showlegend=False)
st.plotly_chart(fig_products, use_container_width=True)
