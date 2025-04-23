# RFM Page.py

import streamlit as st
from datetime import timedelta
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="RFM Analysis", layout="wide")
st.title("🧮 RFM Analysis - Customer Segmentation")

# --- Load Raw Data ---
# Change1: Added order timestamp extraction
@st.cache_data
def load_data():
    from analysis.Preprocessing import full_orders
    full_orders['order_purchase_timestamp'] = pd.to_datetime(full_orders['order_purchase_timestamp'])
    full_orders['order_month'] = full_orders['order_purchase_timestamp'].dt.to_period("M").astype(str)
    return full_orders

full_orders = load_data()

# Change2: Added date filtering for dynamic RFM analysis
min_date = full_orders['order_purchase_timestamp'].min()
max_date = full_orders['order_purchase_timestamp'].max()

st.sidebar.header("🗓️ RFM Timeframe")
selected_date = st.sidebar.slider("Select reference end date for RFM", min_value=min_date.date(), max_value=max_date.date(), value=max_date.date())

# --- RFM Calculation ---
@st.cache_data
def calculate_rfm(df, ref_date):
    df['order_purchase_date'] = pd.to_datetime(df['order_purchase_timestamp'])
    reference_date = pd.to_datetime(ref_date)

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

rfm_df = calculate_rfm(full_orders, selected_date)

# --- Customer Group Tagging ---
@st.cache_data
def add_rfm_tags(rfm_df):
    rfm_df['CustomerGroup'] = rfm_df['RFM_Score'].apply(
        lambda x: 'High-value' if int(x) >= 444 else ('Medium-value' if int(x) >= 222 else 'Low-value')
    )
    rfm_df['BehaviorSegment'] = rfm_df.apply(lambda row:
        "Champions" if row['R'] == 4 and row['F'] == 4 and row['M'] == 4 else
        "Loyal Customers" if row['R'] >= 3 and row['F'] >= 3 else
        "Recent Customers" if row['R'] == 4 else
        "Frequent Buyers" if row['F'] == 4 else
        "Big Spenders" if row['M'] == 4 else
        "Others", axis=1)
    return rfm_df

rfm_df = add_rfm_tags(rfm_df)

# --- Segment Summary ---
@st.cache_data
def get_rfm_summary(df):
    return df.groupby("CustomerGroup").agg({
        "Recency": "mean",
        "Frequency": "mean",
        "Monetary": "mean",
        "customer_unique_id": "count"
    }).rename(columns={"customer_unique_id": "CustomerCount"}).round(2).reset_index()

rfm_summary = get_rfm_summary(rfm_df)
st.subheader("📊 RFM Segment Summary")
st.dataframe(rfm_summary)
st.caption("Insight: High-value customers are frequent, recent, and big spenders. Target them with loyalty perks.")  # Change3

# --- Segment Distribution Plot ---
fig1 = px.bar(
    rfm_df,
    x="CustomerGroup",
    title="Customer Segments Distribution",
    labels={"CustomerGroup": "Customer Group"},
    color="CustomerGroup",
    color_discrete_sequence=px.colors.qualitative.Set2
)
st.plotly_chart(fig1)
st.caption("Insight: Monitor growth of High-value segment monthly to measure retention success.")  # Change4

# --- Behavior Segments Table ---
st.subheader("🧠 Behavioral Segments")
st.dataframe(rfm_df["BehaviorSegment"].value_counts().reset_index().rename(columns={"index": "Segment", "BehaviorSegment": "Count"}))
st.caption("Insight: Champions and Loyal Customers deserve exclusive offers.")  # Change5

# --- RFM Heatmaps ---
st.subheader("🔥 RFM Heatmaps")

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

rfm_scores = rfm_df[['R', 'F', 'M']].copy()

st.plotly_chart(plot_heatmap(rfm_scores, "R", "F", "Recency vs Frequency", "Frequency Score", "Recency Score"))
st.plotly_chart(plot_heatmap(rfm_scores, "R", "M", "Recency vs Monetary", "Monetary Score", "Recency Score"))
st.plotly_chart(plot_heatmap(rfm_scores, "M", "F", "Monetary vs Frequency", "Frequency Score", "Monetary Score"))
st.caption("Insight: Use these combos to find likely churners (e.g., R=1, F=4).")  # Change6

# --- Product Preferences by Group ---
@st.cache_data
def get_product_preferences(full_orders, rfm_df):
    rfm_orders = full_orders.merge(rfm_df[['customer_unique_id', 'CustomerGroup']], on='customer_unique_id', how='inner')
    product_pref = (
        rfm_orders.groupby(["CustomerGroup", "product_category"]).size().reset_index(name='count')
        .sort_values("count", ascending=False)
    )
    return product_pref

st.subheader("🛍️ Top Products by Customer Group")
product_pref = get_product_preferences(full_orders, rfm_df)

available_groups = sorted(product_pref['CustomerGroup'].unique())
selected_group = st.selectbox("Select Customer Group", available_groups)

filtered_pref = product_pref[product_pref["CustomerGroup"] == selected_group]

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
st.caption("Insight: Align marketing campaigns with category preferences per group.")  # Change7

# --- Export CSV ---
# Change8: CSV download for marketing/export purposes
st.download_button(
    label="📥 Download RFM Segments as CSV",
    data=rfm_df.to_csv(index=False).encode('utf-8'),
    file_name='rfm_segments.csv',
    mime='text/csv'
)

# --- Customer Segment Trend Over Time ---
# Change9: Time trend visualization
st.subheader("📈 Segment Trends Over Time")
time_trend = full_orders.merge(rfm_df[['customer_unique_id', 'CustomerGroup']], on='customer_unique_id', how='inner')
segment_month = time_trend.groupby(['order_month', 'CustomerGroup']).size().reset_index(name='count')
fig_trend = px.line(
    segment_month,
    x='order_month',
    y='count',
    color='CustomerGroup',
    title='Customer Group Trend Over Time'
)
st.plotly_chart(fig_trend, use_container_width=True)
st.caption("Insight: Observe whether High-value segment is growing. Adjust retention strategy accordingly.")
