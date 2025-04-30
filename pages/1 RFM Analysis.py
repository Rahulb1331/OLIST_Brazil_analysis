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
selected_start = st.sidebar.date_input("Start date", min_value=min_date.date(), max_value=max_date.date(), value=min_date.date())
selected_end = st.sidebar.date_input("End date", min_value=min_date.date(), max_value=max_date.date(), value=max_date.date())

mask = (full_orders['order_purchase_timestamp'].dt.date >= selected_start) & (full_orders['order_purchase_timestamp'].dt.date <= selected_end)
filtered_orders = full_orders[mask]

# --- RFM Calculation ---
@st.cache_data
def calculate_rfm(df):
    df['order_purchase_date'] = pd.to_datetime(df['order_purchase_timestamp'])
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

rfm_df = calculate_rfm(filtered_orders)

# --- Customer Group Tagging ---
@st.cache_data
def add_rfm_tags(rfm_df):
    rfm_df['CustomerGroup'] = rfm_df['RFM_Score'].apply(
        lambda x: 'High-value' if int(x) >= 344 else ('Medium-value' if int(x) >= 222 else 'Low-value')
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
if st.checkbox("📌 Show Segment Insights", key="unique_key_rf1"):
    st.info("High-value customers are frequent, recent, and big spenders. Target them with loyalty perks.")

    with st.expander("📌 Segmentation Strategy & RFM Calculation Details"):
        st.info("""
        **Why we use a static, end-of-period segmentation**  
        - **Simplicity:** We compute RFM once on the full history to get three clear, stable groups (Low/Medium/High) for all downstream charts.  
        - **Stability:** Since the dataset is  having many one-time purchasers, monthly re-segmentation produces huge spikes and drop‐offs (noise), not meaningful trends.  
        - **Cross-sectional clarity:** Business users see “Right now these 44.3 K customers are Low-value” and can tie actions (promotions, retention offers) directly to that snapshot.

        **Argument for adding a dynamic “segment flow” view**  
        - If we need to **track cohort movement**—e.g. “Which Low-value customers in Q1 moved to Medium by Q3”—you can recompute RFM quarterly or on a rolling 3-month window.  
        - **Cohort flow charts** (e.g. Sankey diagrams or stacked area charts) will then show true migrations without the jagged noise of one-time buyers.

        **Current setup**  
        **We have kept the core dashboard** on a **static segmentation** basis for all RFM, CLTV, churn, and MBA pages—this provides a consistent grouping and straightforward business actions.  

        ### How RFM Segments Are Computed
        - **Reference Date:** Set to one day after the latest order in the filtered dataset.
        - **Recency (R):** Number of days since a customer’s most recent purchase.  
          - Score 4 if ≤Q1, 3 if between Q1–Q2, 2 if between Q2–Q3, else 1.
        - **Frequency (F):** Total number of orders per customer.  
          - Score 1–4 by quartiles: 1 if ≤Q1 up to 4 if >Q3.
        - **Monetary (M):** Total spend per customer.  
          - Likewise scored by quartiles.
        - **RFM Score:** Concatenate R, F, and M scores to form a three-digit code (e.g. “4-3-2”).  
        - **CustomerGroup Assignment:**  
          - **High-value** if RFM ≥ 344  
          - **Medium-value** if 222 ≤ RFM < 344  
          - **Low-value** otherwise  
        """)


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
if st.checkbox("📌 Show Distribution Insights", key="unique_key_rf2"):
    st.info("Insight: Monitor growth of High-value segment monthly to measure retention success.")


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
if st.checkbox("📌 Show Heatmap Insights", key="unique_key_rf3"):
    st.info("Insight: Use these combos to find likely churners (e.g., R=1, F=4).")

# --- Product Preferences by Group ---
@st.cache_data
def get_product_preferences(full_orders, rfm_df):
    rfm_orders = full_orders.merge(rfm_df[['customer_unique_id', 'BehaviorSegment']], on='customer_unique_id', how='inner')
    product_pref = (
        rfm_orders.groupby(["BehaviorSegment", "product_category"]).size().reset_index(name='count')
        .sort_values("count", ascending=False)
    )
    return product_pref

st.subheader("🛍️ Top Product Preferences by Segment")
product_pref = get_product_preferences(filtered_orders, rfm_df)
segments = sorted(product_pref['BehaviorSegment'].unique())
selected_segment = st.selectbox("Select Segment", segments)
filtered_pref = product_pref[product_pref["BehaviorSegment"] == selected_segment]
fig_products = px.bar(
    filtered_pref,
    x="product_category",
    y="count",
    color="product_category",
    title=f"Top Product Categories - {selected_segment}",
    template="plotly_white"
)
fig_products.update_layout(xaxis_tickangle=-45, showlegend=False)
st.plotly_chart(fig_products, use_container_width=True)
if st.checkbox("📌 Show Product Preference Insights", key="unique_key_rf4"):
    st.info("Insight: Tailor promotions by segment preference — e.g., Frequent Buyers love {filtered_pref.iloc[0]['product_category']}.")

# --- Behavior Segments Table ---
st.subheader("🧠 Behavioral Segments")
st.dataframe(rfm_df["BehaviorSegment"].value_counts().reset_index().rename(columns={"index": "Segment", "BehaviorSegment": "Segment"}))
segment_definitions = {
    "Segment": [
        "Champions", "Loyal Customers", "Recent Customers", "Frequent Buyers", "Big Spenders", "Others"
    ],
    "Definition": [
        "High Recency (R=4), Frequency (F=4), and Monetary (M=4). Most valuable, loyal, and recent customers.",
        "Visit and purchase often (R ≥ 3, F ≥ 3). Strong candidates for loyalty programs.",
        "Purchased very recently (R = 4), but may not yet be frequent or high spenders.",
        "Shop often (F = 4), not necessarily recent or high spenders.",
        "High spenders (M = 4), regardless of frequency or recency.",
        "Customers that don’t strongly qualify in other categories — potential to nurture or churn."
    ]
}
st.expander("📘 Click to view Segment Definitions").table(pd.DataFrame(segment_definitions))
if st.checkbox("📌 Show Behavioral Insights", key="unique_key_rf5"):
    st.info("Insight: Champions and loyal customers can be incentivized by offering them exclusive deals.")


# --- Segment Revenue Contribution ---
revenue_by_segment = rfm_df.groupby("BehaviorSegment")["Monetary"].sum().reset_index().sort_values("Monetary", ascending=False)
st.subheader("💰 Revenue Contribution by Segment")
st.table(revenue_by_segment)
if st.checkbox("📌 Show Revenue Insights", key="unique_key_rf6"):
    st.info("Insight: A few segments often contribute disproportionately — focus retention and upsell there.")

# --- Export CSV ---
st.download_button(
    label="📥 Download RFM Segments as CSV",
    data=rfm_df.to_csv(index=False).encode('utf-8'),
    file_name='rfm_segments.csv',
    mime='text/csv'
)

# --- Customer Segment Orders Over Time ---
st.subheader("📈 Orders by Customer Segment Over Time")
time_trend = filtered_orders.merge(rfm_df[['customer_unique_id', 'CustomerGroup']], on='customer_unique_id', how='inner')
segment_month = time_trend.groupby(['order_month', 'CustomerGroup']).size().reset_index(name='order_count')
fig_trend = px.line(
    segment_month,
    x='order_month',
    y='order_count',
    color='CustomerGroup',
    title='Customer Group Trend Over Time',
    labels={
        'order_month': 'Order Month',
         'order_count': 'Order Count',
         'CustomerGroup': 'Customer Segment'
    }
    )
st.plotly_chart(fig_trend, use_container_width=True)
if st.checkbox("📌 Show Trend Insights", key="unique_key_rf7"):
    st.info("Shows the orders placed by the customer segments over the months")
