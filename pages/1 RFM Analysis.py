import streamlit as st
from datetime import timedelta
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.title("🧮 RFM Analysis - Customer Segmentation")

# Change1: Add date input for dynamic RFM calculations
selected_date = st.date_input("Select RFM Reference Date", pd.to_datetime("2018-09-04"))

# --- Load Raw Data ---
@st.cache_data
def load_data():
    from analysis.Preprocessing import full_orders
    return full_orders
full_orders = load_data()

# --- RFM Calculation ---
@st.cache_data
def calculate_rfm(df, reference_date):
    # ChangeX: Keep as Timestamp to avoid type mismatch
    df['order_purchase_date'] = pd.to_datetime(df['order_purchase_timestamp'])

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

# Change2: Pass selected date for time control
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

# --- RFM Summary Table ---
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
# Change3: Insight under summary
st.caption("High-value customers are your most engaged and profitable. Focus retention strategies here.")
st.dataframe(rfm_summary)

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

# Change4: Add caption to guide user on next actions
st.caption("Target 'High-value' customers for loyalty programs and VIP offers.")

# --- Behavior Segments Table ---
@st.cache_data
def get_behavior_segment_summary(df):
    return df.groupby("BehaviorSegment").size().reset_index(name='count')

st.subheader("🧠 Behavioral Segments")
st.caption("Behavioral segmentation helps personalize communication.")
st.dataframe(get_behavior_segment_summary(rfm_df))
#Change20 - Add Collapsible Segment Definitions Table using st.expander + st.table
segment_definitions = {
    "Segment": [
        "Champions",
        "Loyal Customers",
        "Recent Customers",
        "Frequent Buyers",
        "Big Spenders",
        "Others"
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


# --- RFM Heatmaps ---
st.subheader("🔥 RFM Heatmaps")

# Change5: Add insights above heatmaps
st.caption("Identify clusters of frequent, recent, and high-spending users for targeted campaigns.")

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

# --- Product Preferences ---
@st.cache_data
def get_product_preferences(full_orders, rfm_df):
    rfm_orders = full_orders.merge(rfm_df[['customer_unique_id', 'CustomerGroup']], on='customer_unique_id', how='inner')
    product_pref = (
        rfm_orders.groupby(["CustomerGroup", "product_category"]).size().reset_index(name='count')
        .sort_values("count", ascending=False)
    )
    return product_pref

st.subheader("🛍️ Top Products by Customer Group")
st.caption("Use this to tailor product bundles and promotions.")
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

# Change6: Add CSV download button for business users
st.download_button("📥 Download RFM Segments", rfm_df.to_csv(index=False), file_name="rfm_segments.csv")

# Change7: Add Customer Segment Trend Visualization
@st.cache_data
def segment_trend(df):
    df['order_month'] = df['order_purchase_date'].dt.to_period("M").astype(str)
    trend_data = df.merge(rfm_df[['customer_unique_id', 'CustomerGroup']], on='customer_unique_id', how='inner')
    return trend_data.groupby(['order_month', 'CustomerGroup']).size().reset_index(name='count')

trend_df = segment_trend(full_orders)
fig_trend = px.line(trend_df, x='order_month', y='count', color='CustomerGroup',
                    title="📈 Monthly Customer Group Trend")
st.plotly_chart(fig_trend, use_container_width=True)
