import streamlit as st
import pandas as pd
import calendar
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
import plotly.express as px
from sklearn.metrics import mean_absolute_error, mean_squared_error
import plotly.graph_objects as go
import numpy as np

st.set_page_config(page_title="\ud83d\udcc8 Time Series Dashboard", layout="wide")
st.title("\ud83d\udcc8 Time Series Analysis Dashboard")

@st.cache_data
def load_data():
    from analysis.Preprocessing import full_orders
    return full_orders
full_orders = load_data()

# Preprocessing
df = full_orders.copy()
df["order_purchase_timestamp"] = pd.to_datetime(df["order_purchase_timestamp"])
df["order_month"] = df["order_purchase_timestamp"].dt.to_period("M").astype(str)
df["month"] = df["order_purchase_timestamp"].dt.month

# Regional & Customer Type Segmentation
region_options = sorted(df['customer_state'].dropna().unique())
customer_types = ['All', 'Repeat', 'New']

selected_region = st.sidebar.selectbox("Select Region (State)", ['All'] + region_options)
selected_customer_type = st.sidebar.selectbox("Select Customer Type", customer_types)

if selected_region != 'All':
    df = df[df['customer_state'] == selected_region]

if selected_customer_type == 'New':
    new_customers = df.groupby('customer_id')['order_purchase_timestamp'].min().reset_index()
    new_customers['first_order_month'] = new_customers['order_purchase_timestamp'].dt.to_period("M").astype(str)
    df = df.merge(new_customers[['customer_id', 'first_order_month']], on='customer_id')
    df = df[df['order_month'] == df['first_order_month']]
elif selected_customer_type == 'Repeat':
    counts = df['customer_id'].value_counts()
    repeat_customers = counts[counts > 1].index
    df = df[df['customer_id'].isin(repeat_customers)]

# Aggregate Revenue & Orders
monthly_revenue_pd = df.groupby("order_month")["price"].sum().reset_index(name="total_revenue")
monthly_orders_pd = df.groupby("order_month")["order_id"].nunique().reset_index(name="order_count")

monthly_revenue_pd["rolling_3mo"] = monthly_revenue_pd["total_revenue"].rolling(3).mean()
monthly_orders_pd["rolling_3mo"] = monthly_orders_pd["order_count"].rolling(3).mean()

monthly_avg_revenue_pd = df.groupby("month")["price"].sum().reset_index(name="total_revenue")
monthly_avg_orders_pd = df.groupby("month")["order_id"].nunique().reset_index(name="order_count")

monthly_avg_revenue_pd["month"] = monthly_avg_revenue_pd["month"].apply(lambda x: calendar.month_abbr[x])
monthly_avg_orders_pd["month"] = monthly_avg_orders_pd["month"].apply(lambda x: calendar.month_abbr[x])

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["\ud83d\udcca Historical Trends", "\ud83d\uddd6\ufe0f Seasonality", "\ud83d\udd2e Forecasts", "\ud83d\udce6 Category Trends"])

with tab1:
    st.subheader("Monthly Revenue and Order Trends")
    with st.checkbox("Show Analysis & Recommendations - Historical Trends"):
        st.info("""
        **Analysis Performed:**
        - Aggregated total revenue and order count per month.
        - Added a 3-month rolling average to smooth short-term fluctuations.

        **Purpose:**
        - Identify growth patterns or dips over time.
        - Rolling average helps highlight broader trends without short-term noise.

        **Recommendations:**
        - If a consistent upward trend is observed, consider scaling operations.
        - If dips occur in specific months, investigate root causes like seasonality or operational issues.
        """)

    col1, col2 = st.columns(2)

    with col1:
        fig = px.line(
            monthly_revenue_pd,
            x="order_month",
            y=["total_revenue", "rolling_3mo"],
            title="\ud83d\udcc8 Monthly Revenue (with 3-mo Rolling Avg)",
            markers=True,
            labels={"order_month": "Month", "value": "Revenue", "variable": "Legend"}
        )
        fig.update_layout(template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.line(
            monthly_orders_pd,
            x="order_month",
            y=["order_count", "rolling_3mo"],
            title="\ud83d\uded2 Monthly Order Count (with 3-mo Rolling Avg)",
            markers=True,
            labels={"order_month": "Month", "value": "Orders", "variable": "Legend"}
        )
        fig.update_layout(template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("\ud83d\uddd6\ufe0f Average Revenue and Orders by Calendar Month")
    with st.checkbox("Show Analysis & Recommendations - Seasonality"):
        st.info("""
        **Analysis Performed:**
        - Grouped and visualized revenue and orders by calendar month.
        
        **Purpose:**
        - To detect seasonal patterns. For example, which months consistently perform better.

        **Recommendations:**
        - Months with historically high sales (e.g. November/December) could be targeted for promotions.
        - Slow months may benefit from retention campaigns or sales events.
        """)

    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(
            monthly_avg_revenue_pd,
            x="month",
            y="total_revenue",
            title="\ud83d\udcb0 Revenue by Calendar Month",
            labels={"total_revenue": "Revenue"},
            template="plotly_white"
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.bar(
            monthly_avg_orders_pd,
            x="month",
            y="order_count",
            title="\ud83d\udce6 Order Count by Calendar Month",
            labels={"order_count": "Orders"},
            template="plotly_white"
        )
        st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.subheader("\ud83d\udd2e Prophet Forecasts for Revenue & Orders")
    with st.checkbox("Show Forecast Insights"):
        st.info("""
        **Analysis Performed:**
        - Used Prophet model to forecast revenue and orders for the next 6 months.
        - Visualized actual vs. predicted values with confidence intervals.
        - Included model performance metrics (MAE, RMSE, MAPE).

        **Purpose:**
        - Help predict future demand and guide business planning.

        **Recommendations:**
        - If forecasts show a decline, consider launching marketing campaigns or optimizing product offerings.
        - If a spike is expected, plan for inventory and logistics scaling.
        """)

    # Prepare data
    rev_df = monthly_revenue_pd.rename(columns={"order_month": "ds", "total_revenue": "y"})
    ord_df = monthly_orders_pd.rename(columns={"order_month": "ds", "order_count": "y"})
    rev_df["ds"] = pd.to_datetime(rev_df["ds"])
    ord_df["ds"] = pd.to_datetime(ord_df["ds"])

    rev_model = Prophet()
    ord_model = Prophet()

    rev_model.fit(rev_df)
    ord_model.fit(ord_df)

    future_rev = rev_model.make_future_dataframe(periods=6, freq='M')
    future_ord = ord_model.make_future_dataframe(periods=6, freq='M')

    forecast_rev = rev_model.predict(future_rev)
    forecast_ord = ord_model.predict(future_ord)

    col1, col2 = st.columns(2)
    with col1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=forecast_rev['ds'], y=forecast_rev['yhat'], mode='lines', name='Forecast'))
        fig.add_trace(go.Scatter(x=rev_df['ds'], y=rev_df['y'], mode='lines', name='Actual'))
        fig.update_layout(title='\ud83d\udcc8 Revenue Forecast', xaxis_title='Date', yaxis_title='Revenue')
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=forecast_ord['ds'], y=forecast_ord['yhat'], mode='lines', name='Forecast'))
        fig.add_trace(go.Scatter(x=ord_df['ds'], y=ord_df['y'], mode='lines', name='Actual'))
        fig.update_layout(title='\ud83d\udce6 Order Count Forecast', xaxis_title='Date', yaxis_title='Orders')
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("\ud83d\udcca Revenue Forecast with Confidence Intervals")

    fig_ci = go.Figure()
    fig_ci.add_trace(go.Scatter(x=forecast_rev['ds'], y=forecast_rev['yhat'], name='Forecast', line=dict(color='blue')))
    fig_ci.add_trace(go.Scatter(x=forecast_rev['ds'], y=forecast_rev['yhat_upper'], name='Upper Bound', line=dict(dash='dot', color='lightblue')))
    fig_ci.add_trace(go.Scatter(x=forecast_rev['ds'], y=forecast_rev['yhat_lower'], name='Lower Bound', line=dict(dash='dot', color='lightblue'), fill='tonexty', fillcolor='rgba(173,216,230,0.2)'))
    fig_ci.update_layout(title="\ud83d\udca1 Revenue Forecast with Confidence Intervals", xaxis_title="Date", yaxis_title="Revenue")
    st.plotly_chart(fig_ci, use_container_width=True)

    st.markdown("---")
    st.subheader("\ud83d\udccf Forecast Accuracy Metrics")

    common_dates = rev_df["ds"].isin(forecast_rev["ds"])
    actual = rev_df[common_dates].set_index("ds")
    predicted = forecast_rev.set_index("ds").loc[actual.index]

    mae = mean_absolute_error(actual['y'], predicted['yhat'])
    rmse = np.sqrt(mean_squared_error(actual['y'], predicted['yhat']))
    mape = np.mean(np.abs((actual['y'] - predicted['yhat']) / actual['y'])) * 100

    st.metric("MAE (Mean Absolute Error)", f"{mae:,.0f}")
    st.metric("RMSE (Root Mean Squared Error)", f"{rmse:,.0f}")
    st.metric("MAPE (Mean Absolute Percentage Error)", f"{mape:.2f}%")

with tab4:
    st.subheader("\ud83d\udce6 Category-wise Revenue Trends")
    categories = sorted(df['product_category'].dropna().unique())
    selected_cat = st.selectbox("Select Product Category", categories)

    with st.checkbox("Show Analysis & Recommendations - Category Trends"):
        st.info("""
        **Analysis Performed:**
        - Summed revenue per product category by month.

        **Purpose:**
        - Understand product performance trends over time.

        **Recommendations:**
        - Focus marketing or inventory efforts on top-performing categories.
        - Monitor underperforming categories to adjust strategy or pricing.
        """)

    cat_df = df[df['product_category'] == selected_cat]
    cat_monthly = cat_df.groupby(cat_df['order_purchase_timestamp'].dt.to_period("M").astype(str))["price"].sum().reset_index()

    fig_cat = px.line(cat_monthly, x="order_purchase_timestamp", y="price", title=f"Revenue Trend: {selected_cat}",
                      labels={"order_purchase_timestamp": "Month", "price": "Revenue"})
    st.plotly_chart(fig_cat, use_container_width=True)
