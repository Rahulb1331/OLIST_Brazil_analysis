import streamlit as st
from analysis.Preprocessing import full_orders
import pandas as pd
import calendar
from prophet import Prophet
import plotly.express as px
from sklearn.metrics import mean_absolute_error, mean_squared_error
import plotly.graph_objects as go
import numpy as np

st.set_page_config(page_title="📈 Time Series Dashboard", layout="wide")
st.title("📈 Time Series Analysis Dashboard")

# Preprocess Spark DataFrame
full_orders["order_purchase_timestamp"] = pd.to_datetime(full_orders["order_purchase_timestamp"])
# Create month column
full_orders["order_month"] = full_orders["order_purchase_timestamp"].dt.to_period("M").astype(str)


# Aggregate Revenue & Orders
monthly_revenue_pd = full_orders.groupby("order_month")["price"].sum().reset_index(name="total_revenue")
monthly_revenue_pd = monthly_revenue_pd[monthly_revenue_pd["total_revenue"] > 100000]

monthly_orders_pd = full_orders.groupby("order_month")["order_id"].nunique().reset_index(name="order_count")
monthly_orders_pd = monthly_orders_pd[monthly_orders_pd["order_count"] > 500]

# Rolling Averages
monthly_revenue_pd["rolling_3mo"] = monthly_revenue_pd["total_revenue"].rolling(3).mean()
monthly_orders_pd["rolling_3mo"] = monthly_orders_pd["order_count"].rolling(3).mean()

# Calendar month revenue & orders (seasonality)
full_orders["month"] = full_orders["order_purchase_timestamp"].dt.month

monthly_avg_revenue_pd = full_orders.groupby("month")["price"].sum().reset_index(name="total_revenue")
monthly_avg_orders_pd = full_orders.groupby("month")["order_id"].nunique().reset_index(name="order_count")

monthly_avg_revenue_pd["month"] = monthly_avg_revenue_pd["month"].apply(lambda x: calendar.month_abbr[x])
monthly_avg_orders_pd["month"] = monthly_avg_orders_pd["month"].apply(lambda x: calendar.month_abbr[x])

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["📊 Historical Trends", "📆 Seasonality", "🔮 Forecasts", "📦 Category Trends"])

with tab1:
    st.subheader("Monthly Revenue and Order Trends")

    col1, col2 = st.columns(2)

    with col1:
        fig = px.line(
            monthly_revenue_pd,
            x="order_month",
            y=["total_revenue", "rolling_3mo"],
            title="📈 Monthly Revenue (with 3-mo Rolling Avg)",
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
            title="🛒 Monthly Order Count (with 3-mo Rolling Avg)",
            markers=True,
            labels={"order_month": "Month", "value": "Orders", "variable": "Legend"}
        )
        fig.update_layout(template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("📆 Average Revenue and Orders by Calendar Month")

    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(
            monthly_avg_revenue_pd,
            x="month",
            y="total_revenue",
            title="💰 Revenue by Calendar Month",
            labels={"total_revenue": "Revenue"},
            template="plotly_white"
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.bar(
            monthly_avg_orders_pd,
            x="month",
            y="order_count",
            title="📦 Order Count by Calendar Month",
            labels={"order_count": "Orders"},
            template="plotly_white"
        )
        st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.subheader("🔮 Prophet Forecasts for Revenue & Orders")

    # Prophet setup
    revenue_df = monthly_revenue_pd.rename(columns={"order_month": "ds", "total_revenue": "y"})
    orders_df = monthly_orders_pd.rename(columns={"order_month": "ds", "order_count": "y"})

    revenue_df["ds"] = pd.to_datetime(revenue_df["ds"])
    orders_df["ds"] = pd.to_datetime(orders_df["ds"])

    # Fit models
    revenue_model = Prophet()
    revenue_model.fit(revenue_df)

    orders_model = Prophet()
    orders_model.fit(orders_df)

    # Forecast 6 months
    future_revenue = revenue_model.make_future_dataframe(periods=6, freq='ME')
    future_orders = orders_model.make_future_dataframe(periods=6, freq='ME')

    forecast_revenue = revenue_model.predict(future_revenue)
    forecast_orders = orders_model.predict(future_orders)

    col1, col2 = st.columns(2)
    with col1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=forecast_revenue['ds'], y=forecast_revenue['yhat'], mode='lines', name='Forecast'))
        fig.add_trace(go.Scatter(x=revenue_df['ds'], y=revenue_df['y'], mode='lines', name='Actual'))
        fig.update_layout(title='📈 Revenue Forecast', xaxis_title='Date', yaxis_title='Revenue')
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=forecast_orders['ds'], y=forecast_orders['yhat'], mode='lines', name='Forecast'))
        fig.add_trace(go.Scatter(x=orders_df['ds'], y=orders_df['y'], mode='lines', name='Actual'))
        fig.update_layout(title='📦 Order Count Forecast', xaxis_title='Date', yaxis_title='Orders')
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("📊 Revenue Forecast with Confidence Intervals")

    fig_ci = go.Figure()
    fig_ci.add_trace(
        go.Scatter(x=forecast_revenue['ds'], y=forecast_revenue['yhat'], name='Forecast', line=dict(color='blue')))
    fig_ci.add_trace(go.Scatter(x=forecast_revenue['ds'], y=forecast_revenue['yhat_upper'], name='Upper Bound',
                                line=dict(dash='dot', color='lightblue')))
    fig_ci.add_trace(go.Scatter(x=forecast_revenue['ds'], y=forecast_revenue['yhat_lower'], name='Lower Bound',
                                line=dict(dash='dot', color='lightblue'),
                                fill='tonexty', fillcolor='rgba(173,216,230,0.2)'))
    fig_ci.update_layout(title="💡 Revenue Forecast with Confidence Intervals", xaxis_title="Date",
                         yaxis_title="Revenue")
    st.plotly_chart(fig_ci, use_container_width=True)

    # Forecast Accuracy Metrics
    st.markdown("---")
    st.subheader("📏 Forecast Accuracy Metrics")
    # Ensure we only evaluate metrics on historical period
    common_dates = revenue_df["ds"].isin(forecast_revenue["ds"])
    actual = revenue_df[common_dates].set_index("ds")
    predicted = forecast_revenue.set_index("ds").loc[actual.index]


    mae = mean_absolute_error(actual['y'], predicted['yhat'])
    rmse = np.sqrt(mean_squared_error(actual['y'], predicted['yhat']))
    mape = np.mean(np.abs((actual['y'] - predicted['yhat']) / actual['y'])) * 100

    st.metric("MAE (Mean Absolute Error)", f"{mae:,.0f}")
    st.metric("RMSE (Root Mean Squared Error)", f"{rmse:,.0f}")
    st.metric("MAPE (Mean Absolute Percentage Error)", f"{mape:.2f}%")

with tab4:
    st.subheader("📦 Category-wise Revenue Trends")
    categories = sorted(full_orders['product_category'].dropna().unique())
    selected_cat = st.selectbox("Select Product Category", categories)

    cat_df = full_orders[full_orders['product_category'] == selected_cat]
    cat_monthly = cat_df.groupby(cat_df['order_purchase_timestamp'].dt.to_period("M").astype(str))["price"].sum().reset_index()
    fig_cat = px.line(cat_monthly, x="order_purchase_timestamp", y="price", title=f"Revenue Trend: {selected_cat}",
                      labels={"order_purchase_timestamp": "Month", "price": "Revenue"})
    st.plotly_chart(fig_cat, use_container_width=True)

