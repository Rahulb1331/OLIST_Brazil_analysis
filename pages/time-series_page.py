import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from Scripts.config import setup_environment

setup_environment()
import streamlit as st
from pyspark.sql.functions import to_date, date_format, sum as _sum, countDistinct, month
from analysis.Preprocessing import full_orders
import pandas as pd
import calendar
from prophet import Prophet
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="📈 Time Series Dashboard", layout="wide")
st.title("📈 Time Series Analysis Dashboard")

# Preprocess Spark DataFrame
full_orders_with_month = full_orders.withColumn("order_month", date_format("order_purchase_timestamp", "yyyy-MM"))

# Aggregate Revenue & Orders
monthly_revenue = full_orders_with_month.groupBy("order_month").agg(
    _sum("price").alias("total_revenue")
).orderBy("order_month").filter("total_revenue > 100000")

monthly_orders = full_orders_with_month.groupBy("order_month").agg(
    countDistinct("order_id").alias("order_count")
).orderBy("order_month").filter("order_count > 500")

# Convert to Pandas
monthly_revenue_pd = monthly_revenue.toPandas()
monthly_orders_pd = monthly_orders.toPandas()

# Calendar month revenue & orders (seasonality)
season_df = full_orders.withColumn("month", month("order_purchase_timestamp"))
monthly_avg_revenue = season_df.groupBy("month").agg(_sum("price").alias("total_revenue")).orderBy("month")
monthly_avg_orders = season_df.groupBy("month").agg(countDistinct("order_id").alias("order_count")).orderBy("month")

monthly_avg_revenue_pd = monthly_avg_revenue.toPandas()
monthly_avg_orders_pd = monthly_avg_orders.toPandas()

monthly_avg_revenue_pd["month"] = monthly_avg_revenue_pd["month"].apply(lambda x: calendar.month_abbr[x])
monthly_avg_orders_pd["month"] = monthly_avg_orders_pd["month"].apply(lambda x: calendar.month_abbr[x])

# Tabs
tab1, tab2, tab3 = st.tabs(["📊 Historical Trends", "📆 Seasonality", "🔮 Forecasts"])

with tab1:
    st.subheader("Monthly Revenue and Order Trends")

    col1, col2 = st.columns(2)

    with col1:
        fig = px.line(
            monthly_revenue_pd,
            x="order_month",
            y="total_revenue",
            title="📈 Monthly Revenue",
            markers=True,
            labels={"order_month": "Month", "total_revenue": "Revenue"}
        )
        fig.update_layout(template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.line(
            monthly_orders_pd,
            x="order_month",
            y="order_count",
            title="🛒 Monthly Order Count",
            markers=True,
            labels={"order_month": "Month", "order_count": "Orders"}
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
