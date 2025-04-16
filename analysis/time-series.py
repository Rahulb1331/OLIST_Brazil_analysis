from pyspark.sql.functions import to_date, date_format, sum as _sum, countDistinct
import plotly.express as px
from Preprocessing import full_orders
import pandas as pd


# LOOK AT THE TREND OVER TIME
# Add a column for order month
full_orders_with_month = full_orders.withColumn(
    "order_month", date_format("order_purchase_timestamp", "yyyy-MM")
)

# Revenue per month
monthly_revenue = full_orders_with_month.groupBy("order_month").agg(
    _sum("price").alias("total_revenue")
).orderBy("order_month")

monthly_revenue = monthly_revenue[monthly_revenue["total_revenue"]>100000]

# Orders per month
monthly_orders = full_orders_with_month.groupBy("order_month").agg(
    countDistinct("order_id").alias("order_count")
).orderBy("order_month")

monthly_orders = monthly_orders[monthly_orders["order_count"]>500]

# Convert to Pandas for Plotly
monthly_revenue_pd = monthly_revenue.toPandas()
monthly_orders_pd = monthly_orders.toPandas()

# Revenue Chart
fig_revenue = px.line(
    monthly_revenue_pd,
    x="order_month",
    y="total_revenue",
    title="Monthly Revenue",
    labels={"order_month": "Month", "total_revenue": "Revenue"},
    markers=True
)
fig_revenue.update_layout(template="plotly_white")
fig_revenue.show()

# Orders Chart
fig_orders = px.line(
    monthly_orders_pd,
    x="order_month",
    y="order_count",
    title="Monthly Order Count",
    labels={"order_month": "Month", "order_count": "Number of Orders"},
    markers=True
)
fig_orders.update_layout(template="plotly_white")
fig_orders.show()


# monthly buying patterns or seasonality

from pyspark.sql.functions import month

# Extract just the month number (1–12)
full_orders_with_month = full_orders.withColumn("month", month("order_purchase_timestamp"))

# Revenue by calendar month
monthly_avg_revenue = full_orders_with_month.groupBy("month").agg(
    _sum("price").alias("total_revenue")
).orderBy("month")

# Orders by calendar month
monthly_avg_orders = full_orders_with_month.groupBy("month").agg(
    countDistinct("order_id").alias("order_count")
).orderBy("month")

# Convert to Pandas
monthly_avg_revenue_pd = monthly_avg_revenue.toPandas()
monthly_avg_orders_pd = monthly_avg_orders.toPandas()

# Month names for better readability
import calendar
monthly_avg_revenue_pd["month"] = monthly_avg_revenue_pd["month"].apply(lambda x: calendar.month_abbr[x])
monthly_avg_orders_pd["month"] = monthly_avg_orders_pd["month"].apply(lambda x: calendar.month_abbr[x])

fig_rev = px.bar(
    monthly_avg_revenue_pd,
    x="month",
    y="total_revenue",
    title="Average Revenue by Calendar Month",
    labels={"total_revenue": "Revenue"},
    template="plotly_white"
)

fig_ord = px.bar(
    monthly_avg_orders_pd,
    x="month",
    y="order_count",
    title="Average Order Count by Calendar Month",
    labels={"order_count": "Orders"},
    template="plotly_white"
)

fig_rev.show()
fig_ord.show()


# FORECASTING
# Example
# monthly_revenue_df: ['month', 'revenue']
# monthly_orders_df: ['month', 'order_count']

print(monthly_revenue_pd.columns)
print(monthly_orders_pd.columns)

# Rename to match Prophet format
revenue_df = monthly_revenue_pd.rename(columns={"order_month": "ds", "total_revenue": "y"})
orders_df = monthly_orders_pd.rename(columns={"order_month": "ds", "order_count": "y"})

# Ensure datetime type
revenue_df['ds'] = pd.to_datetime(revenue_df['ds'])
orders_df['ds'] = pd.to_datetime(orders_df['ds'])

#Fitting the models
from prophet import Prophet

# Initialize and fit
revenue_model = Prophet()
revenue_model.fit(revenue_df)

orders_model = Prophet()
orders_model.fit(orders_df)

# Forecasting for next six months
# Choose forecast horizon
periods = 6

# Generate future dates
future_revenue = revenue_model.make_future_dataframe(periods=periods, freq='ME')
future_orders = orders_model.make_future_dataframe(periods=periods, freq='ME')

# Predict
forecast_revenue = revenue_model.predict(future_revenue)
forecast_orders = orders_model.predict(future_orders)

# Plotting
revenue_model.plot(forecast_revenue)
orders_model.plot(forecast_orders)

import plotly.graph_objects as go

# Revenue
fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=forecast_revenue['ds'], y=forecast_revenue['yhat'], mode='lines', name='Forecast'))
fig1.add_trace(go.Scatter(x=revenue_df['ds'], y=revenue_df['y'], mode='lines', name='Actual'))
fig1.update_layout(title='Revenue Forecast', xaxis_title='Date', yaxis_title='Revenue')

# Orders
fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=forecast_orders['ds'], y=forecast_orders['yhat'], mode='lines', name='Forecast'))
fig2.add_trace(go.Scatter(x=orders_df['ds'], y=orders_df['y'], mode='lines', name='Actual'))
fig2.update_layout(title='Order Count Forecast', xaxis_title='Date', yaxis_title='Orders')

fig1.show()
fig2.show()


# Revenue Forecast with Confidence Intervals
model = Prophet()
model.fit(revenue_df)

# Make future dataframe and forecast
future = model.make_future_dataframe(periods=6, freq='ME')  # adjust as needed
forecast = model.predict(future)

# Plotting forecast with confidence intervals
fig = go.Figure()

# Main forecast line
fig.add_trace(go.Scatter(
    x=forecast['ds'],
    y=forecast['yhat'],
    mode='lines',
    name='Forecasted Revenue',
    line=dict(color='blue')
))

# Upper bound
fig.add_trace(go.Scatter(
    x=forecast['ds'],
    y=forecast['yhat_upper'],
    mode='lines',
    name='Upper Confidence',
    line=dict(dash='dot', color='lightblue'),
    showlegend=True
))

# Lower bound
fig.add_trace(go.Scatter(
    x=forecast['ds'],
    y=forecast['yhat_lower'],
    mode='lines',
    name='Lower Confidence',
    line=dict(dash='dot', color='lightblue'),
    fill='tonexty',
    fillcolor='rgba(173, 216, 230, 0.2)',  # light blue fill
    showlegend=True
))

fig.update_layout(
    title="Monthly Revenue Forecast with Confidence Intervals",
    xaxis_title="Date",
    yaxis_title="Revenue",
    template="plotly_white"
)

fig.show()