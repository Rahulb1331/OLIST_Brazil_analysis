import streamlit as st
import pandas as pd
import calendar
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
import plotly.express as px
from sklearn.metrics import mean_absolute_error, mean_squared_error
import plotly.graph_objects as go
import numpy as np

st.set_page_config(page_title="📈 Time Series Dashboard", layout="wide")
st.title("📈 Time Series Analysis Dashboard")

@st.cache_data
def load_data():
    from analysis.Preprocessing import full_orders
    from analysis.cltv import cltv_df
    return full_orders, cltv_df
full_orders, cltv_df = load_data()

# Preprocessing
df = full_orders.copy()
df["order_purchase_timestamp"] = pd.to_datetime(df["order_purchase_timestamp"])
df["order_month"] = df["order_purchase_timestamp"].dt.to_period("M").astype(str)
df["month"] = df["order_purchase_timestamp"].dt.month

# Merge CLTV Segments
df = df.merge(cltv_df[["customer_unique_id", "CLTV_new_Segment"]], on="customer_unique_id", how="left")

# Regional & Customer Type Segmentation
region_options = sorted(df['customer_state'].dropna().unique())
#customer_types = ['All', 'Repeat', 'New']
cltv_segments = df['CLTV_new_Segment'].dropna().unique().tolist()
cltv_segments.sort()

selected_region = st.sidebar.selectbox("Select Region (State)", ['All'] + region_options)
#selected_customer_type = st.sidebar.selectbox("Select Customer Type", customer_types)
selected_cltv_segment = st.sidebar.selectbox("Select CLTV Segment", ['All'] + cltv_segments)

if selected_region != 'All':
    df = df[df['customer_state'] == selected_region]

if selected_cltv_segment != 'All':
    df = df[df['CLTV_new_Segment'] == selected_cltv_segment]

#if selected_customer_type == 'New':
#    new_customers = df.groupby('customer_id')['order_purchase_timestamp'].min().reset_index()
#    new_customers['first_order_month'] = new_customers['order_purchase_timestamp'].dt.to_period("M").astype(str)
#    df = df.merge(new_customers[['customer_id', 'first_order_month']], on='customer_id')
#    df = df[df['order_month'] == df['first_order_month']]
#elif selected_customer_type == 'Repeat':
#    counts = df[''].value_counts()
#    repeat_customers = counts[counts > 1].index
#    df = df[df['customer_id'].isin(repeat_customers)]

# Aggregate Revenue & Orders
monthly_revenue_pd = df.groupby("order_month")["payment_value"].sum().reset_index(name="total_revenue")
monthly_orders_pd = df.groupby("order_month")["order_id"].nunique().reset_index(name="order_count")

monthly_revenue_pd["rolling_3mo"] = monthly_revenue_pd["total_revenue"].rolling(3).mean()
monthly_orders_pd["rolling_3mo"] = monthly_orders_pd["order_count"].rolling(3).mean()

monthly_avg_revenue_pd = df.groupby("month")["payment_value"].sum().reset_index(name="total_revenue")
monthly_avg_orders_pd = df.groupby("month")["order_id"].nunique().reset_index(name="order_count")

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

    if st.checkbox("Show Analysis & Recommendations - Historical Trends", key="unique_key_ts1"):
        st.info("""
        **Analysis Performed:**
        - Aggregated total revenue and order count per month.
        - Added a 3-month rolling average to smooth short-term fluctuations.

        **Purpose:**
        - Identify growth patterns or dips over time.
        - Rolling average helps highlight broader trends without short-term noise.

        **Recommendations:**
        - A consistent upward trend can be observed, and assuming that the sudden drop in both the orders and revenue after Aug 2018 is due to the lack of data, Olist can consider scaling their operations.
        - The peak operations can be observed for the months of Nov-Jan, even though it can't be confirmed whether it is due to the festive season or due to Olist scaling up as the data for this season is available only once. But since the Olist operations have increased in the subsequent months, it would be more suitable to credit the scaling operations rather than seasonality.
        """)

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

    if st.checkbox("Show Analysis & Recommendations - Seasonality", key="unique_key_ts2"):
        st.info("""
        **Analysis Performed:**
        - Grouped and visualized revenue and orders by calendar month.
        
        **Purpose:**
        - To detect seasonal patterns. For example, which months consistently perform better.

        **Recommendations:**
        - Festive Months (e.g. November/December) could be targeted for promotions, even though the low revenue and order count can be attributed to the lack of data for these months.
        - Slow months can be targeted with retention campaigns or sales events.
        """)

with tab3:
    st.subheader("🔮 Prophet Forecasts for Revenue & Orders")

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
        fig.update_layout(title='📈 Revenue Forecast', xaxis_title='Date', yaxis_title='Revenue')
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=forecast_ord['ds'], y=forecast_ord['yhat'], mode='lines', name='Forecast'))
        fig.add_trace(go.Scatter(x=ord_df['ds'], y=ord_df['y'], mode='lines', name='Actual'))
        fig.update_layout(title='📦 Order Count Forecast', xaxis_title='Date', yaxis_title='Orders')
        st.plotly_chart(fig, use_container_width=True)
    
    if st.checkbox("Show Forecast Insights", key="unique_key_ts3"):
        st.info("""
        **Analysis Performed:**
        - Used Prophet model to forecast revenue and orders for the next 6 months.
        - Visualized actual vs. predicted values with confidence intervals.
        - Included model performance metrics (MAE, RMSE, MAPE).

        **Purpose:**
        - Help predict future demand and guide business planning.

        **Recommendations:**
        - The forecasts show a sharp peak in the month of Nov, and declining heavily by Feb, indicating that seasonality may be attributed to the rise of sales in the months of Nov-Jan, and the sellers on the Olist platform may plan for inventory and logistics scaling.
        - In the off-season the sellers may try launching marketing campaigns or optimizing product offerings.
        """)
        
    st.markdown("---")
    st.subheader("📊 Revenue Forecast with Confidence Intervals")

    fig_ci = go.Figure()
    fig_ci.add_trace(go.Scatter(x=forecast_rev['ds'], y=forecast_rev['yhat'], name='Forecast', line=dict(color='blue')))
    fig_ci.add_trace(go.Scatter(x=forecast_rev['ds'], y=forecast_rev['yhat_upper'], name='Upper Bound', line=dict(dash='dot', color='lightblue')))
    fig_ci.add_trace(go.Scatter(x=forecast_rev['ds'], y=forecast_rev['yhat_lower'], name='Lower Bound', line=dict(dash='dot', color='lightblue'), fill='tonexty', fillcolor='rgba(173,216,230,0.2)'))
    fig_ci.update_layout(title="💡 Revenue Forecast with Confidence Intervals", xaxis_title="Date", yaxis_title="Revenue")
    st.plotly_chart(fig_ci, use_container_width=True)

    # Interactive checkbox to display insights below the graph
    if st.checkbox("Show Revenue Forecast Insights", key ="confidennce_interval"):
        st.info(
            """
            **Revenue Forecast Analysis & Strategic Insights for Olist:**

            - **Forecast Trend Analysis:**  
              The solid blue line represents the predicted revenue trajectory from January 2017 to January 2019. This trend line helps in visualizing the expected growth pattern. An upward trend could signal effective past strategies that might be further capitalized upon, while a downward or flat trend suggests a need for new measures.

            - **Understanding Confidence Intervals:**  
              The dotted lines (upper and lower bounds) alongside the filled area denote the confidence interval around the forecast. Narrow intervals generally indicate higher certainty in the prediction, whereas wider intervals signal increased uncertainty, highlighting periods where external factors or market volatility might have a stronger impact.

            - **Seasonal and Cyclical Patterns Identification:**  
              If there are recurring patterns within the forecast (e.g., periodic peaks or troughs), these may correspond to seasonal trends, festive seasons, or promotional periods. Recognizing such cycles allows Olist to adjust marketing efforts or operational resource allocation proactively.

            **Actionable Recommendations:**
    
            1. **Enhance Marketing Initiatives:**  
               Capitalize on periods with a strong upward forecast by deploying targeted marketing campaigns and upselling strategies to further boost revenue.

            2. **Mitigate Uncertainty During High Variability:**  
               In phases where the confidence interval widens, investigate potential external factors influencing the revenue. This is a good opportunity to review competitive activity or sudden market changes and adjust strategies accordingly.

            3. **Operational Readiness:**  
               Align inventory and operational planning with the forecasted revenue trends. During predicted peak revenue periods, ensure that supply chains and customer service are well-prepared to handle the increased demand.

            4. **Continuous Model Improvement:**  
               Regularly update and refine the forecast model with additional data and identify factors not previously included (such as promotional events or economic indicators) to improve forecasting accuracy.

            These insights provide a strategic roadmap, enabling Olist to synchronize marketing, operations, and sales initiatives with the anticipated revenue trends, thereby fostering a proactive approach to business management.
            """
        )
    
    st.markdown("---")
    st.subheader("📏 Forecast Accuracy Metrics")

    common_dates = rev_df["ds"].isin(forecast_rev["ds"])
    actual = rev_df[common_dates].set_index("ds")
    predicted = forecast_rev.set_index("ds").loc[actual.index]

    mae = mean_absolute_error(actual['y'], predicted['yhat'])
    rmse = np.sqrt(mean_squared_error(actual['y'], predicted['yhat']))
    mape = np.mean(np.abs((actual['y'] - predicted['yhat']) / actual['y'])) * 100

    st.metric("MAE (Mean Absolute Error)", f"{mae:,.0f}")
    st.metric("RMSE (Root Mean Squared Error)", f"{rmse:,.0f}")
    st.metric("MAPE (Mean Absolute Percentage Error)", f"{mape:.2f}%")

    # An interactive checkbox to show insights based on these metrics
    if st.checkbox("Show Forecast Accuracy Insights", key = "metrics_insight):
        st.info(
            """
            **Interpreting the Forecast Accuracy Metrics:**

            - **MAE (222,806):**  
              This metric represents the average absolute error between the forecasted and actual revenue. In isolation, it gives a general sense of error magnitude. However, its value must be contextualized by the overall revenue scale to assess if this error is acceptable.

            - **RMSE (280,451):**  
              Being higher than MAE, the RMSE reflects that there are instances with large deviations. The fact that RMSE is significantly higher suggests that the model occasionally over- or under-predicts by a sizable margin, pointing to some large forecasting errors.

            - **MAPE (23725.54%):**  
              This exceptionally high percentage error typically indicates that the model has encountered instances where the actual values are very low (or near zero) or that outliers are skewing the metric. It suggests caution when interpreting percentage errors in this context and might warrant complementary error metrics or transformations.

            **Strategic Insights & Recommendations for Olist:**
    
            1. **Investigate Outliers and Low-Volume Periods:**  
               A soaring MAPE can be a sign of revenue values that are very small or erratic, which dramatically inflate percentage errors. Analyzing these periods could help identify if special events, data quality issues, or seasonal effects are at play.

            2. **Model Refinement:**  
               The discrepancy between MAE and RMSE indicates that while the model performs reasonably on average, some high-error instances adversely affect the performance. Consider either re-calibrating the model or introducing additional predictors (such as market signals or promotional events) to better capture these fluctuations.

            3. **Enhanced Error Evaluation:**  
               Given the skewed MAPE results, using additional or alternative error measurements—such as symmetric MAPE or incorporating revenue scaling—might provide a more balanced view of forecast performance.

            4. **Operational Preparedness:**  
               Use these error insights as a cue for operational planning. Periods with higher forecast variability may require improved contingency measures such as flexible resource allocation or proactive inventory adjustments.

            By delving into these metrics and refining the forecasting model, Olist can enhance the reliability of future predictions and better drive decision-making in marketing, operational planning, and overall business strategy.
            """
        )

with tab4:
    st.subheader("📦 Category-wise Revenue Trends")
    categories = sorted(df['product_category'].dropna().unique())
    selected_cat = st.selectbox("Select Product Category", categories)


    cat_df = df[df['product_category'] == selected_cat]
    cat_monthly = cat_df.groupby(cat_df['order_purchase_timestamp'].dt.to_period("M").astype(str))["payment_value"].sum().reset_index()

    fig_cat = px.line(cat_monthly, x="order_purchase_timestamp", y="payment_value", title=f"Revenue Trend: {selected_cat}",
                      labels={"order_purchase_timestamp": "Month", "payment_value": "Revenue"})
    st.plotly_chart(fig_cat, use_container_width=True)

    if st.checkbox("Show Analysis & Recommendations - Category Trends", key="unique_key_ts4"):
        st.info("""
        **Analysis Performed:**
        - Summed revenue per product category by month.

        **Purpose:**
        - Understand product performance trends over time.

        **Recommendations:**
        - Since for almost all product categories maximum revenue and sales be seen in the month of Nov, the sellers may plan for inventory and logistics scaling, especially for their most profitable products. They shall focus on marketing or inventory efforts on top-performing categories.
        - Monitor underperforming categories to adjust strategy or pricing, for example, the products in the agro_industry_and_commerce category don't see much revenue across the year except for Nov.
        """)
