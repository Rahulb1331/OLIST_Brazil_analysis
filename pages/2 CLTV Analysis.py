# Scripts/pages/cltv_page.py
import pandas as pd
import numpy as np
import plotly.express as px
from analysis.rfm import run_rfm_analysis
from analysis.cltv import run_cltv_analysis, enrich_cltv_with_segments, model_cltv_lifetimes
from sklearn.metrics import mean_absolute_error, mean_squared_error

# --- Page Config ---
import streamlit as st
st.set_page_config(page_title="Customer Lifetime Value (CLTV)", layout="wide")
st.title("💸 Customer Lifetime Value (CLTV) Analysis")

# --- Caching heavy processing functions ---
@st.cache_data
def load_full_orders():
    from analysis.Preprocessing import full_orders
    return full_orders.dropna(subset=['customer_city', 'customer_state', 'product_category'])

@st.cache_data
def get_rfm(full_orders):
    return run_rfm_analysis(full_orders)

@st.cache_data
def get_cltv(full_orders):
    return run_cltv_analysis(full_orders)

@st.cache_data
def get_enriched_cltv(cltv_df):
    return enrich_cltv_with_segments(cltv_df)

@st.cache_data
def get_lifetimes_model(full_orders):
    return model_cltv_lifetimes(full_orders)

@st.cache_data
def merge_rfm_cltv(rfm_df, cltv_df):
    return pd.merge(
        rfm_df,
        cltv_df[["customer_unique_id", "better_cltv", "cltv_normalized", "CLTV_new_Segment"]],
        on="customer_unique_id",
        how="inner"
    )

# --- Load data ---
with st.spinner("Loading and processing data..."):
    full_orders = load_full_orders()
    rfm_df = get_rfm(full_orders)
    cltv_raw = get_cltv(full_orders)
    cltv_df = get_enriched_cltv(cltv_raw)
    summary_df = get_lifetimes_model(full_orders)
    rfm_cltv_df = merge_rfm_cltv(rfm_df, cltv_df)

# --- 1. Introduction Section ---
st.markdown("""
Customer Lifetime Value (CLTV) helps estimate the total value a customer brings to your business over their entire relationship.

This page models future revenue using **BG/NBD + Gamma-Gamma** models.
""")

with st.expander("ℹ️ What is BG/NBD + Gamma-Gamma?"):
    st.markdown("""
- **BG/NBD (Beta-Geometric/Negative Binomial Distribution)** predicts the number of future transactions.
- **Gamma-Gamma** predicts the average revenue per transaction.
Together, they allow us to estimate a customer's **total future value**.

These models assume that purchase and monetary patterns follow specific probability distributions.
""")

# --- 2. Segment Explanation & Counts ---
st.subheader("🔍 CLTV Segments & Counts")
seg_counts = cltv_df['CLTV_new_Segment'].value_counts().reset_index()
seg_counts.columns = ['CLTV Segment', 'Customer Count']
st.table(seg_counts)
with st.expander("🧩 How segments were defined"):
    st.write(
        "Segments are based on the Pareto principle (80/20 rule) applied to predicted CLTV values:\n"
        "- **High CLTV**: Top 20% of customers contributing the most to revenue\n"
        "- **Medium CLTV**: Middle 60% of customers\n"
        "- **Low CLTV**: Bottom 20% of customers\n\n"
        "This segmentation helps prioritize high-value customers while still monitoring the broader base."
    )

# --- 5. CLTV Segments Explanation ---
st.subheader("🎯 CLTV Segments Defined")

st.dataframe(cltv_df.head(10))

# --- 3. Visualizations ---
st.subheader("📊 CLTV Distributions and Relationships")

if 'log_applied' not in st.session_state:
    st.session_state.log_applied = False

log_toggle = st.toggle("Apply Log Transformation", value=st.session_state.log_applied)
st.session_state.log_applied = log_toggle

cltv_pd = rfm_cltv_df[["cltv_normalized", "CLTV_new_Segment"]].copy()

if st.session_state.log_applied:
    cltv_pd['cltv_transformed'] = np.log1p(cltv_pd['cltv_normalized'] * 1000)
    title_suffix = "(Log Scale)"
else:
    cltv_pd['cltv_transformed'] = cltv_pd['cltv_normalized']
    title_suffix = "(Raw Scale)"


with st.expander("📦 Box Plot of CLTV"):
    fig = px.box(
        cltv_pd,
        x="CLTV_new_Segment",
        y="cltv_transformed",
        color="CLTV_new_Segment",
        title=f"Boxplot: CLTV Distribution by Segment {title_suffix}"
    )
    st.plotly_chart(fig, use_container_width=True)
    if st.checkbox("Show Graph Insights", key = "cltv_box"):
        st.info(
            """
            - **Improved Visibility Through Log Transformation:**  
              Applying a log transformation compresses the range of the highly skewed CLTV data. This makes differences across segments clearer by reducing the impact of extreme high values and helping to standardize the scale.

            - **Segment-Specific Observations:**  
              • **Low CLTV Segment:**  
                The transformed values for this group are concentrated at a lower range. Their tight distribution suggests that these customers have very low lifetime values.  
              • **Medium CLTV Segment:**  
                This segment shows a moderate, narrow spread around a middle range log value, indicating a relatively consistent but modest contribution from these customers.  
              • **High CLTV Segment:**  
                Even after transformation, this group exhibits a wider spread with several outliers above the typical range. The presence of these outliers warns that a small fraction of customers are driving significant value variability.

            - **Business Implications:**  
              The log scale helps in highlighting the subtle differences between segments that are obscured by raw monetary values. For strategic targeting:
                - Olist might explore tailored retention or upsell strategies for the High CLTV group, given their variability and occasional high contribution.  
                - Optimizing engagement in the Medium segment could help lift lower-performing customers without the distortion of extreme outliers.
                - The clear separation in distributions indicates that different segments might need different treatment in Olist's marketing and customer relationship management strategies.

        """
    )

with st.expander("Distribution Plots"):
    # Recency distribution
    fig_recency = px.histogram(summary_df, x="recency", nbins=50, marginal="rug", title="Recency Distribution", opacity=0.7)
    fig_recency.update_layout(bargap=0.1)
    st.plotly_chart(fig_recency, use_container_width=True)

    # T distribution
    fig_T = px.histogram(summary_df, x="T", nbins=50, marginal="rug", title="T Distribution", opacity=0.7)
    fig_T.update_layout(bargap=0.1)
    st.plotly_chart(fig_T, use_container_width=True)

    if st.checkbox("Get Insights", key="distribution"):
        st.info("""
    **Recency vs T in CLTV Modeling**

    - **Recency**: Time between a customer's **first and most recent** purchase.
      - Formula: `recency = last_purchase_date - first_purchase_date`
      - It shows **how recently** a customer has bought from you.

    - **T**: Time between the customer's **first purchase** and the **end of the analysis period**.
      - Formula: `T = current_date - first_purchase_date`
      - It represents **how long** the customer has been observed. 
      - The longer the `T`, the more time the customer has had to make repeat purchases, which is crucial for predicting future behavior.

    **Integrated Insights from Recency and T Distribution Histograms:**

    - **Understanding Customer Behavior over Time:**  
        The recency histogram reveals two distinct segments – one with very recent transactions (low recency) and another with customers inactive for nearly a year (high recency). In parallel, the T distribution histogram displays a heavy right skew with a prominent peak around T = 360, indicating that many customers have been observed for a long time. This juxtaposition suggests that while a significant portion of Olist's customer base is established (long T), only a subset of these actively transact.

    - **Implications for BG/NBD & Gamma Gamma Modeling:**  
      In the BG/NBD model, a low recency combined with a long observation period (high T) bolsters the confidence in predicting future transactions and enhancing CLTV. Customers who are both long-term (high T) and active (low recency) are valuable for targeted promotions and upsell strategies. Conversely, customers with high recency despite a long observation period may indicate potential churn, which could skew the Gamma Gamma model's revenue estimates if not adjusted for properly.

    - **Strategic Recommendations for Olist:**  
      - **Targeted Re-engagement:** Deploy campaigns to re-engage customers with high recency and long observation durations to awaken their dormant behavior.  
      - **Refined Customer Segmentation:** Differentiate between active, established customers and those with a prolonged inactive period even if they’ve been around for a while. Tailor loyalty programs and personalized offers accordingly.  

    Combining these insights provides a holistic understanding of customer lifetime dynamics, enabling more accurate CLTV predictions and more focused strategic initiatives.
    """)


with st.expander("🧬 RFM Vs CLTV Scatterplot"):
    scatter_df = rfm_cltv_df.copy()

    fig5 = px.scatter(
        scatter_df,
        x="RFM_Score",
        y="cltv_normalized",
        color="CLTV_new_Segment",
        title="RFM Score vs CLTV Normalized",
        size_max=10
    )
    st.plotly_chart(fig5, use_container_width=True)
    # --- Integrated Insights using st.info() ---
    if st.checkbox("Show Insights and Inferences", key="rfmvcltv"):
        st.info(
            """
            **Integrated Insights from RFM Score vs Normalized CLTV Analysis:**

            - **Customer Segmentation & Behavior:**  
              Customers with a Monetary (M) score of 4 are consistently classified in the High CLTV segment, even if their Recency (R) and Frequency (F) scores are low (ranging from 1 to 3). This classification results from the fact that I have used the Pareto Principle to categorize customers into High, Medium, or Low CLTV segments, since the top 20% of customers—who make the major contribution to revenue—typically have an M (Monetary) score of 4. Consequently, customers with an overall RFM score of 114 are classified as High Value in CLTV analysis.

            - **The Role of Normalized CLTV:**  
              Normalized CLTV is employed to standardize the raw CLTV values into a 0-to-1 range. This transformation is crucial as it:
                • **Facilitates Comparison:** Standardizing enables a direct comparison across diverse customer segments, irrespective of the inherent scale of raw monetary values.
                • **Highlights Patterns:** The normalized range helps in easily spotting trends, clusters, and outliers, which might have been obscured by large value disparities in raw data.
                • **Integrates with Predictive Models:** Using a normalized metric ensures that subsequent modeling or clustering algorithms are robust and not skewed by extreme values.

            - **Strategic Recommendations for Olist:**  
              • **Target High-Value Customers:** Focus on maintaining and expanding engagement with customers in the High CLTV segment, as their behaviors indicate strong loyalty and potential for additional upsell.  
              • **Boost Medium Segment Engagement:** For customers in the Medium CLTV segment, calculated interventions—such as personalized offers—could help shift them into the higher CLTV bracket.  
              • **Revisit Low CLTV Group:** Investigate the reasons behind the low CLTV scores among customers with higher RFM scores. They may need reactivation campaigns or adjustments in service delivery to enhance satisfaction and transaction frequency.

            By combining the RFM scoring with a normalized CLTV metric, Olist gains a clearer, more actionable overview of customer value dynamics, making it easier to tailor strategic initiatives and improve overall revenue management.
        """
    )

# --- 3. Model Performance Summary ---
st.subheader("📈 Model Performance & Evaluation")
# merge actual vs predicted CLTV for evaluation
eval_df = pd.merge(
    summary_df[['customer_unique_id', 'predicted_cltv']],
    cltv_df[['customer_unique_id', 'better_cltv']],
    on='customer_unique_id', how='inner'
)
mae = mean_absolute_error(eval_df['better_cltv'], eval_df['predicted_cltv'])
rmse = mean_squared_error(eval_df['better_cltv'], eval_df['predicted_cltv']) ** 0.5  # compute RMSE as sqrt(MSE)
with st.expander("Insights on Model Fit"):
    st.info(
        f"""
        ### 📊 BG/NBD & Gamma-Gamma Model Assumptions And Evaluation Metrics Result

        **🔁 BG/NBD (Purchase Frequency Model):**
        - Customers purchase at a constant but varied rate (Poisson process).
        - After each purchase, there's a constant probability the customer becomes inactive.
        - Purchase rates and dropout probabilities vary across customers (Gamma and Beta distributions).
        - Customer behavior is independent of others.

        **💰 Gamma-Gamma (Monetary Value Model):**
        - Transaction value is independent of purchase frequency.
        - Each customer has their own average spend, drawn from a Gamma distribution.
        - Individual transaction amounts are Gamma-distributed around each customer's average.
        
        Models assume customer dropout and spending patterns are stable over time.
        
        **Evaluation metrics:**
        - MAE of {mae:.2f} indicates average deviation of predictions from actual CLTV.
        - RMSE of {rmse:.2f} highlights occasional larger errors.
        - Periodic retraining and parameter tuning can reduce these errors.
        """
    )
with st.expander("Predicted CLTV using BG/NBD + Gamma-Gamma"):
    st.dataframe(summary_df.head(10))

# --- 4. Past vs Predicted Revenue ---
st.subheader("💵 Revenue Comparison: Past vs Predicted")

full_orders['order_purchase_timestamp'] = pd.to_datetime(full_orders['order_purchase_timestamp'])
full_orders['order_purchase_quarter'] = full_orders['order_purchase_timestamp'].dt.to_period('Q')

past_revenue = full_orders.groupby('customer_unique_id').agg(
    past_revenue=('payment_value', 'sum')
).reset_index()

summary_revenue = summary_df.merge(past_revenue, on='customer_unique_id', how='left')
summary_revenue['past_revenue'] = summary_revenue['past_revenue'].fillna(0)


col1, col2 = st.columns(2)

with col1:
    fig1 = px.bar(
        summary_revenue.groupby('cltv_segment')['past_revenue'].sum().reset_index(),
        x="cltv_segment",
        y="past_revenue",
        title="Past 12M Revenue by Segment",
        text_auto=".2s",
        color="cltv_segment"
    )
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    fig2 = px.bar(
        summary_revenue.groupby('cltv_segment')['predicted_cltv'].sum().reset_index(),
        x="cltv_segment",
        y="predicted_cltv",
        title="Predicted 12M Revenue by Segment",
        text_auto=".2s",
        color="cltv_segment"
    )
    st.plotly_chart(fig2, use_container_width=True)

st.info("""
ℹ️ **Note:**  
We can see that the revenue across all three segments in the past 12 month is coming to be greater than the the predicted 12 month revenue by segment. 

This is mainly because, for the previous 12 month we are collecting the **revenue of all the customers** even **one-time purchasers**.  

The **BG/NBD + Gamma-Gamma** model predicts future revenue **only from current customers** based on their past behavior. That is those customers who have made **more than 1 purchases** on the platform. 
It **does not account for new customer acquisitions**.

Thus, the predicted 12-month revenue appears lower than the past 12 months.

BG/NBD (Pareto/NBD) and Gamma-Gamma models we require at least two purchases to estimate recency/frequency parameters. Without a repeat purchase, we can’t infer their “decay rate” or expected future order rate.
""")

st.markdown("---")

# --- Top Products by CLTV Segment ---
st.subheader("🏆 Top Products by CLTV Segment")

top_n = st.slider("Select number of top products to show:", min_value=3, max_value=15, value=5)

# Merge product info with segment info
product_segment_df = pd.merge(
    full_orders[['customer_unique_id', 'product_category']],
    cltv_df[['customer_unique_id', 'CLTV_new_Segment']],
    on='customer_unique_id', how='inner'
)

# Count top products per segment
top_products = (
    product_segment_df
    .groupby(['CLTV_new_Segment', 'product_category'])
    .size()
    .reset_index(name='purchase_count')
)

# Sort and pick top N per segment
top_products = top_products.sort_values(['CLTV_new_Segment', 'purchase_count'], ascending=[True, False])
top_n_products_per_segment = top_products.groupby('CLTV_new_Segment').head(top_n)

# Plotting
fig_top_products = px.bar(
    top_n_products_per_segment,
    x='product_category',
    y='purchase_count',
    color='CLTV_new_Segment',
    barmode='group',
    facet_col='CLTV_new_Segment',
    title=f"Top {top_n} Products Bought by Each CLTV Segment",
    labels={'purchase_count': 'Number of Purchases', 'product_category': 'Product Category'},
    height=500
)
fig_top_products.update_layout(showlegend=False)
st.plotly_chart(fig_top_products, use_container_width=True)

st.info("Visualising the products popularity for each CLTV segment")

st.markdown("---")

# --- 5. Cohort Analysis ---
st.subheader("👥 Customer Cohort Analysis")

cohort_df = full_orders.copy()
cohort_df['first_purchase_quarter'] = cohort_df.groupby('customer_unique_id')['order_purchase_quarter'].transform('min')

cohort_summary = cohort_df.groupby('first_purchase_quarter').agg(
    customer_count=('customer_unique_id', 'nunique'),
    total_revenue=('payment_value', 'sum')
).reset_index()

cohort_summary['first_purchase_quarter'] = cohort_summary['first_purchase_quarter'].astype(str)

fig3 = px.line(
    cohort_summary,
    x="first_purchase_quarter",
    y=["customer_count", "total_revenue"],
    markers=True,
    title="Cohort Analysis: New Customers and Revenue per Quarter"
)
st.plotly_chart(fig3, use_container_width=True)

#
if st.checkbox("🔍 Show Cohort Chart Insights", key = "cohort_analysis"):
    st.info(
        """
        **Methodology followed**  
        - Here, summation is done of all lifetime revenue from every customer whose first purchase fell in Q1 (or Q2, etc.), not the revenue earned during that quarter. Every customer has been grouped by the quarter of *their first order*.  
        - For each cohort (the “first_purchase_quarter”), following have plotted:
          1. **customer_count** = number of unique new customers in that quarter  
          2. **total_revenue** = the *lifetime* revenue those new-customer cohorts have generated so far  

        **What this shows**  
        This is *not* “all sales in Q1/Q2/etc.” — it’s the total spend **ever** of only those customers who joined in each quarter.  
        """
    )


# --- 5. CLTV Growth by Cohort ---
st.subheader("📈 CLTV Growth Over Cohorts")
growth = rfm_cltv_df.copy()
growth = pd.merge(growth, full_orders[['customer_unique_id','order_purchase_timestamp']], on='customer_unique_id', how='left')
growth['first_quarter'] = pd.to_datetime(growth['order_purchase_timestamp']).dt.to_period('Q').astype(str)
grp = growth.groupby(['first_quarter','CLTV_new_Segment']).cltv_normalized.mean().reset_index()
fig3 = px.line(
    grp, x='first_quarter', y='cltv_normalized', color='CLTV_new_Segment',
    markers=True, title="Avg Normalized CLTV by Cohort"
)
st.plotly_chart(fig3, use_container_width=True)

if st.checkbox("Show CLTV Growth Insights", key="cohort"):
    st.info("""
    **Insights on the Graph:**

    - **Distinct Cohort Behavior:**  
      The graph clearly differentiates between customer cohorts. The High CLTV segment shows consistently higher normalized values compared to the Medium and Low segments. This suggests that even though the High-value segment may comprise fewer customers, they are driving a more significant portion of the overall value.

    - **Stability vs. Variability:**  
      While the Low and Medium cohorts exhibit stable trends over time—indicating a baseline level of customer value—the High cohort, despite minor fluctuations, remains markedly elevated. This hints at a dynamic purchasing pattern among the top customers, implying that strategies to nurture these relationships could yield high returns.

    - **Normalized vs. Actual CLTV:**  
      Using normalized CLTV is used here to compare the trends across cohorts, as it puts all values on a comparable scale. This is helpful to understand relative performance over time. 
      """)

# --- 6. Retention by Segment ---
st.subheader("🔄 Retention Rate by CLTV Segment")

# Identify whether a customer made more than one purchase
ret = full_orders.copy()
order_counts = ret.groupby('customer_unique_id').order_id.nunique().reset_index()
order_counts['repeat_flag'] = (order_counts['order_id'] > 1).astype(int)

# Merge with CLTV segments
temp = pd.merge(order_counts[['customer_unique_id', 'repeat_flag']], cltv_df[['customer_unique_id','CLTV_new_Segment']], on='customer_unique_id')

# Compute retention by segment
seg_rate = temp.groupby('CLTV_new_Segment').repeat_flag.mean().reset_index()
fig4 = px.bar(seg_rate, x='CLTV_new_Segment', y='repeat_flag',
    labels={'repeat_flag':'Repeat Purchase Rate'}, title="Repeat Rate by Segment")
st.plotly_chart(fig4, use_container_width=True)

if st.checkbox("Show Retention Insights", key="retention"):
    st.info(
        """
        All the three segments have a low retention rate, further validating the observation that most of the customers are only making one puurchase on the platform. However, if the three segments have to be compared with each other. We can make the following conlusions:
        
        - **Highest Retention in Medium CLTV:**  
          The graph shows that the Medium CLTV segment has the highest repeat purchase (retention) rate, slightly above 0.03. This indicates that customers in this group are the most likely to make repeat purchases, perhaps due to a balanced mix of engagement and spending behavior.
          
        - **Strong, Yet Lower, Retention in High CLTV:**  
        Although the High CLTV segment consists of customers with a high lifetime value, their repeat purchase rate is slightly lower—just under 0.03. This may suggest that while these customers contribute significant value when they purchase, they might do so less frequently.
          
        - **Lowest Retention in Low CLTV:**  
          The Low CLTV segment displays the lowest retention rate, about 0.02. This points to the possibility that customers in this segment are either less engaged or require different strategies to boost repeat activity.

        These insights highlight that while high-spending customers are valuable, the medium CLTV group demonstrates the best retention, suggesting targeted strategies for nurturing this cohort could potentially yield substantial long-term benefits.
        """
    )


# --- 7. User Actions by Segment ---
st.subheader("🚀 Actions Suggested by Customer Segment")

st.markdown("""
- **High Value:** Upselling/cross-selling
- **Medium Value:** Targeted promotions
- **Low Value:** Win-back campaigns, reactivation
""")

# --- 8. Table Download ---
st.subheader("📥 Download CLTV Data")

csv = cltv_df.to_csv(index=False).encode('utf-8')
st.download_button("Download CLTV Data as CSV", data=csv, file_name="cltv_analysis.csv", mime='text/csv')

# --- 9. Filtering Section ---
st.subheader("🔍 Filter Customers")

with st.expander("📂 Filter Options"):
    state_filter = st.multiselect("Select States", full_orders['customer_state'].unique())
    city_filter = st.multiselect("Select Cities", full_orders['customer_city'].unique())
    product_filter = st.multiselect("Select Product Categories", full_orders['product_category'].unique())

    if state_filter or city_filter or product_filter:
        filtered_orders = full_orders[
            (full_orders['customer_state'].isin(state_filter) if state_filter else True) &
            (full_orders['customer_city'].isin(city_filter) if city_filter else True) &
            (full_orders['product_category'].isin(product_filter) if product_filter else True)
        ]
        st.write(f"Filtered Orders: {len(filtered_orders)} transactions")
        st.dataframe(filtered_orders)
