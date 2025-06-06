# mba_app.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
from mlxtend.frequent_patterns import apriori, association_rules, fpgrowth
from sklearn.preprocessing import MultiLabelBinarizer

# Setup
st.set_page_config(page_title="Market Basket Analysis", layout="wide")
st.title("🛒 Market Basket Analysis (MBA)")

@st.cache_data
def load_data():
    from analysis.Preprocessing import full_orders
    from analysis.cltv import cltv_df
    return full_orders, cltv_df

full_orders, cltv_df = load_data()

# Data Preparation
transactions_df = full_orders.groupby(['order_id', 'customer_unique_id'])['product_category'] \
    .apply(set).reset_index().rename(columns={'product_category': 'items'})

segmented_txns = pd.merge(
    transactions_df,
    cltv_df[['customer_unique_id', 'CLTV_new_Segment']],
    on='customer_unique_id',
    how='inner'
)

segments = segmented_txns['CLTV_new_Segment'].dropna().unique()
selected_segment = st.selectbox("Select CLTV Segment", sorted(segments))

# Filter for multi-item transactions
segment_df = segmented_txns[segmented_txns['CLTV_new_Segment'] == selected_segment]
multi_item_txns = segment_df[segment_df['items'].apply(lambda x: len(x) > 1)]

if not multi_item_txns.empty:

    # Clean Items
    cleaned_items = multi_item_txns['items'].apply(
        lambda items: sorted([str(i).strip().lower() for i in items if pd.notnull(i)])
    )

    # One-hot encoding
    mlb = MultiLabelBinarizer()
    itemsets = pd.DataFrame(
        mlb.fit_transform(cleaned_items),
        columns=mlb.classes_,
        index=multi_item_txns.index
    ).astype(bool)

    # Mining frequent itemsets
    frequent_itemsets = fpgrowth(itemsets, min_support=0.001, use_colnames=True)

    # Generate Association Rules
    rules_df = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)
    rules_df = rules_df.sort_values("lift", ascending=False)

    if not rules_df.empty:

        rules_df["rule"] = rules_df["antecedents"].apply(lambda x: ', '.join(x)) + " → " + \
                           rules_df["consequents"].apply(lambda x: ', '.join(x))

        # Revenue Mapping
        order_revenue = full_orders.groupby('order_id')['payment_value'].sum().to_dict()

        # Estimated Revenue per Rule
        rules_df['estimated_revenue'] = rules_df['antecedents'].apply(
            lambda antecedents: sum([
                order_revenue.get(order_id, 0)
                for order_id, items in zip(multi_item_txns['order_id'], multi_item_txns['items'])
                if antecedents.issubset(items)
            ])
        )

        rules_df = rules_df.sort_values(["estimated_revenue", "lift"], ascending=[False, False])

        # --- Insights Section ---
        with st.expander("Insights Behind This Analysis"):
            st.info(
                """
                **What was done:**  
                - Applied **FP-Growth** to discover frequent product combinations.
                - Used **Association Rules** to find cross-sell opportunities.
                - Estimated **revenue potential** per rule.
                - Filtered **only meaningful rules** based on lift > 1.
                - Used a low min_support (0.1%) to find hidden gems.

                """
            )

        # --- Scatterplot: Confidence vs Support (Bias Check) ---
        st.subheader(f"Confidence vs Support Analysis ({selected_segment})")
        fig_bias = px.scatter(
            rules_df, x='support', y='confidence',
            size='lift', color='lift',
            hover_name='rule', title='Support vs Confidence Bias Detection',
            labels={"support": "Support", "confidence": "Confidence"}
        )
        st.plotly_chart(fig_bias, use_container_width=True)

        with st.expander("Show Insights"):
            st.info("""
                **Support:**  
            - The proportion of transactions that contain a specific itemset.  
            - Example: If 5 out of 100 transactions include "milk and bread", support = 5%.

            **Confidence:**  
            - How often items in the consequent (e.g., "bread") appear in transactions that contain the antecedent (e.g., "milk").  
            - Example: If 80% of people who buy milk also buy bread, confidence = 80%.

            **Lift:**  
            - How much more likely items are to be bought together compared to being bought independently.  
            - A lift > 1 indicates a positive association. Higher lift = stronger buying relationship.

            In short:
            - **Support** shows **how popular** a combination is.
            - **Confidence** shows **how reliable** the rule is.
            - **Lift** shows **how much stronger** the buying pattern is than random chance.

            **Why Bias Matters?**
            - Rules with **high confidence but very low support** are often misleading.
            - They may look strong but occur rarely, making them risky to act on.
            - So we have to **check both confidence AND support** before trusting a rule!
            """)

        # --- Scatterplot ---
        st.subheader(f"📈 Association Rules Scatter Plot for {selected_segment}")
        fig1 = px.scatter(
            rules_df,
            x="support",
            y="confidence",
            size="lift",
            color="estimated_revenue",
            hover_name="rule",
            title=f"Rules: Support vs Confidence vs Revenue ({selected_segment})",
            labels={"support": "Support", "confidence": "Confidence", "estimated_revenue": "Revenue"}
        )
        st.plotly_chart(fig1, use_container_width=True)

        # --- Combined Insights in an Expander ---
        with st.expander("Show Combined Market Basket Analysis Association Rules Insights"):
            st.info(
                """
            **Combined Market Basket Analysis Insights:**

            - **Association Rule Dynamics:**  
              Across all CLTV segments, the scatter plots highlight that while the support and confidence for association rules tend to be low, even modest changes in these values are associated with significant differences in revenue.  
        
            - **High CLTV Segment:**  
              In this segment, certain rules (for example, "furniture_decor -> furniture_bedroom") exhibit very high revenue despite having low support and confidence. This suggests that high-value purchases, though infrequent, have an outsized influence on the overall revenue, indicating that premium bundling strategies or targeted upselling for select products could yield significant returns.
        
            - **Medium CLTV Segment:**  
              The rules here generate moderate revenue levels. This indicates a balanced purchasing behavior where customers occasionally commit to associations that still contribute substantially to revenue. This segment could benefit from tailored cross-selling initiatives aimed at moving moderate purchases into higher value transactions.
        
            - **Low CLTV Segment:**  
              Although the rules in this segment (for example, "housewares -> bed_bath_table") tend to occur more frequently—as seen by relatively higher support—the revenue impact remains low. This points toward a volume-centric buying behavior where frequent, yet lower-valued, purchases predominate. Strategies here might focus on nudging customers towards higher-value add-ons or complementary products, potentially lifting the average transaction revenue.
        
            - **Strategic Implications:**  
              These insights collectively suggest that Olist can optimize marketing and bundling strategies by treating each CLTV segment distinctly. While high CLTV customers might be best targeted through premium, high-margin offers, medium CLTV customers could be encouraged to explore complementary products, and low CLTV segments might yield benefit from tactics that increase the average order value through frequent but enhanced cross-sell opportunities.                
            """
                )

        
        # --- Dead-End Products Section ---
        st.subheader("Dead-End Products (No Follow-up Sales)")
        dead_ends = set()
        for a, c in zip(rules_df['antecedents'], rules_df['consequents']):
            dead_ends.update(c)
        all_products = set(mlb.classes_)
        products_with_no_consequents = sorted(all_products - dead_ends)

        st.write(f"🧩 {len(products_with_no_consequents)} products found that **don't lead to other purchases**.")
        if products_with_no_consequents:
            st.dataframe(pd.DataFrame(products_with_no_consequents, columns=["Dead-End Product"]), use_container_width=True)

        with st.expander("Why Look at Dead-Ends?"):
            st.info("""
                - Products that don't lead to cross-sales are **low leverage**.
                - You may want to:
                    - Deprioritize them in promotions.
                    - Bundle them with higher-impact products.
            """)

        # --- Network Graph ---
        st.subheader("Network Graph of Association Rules")
        G = nx.DiGraph()
        for _, row in rules_df.iterrows():
            G.add_edge(', '.join(row["antecedents"]), ', '.join(row["consequents"]), weight=row["lift"])

        pos = nx.spring_layout(G, seed=42)
        edge_x, edge_y, node_x, node_y = [], [], [], []

        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)

        edge_trace = go.Scatter(x=edge_x, y=edge_y, mode='lines', line=dict(width=1, color='#888'), hoverinfo='none')
        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode='markers+text',
            text=list(G.nodes()),
            hoverinfo='text',
            marker=dict(size=20, color='skyblue'),
            textposition='top center'
        )

        fig2 = go.Figure(data=[edge_trace, node_trace],
                         layout=go.Layout(title='Association Rules Network', showlegend=False,
                                          margin=dict(b=20, l=5, r=5, t=40)))
        st.plotly_chart(fig2, use_container_width=True)

        st.info("""
        - Nodes = Products, Arrows = Rules.
        - Hubs show **good cross-sell opportunities**.
        - Isolated nodes could be **dead-ends**.
            """)
        with st.expander("Show Insights about the Network Graph"):
            # Integrated Insights: Display only when the checkbox is checked
            st.info(
            """
                **Integrated Insights from Market Basket Analysis Across CLTV Segments:**

                - **High CLTV Segment:**  
                    The network graph for high CLTV customers shows strong, well-defined clusters of product associations. This indicates that these customers tend to make deliberate and strategic purchases with specific high-value products frequently bought together. Such tight-knit patterns suggest clear buying habits and loyalty, making these customers prime candidates for premium bundle offerings and targeted cross-selling.
    
                - **Medium CLTV Segment:**  
                    In the medium CLTV segment, the network graph reveals moderately dense connections among product nodes. This pattern suggests a blend of planned and opportunistic buying behaviors. Although these customers exhibit some product associations, there’s also an element of variability that provides an opportunity to enhance basket sizes with tailored promotions or dynamic product bundles.

                - **Low CLTV Segment:**  
                    The network graph for low CLTV customers is noticeably more dispersed, with fewer and less intense product connections. This dispersion points to more sporadic purchasing habits, where customers are less likely to buy complementary products together. Such insights reveal an opportunity to encourage cross-category purchasing through targeted recommendations or promotional strategies.

                **Overall Observations and Strategic Implications:**  
                The contrast in connectivity across the networks highlights distinct consumer behaviors:
                    • **Tight clusters in High CLTV** indicate predictable, high-intent buying, supporting strategies that reinforce existing popular bundles.
                    • **Moderate connections in Medium CLTV** suggest room to nurture customer behavior, potentially elevating their future lifetime value with interactive promotions.
                    • **Loose patterns in Low CLTV** offer an opportunity to design interventions aimed at stimulating complementary purchases to enhance overall basket value.

                By understanding these inter-product relationships across different customer segments, Olist can tailor marketing strategies—ranging from bespoke cross-sell opportunities to curated product bundles—that engage customers effectively and drive revenue growth.
            """
            )
        # --- Top Revenue-Generating Bundles ---
        st.subheader("Top 10 Revenue-Generating Product Bundles")
        top_bundles = rules_df.sort_values('estimated_revenue', ascending=False).head(10)
        st.dataframe(top_bundles.reset_index(drop=True)[['rule', 'antecedent support', 'consequent support', 'support', 'confidence', 'lift', 'estimated_revenue']], use_container_width=True)


        # --- Strategic Filters (Tactical Sliders) ---
        st.subheader("Tactical Filters for Association Rules")
        col1, col2, col3 = st.columns(3)

        with col1:
            min_conf = st.slider("Min Confidence", 0.0, 1.0, 0.1, 0.01)
        with col2:
            min_lift = st.slider("Min Lift", 0.0, float(rules_df['lift'].max()), 1.2, 0.1)
        with col3:
            min_support = st.slider("Min Support", 0.0, float(rules_df['support'].max()), 0.01, 0.001)

        filtered_rules = rules_df[
            (rules_df['confidence'] >= min_conf) &
            (rules_df['lift'] >= min_lift) &
            (rules_df['support'] >= min_support)
        ]

        st.dataframe(filtered_rules[['rule', 'support', 'confidence', 'lift', 'estimated_revenue']].reset_index(drop=True), use_container_width=True)

        # --- Conversion / Recommendation Impact ---
        st.subheader("Predicted Impact if Bundled")
        filtered_rules['predicted_uplift_revenue'] = filtered_rules['estimated_revenue'] * (filtered_rules['lift'] - 1)
        top_uplift = filtered_rules.sort_values('predicted_uplift_revenue', ascending=False) #.head(10)

        st.dataframe(top_uplift[['rule', 'lift', 'estimated_revenue', 'predicted_uplift_revenue']], use_container_width=True)

        with st.expander("How to Interpret This?"):
            st.info("""
                **Top Revenue Bundles** show which combinations of products bring the most sales value.
                
                **Action Points:**
                    - Olist can bundle these products together in special offers.
                    - They may also promote them in upsell/cross-sell campaigns to high CLTV segments.
                    - Prioritize stocking and marketing these bundles.
                    
                - Predicted uplift shows **extra revenue** if we bundle and successfully cross-sell.
                
                - Higher is the uplift better is the bundling opportunity.
            """)

        with st.expander("How Was Bundle Revenue Calculated?"):
            st.info(
                """
            **How We Predict Bundle Revenue:**
    
            - We first calculate the total historical revenue where the antecedent products were purchased together.
            - Then we adjust it by the **strength** of the relationship between the products using **Lift**.

            **What is Lift?**
            - Lift measures how much more likely two products are bought together compared to random chance.
            - If Lift = 1 ➔ no relationship (pure chance).
            - If Lift > 1 ➔ positive association (buying A makes buying B more likely).

            **Why use (Lift - 1)?**
            - **Lift - 1** captures only the *extra power* of the association beyond random chance.
            - Without adjusting, we'd wrongly assume all sales were due to the association (which is not true).

            **Final Calculation:**
            ```
            Adjusted Estimated Revenue ≈ Base Revenue × (Lift - 1)
            ```

            **Example:**
            - Base revenue for a bundle = $10,000
            - Lift = 1.5
            - Extra revenue = $10,000 × (1.5 - 1) = $5,000 additional sales attributable to the bundling effect.

            **In Simple Terms:**  
            ➔ We are isolating the true "bonus" effect of bundling, not just what would happen randomly.

            """
        )

        # --- "So What?" Actionable Insights ---
        st.subheader("What can Olist DO?")
        with st.expander("Action Plan Based on Findings"):
            st.success("""
                🎯 **Action Recommendations:**
                - Build bundles around top high-lift rules.
                - Avoid dead-end products unless bundled with popular hubs.
                - Promote bundles more aggressively to high-CLTV segments.
                - Use "Top Revenue" bundles in targeted marketing.
                - Monitor conversion rates post-recommendation for continuous learning.
            """)

        st.markdown("---")

else:
    st.warning("Not enough multi-item transactions in this segment.")
