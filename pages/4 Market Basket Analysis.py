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
    from analysis.cltv import summary
    return full_orders, summary

full_orders, summary = load_data()

# Data Preparation
transactions_df = full_orders.groupby(['order_id', 'customer_unique_id'])['product_category'] \
    .apply(set).reset_index().rename(columns={'product_category': 'items'})

segmented_txns = pd.merge(
    transactions_df,
    summary[['customer_unique_id', 'cltv_segment']],
    on='customer_unique_id',
    how='inner'
)

segments = segmented_txns['cltv_segment'].dropna().unique()
selected_segment = st.selectbox("Select CLTV Segment", sorted(segments))

# User Algorithm Choice
algo_choice = st.radio(
    "Choose Algorithm for Mining Frequent Itemsets:",
    ["Apriori", "FP-Growth", "Both"],
    horizontal=True
)

# Filter for multi-item transactions
segment_df = segmented_txns[segmented_txns['cltv_segment'] == selected_segment]
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
    if algo_choice == "Apriori":
        frequent_itemsets = apriori(itemsets, min_support=0.001, use_colnames=True)
    elif algo_choice == "FP-Growth":
        frequent_itemsets = fpgrowth(itemsets, min_support=0.001, use_colnames=True)
    else:  # Both
        frequent_itemsets_apriori = apriori(itemsets, min_support=0.001, use_colnames=True)
        frequent_itemsets_fpgrowth = fpgrowth(itemsets, min_support=0.001, use_colnames=True)
        frequent_itemsets = pd.concat([frequent_itemsets_apriori, frequent_itemsets_fpgrowth]).drop_duplicates()

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
        with st.expander("🔍 Insights Behind This Analysis"):
            st.info(
                """
                **What We Did:**  
                - Cleaned and prepared transaction data.
                - Applied either **Apriori** or **FP-Growth** to discover frequent product combinations.
                - Generated association rules with key metrics: Support, Confidence, Lift.
                - Estimated the total **Revenue per Rule** based on historical transactions.

                **Why It Matters:**  
                - Not all frequent rules are valuable; revenue-weighted rules identify *high-impact opportunities*.
                - FP-Growth is faster and more scalable than Apriori for large datasets.

                **Recommendations:**  
                - Focus on rules with high **Lift** (>1.2) and high **Revenue**.
                - Bundle cross-selling products identified in Top Revenue Rules.
                - Build targeted promotions based on segment-specific MBA insights.
                """
            )

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

        # --- Network Graph ---
        st.subheader("🌐 Network Graph of Association Rules")
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

        st.markdown("---")

        # --- Top Revenue-Generating Bundles ---
        st.subheader("💰 Top 10 Revenue-Generating Product Bundles")
        top_bundles = rules_df[['rule', 'support', 'confidence', 'lift', 'estimated_revenue']].head(10)
        st.dataframe(top_bundles.reset_index(drop=True), use_container_width=True)

        with st.expander("🔎 Why Focus on Revenue Bundles?"):
            st.info(
                """
                **Top Revenue Bundles** show which combinations of products bring the most sales value.
                **Action Points:**
                - Bundle these products together in special offers.
                - Promote them in upsell/cross-sell campaigns to high CLTV segments.
                - Prioritize stocking and marketing these bundles.
                """
            )

        st.markdown("---")

        # --- Filter Rules (Custom Sliders) ---
        st.subheader("🔍 Custom Filter for Association Rules")
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

        st.markdown("---")

        # --- Product Recommendation Engine ---
        st.subheader("💡 Rule-Based Product Recommendations")

        unique_items = sorted(set(
            item for subset in rules_df['antecedents'] for item in subset
        ))

        selected_product = st.selectbox("Select a Product for Recommendations", unique_items)

        reco_rules = rules_df[rules_df['antecedents'].apply(lambda x: selected_product in x)]

        if not reco_rules.empty:
            reco_display = reco_rules[["consequents", "confidence", "lift", "estimated_revenue"]].copy()
            reco_display["consequents"] = reco_display["consequents"].apply(lambda x: ', '.join(x))
            st.write(f"📦 Products often bought with **{selected_product}**:")
            st.dataframe(reco_display.reset_index(drop=True), use_container_width=True)
        else:
            st.info("No association rules found for this product.")
    else:
        st.warning("No valid association rules found for this segment.")
else:
    st.warning("Not enough multi-item transactions in this segment.")

