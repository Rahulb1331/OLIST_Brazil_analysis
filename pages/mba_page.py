# mba_app.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.preprocessing import MultiLabelBinarizer
from analysis.Preprocessing import full_orders
from analysis.cltv import summary

# Setup
st.set_page_config(page_title="Market Basket Analysis", layout="wide")
st.title("🛒 Market Basket Analysis (MBA)")

# Prepare Data
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

# MBA for selected segment
segment_df = segmented_txns[segmented_txns['cltv_segment'] == selected_segment]
multi_item_txns = segment_df[segment_df['items'].apply(lambda x: len(x) > 1)]

if not multi_item_txns.empty:
    # One-hot encode for apriori
    #mlb = MultiLabelBinarizer()
    # Clean items: convert to string, drop NaNs, and sort
    cleaned_items = multi_item_txns['items'].apply(
        lambda items: sorted([str(i) for i in items if pd.notnull(i)])
    )

    mlb = MultiLabelBinarizer()
    itemsets = pd.DataFrame(
        mlb.fit_transform(cleaned_items),
        columns=mlb.classes_,
        index=multi_item_txns.index
    ).astype(bool)
    
    #itemsets = pd.DataFrame(mlb.fit_transform(multi_item_txns['items']),
     #                       columns=mlb.classes_,
      #                      index=multi_item_txns.index).astype(bool)

    frequent_itemsets = apriori(itemsets, min_support=0.001, use_colnames=True)
    rules_df = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)
    rules_df = rules_df.sort_values("lift", ascending=False)

    if not rules_df.empty:
        rules_df["rule"] = rules_df["antecedents"].apply(lambda x: ', '.join(x)) + \
                            " → " + rules_df["consequents"].apply(lambda x: ', '.join(x))

        # Visuals
        st.subheader(f"Association Rules for {selected_segment}")
        fig1 = px.scatter(
            rules_df,
            x="support",
            y="confidence",
            size="lift",
            color="lift",
            hover_name="rule",
            title=f"Rules Scatter Plot ({selected_segment})",
            labels={"support": "Support", "confidence": "Confidence", "lift": "Lift"}
        )
        st.plotly_chart(fig1, use_container_width=True)

        st.subheader("Network Graph of Rules")

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
    else:
        st.warning("No valid association rules found for this segment.")
else:
    st.warning("Not enough multi-item transactions in this segment.")

st.markdown("---")

st.subheader("🔍 Filter Association Rules")

# Filter sliders
col1, col2, col3 = st.columns(3)
with col1:
    min_conf = st.slider("Min Confidence", 0.0, 1.0, 0.1, 0.01)
with col2:
    min_lift = st.slider("Min Lift", 0.0, float(rules_df['lift'].max()), 1.0, 0.1)
with col3:
    min_support = st.slider("Min Support", 0.0, float(rules_df['support'].max()), 0.01, 0.001)

# Filter rules
filtered_rules = rules_df[
    (rules_df['confidence'] >= min_conf) &
    (rules_df['lift'] >= min_lift) &
    (rules_df['support'] >= min_support)
]

# Top N selector
top_n = st.slider("Show Top N Rules", 5, 50, 10)
top_rules = filtered_rules.head(top_n)

# Display table
st.dataframe(top_rules[["rule", "support", "confidence", "lift"]].reset_index(drop=True), use_container_width=True)

st.markdown("---")

st.subheader("💡 Rule-Based Product Recommendations")

# Flatten antecedents and consequents to extract unique items
unique_items = sorted(set(
    item for subset in rules_df['antecedents'] for item in subset
))

selected_product = st.selectbox("Select a Product", unique_items)

# Find rules where selected product is in antecedents
reco_rules = rules_df[rules_df['antecedents'].apply(lambda x: selected_product in x)]

if not reco_rules.empty:
    reco_rules_display = reco_rules[["consequents", "confidence", "lift"]].copy()
    reco_rules_display["consequents"] = reco_rules_display["consequents"].apply(lambda x: ', '.join(x))
    st.write(f"📦 Products often bought with **{selected_product}**:")
    st.dataframe(reco_rules_display.reset_index(drop=True), use_container_width=True)
else:
    st.info("No association rules found for this product.")
