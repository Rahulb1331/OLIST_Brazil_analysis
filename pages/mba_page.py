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
    mlb = MultiLabelBinarizer()
    itemsets = pd.DataFrame(mlb.fit_transform(multi_item_txns['items']),
                            columns=mlb.classes_,
                            index=multi_item_txns.index).astype(bool)

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

