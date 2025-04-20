from analysis.Preprocessing import full_orders
from analysis.cltv import summary
import plotly.express as px
import plotly.graph_objects as go
from mlxtend.frequent_patterns import apriori, association_rules
import pandas as pd
import numpy as np
import networkx as nx

# Prepare data: group product names per order
transactions_df = full_orders.groupby("order_id")['product_category'].unique().reset_index()
transactions_df['items'] = transactions_df['product_category']

# Check orders with ≥ 2 items
multi_item_txns = transactions_df[transactions_df['items'].apply(len) > 1]
print(f"Multi-item transactions: {len(multi_item_txns)}") # just to see how many we get

# Clean 'items' to ensure no NaNs and consistent string types
multi_item_txns['items'] = multi_item_txns['items'].apply(
    lambda items: [str(i).strip().lower() for i in items if pd.notnull(i)] if isinstance(items, (list, pd.Series, np.ndarray)) else []
)

# Re-check for any empty rows (can happen after filtering nulls)
multi_item_txns = multi_item_txns[multi_item_txns['items'].apply(len) > 1]

# Extract all unique items safely

# Convert to one-hot encoded format for mlxtend.apriori
all_items = sorted(set(item for sublist in multi_item_txns['items'] for item in sublist))
encoded_df = pd.DataFrame(0, index=multi_item_txns.index, columns=all_items)
for idx, row in multi_item_txns.iterrows():
    encoded_df.loc[idx, row['items']] = 1

# Apply Apriori algorithm
frequent_itemsets = apriori(encoded_df, min_support=0.001, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.1)
rules.sort_values(by='lift', ascending=False, inplace=True)

# Creating readable rule labels
rules["rule"] = rules["antecedents"].apply(lambda x: ", ".join(x)) + " → " + rules["consequents"].apply(lambda x: ", ".join(x))


# Plotly bubble chart to visualize your top association rules
fig = px.scatter(
    rules,
    x="confidence",
    y="lift",
    size="support",
    hover_name="rule",
    title="Market Basket Rules: Confidence vs Lift",
    size_max=40,
    color="lift",  # Optional: color by lift to highlight strong associations
    color_continuous_scale="Turbo"
)

fig.update_layout(
    xaxis_title="Confidence",
    yaxis_title="Lift",
    template="plotly_white"
)

fig.show()

# Filtering the transactions to get the transactions with support at least 0.005 to avoid niche or singular transactions with high lift and then ordering by lift and confidence
rules_filtered = rules[rules['support'] >= 0.005].sort_values(by=["lift", "confidence"], ascending=False)
rules_filtered["rule"] = rules_filtered["antecedents"].astype(str) + " → " + rules_filtered["consequents"].astype(str)

# Now plot
fig = px.scatter(
    rules_filtered,
    x="support",
    y="confidence",
    size="lift",
    color="lift",
    hover_name="rule",
    title="Association Rules (Support vs Confidence, sized by Lift)",
    labels={"support": "Support", "confidence": "Confidence", "lift": "Lift"}
)
fig.update_layout(template="plotly_white")
fig.show()


# Network graph
rules_net = rules_filtered.copy()
rules_net["antecedent"] = rules_net["antecedents"].apply(lambda x: ", ".join(x))
rules_net["consequent"] = rules_net["consequents"].apply(lambda x: ", ".join(x))

# Create a graph
G = nx.DiGraph()

# Add nodes and edges
for _, row in rules_net.iterrows():
    G.add_node(row["antecedent"])
    G.add_node(row["consequent"])
    G.add_edge(row["antecedent"], row["consequent"], weight=row["lift"], confidence=row["confidence"], support=row["support"])

# Get positions using spring layout
pos = nx.spring_layout(G, seed=42)

# Prepare for plotly
edge_x = []
edge_y = []
edge_text = []
for edge in G.edges(data=True):
    x0, y0 = pos[edge[0]]
    x1, y1 = pos[edge[1]]
    edge_x.extend([x0, x1, None])
    edge_y.extend([y0, y1, None])
    edge_text.append(f"{edge[0]} → {edge[1]}<br>Confidence: {edge[2]['confidence']}, Lift: {edge[2]['weight']}")

# Node positions
node_x, node_y = zip(*[pos[n] for n in G.nodes()])


# Plot edges
edge_trace = go.Scatter(
    x=edge_x,
    y=edge_y,
    line=dict(width=1, color='#888'),
    hoverinfo='text',
    mode='lines'
)

# Plot nodes
node_trace = go.Scatter(
    x=node_x,
    y=node_y,
    mode='markers+text',
    hoverinfo='text',
    marker=dict(
        showscale=True,
        colorscale='YlGnBu',
        size=20,
        color=[G.degree(n) for n in G.nodes()],
        colorbar=dict(thickness=15, title='Node Connections', xanchor='left')
    ),
    text=list(G.nodes())
)

# Combine the plot
fig = go.Figure(data=[edge_trace, node_trace],
                layout=go.Layout(
                    title='Association Rules Network Graph',
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(b=20, l=5, r=5, t=40),
                    xaxis=dict(showgrid=False, zeroline=False),
                    yaxis=dict(showgrid=False, zeroline=False)
                ))
fig.show()


# Segmented MBA using CLTV

# Joining CLTV Segments with Transactions
segmented_txns = full_orders.merge(summary[['customer_unique_id', 'cltv_segment']], on='customer_unique_id', how='inner')
segmented_txns = segmented_txns.groupby(['order_id', 'customer_unique_id', 'cltv_segment'])['product_category'].unique().reset_index()
segmented_txns.rename(columns={'product_category': 'items'}, inplace=True)

segments = segmented_txns['cltv_segment'].unique()
segment_models = {}

for segment in segments:
    print(f"Running MBA for segment: {segment}")

    segment_df = segmented_txns[segmented_txns['cltv_segment'] == segment]
    segment_df['items'] = segment_df['items'].apply(
        lambda x: [str(i).strip().lower() for i in x if pd.notnull(i)] if isinstance(x, (list, pd.Series, np.ndarray)) else []
    )

    multi_item_txns = segment_df[segment_df['items'].apply(len) > 1]

    if not multi_item_txns.empty:
        all_items = sorted(set(item for sublist in multi_item_txns['items'] for item in sublist))
        encoded_segment = pd.DataFrame(0, index=multi_item_txns.index, columns=all_items)
        for idx, row in multi_item_txns.iterrows():
            encoded_segment.loc[idx, row['items']] = 1

        frequent_itemsets = apriori(encoded_segment, min_support=0.001, use_colnames=True)
        rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.1)
        rules.sort_values(by='lift', ascending=False, inplace=True)
        segment_models[segment] = rules
    else:
        print(f"Not enough multi-item transactions for {segment}")

# Converting Rules to Pandas for Visualization
# Convert one segment to Pandas (we can choose dynamically in Streamlit)
segment = "Very High"  # example
rules_df = segment_models[segment]

# Add readable rule text for visuals
rules_df['rule'] = rules_df['antecedents'].apply(lambda x: ", ".join(x)) + " → " + rules_df['consequents'].apply(lambda x: ", ".join(x))

# Plotting
fig = px.scatter(
    rules_df,
    x="support",
    y="confidence",
    size="lift",
    color="lift",
    hover_name="rule",
    title=f"Association Rules for {segment} CLTV Segment",
    labels={"support": "Support", "confidence": "Confidence", "lift": "Lift"}
)
fig.update_layout(template="plotly_white")
fig.show()
