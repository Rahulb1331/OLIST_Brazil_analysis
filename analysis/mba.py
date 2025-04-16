from analysis.Preprocessing import full_orders
from pyspark.ml.fpm import FPGrowth
from pyspark.sql.functions import collect_set
from pyspark.sql.functions import size
import plotly.express as px
import plotly.graph_objects as go
from analysis.cltv import summary

# Prepare data: group product names per order
transactions_df = full_orders.groupBy("order_id") \
    .agg(collect_set("product_category").alias("items"))

transactions_df.select(size("items").alias("num_items")).summary().show()

# Check orders with ≥ 2 items
multi_item_txns = transactions_df.filter(size("items") > 1)
print(multi_item_txns.count())  # just to see how many we get

# Fit FP-Growth model
fp_growth = FPGrowth(itemsCol="items", minSupport=0.001, minConfidence=0.1)
model = fp_growth.fit(multi_item_txns)

# Frequent itemsets
frequent_itemsets = model.freqItemsets
frequent_itemsets.show(truncate=False)

# Association rules
rules = model.associationRules
rules.orderBy("lift", ascending=False).show(truncate=False)

# Predictions (if you want to apply it)
predictions = model.transform(transactions_df)


# Plotly bubble chart to visualize your top association rules

# Converting rules to pandas
rules_df = rules.toPandas()

# Creating readable rule labels
rules_df["rule"] = rules_df["antecedent"].apply(lambda x: ", ".join(x)) + " → " + rules_df["consequent"].apply(lambda x: ", ".join(x))

fig = px.scatter(
    rules_df,
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
rules_filtered = rules.filter("support >= 0.005") \
    .orderBy(["lift", "confidence"], ascending=[False, False])

rules_filtered.show(20)

# Convert Spark DataFrame to Pandas
rules_pd = rules_filtered.toPandas()

# Add a readable rule label for hover
rules_pd["rule"] = rules_pd["antecedent"].astype(str) + " → " + rules_pd["consequent"].astype(str)

# Now plot
fig = px.scatter(
    rules_pd,
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

import networkx as nx

# Sample filtered rules DataFrame (you can replace this with your actual filtered rules from Spark)
# First, convert Spark DataFrame to pandas
rules_pdf = rules_filtered.toPandas()

# Flatten the lists for antecedents and consequents into strings
rules_pdf["antecedent"] = rules_pdf["antecedent"].apply(lambda x: ", ".join(x))
rules_pdf["consequent"] = rules_pdf["consequent"].apply(lambda x: ", ".join(x))


# Create a graph
G = nx.DiGraph()

# Add nodes and edges
for _, row in rules_pdf.iterrows():
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
node_x = []
node_y = []
for node in G.nodes():
    x, y = pos[node]
    node_x.append(x)
    node_y.append(y)

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
from pyspark.sql import SparkSession

# Make sure Spark session is active
spark = SparkSession.builder.getOrCreate()

# Convert pandas DataFrame to Spark
summary_spark = spark.createDataFrame(summary)

transactions_df = full_orders.groupBy("order_id", "customer_unique_id") \
    .agg(collect_set("product_category").alias("items"))


transactions_df.printSchema()

# Joining CLTV Segments with Transactions
segmented_txns = transactions_df.join(
    summary_spark.select("customer_unique_id", "cltv_segment"),
    on="customer_unique_id",
    how="inner"
)


# Looping Over Segments and Run FP-Growth

segments = segmented_txns.select("cltv_segment").distinct().rdd.flatMap(lambda x: x).collect()

segment_models = {}
for segment in segments:
    print(f"Running MBA for segment: {segment}")

    segment_df = segmented_txns.filter(segmented_txns["cltv_segment"] == segment)
    multi_item_txns = segment_df.filter(size("items") > 1)

    if multi_item_txns.count() > 0:
        fp_growth = FPGrowth(itemsCol="items", minSupport=0.001, minConfidence=0.1)
        model = fp_growth.fit(multi_item_txns)
        rules = model.associationRules.orderBy("lift", ascending=False)
        segment_models[segment] = rules
    else:
        print(f"Not enough multi-item transactions for {segment}")

# Converting Rules to Pandas for Visualization
# Convert one segment to Pandas (we can choose dynamically in Streamlit)
segment = "Very High"  # example
rules_df = segment_models[segment].toPandas()

# Add readable rule text for visuals
rules_df["rule"] = rules_df["antecedent"].astype(str) + " → " + rules_df["consequent"].astype(str)

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