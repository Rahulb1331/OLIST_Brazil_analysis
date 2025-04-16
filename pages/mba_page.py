import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from Scripts.config import setup_environment
setup_environment()
# mba_app.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pyspark.sql.functions import collect_set, size
from pyspark.ml.fpm import FPGrowth
import networkx as nx
from pyspark.sql import SparkSession
from analysis.Preprocessing import full_orders
from analysis.cltv import summary

# Setup
st.set_page_config(page_title="Market Basket Analysis", layout="wide")
st.title("🛒 Market Basket Analysis (MBA)")

# Prepare Spark and data
spark = SparkSession.builder.getOrCreate()
summary_spark = spark.createDataFrame(summary)

transactions_df = full_orders.groupBy("order_id", "customer_unique_id") \
    .agg(collect_set("product_category").alias("items"))

segmented_txns = transactions_df.join(
    summary_spark.select("customer_unique_id", "cltv_segment"),
    on="customer_unique_id",
    how="inner"
)

segments = segmented_txns.select("cltv_segment").distinct().rdd.flatMap(lambda x: x).collect()
selected_segment = st.selectbox("Select CLTV Segment", segments)

# MBA for selected segment
segment_df = segmented_txns.filter(segmented_txns["cltv_segment"] == selected_segment)
multi_item_txns = segment_df.filter(size("items") > 1)

if multi_item_txns.count() > 0:
    fp_growth = FPGrowth(itemsCol="items", minSupport=0.001, minConfidence=0.1)
    model = fp_growth.fit(multi_item_txns)
    rules = model.associationRules.orderBy("lift", ascending=False)
    rules_df = rules.toPandas()

    rules_df["rule"] = rules_df["antecedent"].astype(str) + " → " + rules_df["consequent"].astype(str)

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
    rules_df["antecedent"] = rules_df["antecedent"].apply(lambda x: ", ".join(x))
    rules_df["consequent"] = rules_df["consequent"].apply(lambda x: ", ".join(x))

    G = nx.DiGraph()
    for _, row in rules_df.iterrows():
        G.add_edge(row["antecedent"], row["consequent"], weight=row["lift"])

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
    st.warning("Not enough multi-item transactions in this segment.")
