import pandas as pd
import numpy as np
from datetime import datetime
import streamlit as st

st.title("Preprocessed Datasets")

# Helper to convert Google Drive links to direct download URLs
def gdrive_to_direct_link(url):
    file_id = url.split("/d/")[1].split("/")[0]
    return f"https://drive.google.com/uc?export=download&id={file_id}"

# Dataset download links
dataset_links = {
    "olist_orders_dataset": "https://drive.google.com/file/d/1IW8RCm8SsxMTnxwBhbY2ki7UQFbjpWJQ/view?usp=sharing",
    "olist_customers_dataset": "https://drive.google.com/file/d/1GlfbdUR7Htaoa23ZaDaJU2BONVpd-46s/view?usp=sharing",
    "olist_order_items_dataset": "https://drive.google.com/file/d/1fzKgJiI8nrpOioDMNEH3FTGjNk38na4K/view?usp=sharing",
    "olist_geolocation_dataset": "https://drive.google.com/file/d/14Ov5-Ulw1pRPQwl-d1HGEDkrdeqAZdsf/view?usp=sharing",
    "olist_order_payments_dataset": "https://drive.google.com/file/d/1Yhb25SAM6uYOKb3LuiZNI87MwpzcMzcm/view?usp=sharing",
    "olist_order_reviews_dataset": "https://drive.google.com/file/d/129XEZCdH-e7LS6RxwJ8yIzTBEO2zSJIZ/view?usp=sharing",
    "olist_products_dataset": "https://drive.google.com/file/d/17jhNuSGKgXTWSop0vsjPGw9CP6eBJto7/view?usp=sharing",
    "olist_sellers_dataset": "https://drive.google.com/file/d/1vhjeb7QmtXiMWBELCylT4vL9-s8s1P_s/view?usp=sharing",
    "product_category_name_translation": "https://drive.google.com/file/d/1viI3NGEKJoN0M8I0DhTGzE47wGRfNB2r/view?usp=sharing"
}

# Download and load into DataFrames
dfs = {}

for name, url in dataset_links.items():
    direct_url = gdrive_to_direct_link(url)
    if name == "olist_order_reviews_dataset":
        dfs[name] = pd.read_csv(direct_url, dtype={"review_score": "Int64"}, keep_default_na=True)
    else:
        dfs[name] = pd.read_csv(direct_url)

# Assign to variables
orders = dfs["olist_orders_dataset"]
customers = dfs["olist_customers_dataset"]
geolocation = dfs["olist_geolocation_dataset"]
product_category = dfs["product_category_name_translation"]
sellers = dfs["olist_sellers_dataset"]
products = dfs["olist_products_dataset"]
order_reviews = dfs["olist_order_reviews_dataset"]
order_payments = dfs["olist_order_payments_dataset"]
order_items = dfs["olist_order_items_dataset"]

# --- Preprocessing Steps ---

# Convert Dates
orders["order_purchase_timestamp"] = pd.to_datetime(orders["order_purchase_timestamp"], errors="coerce")
order_reviews["review_creation_date"] = pd.to_datetime(order_reviews["review_creation_date"], errors="coerce")

order_reviews["review_score"] = order_reviews["review_score"].astype(int)

# Drop invalid review_score or review_id
order_reviews.dropna(subset=["review_score", "review_id"], inplace=True)

# Merge product categories
products = products.merge(
    product_category,
    how="left",
    on="product_category_name"
).drop(columns=["product_category_name"]).rename(columns={"product_category_name_english": "product_category"})

# Drop rows missing product_category
products.dropna(subset=["product_category"], inplace=True)

# Impute numerical product data with column mean
num_cols = ["product_weight_g", "product_length_cm", "product_height_cm", "product_width_cm"]
for col_name in num_cols:
    mean_val = products[col_name].mean()
    products[col_name] = products[col_name].fillna(mean_val)

# Add sentiment column to reviews
def get_sentiment(score):
    if score >= 4:
        return "Positive"
    elif score == 3:
        return "Neutral"
    else:
        return "Negative"

order_reviews["review_sentiment"] = order_reviews["review_score"].apply(get_sentiment)

# Filter invalid prices
order_items = order_items[(order_items["price"] > 0) & (order_items["freight_value"] >= 0)]

# Build full_orders
orders_with_customers = orders.merge(customers, on="customer_id", how="left")
orders_items_merged = orders_with_customers.merge(order_items, on="order_id", how="left")
full_orders = orders_items_merged.merge(products, on="product_id", how="left")
full_orders = full_orders.merge(order_payments, on="order_id", how="left")

st.dataframe(full_orders.head(10))
st.dataframe(geolocation.head(10))
st.dataframe(order_reviews.head(10))
st.dataframe(sellers.head(10))
st.dataframe(order_items.head(10))


# caching using a lightweight decorator 
@st.cache_data
def load_data():
    return full_orders, geolocation, order_reviews, sellers, order_items

full_orders, geolocation, order_reviews, sellers, order_items = load_data()

__all__ = [
    "full_orders",
    "geolocation",
    "order_reviews",
    "sellers",
    "order_items"
]
