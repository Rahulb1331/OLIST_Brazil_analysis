import streamlit as st
from PIL import Image

# Page config
st.set_page_config(page_title="Olist E-commerce Analysis", layout="wide")

# App title
st.title("Olist E-commerce Dataset Analysis")
st.markdown("#### Welcome to the interactive dashboard of Olist's Brazilian e-commerce dataset.")

# Introduction
st.markdown("""
This dashboard presents a comprehensive analysis of the **Olist Store dataset**,
which includes orders, customer behavior, product performance, seller data, and more
from a Brazilian e-commerce platform.

The dataset was originally published on [Kaggle](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce).
""")

st.markdown("---")

st.subheader("Project Objectives")
st.markdown("""
- Analyze customer purchase behavior and lifetime value
- Predict potential churn and purchasing trends
- Uncover insights from customer reviews and sentiment
- Identify top-performing and underperforming products and sellers
- Analyze product categories, freight, and delivery impact
- Perform Market Basket Analysis to discover frequently bought items together
- Explore geolocation-based patterns
""")

st.markdown("---")

# Sidebar Navigation (Multi-page Streamlit app)


# Optional: Project Image (add an image to the folder if you want)
# image = Image.open("images/olist_banner.png")
# st.image(image, use_column_width=True)

st.markdown("Made with Streamlit by Rahul Bamal | Dataset: Olist Store - Brazilian E-Commerce")

