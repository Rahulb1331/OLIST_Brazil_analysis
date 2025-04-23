import streamlit as st

st.title("Preprocessed Datasets")

# caching using a lightweight decorator 
# --- Cached loading functions ---

@st.cache_data
def load_full_orders():
    from analysis.Preprocessing import full_orders
    return full_orders

@st.cache_data
def load_geolocation():
    from analysis.Preprocessing import geolocation
    return geolocation

@st.cache_data
def load_order_reviews():
    from analysis.Preprocessing import order_reviews
    return order_reviews

@st.cache_data
def load_sellers():
    from analysis.Preprocessing import sellers
    return sellers

@st.cache_data
def load_order_items():
    from analysis.Preprocessing import order_items
    return order_items

# Load cached data
full_orders = load_full_orders()
geolocation = load_geolocation()
order_reviews = load_order_reviews()
sellers = load_sellers()
order_items = load_order_items()

# Display dataframes
st.subheader("Full Orders dataset")
st.dataframe(full_orders.head(10))
st.markdown("---")

st.subheader("Geolocation dataset")
st.dataframe(geolocation.head(10))
st.markdown("---")

st.subheader("Order Reviews dataset")
st.dataframe(order_reviews.head(10))
st.markdown("---")

st.subheader("Sellers dataset")
st.dataframe(sellers.head(10))
st.markdown("---")

st.subheader("Order Items dataset")
st.dataframe(order_items.head(10))
