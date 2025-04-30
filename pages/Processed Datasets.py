import streamlit as st
import pandas as pd

st.set_page_config(page_title="Preprocessed Data Overview", layout="wide")
st.title("🧹 Preprocessed Datasets Overview")

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

@st.cache_data
def load_cltv_df():
    from analysis.cltv import cltv_df
    return cltv_df


# --- Load data ---
full_orders = load_full_orders()
geolocation = load_geolocation()
order_reviews = load_order_reviews()
sellers = load_sellers()
order_items = load_order_items()
cltv_df = load_cltv_df()

# --- Helper functions ---
def show_dataset_summary(df, name, description, keys_used=None, used_in=None):
    st.subheader(f"📦 {name}")
    st.markdown(f"**Description:** {description}")
    
    if keys_used:
        st.markdown(f"**Merge Keys:** {', '.join(keys_used)}")
    if used_in:
        st.markdown(f"**Used In:** {', '.join(used_in)}")

    # Row/column stats
    st.markdown(f"**Shape:** {df.shape[0]:,} rows × {df.shape[1]:,} columns")

    with st.expander("🔍 Schema Overview"):
        schema = pd.DataFrame({
            "Column": df.columns,
            "Data Type": df.dtypes.values,
            "Nulls (%)": (df.isnull().mean() * 100).round(2).values
        })
        st.dataframe(schema)

    row_limit = st.slider(f"Show preview rows for {name}", 5, 50, 10, key=name)
    st.dataframe(df.head(row_limit))
    st.markdown("---")

# --- Dataset Grouping ---
st.markdown("### 🔗 Core Datasets")
show_dataset_summary(
    full_orders,
    "Full Orders",
    "Merged dataset containing customer, order, payment, and product information. Core to RFM, CLTV, and segmentation.",
    keys_used=["customer_unique_id", "order_id"],
    used_in=["RFM", "CLTV", "Cohort Analysis"]
)

show_dataset_summary(
    order_items,
    "Order Items",
    "Contains individual order lines including products and sellers. Used for revenue calculation and basket size.",
    keys_used=["order_id", "product_id"],
    used_in=["CLTV", "Revenue Tracking"]
)

show_dataset_summary(
    cltv_df,
    "CLTV DataFrame",
    "Customer-level dataset containing predicted CLTV, probability of repeat, frequency, recency, monetary value, and segment tags. Central to predictive modeling.",
    keys_used=["customer_unique_id"],
    used_in=["CLTV Modeling", "Segmentation", "Revenue Forecasting"]
)

st.markdown("### 🧩 Auxiliary Datasets")

show_dataset_summary(
    geolocation,
    "Geolocation",
    "Mapping of ZIP codes to city, state, and lat/long. Useful for customer location enrichment.",
    keys_used=["customer_zip_code_prefix"],
    used_in=["Geo Visualizations", "Regional Insights"]
)

show_dataset_summary(
    order_reviews,
    "Order Reviews",
    "Customer review scores and comments. Used for NPS analysis and sentiment modeling.",
    keys_used=["order_id"],
    used_in=["Review Analysis", "Customer Experience"]
)

show_dataset_summary(
    sellers,
    "Sellers",
    "Details of marketplace sellers including location and unique seller IDs.",
    keys_used=["seller_id"],
    used_in=["Seller Performance"]
)

# --- Footer ---
st.markdown("This page is a reference for the different datasets used for the different analysis and understanding their structure, purpose, and linkages.")
