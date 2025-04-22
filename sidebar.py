# sidebar.py
import streamlit as st

def render_sidebar():
    with st.sidebar:
        st.markdown("### Navigation")
        st.page_link("pages/rfm_page.py", label="RFM Analysis")
        st.page_link("pages/cltv_page.py", label="CLTV Prediction")
        st.page_link("pages/churn_page.py", label="Churn Prediction")
        st.page_link("pages/sentiment_page.py", label="Sentiment Analysis")
        st.page_link("pages/time_series_page.py", label="Time-Series Forecasting")
        st.page_link("pages/geo_page.py", label="Geo Visualization")
        st.page_link("pages/mba_page.py", label="Market Basket Analysis")
        st.page_link("pages/preprocessing.py", label="Preprocessing")
