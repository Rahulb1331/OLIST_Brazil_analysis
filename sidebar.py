# sidebar.py
import streamlit as st

def render_sidebar():
    with st.sidebar:
        st.markdown("### Navigation")
        st.markdown("Use the links below to explore each section:")
        st.markdown("- [RFM Analysis](#rfm-page)")
        st.markdown("- [CLTV Prediction](#cltv-page)")
        st.markdown("- [Churn Prediction](#churn-page)")
        st.markdown("- [Sentiment Analysis](#sentiment-page)")
        st.markdown("- [Time-Series Forecasting](#time-series-page)")
        st.markdown("- [Geo Visualization](#geo-page)")
        st.markdown("- [Market Basket Analysis](#mba-page)")
        st.markdown("- [Preprocessing](#preprocessing)")
