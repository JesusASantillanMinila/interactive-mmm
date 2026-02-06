import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from data_generation import generate_mmm_data
from mmm_analysis import run_bayesian_mmm

st.set_page_config(page_title="MMM Data Simulator", layout="wide")
st.title("📈 MMM Synthetic Data Generator")

# Sidebar Inputs
st.sidebar.header("Global Parameters")
yoy_trend = st.sidebar.slider("Annual Trend (YoY %)", 0.0, 0.50, 0.10)
peak_month = st.sidebar.slider("Peak Seasonality Month", 1, 12, 12)
total_spend = st.sidebar.number_input("Total 3-Year Ad Spend ($)", value=1000000)

st.sidebar.header("Channel Settings")
channels = []
spend_shares = []

cols = st.sidebar.columns(2)
for i in range(4):
    name = cols[0].text_input(f"Channel {i+1}", value=f"Channel_{i+1}", key=f"name_{i}")
    share = cols[1].number_input(f"Share %", value=25, key=f"share_{i}")
    channels.append(name)
    spend_shares.append(share / 100)

# Validation and Execution
if st.button("Generate Dataset"):
    if sum(spend_shares) != 1.0:
        st.error("Spend shares must add up to 100%!")
    else:
        df = generate_mmm_data(channels, yoy_trend, peak_month, total_spend, spend_shares)
        st.session_state['mmm_df'] = df
        
        # Plotting
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Ad Spend Over Time")
            spend_cols = [f'spend_{c}' for c in channels]
            fig_spend = px.line(df, x='date', y=spend_cols, title="Weekly Spend per Channel")
            st.plotly_chart(fig_spend, use_container_width=True)
            
        with col2:
            st.subheader("Total Demand (Sales)")
            fig_demand = px.area(df, x='date', y='demand', title="Total Demand (Trend + Seasonality + Marketing)")
            st.plotly_chart(fig_demand, use_container_width=True)
            
        st.success("Data generated successfully!")
        st.dataframe(df.head(10))

# --- ANALYSIS SECTION ---
# ... (inside the Analysis section of app.py)
if 'mmm_df' in st.session_state:
    st.header("🔮 Bayesian MMM Analysis")
    df = st.session_state['mmm_df']
    
    with st.spinner("Running MCMC Sampling (this takes a few seconds)..."):
        roi_df, trace, preds = run_bayesian_mmm(df, channels)
    
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.subheader("ROI Credible Intervals")
        # Show the uncertainty
        fig_uncertainty = az.plot_forest(trace, var_names=["beta"], combined=True)
        st.pyplot(fig_uncertainty[0].figure)
        st.caption("This chart shows the model's confidence. Narrower bars = more certainty.")

    with col_b:
        fig_roi = px.bar(roi_df, x="Channel", y="ROI", title="Mean Predicted ROI")
        st.plotly_chart(fig_roi)
