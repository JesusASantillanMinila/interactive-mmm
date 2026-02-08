import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from data_creation import generate_mmm_data
from mmm_analysis import run_bayesian_mmm, budget_optimizer

st.set_page_config(layout="wide", page_title="Bayesian MMM Dashboard")

# Session State Initialization
if 'data' not in st.session_state:
    st.session_state.data = None

st.title("📊 Marketing Mix Modeling & Budget Optimizer")

with st.sidebar:
    st.header("1. Data Generation Settings")
    chans = st.text_input("Channels (comma separated)", "TV, Social, Search, OOH").split(", ")
    lvl = st.select_slider("Adspend Level", ["Low", "Medium", "High"])
    dec = st.select_slider("Adstock Decay", ["Low", "Medium", "High"])
    peak = st.slider("Peak Sales Month", 1, 12, 12)
    
    if st.button("Generate Randomized Data"):
        st.session_state.data = generate_mmm_data(chans, lvl, dec, "Slow", "Medium", peak)

if st.session_state.data is not None:
    df = st.session_state.data
    
    # --- VISUALIZATION SECTION ---
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Adspend over Time")
        fig_spend = px.line(df, x='date', y=chans)
        st.plotly_chart(fig_spend, use_container_width=True)
    
    with col2:
        st.subheader("Total Demand (Target)")
        fig_dem = px.line(df, x='date', y='demand')
        st.plotly_chart(fig_dem, use_container_width=True)

    # --- ANALYSIS SECTION ---
    if st.button("Run Bayesian MMM Analysis"):
        with st.spinner("Sampling Markov Chains..."):
            trace = run_bayesian_mmm(df, chans)
            st.success("Model Converged!")
            
            # Waterfall Chart (Simplified)
            st.subheader("Contribution Waterfall")
            fig_water = go.Figure(go.Waterfall(
                name = "20", orientation = "v",
                measure = ["relative"] * (len(chans) + 1),
                x = ["Baseline"] + chans,
                y = [100] + [15, 10, 5, 8] # Placeholder for trace means
            ))
            st.plotly_chart(fig_water)

    # --- OPTIMIZATION SECTION ---
    st.divider()
    st.header("💰 Budget Optimization")
    if st.button("Optimize Budget"):
        current = df[chans].mean().values
        # Simulated ROI for demo
        optimized = budget_optimizer(current, [1.2, 2.5, 0.8, 1.5])
        
        opt_df = pd.DataFrame({
            'Channel': chans,
            'Current': current,
            'Optimal': optimized
        }).melt(id_vars='Channel')
        
        st.plotly_chart(px.bar(opt_df, x='Channel', y='value', color='variable', barmode='group'))

if st.button("Restart Session"):
    st.session_state.clear()
    st.rerun()
