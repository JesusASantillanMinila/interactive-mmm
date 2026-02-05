import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pymc_marketing.mmm.transformers import geometric_adstock, logistic_saturation

# Page configuration
st.set_page_config(page_title="MMM Synthetic Data Generator", layout="wide")

st.title("📊 Marketing Mix Model: Synthetic Data Dashboard")
st.markdown("""
This dashboard simulates sales data based on **Trend**, **Seasonality**, and **Marketing Spend** transformed via Adstock and Saturation functions.
""")

# --- 1. Setup Timeframe ---
n_weeks = 156  # 3 years of weekly data
dates = pd.date_range(start="2021-01-01", periods=n_weeks, freq="W")
df = pd.DataFrame({"date": dates})

# --- 2. Generate Baseline ---
df["trend"] = np.linspace(100, 150, n_weeks)
df["seasonality"] = 20 * np.sin(2 * np.pi * df.index / 52.18)

# --- 3. Generate Ad Spend ---
rng = np.random.default_rng(42)
df["tv_spend"] = 500 * (np.sin(2 * np.pi * df.index / 13) + 1)
df["social_spend"] = rng.gamma(shape=2, scale=100, size=n_weeks)
df["search_spend"] = 200 + 0.5 * df["trend"] + rng.normal(0, 20, n_weeks)
df["display_spend"] = rng.uniform(50, 300, size=n_weeks)

# --- 4. Apply MMM Transformations ---
channels = ["tv", "social", "search", "display"]
adstock_alphas = [0.6, 0.2, 0.1, 0.3]
saturation_lambdas = [1.5, 0.8, 0.5, 1.2]
betas = [0.15, 0.25, 0.40, 0.10]

media_contribution = np.zeros(n_weeks)

for i, ch in enumerate(channels):
    spend = df[f"{ch}_spend"].values
    
    # Adstock & Saturation using pymc-marketing
    adstocked = geometric_adstock(spend, alpha=adstock_alphas[i]).eval()
    saturated = logistic_saturation(adstocked, lam=saturation_lambdas[i]).eval()
    
    df[f"{ch}_contribution"] = saturated * betas[i]
    media_contribution += df[f"{ch}_contribution"]

# --- 5. Final Demand ---
noise = rng.normal(0, 5, n_weeks)
df["sales"] = df["trend"] + df["seasonality"] + media_contribution + noise

# --- 6. Visualization ---
col1, col2 = st.columns([3, 1])

with col1:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Plot Sales
    ax1.plot(df["date"], df["sales"], color="black", linewidth=2, label="Total Sales")
    ax1.fill_between(df["date"], df["trend"] + df["seasonality"], alpha=0.2, label="Baseline (Trend+Season)")
    ax1.set_title("Synthetic Demand (Sales) over Time", fontsize=12)
    ax1.set_ylabel("Sales Units")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot Ad Spend
    for ch in channels:
        ax2.plot(df["date"], df[f"{ch}_spend"], label=f"{ch.upper()} Spend")

    ax2.set_title("Marketing Ad Spend per Channel", fontsize=12)
    ax2.set_ylabel("Spend ($)")
    ax2.set_xlabel("Date")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig)

with col2:
    st.subheader("Data Preview")
    st.dataframe(df[["date", "sales"] + [f"{c}_spend" for c in channels]].head(15))
    
    st.download_button(
        label="Download CSV",
        data=df.to_csv(index=False),
        file_name="synthetic_mmm_data.csv",
        mime="text/csv",
    )
