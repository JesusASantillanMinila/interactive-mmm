import pandas as pd
import numpy as np

def generate_mmm_data(channels, yoy_trend, peak_month, total_spend, spend_dist):
    # 1. Setup Timeframe (3 years)
    dates = pd.date_range(start="2023-01-01", periods=36*4, freq='W') # Weekly for 3 years
    df = pd.DataFrame({'date': dates})
    n = len(df)

    # 2. Generate Adspend with Noise
    for i, channel in enumerate(channels):
        base_spend = (total_spend * spend_dist[i]) / n
        # Adding random noise to spend
        df[f'spend_{channel}'] = base_spend * np.random.uniform(0.5, 1.5, n)

    # 3. Adstock & Saturation Logic
    def apply_adstock(spend, decay=0.7):
        adstock = np.zeros(len(spend))
        for t in range(1, len(spend)):
            adstock[t] = spend[t] + decay * adstock[t-1]
        return adstock

    def apply_saturation(x, alpha=2, gamma=0.5):
        # Hill function: x^alpha / (x^alpha + gamma^alpha)
        return (x**alpha) / (x**alpha + gamma**alpha)

    # 4. Build Demand Components
    # Trend
    trend_factor = (1 + yoy_trend) ** (np.arange(n) / 52)
    
    # Seasonality (Sine wave peaking at user-defined month)
    month_series = df['date'].dt.month
    seasonality = np.sin(2 * np.pi * (month_series - peak_month + 3) / 12) + 1.5

    # 5. Calculate Final Demand
    total_ad_effect = 0
    for channel in channels:
        adstocked = apply_adstock(df[f'spend_{channel}'])
        saturated = apply_saturation(adstocked)
        total_ad_effect += saturated * 1000 # Scaling factor

    noise = np.random.normal(0, 0.05 * total_ad_effect.mean(), n)
    
    df['demand'] = (total_ad_effect + 500) * trend_factor * seasonality + noise
    
    return df
