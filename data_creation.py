import numpy as np
import pandas as pd

def generate_mmm_data(channels, adspend_level, decay_speed, delay_val, sat_speed, peak_month):
    np.random.seed(42)
    periods = 152  # 3 years of weekly data
    
    end_date = datetime.now()
    start_date = end_date - pd.DateOffset(years=3)
    
    date_rng = pd.date_range(start=start_date, periods=periods, freq='W')
    
    # Mapping qualitative inputs to numeric ranges
    map_levels = {"Low": (100, 500), "Medium": (500, 1500), "High": (1500, 5000)}
    map_decay = {"Low": 0.1, "Medium": 0.3, "High": 0.6}
    map_delay = {"Immediate": 0, "Slow": 1, "Very Slow": 3}
    map_sat = {"Low": 0.8, "Medium": 0.5, "High": 0.2}

    df = pd.DataFrame({'date': date_rng})
    
    # 1. Base Demand: Trend + Seasonality
    time = np.arange(periods)
    trend = 0.05 * time + 10 
    seasonality = 5 * np.sin(2 * np.pi * (date_rng.month - peak_month) / 12)
    noise = np.random.normal(0, 1, periods)
    demand = 100 + trend + seasonality + noise
    
    # 2. Adspend & Adstock/Saturation
    spend_data = {}
    true_contributions = np.zeros(periods)
    
    for col in channels:
        # Generate Spend
        low, high = map_levels[adspend_level]
        spend = np.random.gamma(shape=2, scale=high/2, size=periods)
        spend_data[col] = spend
        
        # Apply Adstock (Simple Decay)
        alpha = map_decay[decay_speed]
        delay = map_delay[delay_val]
        adstocked_spend = np.zeros(periods)
        for t in range(periods):
            for i in range(0, min(t, 10)): # 10 week window
                adstocked_spend[t] += spend[t-i] * (alpha**(i + delay))
        
        # Apply Saturation (Hill Function)
        beta = map_sat[sat_speed]
        contribution = (adstocked_spend**0.8) / (adstocked_spend**0.8 + beta*100)
        true_contributions += contribution * 10 # Scale factor
        
    df = pd.concat([df, pd.DataFrame(spend_data)], axis=1)
    df['demand'] = demand + true_contributions
    
    return df
