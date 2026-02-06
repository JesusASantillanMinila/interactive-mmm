import pandas as pd
import numpy as np
import pymc as pm
import pymc as az

def run_bayesian_mmm(df, channels):
    X = df[[f'spend_{c}' for c in channels]].values
    y = df['demand'].values
    
    # 1. Define the Bayesian Model
    with pm.Model() as model:
        # Priors: We assume effects are positive (HalfNormal)
        beta = pm.HalfNormal("beta", sigma=10, shape=len(channels))
        intercept = pm.Normal("intercept", mu=y.mean(), sigma=10)
        sigma = pm.HalfNormal("sigma", sigma=10)
        
        # Likelihood
        mu = intercept + pm.math.dot(X, beta)
        outcome = pm.Normal("y", mu=mu, sigma=sigma, observed=y)
        
        # Inference: Sampling from the posterior
        trace = pm.sample(1000, tune=1000, chains=2, target_accept=0.9)

    # 2. Extract Results
    summary = az.summary(trace, var_names=["beta"])
    post_mean_beta = summary["mean"].values
    
    # Calculate Contribution & ROI
    roi_data = []
    for i, channel in enumerate(channels):
        total_spend = df[f'spend_{channel}'].sum()
        total_contrib = (df[f'spend_{channel}'] * post_mean_beta[i]).sum()
        roi = total_contrib / total_spend if total_spend > 0 else 0
        roi_data.append({"Channel": channel, "ROI": roi, "Contribution": total_contrib})
    
    preds = intercept.mean() + np.dot(X, post_mean_beta)
    return pd.DataFrame(roi_data), trace, preds
