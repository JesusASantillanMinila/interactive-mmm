import pymc as pm
import numpy as np
from scipy.optimize import minimize

def run_bayesian_mmm(df, channels):
    # Simplified Bayesian Model
    with pm.Model() as model:
        # Priors
        sigma = pm.HalfNormal("sigma", sigma=1)
        intercept = pm.Normal("intercept", mu=df['demand'].mean(), sigma=10)
        
        # Channel Effects
        channel_contributions = 0
        for chan in channels:
            beta = pm.HalfNormal(f"beta_{chan}", sigma=5)
            channel_contributions += beta * df[chan]
            
        mu = intercept + channel_contributions
        y = pm.Normal("y", mu=mu, sigma=sigma, observed=df['demand'])
        
        trace = pm.sample(500, tune=500, chains=2, return_inferencedata=True)
    return trace

def budget_optimizer(current_spend, rois):
    # Objective: Maximize sum(Spend * ROI) subject to Total Budget
    total_budget = sum(current_spend)
    num_channels = len(current_spend)
    
    def objective(x):
        return -np.sum(x * rois) # Negative for minimization

    cons = ({'type': 'eq', 'fun': lambda x:  np.sum(x) - total_budget})
    bnds = tuple((0, total_budget) for _ in range(num_channels))
    
    res = minimize(objective, current_spend, method='SLSQP', bounds=bnds, constraints=cons)
    return res.x
