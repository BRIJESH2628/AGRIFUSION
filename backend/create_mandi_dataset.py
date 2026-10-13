import os
import numpy as np
import pandas as pd

# Set random seed for reproducibility
np.random.seed(42)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.abspath(os.path.join(BASE_DIR, '..', 'Dataset'))
os.makedirs(DATASET_DIR, exist_ok=True)
csv_path = os.path.join(DATASET_DIR, 'mandi_prices.csv')

# 22 crops with realistic baseline modal price (INR/quintal), annual inflation/trend %, seasonality amplitude %, peak month
CROP_PARAMS = {
    "rice":        {"base": 2400, "trend": 0.04, "season_amp": 0.08, "peak_month": 11},
    "maize":       {"base": 1900, "trend": 0.05, "season_amp": 0.10, "peak_month": 10},
    "chickpea":    {"base": 5200, "trend": 0.03, "season_amp": 0.12, "peak_month": 4},
    "kidneybeans": {"base": 8200, "trend": 0.04, "season_amp": 0.09, "peak_month": 5},
    "pigeonpeas":  {"base": 7800, "trend": 0.06, "season_amp": 0.14, "peak_month": 1},
    "mungbean":    {"base": 7200, "trend": 0.04, "season_amp": 0.11, "peak_month": 9},
    "blackgram":   {"base": 6800, "trend": 0.05, "season_amp": 0.13, "peak_month": 10},
    "lentil":      {"base": 6100, "trend": 0.03, "season_amp": 0.10, "peak_month": 3},
    "mothbeans":   {"base": 5800, "trend": 0.04, "season_amp": 0.12, "peak_month": 11},
    "pomegranate": {"base": 8500, "trend": 0.05, "season_amp": 0.18, "peak_month": 2},
    "banana":      {"base": 2100, "trend": 0.03, "season_amp": 0.07, "peak_month": 7},
    "mango":       {"base": 4500, "trend": 0.06, "season_amp": 0.35, "peak_month": 5},
    "grapes":      {"base": 5800, "trend": 0.04, "season_amp": 0.25, "peak_month": 3},
    "watermelon":  {"base": 1400, "trend": 0.03, "season_amp": 0.22, "peak_month": 5},
    "muskmelon":   {"base": 1800, "trend": 0.04, "season_amp": 0.20, "peak_month": 5},
    "apple":       {"base": 8800, "trend": 0.05, "season_amp": 0.22, "peak_month": 9},
    "orange":      {"base": 3600, "trend": 0.04, "season_amp": 0.16, "peak_month": 12},
    "papaya":      {"base": 2200, "trend": 0.03, "season_amp": 0.09, "peak_month": 8},
    "coconut":     {"base": 2800, "trend": 0.04, "season_amp": 0.08, "peak_month": 6},
    "cotton":      {"base": 6500, "trend": 0.05, "season_amp": 0.12, "peak_month": 11},
    "jute":        {"base": 5100, "trend": 0.04, "season_amp": 0.10, "peak_month": 8},
    "coffee":      {"base": 14500, "trend": 0.07, "season_amp": 0.15, "peak_month": 1}
}

# Weekly dates from Jan 2021 through August 2026 (approx 295 weeks)
dates = pd.date_range(start="2021-01-03", end="2026-08-30", freq="W")

rows = []
for crop, params in CROP_PARAMS.items():
    base = params["base"]
    trend_rate = params["trend"]
    season_amp = params["season_amp"]
    peak_m = params["peak_month"]
    
    for idx, d in enumerate(dates):
        years_elapsed = idx / 52.0
        # Annual trend
        trend_factor = (1 + trend_rate) ** years_elapsed
        # Annual seasonality wave
        month_angle = (d.month - peak_m) * (2 * np.pi / 12)
        season_factor = 1 + season_amp * np.cos(month_angle)
        # Noise
        noise = np.random.normal(1.0, 0.03)
        
        modal_price = round(base * trend_factor * season_factor * noise, 2)
        # Add occasional outlier spike (0.5% chance) to test IQR cleaning
        if np.random.rand() < 0.005:
            modal_price = round(modal_price * 2.2, 2)
            
        rows.append({
            "crop": crop,
            "date": d.strftime("%Y-%m-%d"),
            "modal_price": modal_price
        })

df_prices = pd.DataFrame(rows)
df_prices.to_csv(csv_path, index=False)
print(f"Generated {len(df_prices)} rows of mandi prices across {df_prices['crop'].nunique()} crops.")
print(f"Saved to: {csv_path}")
