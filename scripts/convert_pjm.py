import pandas as pd
import numpy as np
import holidays
import os
from datetime import datetime

def convert_pjm_data():
    # Paths
    input_path = os.path.join("data", "external", "PJME_hourly.csv")
    output_dir = os.path.join("data", "pjm")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    print(f"Reading {input_path}...")
    df = pd.read_csv(input_path)
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    df = df.sort_values('Datetime').reset_index(drop=True)
    
    # 1. Generate load.csv
    print("Generating load.csv...")
    load_df = df.rename(columns={'Datetime': 'timestamp', 'PJME_MW': 'load'})
    load_df['region'] = 'PJME'
    load_df['region_type'] = 'real_grid'
    
    # Save load.csv
    load_df.to_csv(os.path.join(output_dir, "load.csv"), index=False)
    
    # Get time range for other files
    timestamps = load_df['timestamp']
    
    # 2. Generate holiday.csv
    print("Generating holiday.csv...")
    us_holidays = holidays.US()
    
    holiday_data = []
    for ts in timestamps:
        is_hol = ts in us_holidays
        is_weekend = ts.weekday() >= 5
        is_workday = not (is_hol or is_weekend)
        
        # Simple pre/post holiday logic (can be improved)
        pre_hol = (ts + pd.Timedelta(days=1)) in us_holidays
        post_hol = (ts - pd.Timedelta(days=1)) in us_holidays
        
        holiday_data.append({
            'timestamp': ts,
            'is_holiday': int(is_hol),
            'is_weekend': is_weekend,
            'is_workday': int(is_workday),
            'pre_holiday': int(pre_hol),
            'post_holiday': int(post_hol)
        })
        
    holiday_df = pd.DataFrame(holiday_data)
    holiday_df.to_csv(os.path.join(output_dir, "holiday.csv"), index=False)
    
    # 3. Generate weather.csv (procedural weather features)
    print("Generating weather.csv...")
    # PJM is in US East, let's simulate seasonal weather
    # Winter (Jan) ~ 0C, Summer (Jul) ~ 25C
    
    weather_data = []
    for ts in timestamps:
        # Day of year (0-365)
        doy = ts.dayofyear
        hour = ts.hour
        
        # Seasonal temp pattern (cosine wave)
        # Peak in summer (approx day 200), trough in winter
        seasonal_temp = 12.5 - 12.5 * np.cos(2 * np.pi * (doy - 20) / 365)
        
        # Daily temp pattern (peak around 2pm)
        daily_temp = 5 * np.cos(2 * np.pi * (hour - 14) / 24)
        
        # Random noise
        noise = np.random.normal(0, 3)
        
        temp = seasonal_temp + daily_temp + noise
        
        # Humidity (inverse to temp roughly)
        humidity = 60 + np.random.normal(0, 10)
        humidity = max(0, min(100, humidity))
        
        weather_data.append({
            'timestamp': ts,
            'temp': temp,
            'humidity': humidity,
            'wind': max(0, np.random.normal(5, 2)),
            'rain': 0.0 if np.random.random() > 0.1 else np.random.exponential(1),
            'feels_like_temp': temp,
            'heat_index': temp,
            'weather_comfort': 50
        })
        
    weather_df = pd.DataFrame(weather_data)
    weather_df.to_csv(os.path.join(output_dir, "weather.csv"), index=False)
    
    # 4. Generate data_info.json
    print("Generating data_info.json...")
    info = {
        "dataset_info": {
            "name": "PJM East Hourly Load (Real + Generated Weather)",
            "description": "Real load data from PJM East region with generated weather/holiday features",
            "time_range": f"{timestamps.min()} to {timestamps.max()}",
            "frequency": "H",
            "regions": ["PJME"],
            "total_records": len(load_df)
        },
        "data_files": {
            "load.csv": "Real PJM East load data",
            "weather.csv": "Generated weather data based on seasonal patterns",
            "holiday.csv": "US holidays generated via holidays library"
        }
    }
    import json
    with open(os.path.join(output_dir, "data_info.json"), 'w', encoding='utf-8') as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
        
    print("Conversion complete! Data saved to data/pjm/")

if __name__ == "__main__":
    convert_pjm_data()
