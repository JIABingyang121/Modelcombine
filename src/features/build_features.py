from __future__ import annotations
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
import holidays


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    ts = pd.to_datetime(df["timestamp"]) 
    df["hour"] = ts.dt.hour
    df["dow"] = ts.dt.dayofweek
    df["is_weekend"] = (df["dow"].isin([5, 6])).astype(int)
    df["day"] = ts.dt.day
    df["month"] = ts.dt.month
    return df


def add_holiday_feature(df: pd.DataFrame, country: str = "CN") -> pd.DataFrame:
    df = df.copy()
    years = sorted({pd.to_datetime(x).year for x in df["timestamp"]})
    cal = holidays.country_holidays(country, years=years)
    df["is_holiday"] = df["timestamp"].apply(lambda x: 1 if pd.to_datetime(x).date() in cal else 0).astype(int)
    return df


def join_weather(load_df: pd.DataFrame, weather_df: pd.DataFrame) -> pd.DataFrame:
    w = weather_df.copy()
    w["timestamp"] = pd.to_datetime(w["timestamp"]) 
    d = load_df.copy()
    d["timestamp"] = pd.to_datetime(d["timestamp"]) 
    return d.merge(w, on="timestamp", how="left")


def add_lag_rolling(df: pd.DataFrame, lags: List[int], rolling_cfg: List[Dict]) -> pd.DataFrame:
    df = df.sort_values(["region", "timestamp"]).copy()
    for region, g in df.groupby("region"):
        idx = g.index
        for l in lags:
            df.loc[idx, f"lag_{l}"] = g["load"].shift(l)
        for rc in rolling_cfg:
            window = rc["window"]
            stat = rc["stat"]
            if stat == "mean":
                df.loc[idx, f"roll{window}_mean"] = g["load"].shift(1).rolling(window).mean()
            elif stat == "std":
                df.loc[idx, f"roll{window}_std"] = g["load"].shift(1).rolling(window).std()
    return df


def build_matrix(df: pd.DataFrame, target_col: str = "load") -> Tuple[pd.DataFrame, pd.Series]:
    # 排除非数值列和特定列
    exclude_cols = ["load", "timestamp", "region", "region_type"]
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    # 只保留数值类型的列
    X = df[feature_cols].select_dtypes(include=[np.number])
    y = df[target_col]
    return X, y
