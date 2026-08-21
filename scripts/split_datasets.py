import json
from pathlib import Path
from typing import Dict

import pandas as pd


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def time_ordered_split(df: pd.DataFrame, time_col: str, train_frac: float = 0.7,
                       val_frac: float = 0.15, test_frac: float = 0.15,
                       dataset_name: str = "") -> Dict[str, pd.DataFrame]:
    total = train_frac + val_frac + test_frac
    if not abs(total - 1.0) < 1e-6:
        raise ValueError("Split fractions must sum to 1.0")

    if time_col not in df.columns:
        raise ValueError(f"Missing time column '{time_col}' in {dataset_name or 'dataset'}")

    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.dropna(subset=[time_col]).sort_values(time_col).reset_index(drop=True)

    n = len(df)
    if n < 3:
        raise ValueError(f"Not enough rows in {dataset_name or 'dataset'} for splitting (need >=3)")

    train_end = max(1, int(n * train_frac))
    val_end = max(train_end + 1, int(n * (train_frac + val_frac)))

    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]
    if len(test_df) == 0:
        # If rounding left nothing for test, move one row from val to test
        val_df, test_df = val_df.iloc[:-1], pd.concat([val_df.iloc[-1:], test_df])

    return {"train": train_df, "val": val_df, "test": test_df}


def split_pjm() -> None:
    src = Path("data/pjm/load.csv")
    if not src.exists():
        print("PJM data not found at data/pjm/load.csv; skipping.")
        return

    df = pd.read_csv(src, parse_dates=["timestamp"], engine="python")
    splits = time_ordered_split(df, time_col="timestamp", dataset_name="PJM")

    out_dir = Path("data/splits/pjm")
    ensure_dir(out_dir)
    for name, part in splits.items():
        part.to_csv(out_dir / f"{name}.csv", index=False)
    print("PJM split完成，输出目录: data/splits/pjm")


def split_aemo_vic() -> None:
    # 尝试多个可能的位置
    candidates = [
        Path("data/aemo_vic/load.csv"),
        Path("data/external/aemo_vic/load.csv")
    ]
    src = None
    for c in candidates:
        if c.exists():
            src = c
            break
            
    if src is None:
        print("AEMO VIC data not found at data/aemo_vic/load.csv or data/external/aemo_vic/load.csv; skipping.")
        return

    # 先读取不带 parse_dates，避免列名不匹配报错
    df = pd.read_csv(src)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    else:
        # 尝试查找可能是时间列的字段
        print(f"Warning: 'timestamp' col not found in {src}. Columns: {df.columns.tolist()}")
        if "SETTLEMENTDATE" in df.columns:
            df["timestamp"] = pd.to_datetime(df["SETTLEMENTDATE"])
            df = df.rename(columns={"TOTALDEMAND": "load"})
        elif "ds" in df.columns: # 处理用户提供的 aemo load.csv 格式 (ds, y)
            df["timestamp"] = pd.to_datetime(df["ds"])
            df = df.rename(columns={"y": "load"})
    
    if "region" not in df.columns:
        df["region"] = "VIC1"

    splits = time_ordered_split(df, time_col="timestamp", dataset_name="AEMO VIC")

    out_dir = Path("data/splits/aemo_vic")
    ensure_dir(out_dir)
    for name, part in splits.items():
        part.to_csv(out_dir / f"{name}.csv", index=False)
    print("AEMO VIC split完成，输出目录: data/splits/aemo_vic")


def split_aemo_nsw() -> None:
    path_2024 = Path("data/external/aemo_nsw/load_2024.csv")
    path_2025 = Path("data/external/aemo_nsw/load_2025.csv")
    
    dfs = []
    if path_2024.exists():
        dfs.append(pd.read_csv(path_2024))
    if path_2025.exists():
        dfs.append(pd.read_csv(path_2025))
        
    if not dfs:
        print("AEMO NSW data not found (load_2024.csv/load_2025.csv); skipping.")
        return

    df = pd.concat(dfs, ignore_index=True)
    
    if "ds" in df.columns:
         df["timestamp"] = pd.to_datetime(df["ds"])
    if "y" in df.columns:
         df = df.rename(columns={"y": "load"})
         
    if "timestamp" not in df.columns or "load" not in df.columns:
        print("AEMO NSW columns mismatch. Need ds/y or timestamp/load.")
        return

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    
    if "region" not in df.columns:
        df["region"] = "NSW1"

    splits = time_ordered_split(df, time_col="timestamp", dataset_name="AEMO NSW")

    out_dir = Path("data/splits/aemo_nsw")
    ensure_dir(out_dir)
    for name, part in splits.items():
        part.to_csv(out_dir / f"{name}.csv", index=False)
    print("AEMO NSW split完成，输出目录: data/splits/aemo_nsw")


if __name__ == "__main__":
    split_pjm()
    split_aemo_vic()
    split_aemo_nsw()
