"""按配置为每个数据集生成 time/lag/rolling/holiday 特征。

配置的 ``splits`` 段决定要处理哪些数据集以及它们的 train/val/test 目录：

- 旧口径：值为相对/绝对路径（``data/splits/pjm``），直接使用；
- 正式实验口径：值为子目录名（``pjm_rto``），配合 ``--splits-root`` 解析到私有
  split 目录（``<splits-root>/pjm_rto``）。新增数据集只需在配置里登记，不再改代码。

特征定义（lags / rolling / holiday 日历）全部来自配置，按数据集取。
"""
import argparse
from pathlib import Path
from typing import Dict, List, Optional

import holidays
import numpy as np
import pandas as pd
import yaml


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_config(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def add_time_features(df: pd.DataFrame, time_col: str) -> pd.DataFrame:
    df = df.copy()
    ts = pd.to_datetime(df[time_col])
    df["hour"] = getattr(ts.dt, "hour", pd.Series(np.nan, index=df.index))
    df["dayofweek"] = ts.dt.dayofweek
    df["month"] = ts.dt.month
    df["is_weekend"] = ts.dt.dayofweek.isin([5, 6]).astype(int)
    df["date"] = ts.dt.date
    return df


def add_holiday(df: pd.DataFrame, time_col: str, country: str) -> pd.DataFrame:
    df = df.copy()
    ts = pd.to_datetime(df[time_col])
    years = sorted({t.year for t in ts.dropna()})
    try:
        cal = holidays.country_holidays(country, years=years)
    except Exception:
        cal = holidays.CN(years=years)
    df["is_holiday"] = ts.dt.date.apply(lambda d: 1 if d in cal else 0)
    return df


def add_lag_roll_grouped(
    df: pd.DataFrame,
    entity_cols: List[str],
    time_col: str,
    target_col: str,
    lags: List[int],
    roll_windows: List[int],
) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values(entity_cols + [time_col]) if entity_cols else df.sort_values(time_col)
    group_obj = df.groupby(entity_cols) if entity_cols else [(None, df)]

    for key, g in group_obj:
        idx = g.index
        for l in lags:
            df.loc[idx, f"lag_{l}"] = g[target_col].shift(l)
        for w in roll_windows:
            rolled = g[target_col].shift(1).rolling(w)
            df.loc[idx, f"roll{w}_mean"] = rolled.mean()
            df.loc[idx, f"roll{w}_std"] = rolled.std()
    return df


def select_entity_cols(df: pd.DataFrame, candidates: List[str]) -> List[str]:
    return [c for c in candidates if c in df.columns]


def save_splits(out_root: Path, name: str, splits: Dict[str, pd.DataFrame]) -> None:
    ensure_dir(out_root / name)
    for split, part in splits.items():
        out_path = out_root / name / f"{split}.csv"
        part.to_csv(out_path, index=False)
        print(f"[saved] {out_path} rows={len(part)}")


def load_split(root: Path, split: str) -> pd.DataFrame:
    path = root / f"{split}.csv"
    return pd.read_csv(path)


def resolve_split_root(raw: str, splits_root: Optional[Path]) -> Path:
    """``--splits-root`` 给出时把配置值当子目录名；否则按路径使用。"""
    root = Path(raw)
    if splits_root is not None and not root.is_absolute():
        return Path(splits_root) / raw
    return root


def process_dataset(
    dataset: str, cfg: Dict, out_root: Path, splits_root: Optional[Path] = None
) -> None:
    root = resolve_split_root(cfg["splits"][dataset], splits_root)
    lags = cfg["features"]["lags"][dataset]
    rolls = cfg["features"]["rolling"][dataset]
    calendar = cfg["features"]["holiday"][dataset].get("calendar", "US")
    splits = {}
    for split in ["train", "val", "test"]:
        df = load_split(root, split)
        df = add_time_features(df, "timestamp")
        df = add_holiday(df, "timestamp", calendar)
        df = add_lag_roll_grouped(
            df,
            entity_cols=select_entity_cols(df, ["region"]),
            time_col="timestamp",
            target_col="load",
            lags=lags,
            roll_windows=rolls,
        )
        df = df.dropna().reset_index(drop=True)
        splits[split] = df
    save_splits(out_root, dataset, splits)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/exp_comparativetest1.yaml", help="配置文件路径")
    parser.add_argument(
        "--splits-root",
        type=Path,
        default=None,
        help="可选：把配置里的 splits 值当作该目录下的子目录名（正式实验私有 split 目录）",
    )
    parser.add_argument("--out", default="data/features", help="输出特征目录")
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    out_root = Path(args.out)
    ensure_dir(out_root)

    for dataset in cfg["splits"]:
        process_dataset(dataset, cfg, out_root, splits_root=args.splits_root)


if __name__ == "__main__":
    main()
