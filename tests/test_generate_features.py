"""generate_features.py 的配置驱动契约测试。

守两件事：新数据集只靠配置登记即可处理（--splits-root 解析子目录）；旧的
路径式配置在缺省 --splits-root 时行为不变。
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent


def _write_splits(root: Path, dataset: str, rows: int = 200) -> None:
    split_dir = root / dataset
    split_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        ts = pd.date_range("2024-01-01", periods=rows, freq="h")
        pd.DataFrame(
            {"timestamp": ts, "load": 100 + np.arange(rows, dtype=float)}
        ).to_csv(split_dir / f"{split}.csv", index=False)


def _config(tmp_path: Path, splits: dict) -> Path:
    payload = {
        "splits": splits,
        "features": {
            "lags": {ds: [1, 24] for ds in splits},
            "rolling": {ds: [3] for ds in splits},
            "holiday": {ds: {"enabled": True, "calendar": "US" if "pjm" in ds else "AU"} for ds in splits},
        },
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return path


def _run(config: Path, out: Path, *, splits_root: Path | None = None):
    cmd = [
        sys.executable, "scripts/generate_features.py",
        "--config", str(config), "--out", str(out),
    ]
    if splits_root is not None:
        cmd += ["--splits-root", str(splits_root)]
    return subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)


def test_new_dataset_is_registered_by_config_only(tmp_path):
    splits_root = tmp_path / "splits"
    _write_splits(splits_root, "pjm_rto")
    config = _config(tmp_path, {"pjm_rto": "pjm_rto"})
    out = tmp_path / "features"

    proc = _run(config, out, splits_root=splits_root)
    assert proc.returncode == 0, proc.stdout + proc.stderr

    train = pd.read_csv(out / "pjm_rto" / "train.csv")
    expected = {
        "timestamp", "load", "hour", "dayofweek", "month", "is_weekend",
        "is_holiday", "lag_1", "lag_24", "roll3_mean", "roll3_std",
    }
    assert expected <= set(train.columns)
    assert not train.isna().any().any()
    stamps = pd.to_datetime(train["timestamp"])
    assert (stamps.diff().dropna() == pd.Timedelta(hours=1)).all()


def test_path_style_config_without_splits_root_is_unchanged(tmp_path):
    raw = tmp_path / "raw"
    _write_splits(raw, "legacy")
    config = _config(tmp_path, {"legacy": str(raw / "legacy")})
    out = tmp_path / "features"

    proc = _run(config, out)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    train = pd.read_csv(out / "legacy" / "train.csv")
    assert "lag_1" in train.columns and len(train) > 0


def test_multiple_datasets_in_one_config(tmp_path):
    splits_root = tmp_path / "splits"
    _write_splits(splits_root, "pjm_rto")
    _write_splits(splits_root, "aemo_vic")
    config = _config(tmp_path, {"pjm_rto": "pjm_rto", "aemo_vic": "aemo_vic"})
    out = tmp_path / "features"

    proc = _run(config, out, splits_root=splits_root)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert (out / "pjm_rto" / "train.csv").exists()
    assert (out / "aemo_vic" / "train.csv").exists()
