"""MoLE 固定自适应切分的训练包装入口。"""
from __future__ import annotations

import json
import subprocess
import sys

import pandas as pd
import pytest

from src.models.external_adapters import (
    ExternalAdapterError,
    MOLE_SPLIT_POLICY,
    MOLE_TRAIN_SNIPPET,
    hyperparameters,
    write_itransformer_splits,
)


def _series(rows: int) -> pd.DataFrame:
    return pd.DataFrame({
        "timestamp": pd.date_range("2025-01-01", periods=rows, freq="h"),
        "load": range(rows),
    })


_STUB_LOADER = '''
class Dataset_Custom:
    def __init__(self, config, root_path, flag='train', size=None, features='S',
                 data_path='train.csv', target='load', scale=True, timeenc=1, freq='h'):
        self.args = config
        self.seq_len, self.label_len, self.pred_len = size
        self.set_type = {'train': 0, 'val': 1, 'test': 2}[flag]
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.root_path = root_path
        self.__read_data__()
        self.collect_all_data()

    def __read_data__(self):
        raise AssertionError('official proportional reader must be replaced')

    def collect_all_data(self):
        self.length = len(self.data_x) - self.seq_len - self.pred_len + 1

    def __len__(self):
        return self.length
'''

_STUB_FACTORY = '''
from data_provider.data_loader import Dataset_Custom
data_dict = {'custom': Dataset_Custom}

def data_provider(args, flag):
    Data = data_dict[args.data]
    return Data(config=args, root_path=args.root_path, data_path=args.data_path,
                flag=flag, size=[args.seq_len, args.label_len, args.pred_len],
                features=args.features, target=args.target, timeenc=1,
                freq=args.freq), None
'''

_STUB_RUN = '''
if __name__ == '__main__':
    import argparse, json
    from data_provider.data_factory import data_provider
    parser = argparse.ArgumentParser()
    for name in ('data', 'root_path', 'data_path', 'features', 'target', 'freq'):
        parser.add_argument('--' + name)
    for name in ('seq_len', 'label_len', 'pred_len'):
        parser.add_argument('--' + name, type=int)
    args, _ = parser.parse_known_args()
    result = {flag: len(data_provider(args, flag)[0]) for flag in ('train', 'val', 'test')}
    open('wrapper_result.json', 'w').write(json.dumps(result))
'''


def test_mole_wrapper_replaces_only_the_fixed_split_reader(tmp_path):
    repo = tmp_path / "official_mole"
    (repo / "data_provider").mkdir(parents=True)
    (repo / "data_provider" / "__init__.py").write_text("", encoding="utf-8")
    (repo / "data_provider" / "data_loader.py").write_text(
        _STUB_LOADER, encoding="utf-8"
    )
    (repo / "data_provider" / "data_factory.py").write_text(
        _STUB_FACTORY, encoding="utf-8"
    )
    (repo / "utils").mkdir()
    (repo / "utils" / "__init__.py").write_text("", encoding="utf-8")
    (repo / "utils" / "timefeatures.py").write_text(
        "import numpy as np\n"
        "def time_features(dates, freq='h'):\n"
        "    assert hasattr(dates, 'hour'), type(dates)\n"
        "    return np.zeros((4, len(dates)))\n",
        encoding="utf-8",
    )
    (repo / "run_longExp.py").write_text(_STUB_RUN, encoding="utf-8")

    split_root = repo / "split"
    plan = write_itransformer_splits(
        _series(3695), split_root, seq_len=336, pred_len=720, batch_size=8,
    )
    before = {p: p.read_bytes() for p in sorted(repo.rglob("*.py"))}

    completed = subprocess.run(
        [sys.executable, "-u", "-c", MOLE_TRAIN_SNIPPET,
         "--data", "custom", "--root_path", f"{split_root}/",
         "--data_path", "train.csv", "--features", "S", "--target", "load",
         "--freq", "h", "--seq_len", "336", "--label_len", "336",
         "--pred_len", "720"],
        cwd=repo, capture_output=True, text=True,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert json.loads((repo / "wrapper_result.json").read_text()) == plan["loader_samples"]
    assert plan["loader_samples"]["val"] == 8
    after = {p: p.read_bytes() for p in sorted(repo.rglob("*.py"))}
    assert after == before, "MoLE 官方仓库源码文件不得被修改"


def test_mole_uses_the_frozen_split_policy():
    assert MOLE_SPLIT_POLICY == "adaptive_val_at_least_one_full_batch"
    base = {
        "seq_len": 336, "t_dim": 4, "train_epochs": 40, "batch_size": 8,
        "patience": 6, "learning_rate": 0.005, "des": "final",
    }
    with pytest.raises(ExternalAdapterError, match="split_policy"):
        hyperparameters({"hyperparameters": base}, "mole")
    with pytest.raises(ExternalAdapterError, match="当前实现是"):
        hyperparameters(
            {"hyperparameters": {**base, "split_policy": "fixed_70_10_20"}},
            "mole",
        )
    assert hyperparameters(
        {"hyperparameters": {**base, "split_policy": MOLE_SPLIT_POLICY}}, "mole"
    )["split_policy"] == MOLE_SPLIT_POLICY
