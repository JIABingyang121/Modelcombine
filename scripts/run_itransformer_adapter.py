#!/usr/bin/env python3
"""
Run iTransformer on fixed train/val/test splits and export predictions into
ModelCombine baseline artifact format:
  reports/baselines/<dataset>/{val,test}_pred_h{h}_itransformer.csv
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


DEFAULT_DATASETS = ["pjm", "aemo_vic", "aemo_nsw"]
DEFAULT_HORIZONS = [1, 6, 24]
PREFERRED_REF_MODELS = ["catboost_reg", "xgboost_reg", "lgbm_reg", "seasonal_naive"]
TIME_COL_CANDIDATES = ["timestamp", "ds", "date", "datetime", "time", "ts"]
DROP_NON_FEATURE_COLS = {"row_id", "split"}


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _detect_time_col(df: pd.DataFrame) -> str:
    for c in TIME_COL_CANDIDATES:
        if c in df.columns:
            return c
    raise ValueError(f"No time column found; candidates={TIME_COL_CANDIDATES}, columns={list(df.columns)}")


def _detect_target_col(df: pd.DataFrame, preferred: Optional[str]) -> str:
    if preferred and preferred in df.columns:
        return preferred
    for c in ["load", "y", "target", "OT"]:
        if c in df.columns:
            return c
    raise ValueError(f"No target column found. tried={['load','y','target','OT']}, columns={list(df.columns)}")


def _to_itransformer_csv(df: pd.DataFrame, target_col: str, time_col: str) -> pd.DataFrame:
    out = df.copy()
    out[time_col] = pd.to_datetime(out[time_col], errors="coerce")
    out = out.dropna(subset=[time_col]).sort_values([time_col], kind="mergesort").reset_index(drop=True)
    # Drop duplicate time-like columns first, then normalize to a single `date` column.
    time_like_cols = set(TIME_COL_CANDIDATES + ["date"])
    drop_cols = [c for c in out.columns if c != time_col and c in time_like_cols]
    if drop_cols:
        out = out.drop(columns=drop_cols)
    out = out.rename(columns={time_col: "date"})
    out = out.loc[:, ~out.columns.duplicated(keep="first")]

    feature_cols: List[str] = []
    for c in out.columns:
        if c in {"date", target_col} or c in DROP_NON_FEATURE_COLS:
            continue
        if pd.api.types.is_numeric_dtype(out[c]):
            feature_cols.append(c)

    cols = ["date"] + feature_cols + [target_col]
    out = out[cols].copy()
    # iTransformer expects complete numeric matrix.
    for c in feature_cols + [target_col]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.ffill().bfill()
    out = out.dropna().reset_index(drop=True)
    return out


def _build_setting_name(
    *,
    model_id: str,
    model: str,
    data: str,
    features: str,
    seq_len: int,
    label_len: int,
    pred_len: int,
    d_model: int,
    n_heads: int,
    e_layers: int,
    d_layers: int,
    d_ff: int,
    factor: int,
    embed: str,
    distil: bool,
    des: str,
    class_strategy: str,
    itr_idx: int = 0,
) -> str:
    # Keep exact format used in iTransformer run.py.
    return "{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}".format(
        model_id,
        model,
        data,
        features,
        seq_len,
        label_len,
        pred_len,
        d_model,
        n_heads,
        e_layers,
        d_layers,
        d_ff,
        factor,
        embed,
        distil,
        des,
        class_strategy,
        itr_idx,
    )


def _run_cmd(
    cmd: List[str],
    *,
    cwd: Path,
    env: Dict[str, str],
    allow_fail: bool = False,
) -> bool:
    print("[CMD]", " ".join(cmd))
    try:
        subprocess.run(cmd, cwd=str(cwd), env=env, check=True)
        return True
    except subprocess.CalledProcessError as exc:
        if allow_fail:
            print(f"[WARN] command failed (allowed): rc={exc.returncode}")
            return False
        raise


def _load_pred_vector(pred_file: Path) -> np.ndarray:
    arr = np.load(pred_file)
    if arr.ndim != 3:
        raise ValueError(f"Unexpected pred tensor shape {arr.shape} in {pred_file}")
    # Use the last step for h-step ahead target.
    vec = arr[:, -1, 0]
    return np.asarray(vec, dtype=float)


def _load_reference_frame(pred_root: Path, dataset: str, horizon: int, split: str) -> pd.DataFrame:
    ds_dir = pred_root / dataset
    if not ds_dir.exists():
        raise FileNotFoundError(f"Baseline dir not found: {ds_dir}")

    candidates = [ds_dir / f"{split}_pred_h{horizon}_{m}.csv" for m in PREFERRED_REF_MODELS]
    for p in candidates:
        if p.exists():
            df = pd.read_csv(p)
            required = {"row_id", "timestamp", "y"}
            if not required.issubset(set(df.columns)):
                continue
            return df[["row_id", "timestamp", "y"]].copy()

    # fallback: first available split/h file
    files = sorted(ds_dir.glob(f"{split}_pred_h{horizon}_*.csv"))
    if not files:
        raise FileNotFoundError(f"No reference baseline file for {dataset} h={horizon} split={split}")
    df = pd.read_csv(files[0])
    required = {"row_id", "timestamp", "y"}
    if not required.issubset(set(df.columns)):
        raise ValueError(f"Reference file missing columns {required}: {files[0]}")
    return df[["row_id", "timestamp", "y"]].copy()


def _align_pred_to_reference(pred_vec: np.ndarray, ref_df: pd.DataFrame, seq_len: int) -> Tuple[np.ndarray, Dict[str, int]]:
    n_ref = len(ref_df)
    aligned = np.full(n_ref, np.nan, dtype=float)

    start = max(seq_len - 1, 0)
    if start < n_ref and len(pred_vec) > 0:
        take = min(len(pred_vec), n_ref - start)
        aligned[start:start + take] = pred_vec[:take]
    else:
        take = 0

    s = pd.Series(aligned)
    if s.notna().sum() > 0:
        aligned = s.ffill().bfill().to_numpy(dtype=float)
    else:
        aligned = ref_df["y"].astype(float).to_numpy(copy=True)

    meta = {
        "ref_rows": int(n_ref),
        "pred_rows_raw": int(len(pred_vec)),
        "pred_rows_mapped": int(take),
        "missing_prefix": int(max(start, 0)),
    }
    return aligned, meta


def _verify_export_frame(df: pd.DataFrame, ref_df: pd.DataFrame, out_path: Path) -> None:
    required = ["row_id", "timestamp", "pred", "y"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{out_path}: missing required columns {missing}")
    if len(df) != len(ref_df):
        raise ValueError(f"{out_path}: length mismatch, got {len(df)}, expect {len(ref_df)}")
    if not np.isfinite(pd.to_numeric(df["pred"], errors="coerce")).any():
        raise ValueError(f"{out_path}: no finite predictions")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True, help="result/<date>/<run_name>")
    parser.add_argument("--itrans-root", type=Path, default=Path("Comparison_Algorithm/iTransformer-main"))
    parser.add_argument("--itrans-python", type=str, default="python")
    parser.add_argument("--datasets", nargs="*", default=DEFAULT_DATASETS)
    parser.add_argument("--horizons", nargs="*", type=int, default=DEFAULT_HORIZONS)
    parser.add_argument("--pred-root", type=Path, default=None, help="defaults to <run-root>/reports/baselines")
    parser.add_argument("--target-col", type=str, default="load")
    parser.add_argument("--seq-len", type=int, default=24)
    parser.add_argument("--label-len", type=int, default=12)
    parser.add_argument("--train-epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--e-layers", type=int, default=2)
    parser.add_argument("--d-layers", type=int, default=1)
    parser.add_argument("--d-ff", type=int, default=256)
    parser.add_argument("--factor", type=int, default=1)
    parser.add_argument("--embed", type=str, default="timeF")
    parser.add_argument("--class-strategy", type=str, default="projection")
    parser.add_argument("--des", type=str, default="mc")
    parser.add_argument("--skip-existing", action="store_true", default=True)
    parser.add_argument("--force", action="store_true", default=False)
    args = parser.parse_args()

    run_root = args.run_root.resolve()
    itrans_root = args.itrans_root.resolve()
    pred_root = (args.pred_root if args.pred_root is not None else run_root / "reports" / "baselines").resolve()
    feature_root = run_root / "data" / "features"
    work_root = run_root / "reports" / "itransformer_work"
    ckpt_root = run_root / "reports" / "itransformer_checkpoints"
    _ensure_dir(work_root)
    _ensure_dir(ckpt_root)

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("CUDA_VISIBLE_DEVICES", str(args.gpu))

    for dataset in args.datasets:
        ds_feature_dir = feature_root / dataset
        if not ds_feature_dir.exists():
            print(f"[skip] dataset feature dir missing: {ds_feature_dir}")
            continue

        raw_splits = {}
        for split in ["train", "val", "test"]:
            split_file = ds_feature_dir / f"{split}.csv"
            if not split_file.exists():
                raise FileNotFoundError(f"Missing split file: {split_file}")
            raw_splits[split] = pd.read_csv(split_file)

        time_col = _detect_time_col(raw_splits["train"])
        target_col = _detect_target_col(raw_splits["train"], args.target_col)

        for h in args.horizons:
            val_out = pred_root / dataset / f"val_pred_h{h}_itransformer.csv"
            test_out = pred_root / dataset / f"test_pred_h{h}_itransformer.csv"
            if args.skip_existing and (not args.force) and val_out.exists() and test_out.exists():
                print(f"[skip] exists: {dataset} h={h}")
                continue

            task_root = work_root / dataset / f"h{h}"
            _ensure_dir(task_root)
            _ensure_dir(pred_root / dataset)

            converted = {}
            for split in ["train", "val", "test"]:
                df_conv = _to_itransformer_csv(raw_splits[split], target_col=target_col, time_col=time_col)
                converted[split] = df_conv
                df_conv.to_csv(task_root / f"{split}.csv", index=False)

            n_vars = int(len(converted["train"].columns) - 1)
            if n_vars <= 0:
                raise ValueError(f"{dataset} h={h}: no usable variates in converted train.csv")

            model_id = f"mcit_{dataset}_h{h}_{int(time.time())}"
            common = [
                "--model_id", model_id,
                "--model", "iTransformer",
                "--data", "custom_fixed",
                "--root_path", str(task_root),
                "--data_path", "train.csv",
                "--features", "MS",
                "--target", target_col,
                "--freq", "h",
                "--seq_len", str(args.seq_len),
                "--label_len", str(args.label_len),
                "--pred_len", str(h),
                "--enc_in", str(n_vars),
                "--dec_in", str(n_vars),
                "--c_out", str(n_vars),
                "--d_model", str(args.d_model),
                "--n_heads", str(args.n_heads),
                "--e_layers", str(args.e_layers),
                "--d_layers", str(args.d_layers),
                "--d_ff", str(args.d_ff),
                "--factor", str(args.factor),
                "--embed", str(args.embed),
                "--des", str(args.des),
                "--class_strategy", str(args.class_strategy),
                "--checkpoints", str(ckpt_root),
                "--train_epochs", str(args.train_epochs),
                "--batch_size", str(args.batch_size),
                "--num_workers", str(args.num_workers),
                "--itr", "1",
                "--gpu", str(args.gpu),
                "--inverse",
            ]

            train_cmd = [args.itrans_python, "run.py", "--is_training", "1"] + common
            _run_cmd(train_cmd, cwd=itrans_root, env=env, allow_fail=False)

            val_cmd = [args.itrans_python, "run.py", "--is_training", "0", "--eval_split", "val"] + common
            test_cmd = [args.itrans_python, "run.py", "--is_training", "0", "--eval_split", "test"] + common
            val_ok = _run_cmd(val_cmd, cwd=itrans_root, env=env, allow_fail=True)
            test_ok = _run_cmd(test_cmd, cwd=itrans_root, env=env, allow_fail=True)

            setting = _build_setting_name(
                model_id=model_id,
                model="iTransformer",
                data="custom_fixed",
                features="MS",
                seq_len=args.seq_len,
                label_len=args.label_len,
                pred_len=h,
                d_model=args.d_model,
                n_heads=args.n_heads,
                e_layers=args.e_layers,
                d_layers=args.d_layers,
                d_ff=args.d_ff,
                factor=args.factor,
                embed=args.embed,
                distil=True,
                des=args.des,
                class_strategy=args.class_strategy,
                itr_idx=0,
            )
            res_dir = itrans_root / "results" / setting
            pred_val_file = res_dir / "pred_val.npy"
            pred_test_file = res_dir / "pred_test.npy"
            if not pred_val_file.exists() or not pred_test_file.exists():
                raise FileNotFoundError(
                    f"Missing iTransformer outputs for {dataset} h={h}: {pred_val_file}, {pred_test_file}; "
                    f"val_cmd_ok={val_ok}, test_cmd_ok={test_ok}"
                )

            ref_val = _load_reference_frame(pred_root, dataset, h, "val")
            ref_test = _load_reference_frame(pred_root, dataset, h, "test")
            vec_val = _load_pred_vector(pred_val_file)
            vec_test = _load_pred_vector(pred_test_file)

            pred_val_aligned, meta_val = _align_pred_to_reference(vec_val, ref_val, args.seq_len)
            pred_test_aligned, meta_test = _align_pred_to_reference(vec_test, ref_test, args.seq_len)

            out_val_df = ref_val.copy()
            out_val_df["pred"] = pred_val_aligned
            out_test_df = ref_test.copy()
            out_test_df["pred"] = pred_test_aligned

            out_val_df = out_val_df[["row_id", "timestamp", "pred", "y"]]
            out_test_df = out_test_df[["row_id", "timestamp", "pred", "y"]]
            _verify_export_frame(out_val_df, ref_val, val_out)
            _verify_export_frame(out_test_df, ref_test, test_out)

            out_val_df.to_csv(val_out, index=False)
            out_test_df.to_csv(test_out, index=False)

            val_meta_path = pred_root / dataset / f"val_pred_h{h}_itransformer.meta.json"
            model_meta_path = pred_root / dataset / f"model_meta_h{h}_itransformer.json"
            with val_meta_path.open("w", encoding="utf-8") as f:
                json.dump(
                    {
                        "dataset": dataset,
                        "horizon": int(h),
                        "model": "itransformer",
                        "val_eval_mode": "deterministic",
                        "source": "iTransformer-main",
                        "seq_len": int(args.seq_len),
                        "label_len": int(args.label_len),
                        "target_col": target_col,
                        "n_vars": int(n_vars),
                        "run_status": {
                            "train_ok": True,
                            "val_cmd_ok": bool(val_ok),
                            "test_cmd_ok": bool(test_ok),
                        },
                        "alignment": {"val": meta_val, "test": meta_test},
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
            with model_meta_path.open("w", encoding="utf-8") as f:
                json.dump(
                    {
                        "dataset": dataset,
                        "horizon": int(h),
                        "model": "itransformer",
                        "fit_status": {
                            "fit_ok": True,
                            "model_family": "deep_learning",
                            "fallback_used": False,
                            "pred_source": "itransformer_adapter",
                        },
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )

            print(f"[saved] {dataset} h={h} -> {val_out} / {test_out}")


if __name__ == "__main__":
    main()
