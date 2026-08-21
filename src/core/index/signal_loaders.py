"""Load real latency/drift signal records from report artifacts."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def load_latency_records_from_ablation_profile(path: str | Path) -> List[Dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    data = _read_json(path)
    if not isinstance(data, dict):
        return []
    records: List[Dict[str, Any]] = []
    for dataset, horizons in data.items():
        if not isinstance(horizons, dict):
            continue
        for horizon_raw, payload in horizons.items():
            horizon = _coerce_int(horizon_raw)
            if horizon is None or not isinstance(payload, dict):
                continue
            resources = payload.get("resources", {})
            if not isinstance(resources, dict):
                continue
            for strategy_id, metrics in resources.items():
                if not isinstance(metrics, dict):
                    continue
                latency_ms = _coerce_float(metrics.get("predict_time_ms"))
                if latency_ms is None:
                    continue
                record = {
                    "scenario_id": _scenario_id(dataset, horizon),
                    "dataset": str(dataset),
                    "horizon": int(horizon),
                    "model_id": str(strategy_id),
                    "strategy_id": str(strategy_id),
                    "latency_ms": float(latency_ms),
                    "source": str(path),
                }
                optional_map = {
                    "train_time_ms": "train_time_ms",
                    "memory_mb": "rss_mb",
                    "models_used": "models_used",
                }
                for out_key, in_key in optional_map.items():
                    value = metrics.get(in_key)
                    if value is not None:
                        record[out_key] = value
                records.append(record)
    return records


def load_latency_records_from_kg_results(path: str | Path) -> List[Dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    data = _read_json(path)
    records: List[Dict[str, Any]] = []
    for dataset, horizon_raw, payload in _iter_task_payloads(data):
        meta = payload.get("_meta", {}) if isinstance(payload, dict) else {}
        runtime_sec = _coerce_float(meta.get("runtime_protocol_b_sec"))
        if runtime_sec is None:
            continue
        horizon = int(horizon_raw)
        solver_meta = meta.get("protocol_b_solver", {})
        record = {
            "scenario_id": _scenario_id(dataset, horizon),
            "dataset": str(dataset),
            "horizon": horizon,
            "model_id": "kg_protocol_b",
            "strategy_id": "kg_protocol_b",
            "latency_ms": float(runtime_sec) * 1000.0,
            "runtime_sec": float(runtime_sec),
            "source": str(path),
        }
        if isinstance(solver_meta, dict):
            if solver_meta.get("trace_path") is not None:
                record["trace_path"] = solver_meta.get("trace_path")
            if solver_meta.get("trace_stages") is not None:
                record["trace_stages"] = solver_meta.get("trace_stages")
        records.append(record)
    return records


def load_drift_records_from_kg_results(path: str | Path) -> List[Dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    data = _read_json(path)
    records: List[Dict[str, Any]] = []
    for dataset, horizon_raw, payload in _iter_task_payloads(data):
        horizon = int(horizon_raw)
        guard_config = _extract_protocol_b_guard_config(payload)
        if not guard_config:
            continue
        median_psi = _coerce_float(guard_config.get("drift_median_psi"))
        drift_level = (
            guard_config.get("drift_level_effective")
            or guard_config.get("drift_level_raw")
        )
        if median_psi is None and drift_level is None:
            continue
        records.append({
            "scenario_id": _scenario_id(dataset, horizon),
            "dataset": str(dataset),
            "horizon": horizon,
            "model_id": "kg_protocol_b",
            "strategy_id": "kg_protocol_b",
            "drift_level": str(drift_level or "unknown"),
            "drift_level_raw": guard_config.get("drift_level_raw"),
            "drift_level_source": guard_config.get("drift_level_source"),
            "median_psi": float(median_psi) if median_psi is not None else 0.0,
            "source": str(path),
        })
    return records


def build_signal_records_from_reports(
    *,
    ablation_profile_paths: Optional[Iterable[str | Path]] = None,
    kg_result_paths: Optional[Iterable[str | Path]] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    latency_records: List[Dict[str, Any]] = []
    drift_records: List[Dict[str, Any]] = []
    for path in ablation_profile_paths or []:
        latency_records.extend(load_latency_records_from_ablation_profile(path))
    for path in kg_result_paths or []:
        latency_records.extend(load_latency_records_from_kg_results(path))
        drift_records.extend(load_drift_records_from_kg_results(path))
    return {"latency": latency_records, "drift": drift_records}


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _iter_task_payloads(data: Any):
    if not isinstance(data, dict):
        return
    for dataset, horizons in data.items():
        if not isinstance(horizons, dict):
            continue
        for horizon_raw, payload in horizons.items():
            horizon = _coerce_int(horizon_raw)
            if horizon is None or not isinstance(payload, dict):
                continue
            yield str(dataset), horizon, payload


def _extract_protocol_b_guard_config(payload: Dict[str, Any]) -> Dict[str, Any]:
    meta = payload.get("_meta", {}) if isinstance(payload, dict) else {}
    protocol_b_meta = meta.get("protocol_b", {})
    if isinstance(protocol_b_meta, dict) and isinstance(protocol_b_meta.get("guard_config"), dict):
        return dict(protocol_b_meta["guard_config"])
    protocol_b = payload.get("protocol_B") or payload.get("kg_protocol_b") or {}
    if isinstance(protocol_b, dict):
        for split in ("val", "test"):
            split_payload = protocol_b.get(split, {})
            if not isinstance(split_payload, dict):
                continue
            weight_meta = split_payload.get("weight_meta", {})
            if isinstance(weight_meta, dict) and isinstance(weight_meta.get("guard_config"), dict):
                return dict(weight_meta["guard_config"])
    return {}


def _scenario_id(dataset: str, horizon: int) -> str:
    return f"{dataset}_h{int(horizon)}"


def _coerce_int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _coerce_float(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out
