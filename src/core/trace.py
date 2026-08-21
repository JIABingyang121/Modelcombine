"""SelectionTrace：记录模型调度推理链，而不是只记录最终选择。"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


@dataclass
class SelectionTrace:
    """一次模型调度的可审计过程记录。

    Phase 2 只定义共享数据结构，不接入系统 A/B 的决策流程。
    """

    scenario_id: str
    timestamp: str = field(default_factory=_utc_now_iso)
    stages: List[Dict[str, Any]] = field(default_factory=list)
    candidates_considered: List[str] = field(default_factory=list)
    candidates_rejected: Dict[str, str] = field(default_factory=dict)
    final_selection: List[str] = field(default_factory=list)
    final_weights: Dict[str, float] = field(default_factory=dict)
    evidence_refs: List[str] = field(default_factory=list)

    def add_stage(
        self,
        stage: str,
        inputs: Optional[Dict[str, Any]] = None,
        outputs: Optional[Dict[str, Any]] = None,
        duration_ms: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        record: Dict[str, Any] = {"stage": stage}
        if inputs is not None:
            record["inputs"] = inputs
        if outputs is not None:
            record["outputs"] = outputs
        if duration_ms is not None:
            record["duration_ms"] = float(duration_ms)
        if metadata is not None:
            record["metadata"] = metadata
        self.stages.append(record)

    def reject(self, candidate_id: str, reason: str) -> None:
        self.candidates_rejected[candidate_id] = reason

    def consider(self, candidates: List[str]) -> None:
        seen = set(self.candidates_considered)
        for candidate in candidates:
            if candidate not in seen:
                self.candidates_considered.append(candidate)
                seen.add(candidate)

    def set_final(self, selection: List[str], weights: Dict[str, float]) -> None:
        self.final_selection = list(selection)
        self.final_weights = dict(weights)

    def add_evidence_ref(self, evidence_ref: str) -> None:
        self.evidence_refs.append(evidence_ref)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "timestamp": self.timestamp,
            "stages": self.stages,
            "candidates_considered": self.candidates_considered,
            "candidates_rejected": self.candidates_rejected,
            "final_selection": self.final_selection,
            "final_weights": self.final_weights,
            "evidence_refs": self.evidence_refs,
        }

    def save_json(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load_json(cls, path: str | Path) -> "SelectionTrace":
        path = Path(path)
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SelectionTrace":
        return cls(
            scenario_id=data["scenario_id"],
            timestamp=data.get("timestamp", _utc_now_iso()),
            stages=list(data.get("stages", [])),
            candidates_considered=list(data.get("candidates_considered", [])),
            candidates_rejected=dict(data.get("candidates_rejected", {})),
            final_selection=list(data.get("final_selection", [])),
            final_weights=dict(data.get("final_weights", {})),
            evidence_refs=list(data.get("evidence_refs", [])),
        )
