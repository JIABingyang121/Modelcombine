"""EvidenceStore：保存调度证据指针与摘要，不复制原始数据。"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def _stable_evidence_id(payload: Dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return "ev_" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class EvidenceRecord:
    scenario_id: str
    data_slice_ref: str
    feature_snapshot_ref: str = ""
    training_log_ref: str = ""
    residual_summary: Dict[str, float] = field(default_factory=dict)
    drift_events: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    evidence_id: str = ""

    def __post_init__(self) -> None:
        if not self.evidence_id:
            object.__setattr__(self, "evidence_id", _stable_evidence_id(self._identity_payload()))

    def _identity_payload(self) -> Dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "data_slice_ref": self.data_slice_ref,
            "feature_snapshot_ref": self.feature_snapshot_ref,
            "training_log_ref": self.training_log_ref,
            "residual_summary": self.residual_summary,
            "drift_events": self.drift_events,
            "metadata": self.metadata,
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            **self._identity_payload(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvidenceRecord":
        return cls(
            evidence_id=data.get("evidence_id", ""),
            scenario_id=data["scenario_id"],
            data_slice_ref=data["data_slice_ref"],
            feature_snapshot_ref=data.get("feature_snapshot_ref", ""),
            training_log_ref=data.get("training_log_ref", ""),
            residual_summary=dict(data.get("residual_summary", {})),
            drift_events=list(data.get("drift_events", [])),
            metadata=dict(data.get("metadata", {})),
        )


class EvidenceStore:
    """按 evidence_id 存取证据记录。

    这里只保存路径/时间范围等指针和摘要统计，避免复制原始数据或完整残差序列。
    """

    def __init__(self, records: Optional[List[EvidenceRecord]] = None):
        self._records: Dict[str, EvidenceRecord] = {}
        for record in records or []:
            self.add(record)

    def add(self, record: EvidenceRecord) -> str:
        self._records[record.evidence_id] = record
        return record.evidence_id

    def get(self, evidence_id: str) -> Optional[EvidenceRecord]:
        return self._records.get(evidence_id)

    def query_by_scenario(self, scenario_id: str) -> List[EvidenceRecord]:
        return [r for r in self._records.values() if r.scenario_id == scenario_id]

    def query_by_drift_event(self, drift_event: str) -> List[EvidenceRecord]:
        return [r for r in self._records.values() if drift_event in r.drift_events]

    def query_by_data_slice(self, data_slice_prefix: str) -> List[EvidenceRecord]:
        return [r for r in self._records.values() if r.data_slice_ref.startswith(data_slice_prefix)]

    def to_list(self) -> List[Dict[str, Any]]:
        return [record.to_dict() for record in self._records.values()]

    def save_json(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_list(), ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load_json(cls, path: str | Path) -> "EvidenceStore":
        path = Path(path)
        if not path.exists():
            return cls()
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls([EvidenceRecord.from_dict(item) for item in data])
