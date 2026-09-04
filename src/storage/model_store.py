"""SQLite 模型库：训练模型实例、场景、数据特征与最佳组合关系的持久化。

只用标准库 ``sqlite3``。表结构见
``docs/superpowers/plans/2026-09-01-sqlite-model-library-scenario-relations.md`` 第 8 节。

这是新在线链路的唯一关系真源；模型和组合器产物文件仍在文件系统，库里只保存
路径与描述。JSON 列一律 ``json.dumps(..., sort_keys=True)``；时间一律 ISO 8601
UTC 文本；写入全部参数化。
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from typing import Any, Mapping, Optional, Sequence

_SCHEMA_STATEMENTS: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS models (
        model_id TEXT PRIMARY KEY,
        model_type TEXT NOT NULL,
        task_type TEXT NOT NULL,
        artifact_path TEXT NOT NULL,
        required_features_json TEXT NOT NULL,
        model_params_json TEXT NOT NULL,
        lifecycle_stage TEXT NOT NULL,
        trained_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS scenarios (
        scenario_id TEXT PRIMARY KEY,
        task_type TEXT NOT NULL,
        business_domain TEXT NOT NULL,
        region TEXT NOT NULL,
        horizon INTEGER NOT NULL,
        freq TEXT NOT NULL,
        signature_json TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS data_profiles (
        data_profile_id INTEGER PRIMARY KEY,
        scenario_id TEXT NOT NULL REFERENCES scenarios(scenario_id),
        data_ref TEXT NOT NULL,
        target_column TEXT NOT NULL,
        features_json TEXT NOT NULL,
        sample_count INTEGER NOT NULL,
        start_at TEXT NOT NULL,
        end_at TEXT NOT NULL,
        signature_json TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS combinations (
        combination_id INTEGER PRIMARY KEY,
        strategy TEXT NOT NULL,
        artifact_path TEXT NOT NULL,
        created_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS combination_members (
        combination_id INTEGER NOT NULL REFERENCES combinations(combination_id),
        model_id TEXT NOT NULL REFERENCES models(model_id),
        member_order INTEGER NOT NULL,
        weight REAL NOT NULL,
        PRIMARY KEY (combination_id, model_id),
        UNIQUE (combination_id, member_order)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS scenario_data_combinations (
        relation_id INTEGER PRIMARY KEY,
        scenario_id TEXT NOT NULL REFERENCES scenarios(scenario_id),
        data_profile_id INTEGER NOT NULL REFERENCES data_profiles(data_profile_id),
        combination_id INTEGER NOT NULL REFERENCES combinations(combination_id),
        validation_mae REAL NOT NULL,
        test_mae REAL,
        use_count INTEGER NOT NULL DEFAULT 0,
        feedback_count INTEGER NOT NULL DEFAULT 0,
        mean_actual_mae REAL,
        last_used_at TEXT,
        updated_at TEXT NOT NULL,
        UNIQUE (scenario_id, data_profile_id, combination_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS prediction_runs (
        prediction_run_id INTEGER PRIMARY KEY,
        relation_id INTEGER NOT NULL REFERENCES scenario_data_combinations(relation_id),
        prediction_ref TEXT NOT NULL,
        created_at TEXT NOT NULL,
        actual_mae REAL,
        feedback_at TEXT
    )
    """,
)


class FeedbackAlreadyRecorded(RuntimeError):
    """对同一个 prediction run 第二次写反馈时抛出。"""


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _dumps(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True)


class ModelStore:
    def __init__(self, database: str) -> None:
        self._conn = sqlite3.connect(database)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")

    @property
    def connection(self) -> sqlite3.Connection:
        return self._conn

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "ModelStore":
        return self

    def __exit__(self, *exc: Any) -> bool:
        self.close()
        return False

    def create_schema(self) -> None:
        with self._conn:
            for statement in _SCHEMA_STATEMENTS:
                self._conn.execute(statement)

    # ------------------------------------------------------------------ models
    def add_model(
        self,
        *,
        model_id: str,
        model_type: str,
        task_type: str,
        artifact_path: str,
        required_features: Sequence[str],
        model_params: Mapping[str, Any],
        lifecycle_stage: str,
        trained_at: Optional[str] = None,
    ) -> str:
        with self._conn:
            self._conn.execute(
                "INSERT INTO models (model_id, model_type, task_type, artifact_path, "
                "required_features_json, model_params_json, lifecycle_stage, trained_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    model_id,
                    model_type,
                    task_type,
                    artifact_path,
                    _dumps(list(required_features)),
                    _dumps(dict(model_params)),
                    lifecycle_stage,
                    trained_at or _utcnow(),
                ),
            )
        return model_id

    def get_model(self, model_id: str) -> Optional[dict]:
        row = self._conn.execute(
            "SELECT * FROM models WHERE model_id = ?", (model_id,)
        ).fetchone()
        if row is None:
            return None
        record = dict(row)
        record["required_features"] = json.loads(record.pop("required_features_json"))
        record["model_params"] = json.loads(record.pop("model_params_json"))
        return record

    # --------------------------------------------------------------- scenarios
    def add_scenario(
        self,
        *,
        scenario_id: str,
        task_type: str,
        business_domain: str,
        region: str,
        horizon: int,
        freq: str,
        signature: Mapping[str, Any],
    ) -> str:
        with self._conn:
            self._conn.execute(
                "INSERT INTO scenarios (scenario_id, task_type, business_domain, region, "
                "horizon, freq, signature_json) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    scenario_id,
                    task_type,
                    business_domain,
                    region,
                    int(horizon),
                    freq,
                    _dumps(dict(signature)),
                ),
            )
        return scenario_id

    def get_scenario(self, scenario_id: str) -> Optional[dict]:
        row = self._conn.execute(
            "SELECT * FROM scenarios WHERE scenario_id = ?", (scenario_id,)
        ).fetchone()
        return None if row is None else self._scenario_record(row)

    def _scenario_record(self, row: sqlite3.Row) -> dict:
        record = dict(row)
        record["signature"] = json.loads(record.pop("signature_json"))
        return record

    def list_scenarios(
        self,
        *,
        task_type: Optional[str] = None,
        business_domain: Optional[str] = None,
        region: Optional[str] = None,
        horizon: Optional[int] = None,
        freq: Optional[str] = None,
    ) -> list[dict]:
        clauses: list[str] = []
        params: list[Any] = []
        for column, value in (
            ("task_type", task_type),
            ("business_domain", business_domain),
            ("region", region),
            ("horizon", None if horizon is None else int(horizon)),
            ("freq", freq),
        ):
            if value is not None:
                clauses.append(f"{column} = ?")
                params.append(value)
        sql = "SELECT * FROM scenarios"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY scenario_id"
        return [self._scenario_record(row) for row in self._conn.execute(sql, params)]

    def list_relations_for_scenario(self, scenario_id: str) -> list[dict]:
        """按在线匹配优先级排序：有真实反馈的关系优先（按 mean_actual_mae 升序），
        其次无反馈关系（按 validation_mae 升序）；数值相同按 relation_id 升序。
        use_count 不参与排序。"""
        rows = self._conn.execute(
            "SELECT * FROM scenario_data_combinations WHERE scenario_id = ? "
            "ORDER BY (feedback_count > 0) DESC, "
            "CASE WHEN feedback_count > 0 THEN mean_actual_mae ELSE validation_mae END ASC, "
            "relation_id ASC",
            (scenario_id,),
        ).fetchall()
        return [dict(row) for row in rows]

    # ----------------------------------------------------------- data_profiles
    def add_data_profile(
        self,
        *,
        scenario_id: str,
        data_ref: str,
        target_column: str,
        features: Sequence[str],
        sample_count: int,
        start_at: str,
        end_at: str,
        signature: Mapping[str, Any],
    ) -> int:
        with self._conn:
            cursor = self._conn.execute(
                "INSERT INTO data_profiles (scenario_id, data_ref, target_column, "
                "features_json, sample_count, start_at, end_at, signature_json) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    scenario_id,
                    data_ref,
                    target_column,
                    _dumps(list(features)),
                    int(sample_count),
                    start_at,
                    end_at,
                    _dumps(dict(signature)),
                ),
            )
            return int(cursor.lastrowid)

    def get_data_profile(self, data_profile_id: int) -> Optional[dict]:
        row = self._conn.execute(
            "SELECT * FROM data_profiles WHERE data_profile_id = ?", (int(data_profile_id),)
        ).fetchone()
        if row is None:
            return None
        record = dict(row)
        record["features"] = json.loads(record.pop("features_json"))
        record["signature"] = json.loads(record.pop("signature_json"))
        return record

    # ------------------------------------------------------------ combinations
    def add_combination(
        self,
        strategy: str,
        artifact_path: str,
        members: Sequence[tuple[str, int, float]],
        *,
        created_at: Optional[str] = None,
    ) -> int:
        created_at = created_at or _utcnow()
        with self._conn:
            cursor = self._conn.execute(
                "INSERT INTO combinations (strategy, artifact_path, created_at) VALUES (?, ?, ?)",
                (strategy, artifact_path, created_at),
            )
            combination_id = int(cursor.lastrowid)
            for model_id, member_order, weight in members:
                self._conn.execute(
                    "INSERT INTO combination_members (combination_id, model_id, member_order, weight) "
                    "VALUES (?, ?, ?, ?)",
                    (combination_id, model_id, int(member_order), float(weight)),
                )
        return combination_id

    def get_combination(self, combination_id: int) -> Optional[dict]:
        row = self._conn.execute(
            "SELECT * FROM combinations WHERE combination_id = ?", (int(combination_id),)
        ).fetchone()
        if row is None:
            return None
        record = dict(row)
        members = self._conn.execute(
            "SELECT model_id, member_order, weight FROM combination_members "
            "WHERE combination_id = ? ORDER BY member_order",
            (int(combination_id),),
        ).fetchall()
        record["members"] = [dict(member) for member in members]
        return record

    # ---------------------------------------------- scenario_data_combinations
    def add_relation(
        self,
        scenario_id: str,
        data_profile_id: int,
        combination_id: int,
        *,
        validation_mae: float,
        test_mae: Optional[float] = None,
        updated_at: Optional[str] = None,
    ) -> int:
        with self._conn:
            cursor = self._conn.execute(
                "INSERT INTO scenario_data_combinations (scenario_id, data_profile_id, "
                "combination_id, validation_mae, test_mae, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    scenario_id,
                    int(data_profile_id),
                    int(combination_id),
                    float(validation_mae),
                    None if test_mae is None else float(test_mae),
                    updated_at or _utcnow(),
                ),
            )
            return int(cursor.lastrowid)

    def get_relation(self, relation_id: int) -> Optional[dict]:
        row = self._conn.execute(
            "SELECT * FROM scenario_data_combinations WHERE relation_id = ?", (int(relation_id),)
        ).fetchone()
        return None if row is None else dict(row)

    # --------------------------------------------------------- prediction_runs
    def record_prediction_run(
        self,
        relation_id: int,
        prediction_ref: str,
        *,
        created_at: Optional[str] = None,
    ) -> int:
        created_at = created_at or _utcnow()
        with self._conn:
            cursor = self._conn.execute(
                "INSERT INTO prediction_runs (relation_id, prediction_ref, created_at) "
                "VALUES (?, ?, ?)",
                (int(relation_id), prediction_ref, created_at),
            )
            run_id = int(cursor.lastrowid)
            updated = self._conn.execute(
                "UPDATE scenario_data_combinations SET use_count = use_count + 1, "
                "last_used_at = ?, updated_at = ? WHERE relation_id = ?",
                (created_at, created_at, int(relation_id)),
            )
            if updated.rowcount != 1:
                raise ValueError(f"relation {relation_id} does not exist")
        return run_id

    def get_prediction_run(self, prediction_run_id: int) -> Optional[dict]:
        row = self._conn.execute(
            "SELECT * FROM prediction_runs WHERE prediction_run_id = ?",
            (int(prediction_run_id),),
        ).fetchone()
        return None if row is None else dict(row)

    def record_feedback(
        self,
        prediction_run_id: int,
        actual_mae: float,
        *,
        feedback_at: Optional[str] = None,
    ) -> None:
        feedback_at = feedback_at or _utcnow()
        actual_mae = float(actual_mae)
        with self._conn:
            updated = self._conn.execute(
                "UPDATE prediction_runs SET actual_mae = ?, feedback_at = ? "
                "WHERE prediction_run_id = ? AND actual_mae IS NULL",
                (actual_mae, feedback_at, int(prediction_run_id)),
            )
            if updated.rowcount != 1:
                raise FeedbackAlreadyRecorded(
                    f"prediction run {prediction_run_id} already has feedback or does not exist"
                )
            relation_id = int(
                self._conn.execute(
                    "SELECT relation_id FROM prediction_runs WHERE prediction_run_id = ?",
                    (int(prediction_run_id),),
                ).fetchone()["relation_id"]
            )
            relation = self._conn.execute(
                "SELECT feedback_count, mean_actual_mae FROM scenario_data_combinations "
                "WHERE relation_id = ?",
                (relation_id,),
            ).fetchone()
            old_count = int(relation["feedback_count"])
            old_mean = relation["mean_actual_mae"]
            if old_count == 0 or old_mean is None:
                new_mean = actual_mae
            else:
                new_mean = (old_mean * old_count + actual_mae) / (old_count + 1)
            self._conn.execute(
                "UPDATE scenario_data_combinations SET feedback_count = feedback_count + 1, "
                "mean_actual_mae = ?, updated_at = ? WHERE relation_id = ?",
                (new_mean, feedback_at, relation_id),
            )
