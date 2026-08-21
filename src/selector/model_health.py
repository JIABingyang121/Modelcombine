"""
模型健康诊断模块 (P1.1)

功能：
1. 对每个 (dataset, horizon, model) 进行多维度健康检查
2. 自动下线不健康模型，并记录原因码
3. 支持定期重评估与自动重上线
4. 输出 model_health.json 供组合策略使用

诊断维度：
- val RMSE 相对 seasonal_naive 的倍率 (> 5× → 下线)
- val 预测方差 / y 方差 (< 0.01 → 常数预测，下线)
- val MAE 排名 (末位且 > 中位数×2 → 下线)
- 场景适配评分 (RMSE 排名 > 中位数 → 场景不适配，标记)
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict
from pathlib import Path
import json
import numpy as np
from datetime import datetime


# ============================================================================
# 健康状态与原因码
# ============================================================================

class HealthStatus:
    ENABLED = "enabled"
    DISABLED = "disabled"
    SCENE_LIMITED = "scene_limited"  # 仅在部分场景启用


class ReasonCode:
    HEALTHY = "healthy"
    RMSE_VS_NAIVE_TOO_HIGH = "rmse_vs_naive_too_high"       # RMSE > 5× seasonal_naive
    CONSTANT_PREDICTION = "constant_prediction"              # 预测方差 < 0.01 × y 方差
    MAE_RANK_LAST_AND_OUTLIER = "mae_rank_last_and_outlier"  # 末位且 > 中位数×2
    TRAIN_INSTABILITY = "train_instability"                  # 训练不稳定 (如 LightGBM 无增益)
    SCENE_MISMATCH = "scene_mismatch"                        # 场景不适配


@dataclass
class ModelHealthRecord:
    """单个模型的健康记录"""
    dataset: str
    horizon: int
    model: str
    status: str = HealthStatus.ENABLED
    reason_code: str = ReasonCode.HEALTHY
    reason_detail: str = ""
    val_rmse: float = float('nan')
    naive_rmse: float = float('nan')
    rmse_ratio: float = float('nan')       # val_rmse / naive_rmse
    pred_var_ratio: float = float('nan')   # pred_var / y_var
    mae_rank: int = 0                       # 1 = 最佳
    mae_rank_total: int = 0
    scene_rank: int = 0                     # RMSE 排名 (用于场景路由)
    last_check_round: int = 0
    last_check_time: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ModelHealthChecker:
    """
    模型健康诊断器
    
    使用方式：
        checker = ModelHealthChecker()
        records = checker.check_all(
            dataset="pjm", horizon=1,
            model_cols=["xgboost_reg", "lgbm_reg", ...],
            val_preds={"xgboost_reg": np.array([...]), ...},
            y_val=np.array([...]),
            naive_rmse=123.4
        )
        enabled = checker.get_enabled_models(records)
    """
    
    def __init__(
        self,
        rmse_naive_threshold: float = 5.0,
        var_ratio_threshold: float = 0.01,
        mae_outlier_factor: float = 2.0,
        scene_rank_limit_enabled: bool = True,
        scene_rank_median_threshold: Optional[bool] = None,
    ):
        """
        Args:
            rmse_naive_threshold: RMSE / naive_RMSE 超过此值则下线
            var_ratio_threshold: pred_var / y_var 低于此值则判定为常数预测
            mae_outlier_factor: MAE 超过中位数的此倍数且排名末位则下线
            scene_rank_limit_enabled: 是否启用场景排名阈值（当前实现为分位阈值）
            scene_rank_median_threshold: 兼容旧参数名（已弃用，传入时覆盖 scene_rank_limit_enabled）
        """
        self.rmse_naive_threshold = rmse_naive_threshold
        self.var_ratio_threshold = var_ratio_threshold
        self.mae_outlier_factor = mae_outlier_factor
        if scene_rank_median_threshold is not None:
            scene_rank_limit_enabled = bool(scene_rank_median_threshold)
        self.scene_rank_limit_enabled = bool(scene_rank_limit_enabled)
        # backward compatibility
        self.scene_rank_median_threshold = self.scene_rank_limit_enabled
    
    def check_all(
        self,
        dataset: str,
        horizon: int,
        model_cols: List[str],
        val_preds: Dict[str, np.ndarray],
        y_val: np.ndarray,
        naive_rmse: float = None,
        check_round: int = 0,
        train_status_map: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> List[ModelHealthRecord]:
        """
        对所有模型执行健康检查
        
        Returns:
            每个模型的 ModelHealthRecord 列表
        """
        records = []
        
        # 预计算全局统计量
        y_var = float(np.var(y_val)) if len(y_val) > 0 else 1.0
        
        # 计算每个模型的 MAE 和 RMSE
        model_mae = {}
        model_rmse = {}
        for m in model_cols:
            if m not in val_preds:
                continue
            pred = val_preds[m]
            if pred is None or len(pred) != len(y_val):
                continue
            valid = ~np.isnan(pred)
            if valid.sum() < 10:
                continue
            model_mae[m] = float(np.mean(np.abs(y_val[valid] - pred[valid])))
            model_rmse[m] = float(np.sqrt(np.mean((y_val[valid] - pred[valid]) ** 2)))
        
        if not model_mae:
            return records
        
        # MAE 排名与中位数
        sorted_by_mae = sorted(model_mae.items(), key=lambda x: x[1])
        mae_ranks = {m: rank + 1 for rank, (m, _) in enumerate(sorted_by_mae)}
        median_mae = float(np.median(list(model_mae.values())))
        
        # RMSE 排名 (用于场景路由)
        sorted_by_rmse = sorted(model_rmse.items(), key=lambda x: x[1])
        rmse_ranks = {m: rank + 1 for rank, (m, _) in enumerate(sorted_by_rmse)}
        # 使用 75th percentile 作为 scene_limited 阈值；小样本任务回退到中位数，
        # 避免候选数量过少时阈值过于极端导致误判。
        n_rank_models = max(len(model_rmse), 1)
        if n_rank_models < 4:
            scene_rank_threshold = float(np.ceil(n_rank_models * 0.5))
        else:
            scene_rank_threshold = float(np.ceil(n_rank_models * 0.75))
        
        now = datetime.now().isoformat()
        
        for m in model_cols:
            record = ModelHealthRecord(
                dataset=dataset,
                horizon=horizon,
                model=m,
                last_check_round=check_round,
                last_check_time=now,
            )
            
            if m not in model_mae:
                record.status = HealthStatus.DISABLED
                record.reason_code = ReasonCode.TRAIN_INSTABILITY
                record.reason_detail = f"模型 {m} 无有效预测"
                records.append(record)
                continue
            
            pred = val_preds[m]
            record.val_rmse = model_rmse.get(m, float('nan'))
            record.mae_rank = mae_ranks.get(m, 0)
            record.mae_rank_total = len(model_mae)
            record.scene_rank = rmse_ranks.get(m, 0)

            status_info = (train_status_map or {}).get(m, {})
            fit_ok = bool(status_info.get("fit_ok", True))
            conv_warn_count = int(status_info.get("convergence_warning_count", 0) or 0)
            forced_scene_limited = False
            if not fit_ok:
                record.status = HealthStatus.DISABLED
                record.reason_code = ReasonCode.TRAIN_INSTABILITY
                detail = status_info.get("error") or status_info.get("fit_status_error") or "fit_ok=False"
                record.reason_detail = f"训练状态异常: {detail}"
                records.append(record)
                continue
            if conv_warn_count > 0:
                forced_scene_limited = True
            
            # 检查1: RMSE 相对 seasonal_naive 的倍率
            if naive_rmse is not None and naive_rmse > 1e-8:
                record.naive_rmse = naive_rmse
                record.rmse_ratio = record.val_rmse / naive_rmse
                if record.rmse_ratio > self.rmse_naive_threshold:
                    record.status = HealthStatus.DISABLED
                    record.reason_code = ReasonCode.RMSE_VS_NAIVE_TOO_HIGH
                    record.reason_detail = (
                        f"RMSE/naive = {record.rmse_ratio:.2f} > {self.rmse_naive_threshold}×"
                    )
                    records.append(record)
                    continue
            
            # 检查2: 常数预测 (预测方差极低)
            pred_var = float(np.var(pred[~np.isnan(pred)])) if np.any(~np.isnan(pred)) else 0.0
            record.pred_var_ratio = pred_var / y_var if y_var > 1e-8 else 0.0
            if record.pred_var_ratio < self.var_ratio_threshold:
                record.status = HealthStatus.DISABLED
                record.reason_code = ReasonCode.CONSTANT_PREDICTION
                record.reason_detail = (
                    f"pred_var/y_var = {record.pred_var_ratio:.6f} < {self.var_ratio_threshold}"
                )
                records.append(record)
                continue
            
            # 检查3: MAE 排名末位且远超中位数
            if (record.mae_rank == len(model_mae) and 
                model_mae[m] > median_mae * self.mae_outlier_factor and
                len(model_mae) >= 3):
                record.status = HealthStatus.DISABLED
                record.reason_code = ReasonCode.MAE_RANK_LAST_AND_OUTLIER
                record.reason_detail = (
                    f"MAE排名末位({record.mae_rank}/{len(model_mae)}), "
                    f"MAE={model_mae[m]:.2f} > {median_mae:.2f}×{self.mae_outlier_factor}"
                )
                records.append(record)
                continue
            
            # 检查4: 场景适配评分 (非下线，仅标记)
            if self.scene_rank_limit_enabled and record.scene_rank > scene_rank_threshold:
                record.status = HealthStatus.SCENE_LIMITED
                record.reason_code = ReasonCode.SCENE_MISMATCH
                record.reason_detail = (
                    f"RMSE排名 {record.scene_rank}/{len(model_rmse)} > 阈值 {scene_rank_threshold:.0f}，场景适配较差"
                )
            elif forced_scene_limited:
                record.status = HealthStatus.SCENE_LIMITED
                record.reason_code = ReasonCode.TRAIN_INSTABILITY
                record.reason_detail = f"训练期收敛告警: convergence_warnings={conv_warn_count}"
            else:
                record.status = HealthStatus.ENABLED
                record.reason_code = ReasonCode.HEALTHY
            
            records.append(record)
        
        return records
    
    @staticmethod
    def get_enabled_models(records: List[ModelHealthRecord]) -> List[str]:
        """获取健康检查通过（enabled + scene_limited）的模型列表"""
        return [
            r.model for r in records 
            if r.status in (HealthStatus.ENABLED, HealthStatus.SCENE_LIMITED)
        ]
    
    @staticmethod
    def get_scene_fit_models(records: List[ModelHealthRecord]) -> List[str]:
        """获取场景适配良好的模型列表（仅 enabled，不含 scene_limited）"""
        return [r.model for r in records if r.status == HealthStatus.ENABLED]
    
    @staticmethod
    def get_disabled_models(records: List[ModelHealthRecord]) -> Dict[str, str]:
        """获取被下线的模型及原因"""
        return {
            r.model: r.reason_detail 
            for r in records 
            if r.status == HealthStatus.DISABLED
        }


def save_model_health(
    all_records: Dict[str, Dict[int, List[ModelHealthRecord]]],
    output_path: Path,
):
    """
    保存模型健康报告到 JSON
    
    Args:
        all_records: {dataset: {horizon: [ModelHealthRecord, ...]}}
        output_path: 输出路径
    """
    result = {}
    for ds, horizons in all_records.items():
        result[ds] = {}
        for h, records in horizons.items():
            result[ds][str(h)] = [r.to_dict() for r in records]
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)


def load_model_health(input_path: Path) -> Dict[str, Dict[int, List[ModelHealthRecord]]]:
    """
    加载模型健康报告
    
    Returns:
        {dataset: {horizon: [ModelHealthRecord, ...]}}
    """
    if not input_path.exists():
        return {}
    
    with open(input_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    
    result = {}
    for ds, horizons in raw.items():
        result[ds] = {}
        for h_str, records_raw in horizons.items():
            h = int(h_str)
            result[ds][h] = [
                ModelHealthRecord(**{k: v for k, v in r.items() if k in ModelHealthRecord.__dataclass_fields__})
                for r in records_raw
            ]
    return result
