#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
电力需求智能预测分析系统 - 主运行脚本

这是一个基于模型组合技术的电力客户数据挖掘系统，专门用于：
- 住宅区用电需求预测
- 充电桩用电需求预测  
- 服务区用电需求预测

系统特点：
1. 多模态数据融合（用电数据 + 天气数据 + 时间特征）
2. 智能模型选择和组合策略
3. 场景相似度计算和历史经验学习
4. 支持多种预测模型（Prophet、XGBoost、LightGBM等）
"""

import argparse
import sys
import os
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.pipeline.main import PowerPredictionPipeline


def _predict_command(argv):
    """在线模型库预测：匹配历史关系 -> 加载产物 -> 预测。输入不需要未来真实值。"""
    parser = argparse.ArgumentParser(prog="run.py predict")
    parser.add_argument("--database", required=True, help="SQLite 模型库路径")
    parser.add_argument("--scenario", required=True, help="场景 JSON 路径")
    parser.add_argument("--features", required=True, help="已准备的未来特征 CSV 路径")
    parser.add_argument("--output", required=True, help="预测输出 CSV 路径")
    args = parser.parse_args(argv)

    # 不复制旧流水线的宽泛异常捕获：输入错误与产物错误以非零退出暴露。
    from src.pipeline.main import library_predict

    trace = library_predict(
        database=args.database,
        scenario=args.scenario,
        features=args.features,
        output=args.output,
    )
    print(f"预测完成: {args.output}")
    print(f"trace: {trace['trace_path']}")
    print(
        f"匹配场景: {trace['scenario_id']} (相似度 {trace['scenario_similarity']:.4f})，"
        f"relation_id={trace['relation_id']}, prediction_run_id={trace['prediction_run_id']}"
    )
    return 0


def _feedback_command(argv):
    """在线反馈：用后来返回的真实值更新对应关系的实际表现统计。"""
    parser = argparse.ArgumentParser(prog="run.py feedback")
    parser.add_argument("--database", required=True, help="SQLite 模型库路径")
    parser.add_argument("--prediction-run-id", type=int, required=True, help="predict 输出的 prediction_run_id")
    parser.add_argument("--actual", required=True, help="真实值 CSV（含 timestamp 和 y）")
    args = parser.parse_args(argv)

    from src.pipeline.main import library_feedback

    result = library_feedback(
        database=args.database,
        prediction_run_id=args.prediction_run_id,
        actual=args.actual,
    )
    print(
        f"反馈已记录: prediction_run_id={result['prediction_run_id']}, "
        f"actual_mae={result['actual_mae']:.4f}"
    )
    print(
        f"关系 {result['relation_id']}: feedback_count={result['feedback_count']}, "
        f"mean_actual_mae={result['mean_actual_mae']:.4f}"
    )
    return 0


def _run_legacy():
    """运行电力需求预测流水线"""
    print("=" * 60)
    print("电力需求智能预测分析系统")
    print("基于模型组合技术的多区域用电预测")
    print("=" * 60)
    
    try:
        # 创建预测流水线
        pipeline = PowerPredictionPipeline()
        print(f"决策后端: {pipeline.backend_mode}（默认 protocol_b；"
              f"设置 MODELCOMBINE_PIPELINE_BACKEND=combinator 可显式回退到旧引擎）")
        
        # 运行完整流水线
        results = pipeline.run_prediction_pipeline()
        
        if "error" not in results:
            print("\n" + "=" * 60)
            print("预测任务完成！")
            print(f"整体性能指标：")
            print(f"  - RMSE: {results['overall']['RMSE']:.4f}")
            print(f"  - MAE:  {results['overall']['MAE']:.4f}")
            print(f"  - MAPE: {results['overall']['MAPE']:.4f}%")
            
            if results['summary']['region_count'] > 1:
                print(f"\n区域分析：")
                print(f"  - 处理区域数量: {results['summary']['region_count']}")
                print(f"  - 最佳预测区域: {results['summary']['best_region']}")
                print(f"  - 平均RMSE: {results['summary']['avg_rmse_by_region']:.4f}")
            
            print(f"\n结果文件保存在: {project_root}/reports/")
            print("  - predictions.csv: 详细预测结果")
            print("  - report.json: 完整评估报告")
            print("  - model_info.json: 模型配置信息（含实际决策后端与 trace 路径）")
            backend = results.get("backend") or {}
            if backend:
                print(f"\n本次决策后端: {backend.get('mode')}")
                for region, info in (backend.get("regions") or {}).items():
                    print(f"  - {region}: 输出来自 {info.get('final_output_from')}"
                          + (f"，yhat_source={info.get('yhat_source')}" if info.get("yhat_source") else "")
                          + (f"，trace={info.get('trace_path')}" if info.get("trace_path") else ""))
            print("=" * 60)
        else:
            print(f"预测失败: {results['error']}")
            return 1
            
    except Exception as e:
        print(f"运行出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


def main():
    argv = sys.argv[1:]
    if argv and argv[0] == "predict":
        return _predict_command(argv[1:])
    if argv and argv[0] == "feedback":
        return _feedback_command(argv[1:])
    return _run_legacy()


if __name__ == "__main__":
    sys.exit(main())