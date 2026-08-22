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

import sys
import os
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.pipeline.main import PowerPredictionPipeline


def main():
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


if __name__ == "__main__":
    sys.exit(main())