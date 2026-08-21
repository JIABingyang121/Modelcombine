from typing import Dict, List, Tuple
from sklearn.metrics import mean_absolute_error
from src.eval.metrics import robust_mae


def select_models_by_history(val_df, model_cols: List[str],
                             strategy: str = "top_k", top_k: int = 3,
                             threshold_ratio: float = 1.5,
                             robust_cfg: Dict[str, float] | None = None) -> Tuple[List[str], Dict[str, float]]:
    """
    根据验证集历史表现动态选择参与组合的模型。（全局选择版本）
    """
    y_val = val_df["y"].values
    robust_cfg = robust_cfg or {}
    use_robust = robust_cfg.get("enable", False)
    score_label = "robust_mae" if use_robust else "mae"

    model_scores = {}
    for m in model_cols:
        if use_robust:
            score = robust_mae(y_val, val_df[m].values, robust_cfg)
        else:
            score = float(mean_absolute_error(y_val, val_df[m].values))
        model_scores[m] = score

    sorted_models = sorted(model_scores.items(), key=lambda x: x[1])
    print(f"    模型验证集表现({score_label}): {dict(sorted_models)}")

    if strategy == "top_k":
        selected = [m for m, _ in sorted_models[:top_k]]
    elif strategy == "threshold":
        best_mae = sorted_models[0][1]
        cutoff = best_mae * threshold_ratio
        selected = [m for m, mae in sorted_models if mae <= cutoff]
    elif strategy == "positive_contrib":
        selected = [sorted_models[0][0]]
        current_best_mae = sorted_models[0][1]
        for m, _ in sorted_models[1:]:
            test_cols = selected + [m]
            X_test_combo = val_df[test_cols].values
            combo_pred = X_test_combo.mean(axis=1)
            combo_mae = mean_absolute_error(y_val, combo_pred)
            if combo_mae < current_best_mae:
                selected.append(m)
                current_best_mae = combo_mae
    else:
        selected = model_cols

    if not selected:
        selected = [sorted_models[0][0]]

    print(f"    动态选择的模型: {selected}")
    return selected, model_scores
