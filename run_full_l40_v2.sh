#!/usr/bin/env bash
# 运行与最小重跑规范: docs/runtime_iteration_workflow.md
set -euo pipefail

# Resolve project root robustly:
# 1) If PROJECT_ROOT is exported, use it.
# 2) Otherwise, default to the directory where this script lives.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$SCRIPT_DIR}"

if [ ! -d "$PROJECT_ROOT" ]; then
  echo "[ERROR] PROJECT_ROOT not found: $PROJECT_ROOT"
  echo "[HINT] export PROJECT_ROOT=$SCRIPT_DIR"
  exit 1
fi

# Basic repo-root sanity check to avoid running in a wrong directory.
if [ ! -f "$PROJECT_ROOT/run_full_l40_v2.sh" ] || [ ! -d "$PROJECT_ROOT/scripts" ]; then
  echo "[ERROR] PROJECT_ROOT does not look like repo root: $PROJECT_ROOT"
  echo "[HINT] use repo root, e.g. export PROJECT_ROOT=$SCRIPT_DIR"
  exit 1
fi

cd "$PROJECT_ROOT"

if [ ! -d "venv" ]; then
  echo "[ERROR] venv not found at: $PROJECT_ROOT/venv"
  exit 1
fi
source venv/bin/activate

RUN_DATE="$(date +%m%d)"
RUN_TIME="$(date +%H%M)"
RUN_NAME="${1:-${RUN_DATE}_${RUN_TIME}_3gpu_full_v2}"
RUN_PROFILE="${MODELCOMBINE_RUN_PROFILE:-full}"  # full | balanced | fast
REUSE_COMPLETED_STEPS="${MODELCOMBINE_REUSE_COMPLETED_STEPS:-1}"
KG_EXPERIMENT_PROFILE="${MODELCOMBINE_KG_EXPERIMENT_PROFILE:-default}"  # default | clean_noql_dash
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

RUN_ROOT="result/${RUN_DATE}/${RUN_NAME}"
FEATURE_DIR="${RUN_ROOT}/data/features"
BASE_TMP="${RUN_ROOT}/reports"
BASELINE_DIR="${BASE_TMP}/baselines"
MODEL_DIR="${BASE_TMP}/modelcombine"
KG_DIR="${BASE_TMP}/combos_kg"
ANALYSIS_DIR="${BASE_TMP}/analysis"
LOG_DIR="${RUN_ROOT}/logs"

# 记录运行前 RUN_ROOT 是否已存在内容（clean 模式门禁基于该值）
RUN_ROOT_PREEXIST_NONEMPTY=0
if [ -d "$RUN_ROOT" ] && find "$RUN_ROOT" -mindepth 1 -print -quit >/dev/null 2>&1; then
  RUN_ROOT_PREEXIST_NONEMPTY=1
fi

mkdir -p "$FEATURE_DIR" "$BASE_TMP" "$BASELINE_DIR" "$MODEL_DIR" "$KG_DIR" "$ANALYSIS_DIR" "$LOG_DIR"

echo "[INFO] PROJECT_ROOT=$PROJECT_ROOT"
echo "[INFO] RUN_ROOT=$RUN_ROOT"
echo "[INFO] RUN_PROFILE=$RUN_PROFILE"
echo "[INFO] REUSE_COMPLETED_STEPS=$REUSE_COMPLETED_STEPS"
echo "[INFO] KG_EXPERIMENT_PROFILE=$KG_EXPERIMENT_PROFILE"

# 实验配置模式：确保可复现且口径不污染
# clean_noql_dash: KG 不含 rl_qms，不含 itransformer 基础节点，强制包含 dash_tta 节点。
if [ "$KG_EXPERIMENT_PROFILE" = "clean_noql_dash" ]; then
  echo "[PROFILE] apply clean_noql_dash constraints"
  REUSE_COMPLETED_STEPS=0
  export MODELCOMBINE_REUSE_COMPLETED_STEPS=0
  export MODELCOMBINE_KG_INCLUDE_RL_QMS=0
  export MODELCOMBINE_KG_INCLUDE_DASH_TTA=1
  # 防止 iTransformer 通过组合池间接污染 KG 扩展节点。
  export MODELCOMBINE_ITRANSFORMER_INJECT_TO_COMBO=0
  export MODELCOMBINE_ITRANSFORMER_INJECT_TO_KG=0
  unset MODELCOMBINE_KG_EXTRA_BASE_CANDIDATES || true
  unset MODELCOMBINE_EXTRA_EXTERNAL_MODELS || true
  export MODELCOMBINE_AUDIT_CANDIDATES="${MODELCOMBINE_AUDIT_CANDIDATES:-dash_tta,stacking_safe,dynamic_stacking,static_weight_safe,mole_router}"
  export MODELCOMBINE_AUDIT_HIGH_DRIFT_PREFERRED_CANDIDATES="${MODELCOMBINE_AUDIT_HIGH_DRIFT_PREFERRED_CANDIDATES:-dash_tta,stacking_safe}"

  if [ "$RUN_ROOT_PREEXIST_NONEMPTY" = "1" ]; then
    echo "[ERROR] clean_noql_dash requires a fresh RUN_ROOT, but found existing contents: $RUN_ROOT"
    echo "[HINT] use a new RUN_NAME (script arg #1), e.g. bash run_full_l40_v2.sh 0224_XXXX_clean_noql_dash"
    exit 2
  fi
fi

echo "[INFO] effective REUSE_COMPLETED_STEPS=$REUSE_COMPLETED_STEPS"

baseline_artifacts_ready() {
  local root="$1"
  local ds h m
  for ds in pjm aemo_vic aemo_nsw; do
    for h in 1 6 24; do
      for m in xgboost_reg lgbm_reg catboost_reg; do
        [ -f "${root}/${ds}/val_pred_h${h}_${m}.csv" ] || return 1
        [ -f "${root}/${ds}/test_pred_h${h}_${m}.csv" ] || return 1
      done
    done
  done
  return 0
}

modelcombine_artifacts_ready() {
  local root="$1"
  [ -f "${root}/metrics.json" ] || return 1
  [ -f "${root}/leaderboard_full.csv" ] || return 1
  [ -f "${root}/routing_config.json" ] || return 1
  return 0
}

modelcombine_has_model() {
  local root="$1"
  local model="$2"
  [ -f "${root}/metrics.json" ] || return 1
  grep -q "\"${model}\"" "${root}/metrics.json"
}

append_csv_token_unique() {
  local csv="${1:-}"
  local token="$2"
  if [ -z "$csv" ]; then
    echo "$token"
    return 0
  fi
  local IFS=','
  local item
  for item in $csv; do
    item="${item// /}"
    if [ "$item" = "$token" ]; then
      echo "$csv"
      return 0
    fi
  done
  echo "${csv},${token}"
}

# 0) 快速语法检查（避免长跑后才发现代码错误）
echo "[0/9] py_compile sanity"
  python3 -m py_compile \
  scripts/train_baselines.py \
  src/eval/entrypoint.py \
  scripts/train_combinations_kg.py \
  scripts/audit_extended_candidates.py \
  scripts/compare_results.py \
  scripts/modelcombine_eval.py \
  scripts/run_itransformer_adapter.py \
  scripts/generate_features.py \
  scripts/split_datasets.py \
  2>&1 | tee "${LOG_DIR}/00_py_compile.log"

# 1) 数据切分
echo "[1/9] split datasets"
python scripts/split_datasets.py 2>&1 | tee "${LOG_DIR}/01_split.log"

for ds in pjm aemo_vic aemo_nsw; do
  for sp in train val test; do
    test -f "data/splits/${ds}/${sp}.csv" || { echo "[ERROR] missing split: data/splits/${ds}/${sp}.csv"; exit 1; }
  done
done

# 2) 特征生成
echo "[2/9] generate features"
python scripts/generate_features.py \
  --config configs/exp_comparativetest1.yaml \
  --out "$FEATURE_DIR" \
  2>&1 | tee "${LOG_DIR}/02_features.log"

for ds in pjm aemo_vic aemo_nsw; do
  for sp in train val test; do
    test -f "${FEATURE_DIR}/${ds}/${sp}.csv" || { echo "[ERROR] missing feature: ${FEATURE_DIR}/${ds}/${sp}.csv"; exit 1; }
  done
done

# 3) Baseline 三卡并行（每卡一个数据集）
if [ "$REUSE_COMPLETED_STEPS" = "1" ] && baseline_artifacts_ready "$BASELINE_DIR"; then
  echo "[3/9] train baselines in parallel (3 GPUs) - skip (reuse existing artifacts)"
  echo "[4/9] merge baseline artifacts - skip (reuse existing artifacts)"
else
  echo "[3/9] train baselines in parallel (3 GPUs)"

  (
    set -o pipefail
    MODELCOMBINE_USE_GPU=1 MODELCOMBINE_GPU_ID=0 CUDA_VISIBLE_DEVICES=0 MODELCOMBINE_CATBOOST_FORCE_CPU=1 \
    python scripts/train_baselines.py \
      --features "$FEATURE_DIR" \
      --out "${BASE_TMP}/baselines_pjm" \
      --datasets pjm \
      2>&1 | tee "${LOG_DIR}/03_baseline_pjm.log"
  ) &
  PID_PJM=$!

  (
    set -o pipefail
    MODELCOMBINE_USE_GPU=1 MODELCOMBINE_GPU_ID=1 CUDA_VISIBLE_DEVICES=1 MODELCOMBINE_CATBOOST_FORCE_CPU=1 \
    python scripts/train_baselines.py \
      --features "$FEATURE_DIR" \
      --out "${BASE_TMP}/baselines_vic" \
      --datasets aemo_vic \
      2>&1 | tee "${LOG_DIR}/03_baseline_vic.log"
  ) &
  PID_VIC=$!

  (
    set -o pipefail
    MODELCOMBINE_USE_GPU=1 MODELCOMBINE_GPU_ID=2 CUDA_VISIBLE_DEVICES=2 MODELCOMBINE_CATBOOST_FORCE_CPU=1 \
    python scripts/train_baselines.py \
      --features "$FEATURE_DIR" \
      --out "${BASE_TMP}/baselines_nsw" \
      --datasets aemo_nsw \
      2>&1 | tee "${LOG_DIR}/03_baseline_nsw.log"
  ) &
  PID_NSW=$!

  PJM_OK=1
  VIC_OK=1
  NSW_OK=1

  wait "$PID_PJM" || PJM_OK=0
  wait "$PID_VIC" || VIC_OK=0
  wait "$PID_NSW" || NSW_OK=0

  if [ "$PJM_OK" -ne 1 ] || [ "$VIC_OK" -ne 1 ]; then
    echo "[ERROR] baseline parallel jobs failed (pjm_ok=$PJM_OK, vic_ok=$VIC_OK, nsw_ok=$NSW_OK)"
    exit 1
  fi

  if [ "$NSW_OK" -ne 1 ]; then
    echo "[WARN] NSW baseline failed on GPU, retry on CPU..."
    (
      set -o pipefail
      MODELCOMBINE_USE_GPU=0 MODELCOMBINE_CATBOOST_FORCE_CPU=1 \
      python scripts/train_baselines.py \
        --features "$FEATURE_DIR" \
        --out "${BASE_TMP}/baselines_nsw" \
        --datasets aemo_nsw \
        2>&1 | tee "${LOG_DIR}/03_baseline_nsw_cpu_retry.log"
    )
  fi

  # 4) 合并 baseline 产物
  echo "[4/9] merge baseline artifacts"
  mkdir -p "$BASELINE_DIR"
  cp -a "${BASE_TMP}/baselines_pjm/pjm" "$BASELINE_DIR/"
  cp -a "${BASE_TMP}/baselines_vic/aemo_vic" "$BASELINE_DIR/"
  cp -a "${BASE_TMP}/baselines_nsw/aemo_nsw" "$BASELINE_DIR/"
fi

# baseline 完整性检查：每个数据集每个 horizon 至少有 xgboost/lgbm/catboost 的 val+test
baseline_artifacts_ready "$BASELINE_DIR" || {
  echo "[ERROR] baseline artifacts incomplete under: $BASELINE_DIR"
  exit 1
}

# 4.5) iTransformer 适配器（按需启用）
ENABLE_ITRANSFORMER="${MODELCOMBINE_ENABLE_ITRANSFORMER:-0}"
if [ "$ENABLE_ITRANSFORMER" = "1" ]; then
  echo "[4.5/9] run iTransformer adapter"
  ITRANS_ROOT="${MODELCOMBINE_ITRANSFORMER_ROOT:-Comparison_Algorithm/iTransformer-main}"
  ITRANS_PY="${MODELCOMBINE_ITRANSFORMER_PYTHON:-python}"
  ITRANS_GPU="${MODELCOMBINE_ITRANSFORMER_GPU:-0}"
  ITRANS_EPOCHS="${MODELCOMBINE_ITRANSFORMER_TRAIN_EPOCHS:-10}"
  ITRANS_BATCH="${MODELCOMBINE_ITRANSFORMER_BATCH_SIZE:-32}"
  ITRANS_WORKERS="${MODELCOMBINE_ITRANSFORMER_NUM_WORKERS:-0}"
  ITRANS_SEQ_LEN="${MODELCOMBINE_ITRANSFORMER_SEQ_LEN:-168}"
  ITRANS_LABEL_LEN="${MODELCOMBINE_ITRANSFORMER_LABEL_LEN:-48}"
  ITRANS_MIN_EPOCHS_HARD="${MODELCOMBINE_ITRANSFORMER_MIN_EPOCHS_HARD:-0}"
  ITRANS_FORCE="${MODELCOMBINE_ITRANSFORMER_FORCE:-0}"
  ITRANS_DATASETS_RAW="${MODELCOMBINE_ITRANSFORMER_DATASETS:-pjm,aemo_vic,aemo_nsw}"
  ITRANS_HORIZONS_RAW="${MODELCOMBINE_ITRANSFORMER_HORIZONS:-1,6,24}"
  ITRANS_DATASETS_NORM="${ITRANS_DATASETS_RAW//,/ }"
  ITRANS_HORIZONS_NORM="${ITRANS_HORIZONS_RAW//,/ }"
  read -r -a ITRANS_DATASETS_ARR <<< "${ITRANS_DATASETS_NORM}"
  read -r -a ITRANS_HORIZONS_ARR <<< "${ITRANS_HORIZONS_NORM}"
  ITRANS_FORCE_ARGS=()
  if [ "$ITRANS_FORCE" = "1" ]; then
    ITRANS_FORCE_ARGS+=(--force)
  fi
  echo "[INFO] iTransformer config: datasets=${ITRANS_DATASETS_RAW} horizons=${ITRANS_HORIZONS_RAW} gpu=${ITRANS_GPU} epochs=${ITRANS_EPOCHS} seq_len=${ITRANS_SEQ_LEN} label_len=${ITRANS_LABEL_LEN} batch=${ITRANS_BATCH} workers=${ITRANS_WORKERS} force=${ITRANS_FORCE}"
  if [ "$ITRANS_EPOCHS" -lt 2 ]; then
    echo "[WARN] iTransformer epochs=${ITRANS_EPOCHS} is very low; comparisons may be unfair."
    if [ "$ITRANS_MIN_EPOCHS_HARD" = "1" ]; then
      echo "[ERROR] MODELCOMBINE_ITRANSFORMER_MIN_EPOCHS_HARD=1 and epochs<2, abort."
      exit 2
    fi
  fi

  [ -f "scripts/run_itransformer_adapter.py" ] || { echo "[ERROR] missing scripts/run_itransformer_adapter.py"; exit 1; }
  [ -d "$ITRANS_ROOT" ] || { echo "[ERROR] missing iTransformer root: $ITRANS_ROOT"; exit 1; }

  python -u scripts/run_itransformer_adapter.py \
    --run-root "$RUN_ROOT" \
    --pred-root "$BASELINE_DIR" \
    --itrans-root "$ITRANS_ROOT" \
    --itrans-python "$ITRANS_PY" \
    --datasets "${ITRANS_DATASETS_ARR[@]}" \
    --horizons "${ITRANS_HORIZONS_ARR[@]}" \
    --gpu "$ITRANS_GPU" \
    --train-epochs "$ITRANS_EPOCHS" \
    --batch-size "$ITRANS_BATCH" \
    --num-workers "$ITRANS_WORKERS" \
    --seq-len "$ITRANS_SEQ_LEN" \
    --label-len "$ITRANS_LABEL_LEN" \
    "${ITRANS_FORCE_ARGS[@]}" \
    2>&1 | tee "${LOG_DIR}/04a_itransformer.log"

  ITRANS_INJECT_TO_COMBO="${MODELCOMBINE_ITRANSFORMER_INJECT_TO_COMBO:-1}"
  ITRANS_INJECT_TO_KG="${MODELCOMBINE_ITRANSFORMER_INJECT_TO_KG:-1}"
  if [ "$ITRANS_INJECT_TO_COMBO" = "1" ]; then
    export MODELCOMBINE_EXTRA_EXTERNAL_MODELS="$(append_csv_token_unique "${MODELCOMBINE_EXTRA_EXTERNAL_MODELS:-}" "itransformer")"
    echo "[INFO] MODELCOMBINE_EXTRA_EXTERNAL_MODELS=${MODELCOMBINE_EXTRA_EXTERNAL_MODELS}"
  else
    echo "[INFO] skip injecting itransformer into modelcombine external pool"
  fi
  if [ "$ITRANS_INJECT_TO_KG" = "1" ]; then
    export MODELCOMBINE_KG_EXTRA_BASE_CANDIDATES="$(append_csv_token_unique "${MODELCOMBINE_KG_EXTRA_BASE_CANDIDATES:-}" "itransformer")"
    echo "[INFO] MODELCOMBINE_KG_EXTRA_BASE_CANDIDATES=${MODELCOMBINE_KG_EXTRA_BASE_CANDIDATES}"
  else
    echo "[INFO] skip injecting itransformer into KG base candidates"
  fi
fi

# 5) ModelCombine 评估（严格 horizon 完整性门禁）
echo "[5/9] modelcombine eval"
export MODELCOMBINE_REUSE_BASELINE_TREE_PREDS="${MODELCOMBINE_REUSE_BASELINE_TREE_PREDS:-1}"
export MODELCOMBINE_EVAL_THREADS="${MODELCOMBINE_EVAL_THREADS:-16}"
if [ "$RUN_PROFILE" = "fast" ]; then
  export MODELCOMBINE_SKIP_COMPLEX_KG_COMPONENT="${MODELCOMBINE_SKIP_COMPLEX_KG_COMPONENT:-1}"
  export MODELCOMBINE_SKIP_SOTA="${MODELCOMBINE_SKIP_SOTA:-1}"
  export MODELCOMBINE_KG_COMPONENT_STRATEGIES="${MODELCOMBINE_KG_COMPONENT_STRATEGIES:-}"
elif [ "$RUN_PROFILE" = "balanced" ]; then
  export MODELCOMBINE_SKIP_COMPLEX_KG_COMPONENT="${MODELCOMBINE_SKIP_COMPLEX_KG_COMPONENT:-0}"
  export MODELCOMBINE_SKIP_SOTA="${MODELCOMBINE_SKIP_SOTA:-0}"
  # balanced 模式默认跳过最慢的 kg_component 子策略（adaptive_bucket/scenario_similarity）
  export MODELCOMBINE_KG_COMPONENT_STRATEGIES="${MODELCOMBINE_KG_COMPONENT_STRATEGIES:-gating_network,soft_gating,scenario_bucket,gating_network_v2}"
else
  export MODELCOMBINE_SKIP_COMPLEX_KG_COMPONENT="${MODELCOMBINE_SKIP_COMPLEX_KG_COMPONENT:-0}"
  export MODELCOMBINE_SKIP_SOTA="${MODELCOMBINE_SKIP_SOTA:-0}"
  export MODELCOMBINE_KG_COMPONENT_STRATEGIES="${MODELCOMBINE_KG_COMPONENT_STRATEGIES:-}"
fi
echo "[INFO] MODELCOMBINE_SKIP_COMPLEX_KG_COMPONENT=$MODELCOMBINE_SKIP_COMPLEX_KG_COMPONENT"
echo "[INFO] MODELCOMBINE_SKIP_SOTA=$MODELCOMBINE_SKIP_SOTA"
echo "[INFO] MODELCOMBINE_KG_COMPONENT_STRATEGIES=${MODELCOMBINE_KG_COMPONENT_STRATEGIES:-<default>}"
SKIP_MODELCOMBINE=0
if [ "$REUSE_COMPLETED_STEPS" = "1" ] && modelcombine_artifacts_ready "$MODEL_DIR"; then
  if [ "$ENABLE_ITRANSFORMER" = "1" ] && ! modelcombine_has_model "$MODEL_DIR" "itransformer"; then
    echo "[INFO] modelcombine artifacts found but missing itransformer; force rerun modelcombine eval"
  else
    SKIP_MODELCOMBINE=1
  fi
fi

if [ "$SKIP_MODELCOMBINE" = "1" ]; then
  echo "[5/9] modelcombine eval - skip (reuse existing artifacts)"
else
  MODELCOMBINE_REQUIRE_FULL_HORIZON_ARTIFACTS=1 \
  python -u scripts/modelcombine_eval.py \
    --features "$FEATURE_DIR" \
    --baselines "$BASELINE_DIR" \
    --out "$MODEL_DIR" \
    2>&1 | tee "${LOG_DIR}/04_modelcombine.log"
fi

test -f "${MODEL_DIR}/routing_config.json" || { echo "[ERROR] missing routing_config.json"; exit 1; }

# 6) 离线候选审计（入 KG 前门禁）
echo "[6/9] audit extended candidates"
AUDIT_JSON="${MODEL_DIR}/candidate_audit.json"
AUDIT_RESCUE_MAX_MAE_GAP_RATIO="${MODELCOMBINE_AUDIT_RESCUE_MAX_MAE_GAP_RATIO:-0.12}"
AUDIT_RESCUE_MIN_CV_IMPROVE_PCT="${MODELCOMBINE_AUDIT_RESCUE_MIN_CV_IMPROVE_PCT:--0.10}"
AUDIT_RESCUE_MAX_TAIL_DEGRADE_RATIO="${MODELCOMBINE_AUDIT_RESCUE_MAX_TAIL_DEGRADE_RATIO:-0.05}"
AUDIT_CANDIDATES_RAW="${MODELCOMBINE_AUDIT_CANDIDATES:-rl_qms,stacking_safe,dynamic_stacking,static_weight_safe,mole_router}"
AUDIT_PREF_RAW="${MODELCOMBINE_AUDIT_HIGH_DRIFT_PREFERRED_CANDIDATES:-rl_qms,stacking_safe}"
AUDIT_CANDIDATES_NORM="${AUDIT_CANDIDATES_RAW//,/ }"
AUDIT_PREF_NORM="${AUDIT_PREF_RAW//,/ }"
read -r -a AUDIT_CANDIDATES_ARR <<< "${AUDIT_CANDIDATES_NORM}"
read -r -a AUDIT_PREF_ARR <<< "${AUDIT_PREF_NORM}"
echo "[INFO] audit rescue thresholds: mae_gap_ratio=${AUDIT_RESCUE_MAX_MAE_GAP_RATIO} min_cv_improve_pct=${AUDIT_RESCUE_MIN_CV_IMPROVE_PCT} max_tail_degrade_ratio=${AUDIT_RESCUE_MAX_TAIL_DEGRADE_RATIO}"
echo "[INFO] audit candidates=${AUDIT_CANDIDATES_RAW}"
echo "[INFO] audit high-drift preferred=${AUDIT_PREF_RAW}"
if [ "$KG_EXPERIMENT_PROFILE" = "clean_noql_dash" ]; then
  if [[ ",${AUDIT_CANDIDATES_RAW}," == *",rl_qms,"* ]] || [[ ",${AUDIT_PREF_RAW}," == *",rl_qms,"* ]]; then
    echo "[ERROR] clean_noql_dash forbids rl_qms in audit candidates/preferred lists"
    echo "[HINT] current candidates=${AUDIT_CANDIDATES_RAW}"
    echo "[HINT] current preferred=${AUDIT_PREF_RAW}"
    exit 2
  fi
  if [[ ",${AUDIT_CANDIDATES_RAW}," != *",dash_tta,"* ]]; then
    echo "[ERROR] clean_noql_dash requires dash_tta in audit candidates list"
    echo "[HINT] current candidates=${AUDIT_CANDIDATES_RAW}"
    exit 2
  fi
fi
python scripts/audit_extended_candidates.py \
  --pred-root "$BASELINE_DIR" \
  --out "$AUDIT_JSON" \
  --datasets pjm aemo_vic aemo_nsw \
  --candidates "${AUDIT_CANDIDATES_ARR[@]}" \
  --high-drift-min-accepted 1 \
  --high-drift-preferred-candidates "${AUDIT_PREF_ARR[@]}" \
  --high-drift-rescue-max-mae-gap-ratio "$AUDIT_RESCUE_MAX_MAE_GAP_RATIO" \
  --high-drift-rescue-min-cv-improve-pct "$AUDIT_RESCUE_MIN_CV_IMPROVE_PCT" \
  --high-drift-rescue-max-tail-degrade-ratio "$AUDIT_RESCUE_MAX_TAIL_DEGRADE_RATIO" \
  --min-accepted-per-task 0 \
  2>&1 | tee "${LOG_DIR}/05_audit_candidates.log"

test -f "$AUDIT_JSON" || { echo "[ERROR] missing candidate audit json: $AUDIT_JSON"; exit 1; }

# 7) KG 训练（扩展池 + 显式健康白名单 + 审计白名单）
echo "[7/9] train KG"
python - <<'PY'
from src.eval.kg.config import _build_kg_model_candidates, _build_extended_pool_strategies
base = _build_kg_model_candidates()
ext = _build_extended_pool_strategies()
print("[CHECK] KG base candidates:", base)
print("[CHECK] KG extended pool:", ext)
PY
if [ "$KG_EXPERIMENT_PROFILE" = "clean_noql_dash" ]; then
  python - <<'PY'
import sys
from src.eval.kg.config import _build_kg_model_candidates, _build_extended_pool_strategies
base = _build_kg_model_candidates()
ext = _build_extended_pool_strategies()
errors = []
if "rl_qms" in ext:
    errors.append("rl_qms must be excluded in clean_noql_dash")
if "dash_tta" not in ext:
    errors.append("dash_tta must be included in clean_noql_dash")
if "itransformer" in base:
    errors.append("itransformer must not be injected into KG base in clean_noql_dash")
if errors:
    print("[ERROR] clean_noql_dash gate failed:")
    for e in errors:
        print("  -", e)
    sys.exit(2)
print("[OK] clean_noql_dash gate passed")
PY
fi

export MODELCOMBINE_KG_SEASONAL_NAIVE_AS_FROZEN_EXPERT="${MODELCOMBINE_KG_SEASONAL_NAIVE_AS_FROZEN_EXPERT:-1}"
export MODELCOMBINE_KG_B_LAST_BLOCK_GUARD_ENABLED="${MODELCOMBINE_KG_B_LAST_BLOCK_GUARD_ENABLED:-1}"
export MODELCOMBINE_KG_B_HIGH_DRIFT_MIN_W_MULTIPLIER="${MODELCOMBINE_KG_B_HIGH_DRIFT_MIN_W_MULTIPLIER:-1.3}"
export MODELCOMBINE_KG_B_COMPLEXITY_PENALTY_ENABLED="${MODELCOMBINE_KG_B_COMPLEXITY_PENALTY_ENABLED:-1}"
export MODELCOMBINE_KG_B_COMPLEXITY_PENALTY_DATASETS="${MODELCOMBINE_KG_B_COMPLEXITY_PENALTY_DATASETS:-aemo_nsw}"
export MODELCOMBINE_KG_B_BEST_SINGLE_SCOPE="${MODELCOMBINE_KG_B_BEST_SINGLE_SCOPE:-base_models_only}"
export MODELCOMBINE_KG_B_GUARD_BEST_SINGLE_SCOPE="${MODELCOMBINE_KG_B_GUARD_BEST_SINGLE_SCOPE:-base_models_only}"
export MODELCOMBINE_KG_B_DATASET_LAST_BLOCK_GUARD_DATASETS="${MODELCOMBINE_KG_B_DATASET_LAST_BLOCK_GUARD_DATASETS:-aemo_nsw}"
export MODELCOMBINE_KG_B_DATASET_LAST_BLOCK_GUARD_RATIO="${MODELCOMBINE_KG_B_DATASET_LAST_BLOCK_GUARD_RATIO:-0.20}"
export MODELCOMBINE_KG_B_DATASET_LAST_BLOCK_GUARD_MAX_DEGRADATION="${MODELCOMBINE_KG_B_DATASET_LAST_BLOCK_GUARD_MAX_DEGRADATION:-0.000}"
export MODELCOMBINE_KG_B_MIN_W_OVERRIDE_DATASETS="${MODELCOMBINE_KG_B_MIN_W_OVERRIDE_DATASETS:-aemo_nsw}"
export MODELCOMBINE_KG_B_MIN_W_OVERRIDE_VALUE="${MODELCOMBINE_KG_B_MIN_W_OVERRIDE_VALUE:-0.12}"
KG_ELIGIBLE_MASE_HARD="${MODELCOMBINE_KG_ELIGIBLE_MASE_HARD:-}"

# soft-pass 配置（按需启用，默认关闭）:
# export MODELCOMBINE_KG_AUDIT_SOFT_PASS_ENABLED=1
# export MODELCOMBINE_KG_AUDIT_SOFT_PASS_RATIO=2.0
# export MODELCOMBINE_KG_AUDIT_SOFT_PASS_DATASETS="aemo_vic,aemo_nsw"
# export MODELCOMBINE_KG_AUDIT_SOFT_PASS_HORIZONS="24"
# export MODELCOMBINE_KG_AUDIT_SOFT_PASS_ALLOWLIST="seasonal_naive,stacking_safe"
# export MODELCOMBINE_KG_AUDIT_SOFT_PASS_MAX_PER_TASK=2

KG_EXTRA_ARGS=()
if [ -n "$KG_ELIGIBLE_MASE_HARD" ]; then
  KG_EXTRA_ARGS+=(--eligible-mase-hard "$KG_ELIGIBLE_MASE_HARD")
fi

python scripts/train_combinations_kg.py \
  --pred-root "$BASELINE_DIR" \
  --raw-root "$FEATURE_DIR" \
  --out-root "$KG_DIR" \
  --extended-pool \
  --strict-extended-pool \
  --min-extended-loaded 3 \
  --candidate-audit "$AUDIT_JSON" \
  --min-audit-accepted 0 \
  --fail-on-protocol-b-error \
  --seed "${MODELCOMBINE_SEED:-42}" \
  --combo-root "$MODEL_DIR" \
  --health-config "$MODEL_DIR/routing_config.json" \
  --datasets pjm aemo_vic aemo_nsw \
  ${KG_EXTRA_ARGS[@]+"${KG_EXTRA_ARGS[@]}"} \
  2>&1 | tee "${LOG_DIR}/06_kg.log"

python - "$KG_DIR/kg_results.json" <<'PY'
import json
import sys
from pathlib import Path

kg_path = Path(sys.argv[1]).resolve()
if not kg_path.exists():
    raise SystemExit(f"[ERROR] missing kg_results.json: {kg_path}")
payload = json.loads(kg_path.read_text(encoding="utf-8"))
expected = {
    "pjm": ["1", "6", "24"],
    "aemo_vic": ["1", "6", "24"],
    "aemo_nsw": ["1", "6", "24"],
}
errors = []
for ds, horizons in expected.items():
    ds_data = payload.get(ds)
    if not isinstance(ds_data, dict):
        errors.append(f"{ds}: missing_dataset_payload")
        continue
    for h in horizons:
        task = ds_data.get(h)
        if not isinstance(task, dict):
            errors.append(f"{ds} h={h}: missing_task_payload")
            continue
        b = task.get("kg_protocol_b")
        if not isinstance(b, dict):
            errors.append(f"{ds} h={h}: missing_kg_protocol_b")
            continue
        if b.get("error"):
            errors.append(f"{ds} h={h}: {b.get('error')}")
            continue
        test_payload = b.get("test")
        if not isinstance(test_payload, dict):
            errors.append(f"{ds} h={h}: missing_protocol_b_test_payload")
            continue
        if test_payload.get("mae") is None:
            errors.append(f"{ds} h={h}: missing_protocol_b_test_mae")
if errors:
    print("[ERROR] KG runtime completeness gate failed:")
    for e in errors:
        print("  -", e)
    raise SystemExit(2)
print("[OK] KG runtime gate passed (complete 3x3 tasks, no protocol_b.error)")
PY

# 8) 对比报告
echo "[8/9] compare results"
python scripts/compare_results.py \
  --baseline-root "$MODEL_DIR" \
  --kg-root "$KG_DIR" \
  --leaderboard-root "$MODEL_DIR" \
  --datasets pjm aemo_vic aemo_nsw \
  --out "${ANALYSIS_DIR}/comparison_report.md" \
  2>&1 | tee "${LOG_DIR}/07_compare.log"

# 9) 打包结果
echo "[9/9] package run artifacts"
tar -czf "${RUN_ROOT}.tar.gz" -C "$(dirname "$RUN_ROOT")" "$(basename "$RUN_ROOT")"
sha256sum "${RUN_ROOT}.tar.gz" | tee "${RUN_ROOT}.tar.gz.sha256"

echo "[DONE] run root: ${RUN_ROOT}"
echo "[DONE] report: ${ANALYSIS_DIR}/comparison_report.md"
echo "[DONE] package: ${RUN_ROOT}.tar.gz"
echo "[DONE] checksum: ${RUN_ROOT}.tar.gz.sha256"
