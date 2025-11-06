#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: reproduce_all.sh [options] [HYDRA_OVERRIDES...]

Options:
  --smoke             Run a short smoke workload instead of the full experiment.
  --no-conda          Skip Conda environment creation/activation (use existing Python).
  --run-id NAME       Override the run identifier used for evaluation outputs (default: paper or smoke).
  --baseline-model M  Baseline model name for statistical tests (auto-detected if omitted).
  -h, --help          Show this help message and exit.
  --                  Treat all following arguments as Hydra overrides.
USAGE
}

log() {
  printf '[reproduce_all] %s\n' "$1" >&2
}

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
ARTIFACTS_ROOT=${ARTIFACTS_ROOT:-"${ROOT_DIR}/artifacts"}
PAPER_ROOT=${PAPER_OUTPUT_DIR:-"${ROOT_DIR}/paper_outputs"}
ENV_FILE="${ROOT_DIR}/environment.yml"
ENV_NAME=${CONDA_ENV_NAME:-$(awk '/^name:/ {print $2}' "$ENV_FILE")}
USE_CONDA=1
SMOKE=0
RUN_ID=""
BASELINE_MODEL=${BASELINE_MODEL:-""}
HYDRA_OVERRIDES=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --smoke)
      SMOKE=1
      shift
      ;;
    --no-conda)
      USE_CONDA=0
      shift
      ;;
    --run-id)
      if [[ $# -lt 2 ]]; then
        echo "--run-id requires a value" >&2
        exit 1
      fi
      RUN_ID="$2"
      shift 2
      ;;
    --baseline-model)
      if [[ $# -lt 2 ]]; then
        echo "--baseline-model requires a value" >&2
        exit 1
      fi
      BASELINE_MODEL="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      while [[ $# -gt 0 ]]; do
        HYDRA_OVERRIDES+=("$1")
        shift
      done
      break
      ;;
    *)
      HYDRA_OVERRIDES+=("$1")
      shift
      ;;
  esac
done

if [[ -z "$RUN_ID" ]]; then
  if [[ $SMOKE -eq 1 ]]; then
    RUN_ID="smoke"
  else
    RUN_ID="paper"
  fi
fi

if [[ $SMOKE -eq 1 ]]; then
  DEFAULT_OVERRIDES=(
    "training.epochs=1"
    "training.device=cpu"
    "data.time_steps=16"
    "data.batch_size=32"
    "data.pairs=[EURUSD]"
    "data.horizons=[1]"
    "data.walkforward.train=64"
    "data.walkforward.val=32"
    "data.walkforward.test=32"
    "data.walkforward.step=32"
    "multirun.seeds=[7,11]"
  )
  HYDRA_OVERRIDES+=("${DEFAULT_OVERRIDES[@]}")
fi

cd "$ROOT_DIR"

if [[ $USE_CONDA -eq 1 ]]; then
  if ! command -v conda >/dev/null 2>&1; then
    echo "Conda is required but was not found. Use --no-conda to bypass this check." >&2
    exit 1
  fi
  log "Ensuring Conda environment '$ENV_NAME' is available"
  if ! conda env list | awk 'NR>2 {print $1}' | grep -Fxq "$ENV_NAME"; then
    conda env create -f "$ENV_FILE"
  else
    conda env update -f "$ENV_FILE" --prune
  fi
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "$ENV_NAME"
fi

log "Installing project in editable mode"
python -m pip install --upgrade pip
python -m pip install -e .

log "Cleaning previous artifacts"
rm -rf "$ARTIFACTS_ROOT" "$PAPER_ROOT"
mkdir -p "$ARTIFACTS_ROOT" "$PAPER_ROOT"

TRAIN_CMD=(python -m src.cli --multirun)
if [[ ${#HYDRA_OVERRIDES[@]} -gt 0 ]]; then
  TRAIN_CMD+=("${HYDRA_OVERRIDES[@]}")
fi

log "Launching multi-run training"
"${TRAIN_CMD[@]}"

log "Populating aggregate summaries"
python scripts/reproduce_all.py --populate-only --artifacts-root "$ARTIFACTS_ROOT" --aggregates-dir "$ARTIFACTS_ROOT/aggregates"

mapfile -t PREDICTION_FILES < <(find "$ARTIFACTS_ROOT" -type f \( -name 'predictions.csv' -o -name 'predictions.parquet' \) | sort)
if [[ ${#PREDICTION_FILES[@]} -eq 0 ]]; then
  echo "No prediction files discovered under $ARTIFACTS_ROOT. Ensure evaluation exports predictions." >&2
  exit 1
fi

CALIBRATION_DIR="$PAPER_ROOT/calibration"
mkdir -p "$CALIBRATION_DIR"
log "Running calibration CLI on ${#PREDICTION_FILES[@]} prediction files"
python -m src.metrics.calibration_cli "${PREDICTION_FILES[@]}" --output-root "$CALIBRATION_DIR"

mapfile -t DM_CACHES < <(find "$ARTIFACTS_ROOT" -type f -name 'dm_cache.csv' | sort)
if [[ ${#DM_CACHES[@]} -eq 0 ]]; then
  echo "No DM cache files found under $ARTIFACTS_ROOT. Run evaluation to generate statistical inputs." >&2
  exit 1
fi
DM_CACHE="${DM_CACHES[0]}"

if [[ -z "$BASELINE_MODEL" ]]; then
  log "Inferring baseline model from $DM_CACHE"
  BASELINE_MODEL=$(python - "$DM_CACHE" <<'PY'
import sys
import pandas as pd
cache = pd.read_csv(sys.argv[1])
if cache.empty or "model" not in cache.columns:
    raise SystemExit("Unable to infer baseline model; provide --baseline-model")
models = sorted({str(m) for m in cache["model"].dropna()})
if not models:
    raise SystemExit("No model labels present in DM cache")
print(models[0])
PY
)
fi

if [[ -z "$BASELINE_MODEL" ]]; then
  echo "Baseline model could not be determined" >&2
  exit 1
fi

STATS_DIR="$PAPER_ROOT/stats"
mkdir -p "$STATS_DIR"

if [[ $SMOKE -eq 1 ]]; then
  BOOTSTRAP=64
  MAX_COMBINATIONS=32
else
  BOOTSTRAP=2000
  MAX_COMBINATIONS=128
fi

log "Running statistical analysis for baseline '$BASELINE_MODEL'"
python -m src.stats.dm "$DM_CACHE" \
  --baseline-model "$BASELINE_MODEL" \
  --metric squared_error \
  --run-id "$RUN_ID" \
  --output-dir "$STATS_DIR" \
  --alpha 0.05 \
  --assumption-alpha 0.05 \
  --newey-west-lag 1 \
  --random-seed 123 \
  --n-bootstrap "$BOOTSTRAP"

python -m src.stats.spa "$DM_CACHE" \
  --baseline-model "$BASELINE_MODEL" \
  --metric squared_error \
  --run-id "$RUN_ID" \
  --output-dir "$STATS_DIR" \
  --alpha 0.05 \
  --assumption-alpha 0.05 \
  --newey-west-lag 1 \
  --random-seed 123 \
  --n-bootstrap "$BOOTSTRAP"

python -m src.stats.mcs "$DM_CACHE" \
  --baseline-model "$BASELINE_MODEL" \
  --metric squared_error \
  --run-id "$RUN_ID" \
  --output-dir "$STATS_DIR" \
  --alpha 0.05 \
  --assumption-alpha 0.05 \
  --newey-west-lag 1 \
  --random-seed 123 \
  --n-bootstrap "$BOOTSTRAP"

python -m src.stats.pbo "$DM_CACHE" \
  --baseline-model "$BASELINE_MODEL" \
  --metric squared_error \
  --run-id "$RUN_ID" \
  --output-dir "$STATS_DIR" \
  --alpha 0.05 \
  --assumption-alpha 0.05 \
  --newey-west-lag 1 \
  --random-seed 123 \
  --max-combinations "$MAX_COMBINATIONS" \
  --n-bootstrap "$BOOTSTRAP"

FIGURES_DIR="$PAPER_ROOT/figures"
TABLES_DIR="$PAPER_ROOT/tables"
MANIFEST_PATH="$PAPER_ROOT/${RUN_ID}_manifest.json"

log "Assembling publication assets"
python scripts/reproduce_all.py \
  --artifacts-root "$ARTIFACTS_ROOT" \
  --tables-dir "$TABLES_DIR" \
  --figures-dir "$FIGURES_DIR" \
  --aggregates-dir "$ARTIFACTS_ROOT/aggregates" \
  --project-root "$ROOT_DIR" \
  --manifest "$MANIFEST_PATH"

log "Reproduction workflow completed successfully"

