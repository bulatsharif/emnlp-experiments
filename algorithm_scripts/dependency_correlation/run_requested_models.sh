#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
EXP_ROOT="$ROOT/emnlp-experiments"
VENV_PATH="/workspace/tasks/emnlp/.venv"
TASKS="gsm8k,humaneval,math500,mtbench"
SAMPLES=10
SAMPLE_OFFSET=0
SAMPLE_SEED=42
MAX_NEW_TOKENS=32
GENERATION_STEPS=8
SINGLE_FORWARD_ONLY=0
TOKENS_PER_STEP=4
MAX_PAIRS_PER_STEP=4
PAIR_SELECTION="confidence"
CONDITION_BATCH_SIZE=8
DEVICE="cuda"
DTYPE="auto"
SDPA_BACKEND="auto"
ALLOW_DOWNLOADS=0
GPUS=""
MAX_PARALLEL=0
OUTDIR="$EXP_ROOT/outputs/dependency_correlation/$(date -u +%Y%m%d_%H%M%S)"

usage() {
  cat <<'EOF'
Usage:
  bash algorithm_scripts/dependency_correlation/run_requested_models.sh [options]

Options:
  --venv PATH
  --outdir PATH
  --tasks LIST
  --samples N
  --sample-offset N
  --sample-seed N
  --max-new-tokens N
  --generation-steps N
  --single-forward-only
  --tokens-per-step N
  --max-pairs-per-step N
  --pair-selection {confidence,any}
  --condition-batch-size N
  --gpus LIST
  --max-parallel N
  --device DEVICE
  --dtype {auto,float32,float16,bfloat16}
  --sdpa-backend {auto,math}
  --allow-downloads
  -h, --help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --venv)
      VENV_PATH="$2"
      shift 2
      ;;
    --outdir)
      OUTDIR="$2"
      shift 2
      ;;
    --tasks)
      TASKS="$2"
      shift 2
      ;;
    --samples)
      SAMPLES="$2"
      shift 2
      ;;
    --sample-offset)
      SAMPLE_OFFSET="$2"
      shift 2
      ;;
    --sample-seed)
      SAMPLE_SEED="$2"
      shift 2
      ;;
    --max-new-tokens)
      MAX_NEW_TOKENS="$2"
      shift 2
      ;;
    --generation-steps)
      GENERATION_STEPS="$2"
      shift 2
      ;;
    --single-forward-only)
      SINGLE_FORWARD_ONLY=1
      shift
      ;;
    --tokens-per-step)
      TOKENS_PER_STEP="$2"
      shift 2
      ;;
    --max-pairs-per-step)
      MAX_PAIRS_PER_STEP="$2"
      shift 2
      ;;
    --pair-selection)
      PAIR_SELECTION="$2"
      shift 2
      ;;
    --condition-batch-size)
      CONDITION_BATCH_SIZE="$2"
      shift 2
      ;;
    --gpus)
      GPUS="$2"
      shift 2
      ;;
    --max-parallel)
      MAX_PARALLEL="$2"
      shift 2
      ;;
    --device)
      DEVICE="$2"
      shift 2
      ;;
    --dtype)
      DTYPE="$2"
      shift 2
      ;;
    --sdpa-backend)
      SDPA_BACKEND="$2"
      shift 2
      ;;
    --allow-downloads)
      ALLOW_DOWNLOADS=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ ! -f "$VENV_PATH/bin/activate" ]]; then
  echo "Virtual environment not found at $VENV_PATH" >&2
  exit 1
fi

source "$VENV_PATH/bin/activate"

python - <<'PY' "$GPUS" "$MAX_PARALLEL" "$OUTDIR"
import importlib
import json
import pathlib
import subprocess
import sys

missing = []
for name in ("torch", "transformers", "datasets", "scipy", "yaml", "tqdm"):
    try:
        importlib.import_module(name)
    except Exception:
        missing.append(name)
if missing:
    raise SystemExit(
        "Missing Python packages in the active environment: " + ", ".join(missing)
    )

gpu_arg = sys.argv[1].strip()
max_parallel_arg = int(sys.argv[2])
outdir = pathlib.Path(sys.argv[3])
outdir.mkdir(parents=True, exist_ok=True)

if gpu_arg:
    gpu_ids = [item.strip() for item in gpu_arg.split(",") if item.strip()]
else:
    query = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index",
            "--format=csv,noheader",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    gpu_ids = [line.strip() for line in query.stdout.splitlines() if line.strip()]

if not gpu_ids:
    raise SystemExit("No CUDA GPUs detected. Pass --gpus or check nvidia-smi.")

parallel = max_parallel_arg if max_parallel_arg > 0 else len(gpu_ids)
parallel = max(1, min(parallel, len(gpu_ids)))

config_path = outdir / ".scheduler_config.json"
config_path.write_text(json.dumps({"gpu_ids": gpu_ids, "parallel": parallel}))
print(f"Using GPUs: {', '.join(gpu_ids)}")
print(f"Max parallel jobs: {parallel}")
PY

mkdir -p "$OUTDIR"
LOG_DIR="$OUTDIR/logs"
mkdir -p "$LOG_DIR"

SCHEDULER_CONFIG="$OUTDIR/.scheduler_config.json"
GPU_LIST="$(python - <<'PY' "$SCHEDULER_CONFIG"
import json
import pathlib
import sys

config = json.loads(pathlib.Path(sys.argv[1]).read_text())
print(",".join(config["gpu_ids"]))
PY
)"
MAX_PARALLEL="$(python - <<'PY' "$SCHEDULER_CONFIG"
import json
import pathlib
import sys

config = json.loads(pathlib.Path(sys.argv[1]).read_text())
print(config["parallel"])
PY
)"

declare -A MODELS=(
  ["Dream-Coder-v0-Instruct-7B"]="$ROOT/Dream-Coder-v0-Instruct-7B"
  ["LLaDA-8B-Instruct"]="$ROOT/LLaDA-8B-Instruct"
  ["LLaDA-MoE-7B-A1B-Instruct"]="$ROOT/LLaDA-MoE-7B-A1B-Instruct"
)

for model_name in "${!MODELS[@]}"; do
  model_path="${MODELS[$model_name]}"
  if [[ ! -d "$model_path" ]]; then
    echo "Model directory not found: $model_path" >&2
    exit 1
  fi
done

run_flags=(
  --samples "$SAMPLES"
  --sample-offset "$SAMPLE_OFFSET"
  --sample-seed "$SAMPLE_SEED"
  --max-new-tokens "$MAX_NEW_TOKENS"
  --generation-steps "$GENERATION_STEPS"
  --tokens-per-step "$TOKENS_PER_STEP"
  --max-pairs-per-step "$MAX_PAIRS_PER_STEP"
  --pair-selection "$PAIR_SELECTION"
  --condition-batch-size "$CONDITION_BATCH_SIZE"
  --device "$DEVICE"
  --dtype "$DTYPE"
  --sdpa-backend "$SDPA_BACKEND"
)
if [[ "$SINGLE_FORWARD_ONLY" == "1" ]]; then
  run_flags+=(--single-forward-only)
fi
if [[ "$ALLOW_DOWNLOADS" == "1" ]]; then
  run_flags+=(--allow-downloads)
fi

cd "$EXP_ROOT"

JOBS_FILE="$OUTDIR/.jobs.tsv"
python - <<'PY' "$JOBS_FILE" "$TASKS"
import pathlib
import sys

jobs_file = pathlib.Path(sys.argv[1])
tasks = [task.strip() for task in sys.argv[2].split(",") if task.strip()]
models = (
    "Dream-Coder-v0-Instruct-7B",
    "LLaDA-8B-Instruct",
    "LLaDA-MoE-7B-A1B-Instruct",
)
with jobs_file.open("w") as handle:
    for model in models:
        for task in tasks:
            handle.write(f"{model}\t{task}\n")
PY

sanitize_task() {
  echo "$1" | tr '/,' '__'
}

launch_job() {
  local gpu_id="$1"
  local model_name="$2"
  local task_name="$3"
  local model_path="${MODELS[$model_name]}"
  local task_dir="$OUTDIR/$model_name/$(sanitize_task "$task_name")"
  local log_path="$LOG_DIR/${model_name}__$(sanitize_task "$task_name").log"
  mkdir -p "$task_dir"
  (
    export CUDA_VISIBLE_DEVICES="$gpu_id"
    python -m algorithm_scripts.dependency_correlation.run \
      --model "$model_path" \
      --tasks "$task_name" \
      --progress-description "$model_name:$task_name" \
      "${run_flags[@]}" \
      --output-jsonl "$task_dir/pairs.jsonl" \
      --output-csv "$task_dir/summary.csv"
  ) >"$log_path" 2>&1 &
  local pid=$!
  PID_TO_GPU["$pid"]="$gpu_id"
  PID_TO_MODEL["$pid"]="$model_name"
  PID_TO_TASK["$pid"]="$task_name"
  PID_TO_LOG["$pid"]="$log_path"
}

declare -a GPU_IDS=()
IFS=',' read -r -a GPU_IDS <<< "$GPU_LIST"
declare -A GPU_BUSY=()
declare -A PID_TO_GPU=()
declare -A PID_TO_MODEL=()
declare -A PID_TO_TASK=()
declare -A PID_TO_LOG=()

for gpu_id in "${GPU_IDS[@]}"; do
  GPU_BUSY["$gpu_id"]=0
done

mapfile -t JOB_LINES < "$JOBS_FILE"
TOTAL_JOBS="${#JOB_LINES[@]}"
LAUNCHED=0
COMPLETED=0

render_progress() {
  local width=32
  local filled=0
  local empty=0
  local bar=""
  if (( TOTAL_JOBS > 0 )); then
    filled=$((COMPLETED * width / TOTAL_JOBS))
  fi
  empty=$((width - filled))
  bar="$(printf '%*s' "$filled" '' | tr ' ' '#')"
  bar+="$(printf '%*s' "$empty" '' | tr ' ' '-')"
  printf '\rJobs [%s] %d/%d' "$bar" "$COMPLETED" "$TOTAL_JOBS"
  if (( COMPLETED == TOTAL_JOBS )); then
    printf '\n'
  fi
}

cleanup_jobs() {
  local pid
  for pid in "${!PID_TO_GPU[@]}"; do
    kill "$pid" 2>/dev/null || true
  done
}

echo
echo "Scheduling $TOTAL_JOBS jobs across GPUs: $GPU_LIST"
trap cleanup_jobs EXIT INT TERM
render_progress

while (( COMPLETED < TOTAL_JOBS )); do
  for gpu_id in "${GPU_IDS[@]}"; do
    if (( LAUNCHED >= TOTAL_JOBS )); then
      break
    fi
    if [[ "${GPU_BUSY[$gpu_id]}" == "1" ]]; then
      continue
    fi
    IFS=$'\t' read -r model_name task_name <<< "${JOB_LINES[$LAUNCHED]}"
    printf '\nLaunching [%d/%d] %s :: %s on GPU %s\n' \
      "$((LAUNCHED + 1))" "$TOTAL_JOBS" "$model_name" "$task_name" "$gpu_id"
    launch_job "$gpu_id" "$model_name" "$task_name"
    GPU_BUSY["$gpu_id"]=1
    ((LAUNCHED += 1))
    render_progress
    if (( LAUNCHED - COMPLETED >= MAX_PARALLEL )); then
      break
    fi
  done

  if (( COMPLETED >= TOTAL_JOBS )); then
    break
  fi

  if ! wait -n -p finished_pid; then
    status=$?
  else
    status=0
  fi

  gpu_id="${PID_TO_GPU[$finished_pid]}"
  model_name="${PID_TO_MODEL[$finished_pid]}"
  task_name="${PID_TO_TASK[$finished_pid]}"
  log_path="${PID_TO_LOG[$finished_pid]}"
  unset PID_TO_GPU["$finished_pid"]
  unset PID_TO_MODEL["$finished_pid"]
  unset PID_TO_TASK["$finished_pid"]
  unset PID_TO_LOG["$finished_pid"]
  GPU_BUSY["$gpu_id"]=0
  ((COMPLETED += 1))

  if (( status != 0 )); then
    printf '\n'
    echo "Job failed: $model_name :: $task_name on GPU $gpu_id"
    echo "Log: $log_path"
    tail -n 40 "$log_path" || true
    exit "$status"
  fi

  printf '\nCompleted [%d/%d] %s :: %s on GPU %s\n' \
    "$COMPLETED" "$TOTAL_JOBS" "$model_name" "$task_name" "$gpu_id"
  render_progress
done

wait
trap - EXIT INT TERM

python - <<'PY' "$OUTDIR"
import csv
import math
import pathlib
import sys

outdir = pathlib.Path(sys.argv[1])
benchmark_names = {
    "gsm8k": "GSM8K",
    "humaneval": "HumanEval",
    "math500": "MATH500",
    "ifeval": "IFEval",
    "mtbench": "MT-Bench",
}
rows = []
for summary_path in sorted(outdir.glob("*/*/summary.csv")):
    model_name = summary_path.parent.parent.name
    with summary_path.open() as handle:
        for row in csv.DictReader(handle):
            rows.append({"model": model_name, **row})

if rows:
    fieldnames = list(rows[0].keys())
    by_model = {}
    for row in rows:
        by_model.setdefault(row["model"], []).append(row)
    for model_name, model_rows in by_model.items():
        model_summary_path = outdir / model_name / "summary.csv"
        model_summary_path.parent.mkdir(parents=True, exist_ok=True)
        with model_summary_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames[1:])
            writer.writeheader()
            writer.writerows([{k: v for k, v in row.items() if k != "model"} for row in model_rows])
    summary_path = outdir / "combined_summary.csv"
    with summary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    paper_rows = []
    for row in rows:
        paper_rows.append(
            {
                "model": row["model"],
                "benchmark": benchmark_names.get(row["task"], row["task"]),
                "pairs": row["pairs"],
                "spearman_attention": row.get("spearman_attention", ""),
                "spearman_logit": row.get("spearman_logit", ""),
                "spearman_hidden": row.get("spearman_hidden", ""),
                "pearson_attention": row.get("pearson_attention", ""),
                "pearson_logit": row.get("pearson_logit", ""),
                "pearson_hidden": row.get("pearson_hidden", ""),
            }
        )
    for row in paper_rows:
        for key, value in list(row.items()):
            if isinstance(value, float) and math.isnan(value):
                row[key] = "NA"
            elif isinstance(value, str) and value.strip().lower() == "nan":
                row[key] = "NA"
    paper_path = outdir / "paper_table.csv"
    with paper_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "model",
                "benchmark",
                "pairs",
                "spearman_attention",
                "spearman_logit",
                "spearman_hidden",
                "pearson_attention",
                "pearson_logit",
                "pearson_hidden",
            ],
        )
        writer.writeheader()
        writer.writerows(paper_rows)
    print(f"\nWrote combined summary to {summary_path}")
    print(f"Wrote paper-ready table to {paper_path}")
else:
    print("\nNo summary rows were produced.")
PY

echo
echo "Finished. Results are under: $OUTDIR"
