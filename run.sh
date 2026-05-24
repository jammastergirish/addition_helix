#!/usr/bin/env bash
# Sweep the helix experiment across all seven models and all eight numeral
# scripts.  Each run uses --pool mean (the honest default) and --sweep
# (layer scan + auto-targeted standard figures at the helix-R² peak).
#
# Three passes:
#   pass 1  -- 7 models x 8 scripts at n_max=100, basis=[2,5,10,100]
#              (paper-default range and basis)
#   pass 2  -- 7 models x 1 script  (babylonian) at n_max=600,
#              basis=[2,5,10,100] (paper basis -- demonstrates the basis
#              is blind to T=60)
#   pass 3  -- 7 models x 1 script  (babylonian) at n_max=600,
#              basis=[2,5,10,60,100] (paper basis + T=60 -- shows how
#              much variance T=60 actually captures)
#
# Passes 2 and 3 exist because Babylonian is base-60 positional: in the
# 0..99 range there is only one 60-wraparound, so the model can't form
# a period-60 circle even if it "wants" to. n_max=600 gives it ten full
# wraparounds, which is the smallest range where T=60 is testable.
# Pass 3 then quantifies how much of the structure the T=60 dimension
# captures, against the paper-default basis used in pass 2.
#
# All passes are idempotent: any combination whose fig_layer_sweep.json
# already exists is skipped. To force a fresh run, delete the relevant
# subdirectory inside out/.
#
# Usage:
#     ./run.sh                # everything
#     ./run.sh gemma          # only models whose id contains "gemma"
#     ./run.sh pythia         # only Pythia
#     ./run.sh 31B            # only the 31B model
#     ./run.sh Llama-3.1-8B   # exact substring works too
#
# Failures: any failing run prints "!! FAILED ..." and the sweep
# continues with the next combo.

set -euo pipefail

# Requires bash 4+ for ${VAR,,} lowercase substitution. macOS ships with
# bash 3.2 for licensing reasons; install a modern bash via Homebrew:
#     brew install bash
if (( BASH_VERSINFO[0] < 4 )); then
  echo "Error: this script requires bash 4 or newer; you have ${BASH_VERSION}."
  echo "Install a modern bash:  brew install bash"
  echo "Then either: re-open your shell so PATH picks up the new bash,"
  echo "        or:  run explicitly with  /opt/homebrew/bin/bash $0  (or /usr/local/bin/bash on Intel)"
  exit 1
fi

cd "$(dirname "$0")"

FILTER="${1:-}"
LOG_DIR="out/_logs"
mkdir -p "${LOG_DIR}"

MODELS=(
  "EleutherAI/pythia-6.9b"
  "meta-llama/Llama-3.1-8B"
  "google/gemma-4-E4B"
  "google/gemma-4-31B"
  "allenai/Olmo-3-1125-32B"
  "Qwen/Qwen2.5-7B"
  "Qwen/Qwen2.5-32B"
)

# Pass 1 -- paper-default range.
SCRIPTS_N100=(
  latin
  arabic
  persian
  devanagari
  chinese
  greek
  roman
  babylonian
)

# Pass 2 -- extended range, only for scripts where the larger window
# answers a question the small window can't.
SCRIPTS_N600=(
  babylonian
)

# Pass 3 -- extended range AND extended basis (adds T=60). Same scripts.
SCRIPTS_N600_BAB_BASIS=(
  babylonian
)
EXT_PERIODS="2,5,10,60,100"

# Match-helper: case-insensitive substring check.
matches_filter() {
  local model="${1,,}"
  local filter="${FILTER,,}"
  [[ -z "${filter}" || "${model}" == *"${filter}"* ]]
}

# Marker = the per-combo layer-sweep JSON. n_max != 100 lands in a
# `_n<n_max>` suffix, and a non-default `--periods` lands in a `_p<...>`
# suffix, so each pass has its own marker file that doesn't collide
# with the others. Must mirror the suffix logic inside main.py.
#
# We use the JSON (not the PNG) as the marker so that any cells produced
# by the pre-JSON-export version of main.py get re-run automatically:
# the PNG alone is no longer a complete output.
marker_for() {
  local model="$1"
  local script="$2"
  local n_max="$3"
  local periods="$4"      # comma-separated; empty means default
  local model_dir="${model//\//__}"
  local pool="mean"
  if (( n_max != 100 )); then
    pool="mean_n${n_max}"
  fi
  if [[ -n "${periods}" && "${periods}" != "2,5,10,100" ]]; then
    local pstr="${periods//,/-}"
    pool="${pool}_p${pstr}"
  fi
  echo "out/${model_dir}/${script}/${pool}/fig_layer_sweep.json"
}

# ----- Plan: count total and todo so the user sees the work ahead -----
total=0
todo=0
for model in "${MODELS[@]}"; do
  matches_filter "${model}" || continue
  for script in "${SCRIPTS_N100[@]}"; do
    total=$((total + 1))
    [[ -f "$(marker_for "${model}" "${script}" 100 "")" ]] || todo=$((todo + 1))
  done
  for script in "${SCRIPTS_N600[@]}"; do
    total=$((total + 1))
    [[ -f "$(marker_for "${model}" "${script}" 600 "")" ]] || todo=$((todo + 1))
  done
  for script in "${SCRIPTS_N600_BAB_BASIS[@]}"; do
    total=$((total + 1))
    [[ -f "$(marker_for "${model}" "${script}" 600 "${EXT_PERIODS}")" ]] || todo=$((todo + 1))
  done
done

echo "==================================================================="
echo "  filter:  '${FILTER:-<none>}'"
echo "  total:   ${total} (model, script, n_max) combos"
echo "  todo:    ${todo}  (the rest already have fig_layer_sweep.png)"
echo "  output:  out/<model>/<script>/mean[_n<n_max>]/"
echo "  logs:    ${LOG_DIR}/"
echo "==================================================================="
echo

# ----- Run one (model, script, n_max) combo -----
i=0
ran=0
skipped=0
failed=0
declare -a failed_list=()

run_combo() {
  local model="$1"
  local script="$2"
  local n_max="$3"
  local periods="${4:-}"     # empty -> paper default
  i=$((i + 1))
  local marker
  marker="$(marker_for "${model}" "${script}" "${n_max}" "${periods}")"

  local label="n=${n_max}"
  if [[ -n "${periods}" && "${periods}" != "2,5,10,100" ]]; then
    label="${label} p=${periods}"
  fi

  if [[ -f "${marker}" ]]; then
    printf "[%2d/%2d]  SKIP    %-30s %-10s %s\n" \
      "${i}" "${total}" "${model}" "${script}" "${label}"
    skipped=$((skipped + 1))
    return 0
  fi

  printf "[%2d/%2d]  RUN     %-30s %-10s %s\n" \
    "${i}" "${total}" "${model}" "${script}" "${label}"

  local log_file
  local log_tag="$(echo "${model}" | tr / _)__${script}"
  if (( n_max != 100 )); then
    log_tag="${log_tag}_n${n_max}"
  fi
  if [[ -n "${periods}" && "${periods}" != "2,5,10,100" ]]; then
    log_tag="${log_tag}_p${periods//,/-}"
  fi
  log_file="${LOG_DIR}/${log_tag}.log"

  local extra_args=()
  if [[ -n "${periods}" ]]; then
    extra_args+=("--periods" "${periods}")
  fi

  if uv run main.py --model "${model}" --script "${script}" \
       --n_max "${n_max}" --sweep "${extra_args[@]}" \
       > "${log_file}" 2>&1; then
    grep -E "(PC1 R²|helix R²|helix/PCA|using peak layer|basis periods)" "${log_file}" \
      | sed 's/^/         /'
    ran=$((ran + 1))
  else
    echo "         !! FAILED  (see ${log_file})"
    failed=$((failed + 1))
    failed_list+=("${model} / ${script} / ${label}")
  fi
}

# ----- Pass 1: n_max = 100 -----
for model in "${MODELS[@]}"; do
  matches_filter "${model}" || continue
  for script in "${SCRIPTS_N100[@]}"; do
    run_combo "${model}" "${script}" 100
  done
done

# ----- Pass 2: n_max = 600, paper basis (Babylonian -- the T=60 test) -----
for model in "${MODELS[@]}"; do
  matches_filter "${model}" || continue
  for script in "${SCRIPTS_N600[@]}"; do
    run_combo "${model}" "${script}" 600
  done
done

# ----- Pass 3: n_max = 600, extended basis with T=60 (Babylonian) ---------
# Quantifies how much variance the added T=60 dimension actually captures.
for model in "${MODELS[@]}"; do
  matches_filter "${model}" || continue
  for script in "${SCRIPTS_N600_BAB_BASIS[@]}"; do
    run_combo "${model}" "${script}" 600 "${EXT_PERIODS}"
  done
done

# ----- Summary -----
echo
echo "==================================================================="
echo "  summary"
echo "==================================================================="
printf "  ran:     %d\n" "${ran}"
printf "  skipped: %d\n" "${skipped}"
printf "  failed:  %d\n" "${failed}"
if (( failed > 0 )); then
  echo "  failed combos:"
  printf "    - %s\n" "${failed_list[@]}"
fi
echo
echo "  fig_layer_sweep locations:"
find out -name "fig_layer_sweep.json" 2>/dev/null | sort | sed 's/^/    /'
