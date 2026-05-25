#!/usr/bin/env bash
# Sweep the helix experiment across all eight models and all twelve numeral
# scripts.  Each run uses --pool mean (the honest default) and --sweep
# (layer scan + auto-targeted standard figures at the helix-R² peak).
#
# Three passes:
#   pass 1  -- 8 models x 12 scripts at n_max=100, basis=[2,5,10,100]
#              (paper-default range and basis)
#   pass 2  -- 8 models x 1 script  (babylonian) at n_max=600,
#              basis=[2,5,10,100] (paper basis -- demonstrates the basis
#              is blind to T=60)
#   pass 3  -- 8 models x 1 script  (babylonian) at n_max=600,
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
  "EleutherAI/gpt-j-6b"
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
  thai
  chinese
  binary
  hexadecimal
  greek
  hebrew
  roman
  babylonian
)

# Extended passes -- one entry per (script, n_max, periods) combination
# that goes beyond the paper-default n=100, basis [2,5,10,100] sweep.
# Format: "script:n_max:periods"  (empty periods = paper default).
#
# Each row exists to surface a specific failure mode of the standard
# protocol on its target script:
#
#   - Babylonian (base 60): T=60 needs many wraps AND inclusion in the
#     basis. Paper protocol misses both, so we run n=600 with both
#     paper-default and +T=60 bases and compare.
#
#   - Binary (base 2): natural periods are 2, 4, 8, 16, 32, 64. Paper
#     basis catches only T=2. Run at n=1024 (16 wraps of T=64) with
#     paper basis vs binary-native powers-of-2.
#
#   - Hexadecimal (base 16): natural periods 16, 32, 64, 128, 256.
#     Paper basis catches almost nothing. Run at n=1024 (4 wraps of
#     T=256) with paper basis vs hex-native periods.
EXTRA_PASSES=(
  "babylonian:600:"                  # baseline at wider window
  "babylonian:600:2,5,10,60,100"     # + T=60  (the Babylonian story)
  "binary:1024:"                     # baseline at wider window
  "binary:1024:2,4,8,16,32,64"       # binary-native (all powers of 2)
  "hexadecimal:1024:"                # baseline at wider window
  "hexadecimal:1024:16,32,64,256"    # hex-native
)

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
  for spec in "${EXTRA_PASSES[@]}"; do
    IFS=':' read -r script n_max periods <<< "${spec}"
    total=$((total + 1))
    [[ -f "$(marker_for "${model}" "${script}" "${n_max}" "${periods}")" ]] || todo=$((todo + 1))
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

  # Run inside a pseudo-terminal via script(1). Three benefits:
  #   1. main.py's stderr looks like a real TTY -> tqdm uses live \r
  #      refresh instead of getting buffered into oblivion by `tee`.
  #   2. HuggingFace's "Loading checkpoint shards" bar also shows live,
  #      so the multi-minute model load isn't silent.
  #   3. script(1) records the session to ${log_file} AND displays it on
  #      the user's terminal in one go -- no buffering acrobatics needed.
  # The log file contains a few ANSI escape codes from tqdm; grep -a
  # treats it as text, and the post-run summary still works.
  if script -q "${log_file}" uv run main.py --model "${model}" \
       --script "${script}" --n_max "${n_max}" --sweep "${extra_args[@]}"; then
    grep -aE "(PC1 R²|helix R²|helix/PCA|using peak layer|basis periods)" \
      "${log_file}" | sed 's/^/         /'
    ran=$((ran + 1))
  else
    echo "         !! FAILED  (see ${log_file})"
    failed=$((failed + 1))
    failed_list+=("${model} / ${script} / ${label}")
  fi
}

# ----- Pass 1: n_max = 100, paper-default basis on every script -----
for model in "${MODELS[@]}"; do
  matches_filter "${model}" || continue
  for script in "${SCRIPTS_N100[@]}"; do
    run_combo "${model}" "${script}" 100
  done
done

# ----- Extended passes (Babylonian, binary, hex wider windows / bases) -----
for model in "${MODELS[@]}"; do
  matches_filter "${model}" || continue
  for spec in "${EXTRA_PASSES[@]}"; do
    IFS=':' read -r script n_max periods <<< "${spec}"
    run_combo "${model}" "${script}" "${n_max}" "${periods}"
  done
done

# ============================================================================
# Post-sweep tools. Each writes its own aggregate / control / comparison
# artefact into out/. Failures in any one don't abort the others.
# ============================================================================
run_tool() {
  local label="$1"; shift
  local logname="$1"; shift
  local logpath="${LOG_DIR}/${logname}.log"
  echo
  echo "==================================================================="
  echo "  ${label}"
  echo "==================================================================="
  if "$@" > "${logpath}" 2>&1; then
    echo "  ok  (log: ${logpath})"
  else
    echo "  !! FAILED  (log: ${logpath})"
  fi
}

# Best-effort: forward --model when --filter was supplied. The Python
# tools accept a single --model only, so we pick the first match.
TOOL_MODEL_ARGS=()
if [[ -n "${FILTER}" ]]; then
  for model in "${MODELS[@]}"; do
    if matches_filter "${model}"; then
      TOOL_MODEL_ARGS+=("--model" "${model}")
      break
    fi
  done
fi

# ----- Pass 4: aggregate -----
# Cheap. Walks out/<model>/<script>/<pool>/fig_layer_sweep.json files
# and writes out/_index.json (the master cell index the React site reads
# first) plus out/_l0_share.csv (the historical L=0 / peak table).
# Doesn't accept --model; always processes every cell.
run_tool "pass 4: aggregate _index.json" "_aggregate" \
  uv run aggregate.py

# ----- Pass 5: random-embedding control -----
# Cheap (no forward passes -- just tokenize + random embedding lookup +
# mean-pool + basis fit). Writes out/_random_embed_control.json. Used in
# Finding 3 to show the L=0 helix is mechanical (Babylonian) vs learned
# (Latin/positional).
run_tool "pass 5: random-embedding control" "_embed_control" \
  uv run embed_control.py "${TOOL_MODEL_ARGS[@]}"

# ----- Pass 6: subspace alignment (CKA) -----
# EXPENSIVE (loads each model, does 100 forward passes per script). The
# script merges into out/_subspace_alignment.json and skips (model,
# script) rows already present, so re-running after the main sweep only
# costs the new cells. By default runs Latin on every model; add
# --scripts all to cover the whole matrix.
run_tool "pass 6: subspace alignment (CKA) -- Latin only by default" "_subspace_align" \
  uv run subspace_align.py "${TOOL_MODEL_ARGS[@]}" --scripts all

# ----- Pass 7: comparison image -----
# Cheap. Stitches per-model layer-sweep PNGs into one comparison image
# at out/_compare/fig_latin_sweep_<N>models.png. Layer counts on the
# header strips are read at runtime from each model's layer_sweep.json
# so they stay in sync.
run_tool "pass 7: cross-model comparison PNG" "_compare" \
  uv run compare.py

# ----- Summary -----
echo
echo "==================================================================="
echo "  summary"
echo "==================================================================="
printf "  main-sweep ran:     %d\n" "${ran}"
printf "  main-sweep skipped: %d\n" "${skipped}"
printf "  main-sweep failed:  %d\n" "${failed}"
if (( failed > 0 )); then
  echo "  failed combos:"
  printf "    - %s\n" "${failed_list[@]}"
fi
echo
echo "  fig_layer_sweep locations:"
find out -name "fig_layer_sweep.json" 2>/dev/null | sort | sed 's/^/    /'
echo
echo "  aggregate artefacts:"
for f in out/_index.json out/_l0_share.csv out/_random_embed_control.json out/_subspace_alignment.json out/_compare/*.png; do
  [[ -e "${f}" ]] && echo "    ${f}"
done
