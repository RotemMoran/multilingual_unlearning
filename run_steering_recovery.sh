#!/usr/bin/env bash
# Steering-vector recovery on the monolingual and interleaved unlearning arms.
#
# For each checkpoint, extract a language-specific vector on retain99_<lang>
# and attack forget01_<lang> (real vector + norm-matched random control).
# Layer set is late; alphas are 0 0.5 1 2 4.
#
#   en                         English-only unlearn, attack en
#   iw                         Hebrew-only unlearn, attack iw
#   en+fr_half                 interleave en/fr, attack en and fr
#   en+iw_half                 interleave en/iw, attack en and iw
#   en+fr+ja_third             interleave en/fr/ja, attack en, fr, ja
#   en+fr+ar_third             interleave en/fr/ar, attack en, fr, ar
#   en+fr+ja+ar_quarter        interleave en/fr/ja/ar, attack en, fr, ja, ar
#
# Each step waits for a free GPU and is skipped when its output already exists.
#
#   ./run_steering_recovery.sh                          run everything
#   ./run_steering_recovery.sh en iw en+fr_half         run only those arms
#   ./run_steering_recovery.sh --list                   show the table and exit
#   FORCE=1 ./run_steering_recovery.sh                  redo completed steps
#   DRY_RUN=1 ./run_steering_recovery.sh                print the commands, run nothing
#
# Budget ~8-10 hours on one L40; run under tmux or nohup.

set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")" || exit 1

PYTHON="${PYTHON:-/labhome/rmoran/scratch/conda/envs/tofu/bin/python}"
# Inference loads one 8B bf16 model (~16 GB); 20 GB leaves headroom for activations.
GPU_FREE_MIB="${GPU_FREE_MIB:-20000}"
FORCE="${FORCE:-0}"
DRY_RUN="${DRY_RUN:-0}"
FAMILY="${FAMILY:-aya-expanse-8B}"
LAYER_SET="${LAYER_SET:-late}"
ALPHAS="${ALPHAS:-0 0.5 1 2 4}"

BASE="$PWD/outputs/tofu_finetuned_5epoch_aya_10_lang_2e5"
VECTOR_DIR="$PWD/steering/vectors"
RESULT_DIR="$PWD/steering/results"
LOG_DIR="$PWD/logs"

# tag | unlearn languages (comma-separated). The tag is the checkpoint suffix.
EXPERIMENTS=(
    "en|en"
    "iw|iw"
    "en+fr_half|en,fr"
    "en+iw_half|en,iw"
    "en+fr+ja_third|en,fr,ja"
    "en+fr+ar_third|en,fr,ar"
    "en+fr+ja+ar_quarter|en,fr,ja,ar"
)

SELECTED=(${@+"$@"})
SUMMARY=()
FAILED=0

log() { printf '[%s] %s\n' "$(date '+%F %T')" "$*"; }

record() {
    SUMMARY+=("$1")
    [ "${1:0:4}" = "FAIL" ] && FAILED=1
    return 0
}

wanted() {
    [ ${#SELECTED[@]} -eq 0 ] && return 0
    local want
    for want in "${SELECTED[@]}"; do
        [ "$want" = "$1" ] && return 0
    done
    return 1
}

has_model() { [ -f "$1/model.safetensors.index.json" ]; }

wait_for_gpu() {
    local waited=0 free
    while true; do
        free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
        if [ "${free:-0}" -ge "$GPU_FREE_MIB" ]; then
            [ "$waited" -gt 0 ] && log "GPU free (${free} MiB) after ${waited}s"
            return 0
        fi
        if [ "$waited" -eq 0 ]; then
            log "waiting for GPU: ${free} MiB free, need ${GPU_FREE_MIB} MiB"
        elif [ $((waited % 600)) -eq 0 ]; then
            log "  still waiting: ${free} MiB free (${waited}s)"
        fi
        sleep 30
        waited=$((waited + 30))
    done
}

run_extract() {
    local tag=$1 lang=$2 model=$3
    local vector="$VECTOR_DIR/aya_grad_diff_${tag}_${lang}.pt"
    local label="$tag: extract vector ($lang)"
    local logfile="$LOG_DIR/steer_extract_${tag}_${lang}.txt"

    if [ -f "$vector" ] && [ "$FORCE" != "1" ]; then
        log "SKIP  $label - ${vector#$PWD/}"
        record "SKIP    $label"
        return 0
    fi

    local cmd=("$PYTHON" steering/extract_steering_vector.py
               --base-model "$BASE"
               --unlearned-model "$model"
               --aux-dataset "./dataset/retain99_${lang}"
               --language "$lang"
               --model-family "$FAMILY"
               --out "$vector")
    if [ "$DRY_RUN" = "1" ]; then
        log "DRY   $label"
        log "        ${cmd[*]} > ${logfile#$PWD/}"
        record "DRY     $label"
        return 0
    fi

    wait_for_gpu
    log "RUN   $label -> ${logfile#$PWD/}"
    if ! "${cmd[@]}" >"$logfile" 2>&1; then
        log "FAIL  $label - see ${logfile#$PWD/}"
        record "FAIL    $label (see ${logfile#$PWD/})"
        return 1
    fi
    if [ ! -f "$vector" ]; then
        log "FAIL  $label - no vector written"
        record "FAIL    $label (no vector saved, see ${logfile#$PWD/})"
        return 1
    fi
    log "DONE  $label -> ${vector#$PWD/}"
    record "OK      $label"
}

run_attack() {
    local tag=$1 lang=$2 model=$3 random=$4
    local vector="$VECTOR_DIR/aya_grad_diff_${tag}_${lang}.pt"
    local stem="aya_grad_diff_${tag}_${lang}_${LAYER_SET}"
    [ "$random" = "1" ] && stem="${stem}_random"
    local out="$RESULT_DIR/${stem}.json"
    local kind="real"
    [ "$random" = "1" ] && kind="random"
    local label="$tag: attack $lang ($kind, $LAYER_SET)"
    local logfile="$LOG_DIR/steer_attack_${tag}_${lang}_${LAYER_SET}"
    [ "$random" = "1" ] && logfile="${logfile}_random"
    logfile="${logfile}.txt"

    if [ -f "$out" ] && [ "$FORCE" != "1" ]; then
        log "SKIP  $label - ${out#$PWD/}"
        record "SKIP    $label"
        return 0
    fi

    local cmd=("$PYTHON" steering/attack_generate.py
               --model "$model"
               --vector "$vector"
               --forget-dataset "./dataset/forget01_${lang}"
               --language "$lang"
               --model-family "$FAMILY"
               --alphas $ALPHAS
               --layer-set "$LAYER_SET"
               --out "$out")
    [ "$random" = "1" ] && cmd+=(--random-vector)

    if [ "$DRY_RUN" = "1" ]; then
        log "DRY   $label"
        log "        ${cmd[*]} > ${logfile#$PWD/}"
        record "DRY     $label"
        return 0
    fi

    if [ ! -f "$vector" ]; then
        log "MISS  $label - no vector at ${vector#$PWD/}"
        record "MISSING $label"
        return 0
    fi

    wait_for_gpu
    log "RUN   $label -> ${logfile#$PWD/}"
    if ! "${cmd[@]}" >"$logfile" 2>&1; then
        log "FAIL  $label - see ${logfile#$PWD/}"
        record "FAIL    $label (see ${logfile#$PWD/})"
        return 1
    fi
    if [ ! -f "$out" ]; then
        log "FAIL  $label - no result written"
        record "FAIL    $label (no json saved, see ${logfile#$PWD/})"
        return 1
    fi
    log "DONE  $label -> ${out#$PWD/}"
    record "OK      $label"
}

if [ "${SELECTED[0]:-}" = "--list" ]; then
    printf '%-24s %-16s %s\n' TAG LANGUAGES CHECKPOINT
    for row in "${EXPERIMENTS[@]}"; do
        IFS='|' read -r tag langs <<<"$row"
        printf '%-24s %-16s %s\n' "$tag" "$langs" "grad_diff_2e-05_forget01_5_${tag}"
    done
    exit 0
fi

mkdir -p "$LOG_DIR" "$VECTOR_DIR" "$RESULT_DIR"

log "python:        $PYTHON"
log "gpu threshold: $GPU_FREE_MIB MiB free"
log "layer set:     $LAYER_SET"
log "alphas:        $ALPHAS"
log "arms:          ${SELECTED[*]:-all}"

for row in "${EXPERIMENTS[@]}"; do
    IFS='|' read -r tag langs <<<"$row"
    wanted "$tag" || continue
    model="$BASE/grad_diff_2e-05_forget01_5_${tag}"
    if [ "$DRY_RUN" != "1" ] && ! has_model "$model"; then
        log "MISS  $tag - no checkpoint at ${model#$PWD/}"
        record "MISSING $tag (no checkpoint)"
        continue
    fi
    IFS=',' read -ra lang_arr <<<"$langs"
    for lang in "${lang_arr[@]}"; do
        run_extract "$tag" "$lang" "$model" || continue
        run_attack "$tag" "$lang" "$model" 0 || continue
        run_attack "$tag" "$lang" "$model" 1 || continue
    done
done

echo
log "summary"
for line in ${SUMMARY[@]+"${SUMMARY[@]}"}; do
    echo "  $line"
done

exit "$FAILED"
