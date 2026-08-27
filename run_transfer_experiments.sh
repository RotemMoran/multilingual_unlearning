#!/usr/bin/env bash
# Cross-lingual unlearning experiment batch. Every experiment unlearns TOFU forget01
# from the 10-language finetuned model and is then evaluated in each of its languages.
#
#   en+iw_2ep       joint en+iw, 2 epochs
#   en+iw_3ep       joint en+iw, 3 epochs
#   en+fr_5ep       joint en+fr, 5 epochs
#   en+fr_3ep       joint en+fr, 3 epochs
#   en+fa_5ep       joint en+fa, 5 epochs
#   en+fa_3ep       joint en+fa, 3 epochs
#   en+ja_5ep       joint en+ja, 5 epochs
#   en+ja_3ep       joint en+ja, 3 epochs
#   en+iw_half_5ep  half the forget set in en, the other half in iw, 5 epochs
#   en+fr_half_5ep  half the forget set in en, the other half in fr, 5 epochs
#
# Batch 3 splits the forget set over three and four languages and evaluates every model
# in all ten languages of the base finetune:
#
#   en+fr+ja_third_5ep       a third of the forget set in each of en, fr, ja
#   en+fr+ar_third_5ep       a third of the forget set in each of en, fr, ar
#   en+fr+ja+ar_quarter_5ep  a quarter of the forget set in each of en, fr, ja, ar
#   base                     the finetuned model before any unlearning, evaluated only
#
# `base` is the reference point for every other row: mix `none` means there is nothing to
# unlearn, so the model is the base finetune itself and only the evaluations run.
#
# Batch 5 completes the half/half sweep: English paired with every one of the other nine
# languages, 20 forget items each, every model evaluated in all ten languages.
#
#   en+ar_half_5ep en+fa_half_5ep en+hi_half_5ep en+id_half_5ep en+ja_half_5ep
#   en+ko_half_5ep en+ru_half_5ep
#
# en+fr_half_5ep is already complete; en+iw_half_5ep reuses its batch 2 model and only
# needs its eight missing evaluations, which is why its eval field is now `all`.
#
# See EXPERIMENTS.md for what each one measures. Outputs, all keyed by experiment id:
#   model    outputs/tofu_finetuned_5epoch_aya_10_lang_2e5/grad_diff_2e-05_forget01_<N>_<tag>
#   logs     logs/forget_<exp_id>.txt, logs/eval_<exp_id>_on_<lang>.txt
#   results  results/<exp_id>/eval_<lang>.csv
#
# Each step waits for a free GPU and is skipped when its output already exists.
#
#   ./run_transfer_experiments.sh                       run everything
#   ./run_transfer_experiments.sh en+fr_5ep en+fa_5ep   run only those experiments
#   ./run_transfer_experiments.sh --list                show the table and exit
#   FORCE=1 ./run_transfer_experiments.sh               redo completed steps
#   DRY_RUN=1 ./run_transfer_experiments.sh             print the commands, run nothing
#
# Batch 2 takes roughly 5-6 hours, batch 3 roughly 7; run under tmux or nohup.

set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")" || exit 1

PYTHON="${PYTHON:-/labhome/rmoran/scratch/conda/envs/tofu/bin/python}"
FORGET_LOSS="${FORGET_LOSS:-grad_diff}"
GPU_FREE_MIB="${GPU_FREE_MIB:-40000}"
FORCE="${FORCE:-0}"
DRY_RUN="${DRY_RUN:-0}"

BASE="$PWD/outputs/tofu_finetuned_5epoch_aya_10_lang_2e5"
LOG_DIR="$PWD/logs"
RESULT_ROOT="$PWD/results"

# the ten languages taught to the base finetune, i.e. every language in which the TOFU
# authors can be evaluated at all
ALL_LANGS="en fr ar fa hi id iw ja ko ru"

# exp_id | forget languages | mix | epochs | evaluation languages ("all" means ALL_LANGS)
# The language tag used in the model directory is the exp_id without its _<N>ep suffix,
# except for mix `none`, which evaluates the base finetune itself.
EXPERIMENTS=(
    "base|-|none|0|all"
    "en+iw_2ep|en,iw|concat|2|en iw"
    "en+iw_3ep|en,iw|concat|3|en iw"
    "en+fr_5ep|en,fr|concat|5|en fr"
    "en+fr_3ep|en,fr|concat|3|en fr"
    "en+fa_5ep|en,fa|concat|5|en fa"
    "en+fa_3ep|en,fa|concat|3|en fa"
    "en+ja_5ep|en,ja|concat|5|en ja"
    "en+ja_3ep|en,ja|concat|3|en ja"
    "en+iw_half_5ep|en,iw|interleave|5|all"
    "en+fr_half_5ep|en,fr|interleave|5|all"
    "en+fr+ja_third_5ep|en,fr,ja|interleave|5|all"
    "en+fr+ar_third_5ep|en,fr,ar|interleave|5|all"
    "en+fr+ja+ar_quarter_5ep|en,fr,ja,ar|interleave|5|all"
    # ordered so the family predictions of batch 5 are answered first; the loop below
    # follows this table, not the command line
    "en+ar_half_5ep|en,ar|interleave|5|all"
    "en+ko_half_5ep|en,ko|interleave|5|all"
    "en+ja_half_5ep|en,ja|interleave|5|all"
    "en+ru_half_5ep|en,ru|interleave|5|all"
    "en+hi_half_5ep|en,hi|interleave|5|all"
    "en+id_half_5ep|en,id|interleave|5|all"
    "en+fa_half_5ep|en,fa|interleave|5|all"
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

# Blocks until the GPU has enough free memory for an 8B run.
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

run_forget() {
    local exp_id=$1 langs=$2 mix=$3 epochs=$4 model=$5
    local label="$exp_id: unlearn [$langs] $mix ${epochs}ep"
    local logfile="$LOG_DIR/forget_${exp_id}.txt"

    if has_model "$model" && [ "$FORCE" != "1" ]; then
        log "SKIP  $label - model already exists"
        record "SKIP    $label"
        return 0
    fi

    local cmd=("$PYTHON" forget.py --config-name forget_joint
               "language=[$langs]" "lang_tag=${exp_id%_*}" "language_mix=$mix"
               "num_epochs=$epochs" "forget_loss=$FORGET_LOSS")
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
    if ! has_model "$model"; then
        log "FAIL  $label - no model written to ${model#$PWD/}"
        record "FAIL    $label (no model saved, see ${logfile#$PWD/})"
        return 1
    fi
    log "DONE  $label"
    record "OK      $label"
}

run_eval() {
    local exp_id=$1 model=$2 lang=$3
    local label="$exp_id: eval on $lang"
    local aggregated="$model/eval_results/ds_sizeNone_${lang}/eval_log_aggregated.json"
    local logfile="$LOG_DIR/eval_${exp_id}_on_${lang}.txt"
    local csv="$RESULT_ROOT/$exp_id/eval_${lang}.csv"
    local config="eval_multilingual"
    [ "$lang" = "en" ] && config="eval_everything"

    local cmd=("$PYTHON" evaluate_util.py --config-name "$config"
               "model_path=$model" "language=$lang")
    # dry run reports before the model check, the model is produced by the forget
    # step that a dry run also skipped
    if [ "$DRY_RUN" = "1" ]; then
        log "DRY   $label"
        log "        ${cmd[*]} > ${logfile#$PWD/}"
        log "        csv -> ${csv#$PWD/}"
        record "DRY     $label"
        return 0
    fi

    if ! has_model "$model"; then
        log "MISS  $label - no model to evaluate"
        record "MISSING $label"
        return 0
    fi
    if [ -f "$aggregated" ] && [ -f "$csv" ] && [ "$FORCE" != "1" ]; then
        log "SKIP  $label - results already exist"
        record "SKIP    $label"
        return 0
    fi

    wait_for_gpu
    log "RUN   $label -> ${logfile#$PWD/}"
    if ! "${cmd[@]}" >"$logfile" 2>&1; then
        log "FAIL  $label - see ${logfile#$PWD/}"
        record "FAIL    $label (see ${logfile#$PWD/})"
        return 0
    fi

    mkdir -p "$RESULT_ROOT/$exp_id"
    if "$PYTHON" aggregate_eval_stat.py "lang=$lang" "ckpt_result=$aggregated" \
            "method_name=$FORGET_LOSS" "save_file=$csv" >>"$logfile" 2>&1; then
        log "DONE  $label -> ${csv#$PWD/}"
        record "OK      $label -> ${csv#$PWD/}"
    else
        log "DONE  $label (aggregation failed, raw logs kept)"
        record "OK      $label (aggregation failed, see ${logfile#$PWD/})"
    fi
}

if [ "${SELECTED[0]:-}" = "--list" ]; then
    printf '%-24s %-14s %-11s %-7s %s\n' EXPERIMENT LANGUAGES MIX EPOCHS EVAL
    for row in "${EXPERIMENTS[@]}"; do
        IFS='|' read -r exp_id langs mix epochs eval_langs <<<"$row"
        [ "$eval_langs" = "all" ] && eval_langs="$ALL_LANGS"
        printf '%-24s %-14s %-11s %-7s %s\n' "$exp_id" "$langs" "$mix" "$epochs" "$eval_langs"
    done
    exit 0
fi

mkdir -p "$LOG_DIR" "$RESULT_ROOT"

log "python:        $PYTHON"
log "forget loss:   $FORGET_LOSS"
log "gpu threshold: $GPU_FREE_MIB MiB free"
log "experiments:   ${SELECTED[*]:-all}"

for row in "${EXPERIMENTS[@]}"; do
    IFS='|' read -r exp_id langs mix epochs eval_langs <<<"$row"
    wanted "$exp_id" || continue
    [ "$eval_langs" = "all" ] && eval_langs="$ALL_LANGS"

    if [ "$mix" = "none" ]; then
        model="$BASE"
    else
        model="$BASE/${FORGET_LOSS}_2e-05_forget01_${epochs}_${exp_id%_*}"
        run_forget "$exp_id" "$langs" "$mix" "$epochs" "$model"
    fi
    for lang in $eval_langs; do
        run_eval "$exp_id" "$model" "$lang"
    done
done

echo
log "summary"
for line in ${SUMMARY[@]+"${SUMMARY[@]}"}; do
    echo "  $line"
done

exit "$FAILED"
