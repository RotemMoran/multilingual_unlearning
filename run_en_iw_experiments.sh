#!/usr/bin/env bash
# English/Hebrew unlearning-transfer experiments:
#
#   1. unlearn Hebrew                       (forget_iw, grad_diff)
#   2. evaluate the Hebrew-unlearned model on English
#   3. evaluate the Hebrew-unlearned model on Hebrew
#   4. evaluate the joint en+iw model       on English
#   5. evaluate the joint en+iw model       on Hebrew
#   6. evaluate the English-unlearned model on Hebrew
#
# Each step waits for the GPU to free up, writes its output to logs/, and is
# skipped when its result already exists. Steps 4-6 need models that earlier runs
# produced; if one is missing the step is reported as missing and the rest carry on.
#
#   ./run_en_iw_experiments.sh          run everything
#   ./run_en_iw_experiments.sh 1 3      run only steps 1 and 3
#   FORCE=1 ./run_en_iw_experiments.sh  redo steps whose results already exist
#
# Expect several hours end to end; run it under tmux or nohup.

set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")" || exit 1

PYTHON="${PYTHON:-/labhome/rmoran/scratch/conda/envs/tofu/bin/python}"
FORGET_LOSS="${FORGET_LOSS:-grad_diff}"
GPU_FREE_MIB="${GPU_FREE_MIB:-40000}"
FORCE="${FORCE:-0}"
# effective batch of 6, split so an 8B model fits on one 46GB card - same setting
# the existing en and en+iw models were trained with
BATCH_SIZE="${BATCH_SIZE:-2}"
GRAD_ACCUM="${GRAD_ACCUM:-3}"

BASE="$PWD/outputs/tofu_finetuned_5epoch_aya_10_lang_2e5"
LOG_DIR="$PWD/logs"
RESULT_DIR="$PWD/results/en_iw_transfer"

EN_MODEL="$BASE/${FORGET_LOSS}_2e-05_forget01_5_en"
IW_MODEL="$BASE/${FORGET_LOSS}_2e-05_forget01_5_iw"
JOINT_MODEL="$BASE/${FORGET_LOSS}_2e-05_forget01_5_en+iw"

STEPS=(${@+"$@"})
[ ${#STEPS[@]} -eq 0 ] && STEPS=(1 2 3 4 5 6)

SUMMARY=()
FAILED=0

log() { printf '[%s] %s\n' "$(date '+%F %T')" "$*"; }

record() {
    SUMMARY+=("$1")
    [ "${1:0:4}" = "FAIL" ] && FAILED=1
    return 0
}

wants() {
    local step
    for step in "${STEPS[@]}"; do
        [ "$step" = "$1" ] && return 0
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

# unlearn one language: step_no, language, config name
run_forget() {
    local step=$1 lang=$2 config=$3
    local label="step $step: forget $lang ($FORGET_LOSS)"
    local model="$BASE/${FORGET_LOSS}_2e-05_forget01_5_${lang}"
    local logfile="$LOG_DIR/forget_${lang}_${FORGET_LOSS}.txt"

    if has_model "$model" && [ "$FORCE" != "1" ]; then
        log "SKIP  $label - $model already holds a final model"
        record "SKIP    $label"
        return 0
    fi

    wait_for_gpu
    log "RUN   $label -> ${logfile#$PWD/}"
    if "$PYTHON" forget.py --config-name "$config" "forget_loss=$FORGET_LOSS" \
            "batch_size=$BATCH_SIZE" "gradient_accumulation_steps=$GRAD_ACCUM" >"$logfile" 2>&1; then
        if has_model "$model"; then
            log "DONE  $label"
            record "OK      $label"
        else
            log "FAIL  $label - no model written to $model"
            record "FAIL    $label (no model saved, see ${logfile#$PWD/})"
        fi
    else
        log "FAIL  $label - see ${logfile#$PWD/}"
        record "FAIL    $label (see ${logfile#$PWD/})"
    fi
}

# evaluate one model in one language: step_no, model dir, short tag, language
run_eval() {
    local step=$1 model=$2 tag=$3 lang=$4
    local label="step $step: eval $tag on $lang"
    local out_dir="$model/eval_results/ds_sizeNone_${lang}"
    local aggregated="$out_dir/eval_log_aggregated.json"
    local logfile="$LOG_DIR/eval_${tag}_${lang}.txt"
    local config="eval_multilingual"
    [ "$lang" = "en" ] && config="eval_everything"

    if ! has_model "$model"; then
        log "MISS  $label - $model has no final model yet"
        record "MISSING $label (no model at ${model#$PWD/})"
        return 0
    fi
    if [ -f "$aggregated" ] && [ "$FORCE" != "1" ]; then
        log "SKIP  $label - $aggregated already exists"
        record "SKIP    $label"
        return 0
    fi

    wait_for_gpu
    log "RUN   $label -> ${logfile#$PWD/}"
    if ! "$PYTHON" evaluate_util.py --config-name "$config" \
            "model_path=$model" "language=$lang" >"$logfile" 2>&1; then
        log "FAIL  $label - see ${logfile#$PWD/}"
        record "FAIL    $label (see ${logfile#$PWD/})"
        return 0
    fi

    # turn the raw eval logs into the model-utility csv
    local csv="$RESULT_DIR/${tag}_eval_${lang}.csv"
    if "$PYTHON" aggregate_eval_stat.py "lang=$lang" "ckpt_result=$aggregated" \
            "method_name=$FORGET_LOSS" "save_file=$csv" >>"$logfile" 2>&1; then
        log "DONE  $label -> ${csv#$PWD/}"
        record "OK      $label -> ${csv#$PWD/}"
    else
        log "DONE  $label (aggregation failed, raw logs are in ${out_dir#$PWD/})"
        record "OK      $label (aggregation failed, see ${logfile#$PWD/})"
    fi
}

mkdir -p "$LOG_DIR" "$RESULT_DIR"

log "python:        $PYTHON"
log "forget loss:   $FORGET_LOSS"
log "batch:         $BATCH_SIZE x $GRAD_ACCUM accumulation steps"
log "steps:         ${STEPS[*]}"
log "gpu threshold: $GPU_FREE_MIB MiB free"

wants 1 && run_forget 1 iw forget_iw
wants 2 && run_eval 2 "$IW_MODEL"    unlearn_iw    en
wants 3 && run_eval 3 "$IW_MODEL"    unlearn_iw    iw
wants 4 && run_eval 4 "$JOINT_MODEL" unlearn_en_iw en
wants 5 && run_eval 5 "$JOINT_MODEL" unlearn_en_iw iw
wants 6 && run_eval 6 "$EN_MODEL"    unlearn_en    iw

echo
log "summary"
for line in ${SUMMARY[@]+"${SUMMARY[@]}"}; do
    echo "  $line"
done

exit "$FAILED"
