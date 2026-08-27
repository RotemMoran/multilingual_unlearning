#!/usr/bin/env bash
# Paper-faithful steering recovery (Xiang et al. 2606.03291, Section 4.4 + Algorithms 1-2).
#
# Batch 4 (run_steering_recovery.sh) derived a per-language vector from the TARGET
# unlearned model on retain data, injected it unnormalised over a fixed block of late
# layers, and scored with ROUGE/chrF. It recovered nothing. Four of those five choices
# diverge from the paper. This driver replaces them:
#
#   * ONE auxiliary unlearned model (f_un_aux), trained on two retain authors that no
#     attacked checkpoint ever forgot. All seven arms share the finetuned parent f_ft, so
#     one vector serves every arm -- and because it is extracted from English only, using
#     it on fr/ar/ja/iw is a direct test of the paper's language-agnosticism claim.
#   * The vector is a unit direction; injection rescales it to ||h|| so one alpha means
#     the same push at every layer.
#   * Injection hits the last position only, over a 3-layer window whose start is SWEPT.
#   * NLI equivalence scoring on top of the lexical metrics.
#   * An f_ft reference pass, so recovery can be read as a share of what unlearning removed.
#
# Phases run in order and each is skip-if-done, so an interrupted run resumes:
#
#   prep       aux splits, aux model, sanity gate, the English steering vector
#   reference  f_ft generations on every forget set (the "before unlearning" ceiling)
#   quick      Appendix K.3 all-layer variant, small alpha, no window sweep  (~1.5h)
#   sweep      Algorithm 2 start-layer sweep, real + random control          (~13h)
#   nli        NLI equivalence scores over every result file
#   summarize  comparison + layer-profile tables
#
#   ./run_steering_replication.sh                       everything, in order
#   ./run_steering_replication.sh --list                show the arms and exit
#   PHASES="prep reference quick" ./run_steering_replication.sh
#   ./run_steering_replication.sh en iw                 restrict to those arms
#   DRY_RUN=1 ./run_steering_replication.sh             print commands, run nothing
#   FORCE=1 ./run_steering_replication.sh               redo completed steps
#   REQUIRE_EXCLUSIVE=0 ./run_steering_replication.sh   share the card (see wait_for_gpu)
#
# Run the sweep phase under tmux; it is an overnight job. By default every step waits for
# an IDLE card, not merely enough free memory -- see wait_for_gpu for why that distinction
# cost two evals the first time round.

set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")" || exit 1

PYTHON="${PYTHON:-/labhome/rmoran/scratch/conda/envs/tofu/bin/python}"
# Inference loads one 8B bf16 model (~16 GB); 20 GB leaves headroom for activations.
GPU_FREE_MIB="${GPU_FREE_MIB:-20000}"
FORCE="${FORCE:-0}"
DRY_RUN="${DRY_RUN:-0}"
FAMILY="${FAMILY:-aya-expanse-8B}"
PHASES="${PHASES:-prep reference quick sweep nli summarize}"

# Xiang et al. use alpha=0.5 for the high-resource pair (English, Chinese) and 0.8
# elsewhere, with N=2. Here English keeps 0.5 and every other language gets 0.8.
ALPHA_EN="${ALPHA_EN:-0.5}"
ALPHA_OTHER="${ALPHA_OTHER:-0.8}"
WINDOW="${WINDOW:-2}"
STRIDE="${STRIDE:-1}"
# Appendix K.3: all layers at once tolerates -- in fact needs -- a much smaller alpha,
# because the per-layer pushes compound.
QUICK_ALPHAS="${QUICK_ALPHAS:-0.05 0.1 0.2}"

BASE="$PWD/outputs/tofu_finetuned_5epoch_aya_10_lang_2e5"
AUX_MODEL="$BASE/grad_diff_2e-05_forget01aux_5_en"
VECTOR="$PWD/steering/vectors/aya_aux_en.pt"
RESULT_DIR="$PWD/steering/results_v2"
LOG_DIR="$PWD/logs"

# tag | languages to attack (comma-separated). The tag is the checkpoint suffix.
EXPERIMENTS=(
    "en|en"
    "iw|iw"
    "en+fr_half|en,fr"
    "en+iw_half|en,iw"
    "en+fr+ja_third|en,fr,ja"
    "en+fr+ar_third|en,fr,ar"
    "en+fr+ja+ar_quarter|en,fr,ja,ar"
)
# Every language appearing above; each needs one f_ft reference pass.
REFERENCE_LANGS=(en fr iw ja ar)

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

phase_on() {
    local p
    for p in $PHASES; do
        [ "$p" = "$1" ] && return 0
    done
    return 1
}

has_model() { [ -f "$1/model.safetensors.index.json" ]; }

alpha_for() { [ "$1" = "en" ] && echo "$ALPHA_EN" || echo "$ALPHA_OTHER"; }

gpu_busy_pids() { nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr -d ' '; }
gpu_free_mib()  { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1; }

# Jobs we must not collide with. Memory alone cannot detect these: a neighbouring run
# spends ~60s loading checkpoint shards off NFS before it touches CUDA, so for that whole
# minute nvidia-smi reports an idle card while a 35 GB allocation is already inbound. That
# is the exact window the first launch attempt drove into. Watching for the *process* is
# what makes "run one after another" reliable.
# The driver scripts are in here as well as their workhorses, so we wait for a whole
# neighbouring BATCH to drain rather than squeezing between two of its evals.
NEIGHBOUR_PATTERN="${NEIGHBOUR_PATTERN:-evaluate_util\.py|aggregate_eval_stat\.py|finetune\.py|forget\.py|run_transfer_experiments\.sh|run_en_iw_experiments\.sh|run_steering_recovery\.sh}"

# Anything this driver spawns shares our process group, so filtering on PGID keeps us from
# ever waiting on ourselves (the prep phase legitimately runs forget.py).
neighbour_pids() {
    local mypgid pid pgid comm
    mypgid=$(ps -o pgid= -p $$ 2>/dev/null | tr -d ' ')
    for pid in $(pgrep -f "$NEIGHBOUR_PATTERN" 2>/dev/null); do
        [ "$pid" = "$$" ] && continue
        pgid=$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d ' ')
        [ -n "$pgid" ] && [ "$pgid" = "$mypgid" ] && continue
        # Skip terminal multiplexers. `tmux new -d <cmd>` leaves a server daemon that
        # keeps <cmd> in its command line for the lifetime of the SESSION, long after the
        # job itself exited -- so matching it means waiting on something that never dies.
        # This cost this run 48 idle hours once; the job it was "waiting for" had finished.
        comm=$(ps -o comm= -p "$pid" 2>/dev/null | tr -d ' ')
        case "$comm" in tmux*|screen*|sshd*) continue ;; esac
        echo "$pid"
    done
}

# Free memory alone is not a safe gate on a single shared card. A neighbouring job that
# cycles through tasks (an eval loop, say) leaves a gap between tasks that looks free, and
# a 16 GB model load started in that gap collides with its next task -- both then OOM.
# That is not hypothetical: it is how the first attempt at this run died, taking two of
# the neighbour's evals with it. So by default we wait for the card to be genuinely idle
# and then confirm it stayed idle, rather than racing into a gap.
#
# Set REQUIRE_EXCLUSIVE=0 to fall back to a plain free-memory threshold, e.g. when this is
# the only job on a larger card.
REQUIRE_EXCLUSIVE="${REQUIRE_EXCLUSIVE:-1}"
SETTLE_SEC="${SETTLE_SEC:-20}"

gpu_ready() {
    # Free memory already accounts for whatever other tenants are using right now, so a
    # small unrelated job on the card is not a reason to wait. What free memory CANNOT see
    # is a neighbouring repo job that has started but is still reading checkpoint shards
    # off NFS, with its 35 GB allocation seconds away -- hence the process check, which is
    # the one that actually prevents the collision.
    [ "$(gpu_free_mib)" -ge "$GPU_FREE_MIB" ] || return 1
    [ "$REQUIRE_EXCLUSIVE" = "1" ] && [ -n "$(neighbour_pids)" ] && return 1
    return 0
}

wait_for_gpu() {
    local waited=0 blocker
    while true; do
        if gpu_ready; then
            # Re-check after a settle window: if a neighbour is mid-startup, this is where
            # we find out, before committing a 16 GB model load.
            if [ "$REQUIRE_EXCLUSIVE" = "1" ]; then
                sleep "$SETTLE_SEC"
                waited=$((waited + SETTLE_SEC))
                gpu_ready || continue
            fi
            [ "$waited" -gt 0 ] && log "GPU idle ($(gpu_free_mib) MiB free) after ${waited}s"
            return 0
        fi
        blocker="free=$(gpu_free_mib)MiB"
        [ -n "$(gpu_busy_pids)" ] && blocker="$blocker gpu_pids=[$(gpu_busy_pids | tr '\n' ' ')]"
        [ -n "$(neighbour_pids)" ] && blocker="$blocker neighbours=[$(neighbour_pids | tr '\n' ' ')]"
        if [ "$waited" -eq 0 ]; then
            log "queued behind other work: $blocker (need ${GPU_FREE_MIB} MiB and an idle card)"
        elif [ $((waited % 600)) -lt 30 ]; then
            log "  still queued (${waited}s): $blocker"
        fi
        sleep 30
        waited=$((waited + 30))
    done
}

# run <label> <logfile> <sentinel-file-or-empty> <cmd...>
# Skips when the sentinel exists, waits for the GPU, logs, and records the outcome.
run() {
    local label=$1 logfile=$2 sentinel=$3
    shift 3

    if [ -n "$sentinel" ] && [ -e "$sentinel" ] && [ "$FORCE" != "1" ]; then
        log "SKIP  $label - ${sentinel#$PWD/}"
        record "SKIP    $label"
        return 0
    fi
    if [ "$DRY_RUN" = "1" ]; then
        log "DRY   $label"
        log "        $* > ${logfile#$PWD/}"
        record "DRY     $label"
        return 0
    fi

    wait_for_gpu
    log "RUN   $label -> ${logfile#$PWD/}"
    if ! "$@" >"$logfile" 2>&1; then
        log "FAIL  $label - see ${logfile#$PWD/}"
        record "FAIL    $label (see ${logfile#$PWD/})"
        return 1
    fi
    if [ -n "$sentinel" ] && [ ! -e "$sentinel" ]; then
        log "FAIL  $label - produced no output at ${sentinel#$PWD/}"
        record "FAIL    $label (no output written)"
        return 1
    fi
    log "DONE  $label"
    record "OK      $label"
}

# One attack invocation. kind=real|random, mode=sweep|alllayers
run_attack() {
    local tag=$1 lang=$2 model=$3 mode=$4 kind=$5
    local stem="aya_aux_${tag}_${lang}_${mode}"
    [ "$kind" = "random" ] && stem="${stem}_random"
    local out="$RESULT_DIR/${stem}.json"

    local cmd=("$PYTHON" steering/attack_generate.py
               --model "$model"
               --vector "$VECTOR"
               --forget-dataset "./dataset/forget01_${lang}"
               --language "$lang"
               --model-family "$FAMILY"
               --window "$WINDOW"
               --out "$out")
    if [ "$mode" = "alllayers" ]; then
        cmd+=(--mode all-layers --alphas $QUICK_ALPHAS)
    else
        cmd+=(--mode sweep --stride "$STRIDE" --alphas "$(alpha_for "$lang")")
    fi
    [ "$kind" = "random" ] && cmd+=(--random-vector)

    if [ "$DRY_RUN" != "1" ] && [ ! -f "$VECTOR" ]; then
        log "MISS  $tag/$lang $mode ($kind) - no vector, run the prep phase first"
        record "MISSING $tag/$lang $mode ($kind) (no vector)"
        return 1
    fi
    run "$tag/$lang $mode ($kind)" \
        "$LOG_DIR/steer2_${stem}.txt" "$out" "${cmd[@]}"
}

# ---------------------------------------------------------------- phase: prep

phase_prep() {
    log "=== phase prep ==="

    run "aux splits" "$LOG_DIR/steer2_aux_splits.txt" \
        "$PWD/dataset/forget01_aux_en" \
        "$PYTHON" steering/make_aux_forget.py || return 1

    if ! has_model "$AUX_MODEL" || [ "$FORCE" = "1" ]; then
        run "train f_un_aux" "$LOG_DIR/steer2_forget_aux_en.txt" \
            "$AUX_MODEL/model.safetensors.index.json" \
            "$PYTHON" forget.py --config-name forget_aux_en || return 1
    else
        log "SKIP  train f_un_aux - ${AUX_MODEL#$PWD/}"
        record "SKIP    train f_un_aux"
    fi

    run "extract steering vector" "$LOG_DIR/steer2_extract_aux_en.txt" "$VECTOR" \
        "$PYTHON" steering/extract_steering_vector.py \
        --base-model "$BASE" \
        --unlearned-model "$AUX_MODEL" \
        --aux-dataset ./dataset/forget01_aux_en \
        --language en \
        --model-family "$FAMILY" \
        --out "$VECTOR" || return 1

    # Sanity gate. If f_un_aux did not actually forget its own authors then the
    # f_ft -> f_un_aux difference is not an unlearning direction and everything downstream
    # is measuring noise. Two alpha=0 generation passes, so no vector is needed.
    local ft_check="$RESULT_DIR/aux_check_ft.json"
    local un_check="$RESULT_DIR/aux_check_un.json"
    run "aux gate: f_ft on aux authors" "$LOG_DIR/steer2_aux_check_ft.txt" "$ft_check" \
        "$PYTHON" steering/attack_generate.py --model "$BASE" \
        --forget-dataset ./dataset/forget01_aux_en --language en \
        --model-family "$FAMILY" --mode window --start-layer 0 --alphas 0 \
        --out "$ft_check"
    run "aux gate: f_un_aux on aux authors" "$LOG_DIR/steer2_aux_check_un.txt" "$un_check" \
        "$PYTHON" steering/attack_generate.py --model "$AUX_MODEL" \
        --forget-dataset ./dataset/forget01_aux_en --language en \
        --model-family "$FAMILY" --mode window --start-layer 0 --alphas 0 \
        --out "$un_check"

    if [ "$DRY_RUN" != "1" ] && [ -f "$ft_check" ] && [ -f "$un_check" ]; then
        "$PYTHON" - "$ft_check" "$un_check" <<'PY'
import json, sys
ft, un = (json.load(open(p)) for p in sys.argv[1:3])
a, b = ft["baseline"], un["baseline"]
print(f"aux sanity gate: f_ft ROUGE-L={a:.4f} -> f_un_aux={b:.4f} (drop {a-b:+.4f})")
if a - b < 0.10:
    print("WARNING: f_un_aux barely forgot its own authors. The f_ft -> f_un_aux")
    print("difference may not be an unlearning direction. Inspect before trusting")
    print("any recovery number downstream.")
else:
    print("gate passed: the auxiliary model genuinely suppressed its forget set.")
PY
    fi
}

# ----------------------------------------------------------- phase: reference

phase_reference() {
    log "=== phase reference (f_ft ceiling) ==="
    local lang
    for lang in "${REFERENCE_LANGS[@]}"; do
        local out="$RESULT_DIR/ft_baseline_${lang}.json"
        # alpha=0 only: no hook is installed, so this is a plain generation pass on the
        # finetuned parent. It is the "before unlearning" score every recovery fraction
        # is measured against.
        run "f_ft baseline ($lang)" "$LOG_DIR/steer2_ft_baseline_${lang}.txt" "$out" \
            "$PYTHON" steering/attack_generate.py --model "$BASE" \
            --forget-dataset "./dataset/forget01_${lang}" --language "$lang" \
            --model-family "$FAMILY" --mode window --start-layer 0 --alphas 0 \
            --out "$out"
    done
}

# ------------------------------------------------------- phases: quick, sweep

phase_attacks() {
    local mode=$1
    log "=== phase $mode ==="
    local row tag langs model lang
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
            run_attack "$tag" "$lang" "$model" "$mode" real
            run_attack "$tag" "$lang" "$model" "$mode" random
        done
    done
}

# ----------------------------------------------------------------- phase: nli

phase_nli() {
    log "=== phase nli ==="
    if [ "$DRY_RUN" = "1" ]; then
        log "DRY   nli scoring over $RESULT_DIR/*.json"
        record "DRY     nli scoring"
        return 0
    fi
    shopt -s nullglob
    local files=("$RESULT_DIR"/*.json)
    shopt -u nullglob
    if [ ${#files[@]} -eq 0 ]; then
        log "SKIP  nli scoring - no result files yet"
        record "SKIP    nli scoring"
        return 0
    fi
    local extra=()
    [ "$FORCE" = "1" ] && extra+=(--force)
    run "nli scoring (${#files[@]} files)" "$LOG_DIR/steer2_nli.txt" "" \
        "$PYTHON" steering/nli_score.py ${extra[@]+"${extra[@]}"} "${files[@]}"
}

# ----------------------------------------------------------- phase: summarize

phase_summarize() {
    log "=== phase summarize ==="
    if [ "$DRY_RUN" = "1" ]; then
        log "DRY   summarize"
        return 0
    fi
    "$PYTHON" steering/summarize_results.py comparison \
        --results-dir "$RESULT_DIR" --csv results/steering_comparison_v2.csv
    echo
    "$PYTHON" steering/summarize_results.py layer-profile --results-dir "$RESULT_DIR"
    echo
    "$PYTHON" steering/summarize_results.py interleave-split \
        --results-dir "$RESULT_DIR" --csv results/steering_interleave_split_v2.csv
}

# ---------------------------------------------------------------------- main

if [ "${SELECTED[0]:-}" = "--list" ]; then
    printf '%-24s %-16s %s\n' TAG LANGUAGES CHECKPOINT
    for row in "${EXPERIMENTS[@]}"; do
        IFS='|' read -r tag langs <<<"$row"
        printf '%-24s %-16s %s\n' "$tag" "$langs" "grad_diff_2e-05_forget01_5_${tag}"
    done
    echo
    echo "vector:     ${VECTOR#$PWD/} (single English vector, shared by every arm)"
    echo "aux model:  ${AUX_MODEL#$PWD/}"
    echo "phases:     $PHASES"
    echo "alphas:     en=$ALPHA_EN other=$ALPHA_OTHER  window=N+$((WINDOW))  quick=$QUICK_ALPHAS"
    exit 0
fi

mkdir -p "$LOG_DIR" "$RESULT_DIR" "$(dirname "$VECTOR")" results

log "python:        $PYTHON"
log "gpu threshold: $GPU_FREE_MIB MiB free"
log "phases:        $PHASES"
log "alphas:        en=$ALPHA_EN other=$ALPHA_OTHER  window=[c, c+$WINDOW]"
log "arms:          ${SELECTED[*]:-all}"

phase_on prep      && { phase_prep || log "prep phase incomplete"; }
phase_on reference && phase_reference
phase_on quick     && phase_attacks alllayers
phase_on sweep     && phase_attacks sweep
phase_on nli       && phase_nli
phase_on summarize && phase_summarize

echo
log "summary"
for line in ${SUMMARY[@]+"${SUMMARY[@]}"}; do
    echo "  $line"
done

exit "$FAILED"
