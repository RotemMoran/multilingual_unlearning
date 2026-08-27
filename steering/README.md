# Steering-recovery experiment (Aya-Expanse 8B)

Does monolingual unlearning only suppress knowledge, leaving it recoverable by an
inference-time activation intervention? Does spreading the same forget items across
languages — half, third, or quarter of the split per language — make that recovery
harder?

The current implementation follows Xiang et al., arXiv `2606.03291` (PDF in the repo
root), Section 4.4 and Algorithms 1-2. A vector is extracted from the finetuned parent
and an **auxiliary** unlearned model on the set that auxiliary model actually unlearned:

```text
s[l] = sum_x ( h_un(x)/‖h_un(x)‖ - h_ft(x)/‖h_ft(x)‖ )   at the last prompt token
g[l] = s[l] / ‖s[l]‖
h    = h - alpha * ‖h‖ * g[l]        last position only, over a swept 3-layer window
```

**Sections 1-4 below describe the superseded Batch 4 method** (unnormalised, target-derived,
answer-token-averaged, fixed `late` layers). They are kept because `steering/results/` was
produced that way and `EXPERIMENTS.md` Batch 4 reports those numbers. For new work use
[the current pipeline](#current-pipeline-batch-6) at the bottom; `attack_generate.py` will
refuse a vector saved under the old convention rather than silently mix the two.

The model is **Aya-Expanse 8B** throughout (`config/model_config.yaml`,
`CohereForAI/aya-expanse-8b`). The common base is the 10-language TOFU finetune:

```text
outputs/tofu_finetuned_5epoch_aya_10_lang_2e5
```

Unlearning uses `grad_diff` (`config/forget_joint.yaml`): `batch_size: 2`,
`gradient_accumulation_steps: 3` (effective batch 6), 5 epochs, lr 2e-5. The forget
split is always TOFU `forget01` (40 items). See `EXPERIMENTS.md` for the full log.

## Unlearning arms already on disk

Joint runs use `language_mix: interleave`: each of the 40 forget items is unlearned
in exactly one language (`item i` goes to language `i % N`). The forget set stays 40
examples no matter how many languages share it, so every arm below has the same
optimizer budget (7 steps per epoch, 35 at 5 epochs) as a monolingual run. The retain
set is the full union of the trained languages.

| Arm | Forget mix | Items per language | Checkpoint |
|---|---|---|---|
| EN | English only | 40 | `.../grad_diff_2e-05_forget01_5_en` |
| IW | Hebrew only | 40 | `.../grad_diff_2e-05_forget01_5_iw` |
| half (fr) | `en+fr_half` | 20 / 20 | `.../grad_diff_2e-05_forget01_5_en+fr_half` |
| half (iw) | `en+iw_half` | 20 / 20 | `.../grad_diff_2e-05_forget01_5_en+iw_half` |
| third (ja) | `en+fr+ja_third` | 14 / 13 / 13 | `.../grad_diff_2e-05_forget01_5_en+fr+ja_third` |
| third (ar) | `en+fr+ar_third` | 14 / 13 / 13 | `.../grad_diff_2e-05_forget01_5_en+fr+ar_third` |
| quarter | `en+fr+ja+ar_quarter` | 10 each | `.../grad_diff_2e-05_forget01_5_en+fr+ja+ar_quarter` |

The batch-4 run of these seven arms is logged in `EXPERIMENTS.md`. Vectors are
`steering/vectors/aya_grad_diff_<tag>_<lang>.pt`; attacks are
`steering/results/aya_grad_diff_<tag>_<lang>_late.json` (and `_late_random.json`).

Steering extract and attack read `dataset/forget01_<lang>` and
`dataset/retain99_<lang>`, not the perturbed eval copies.

Aya decoder blocks are at `model.layers`. Extraction and attack hook that same
tensor. Means are taken over answer tokens only (`labels != -100`).

Because interleave keeps the forget set at 40 rows, these arms are matched on
optimizer steps. A recovery gap between EN and half/third/quarter is not explained
by extra training.

Before attributing a recovery gap to the mix, check that the checkpoints are in a
similar forgetting regime on forget probability / truth ratio / utility
(`summarize_results.py forget-quality`). Spreading thinner does weaken in-language
forgetting (see the batch 3 table in `EXPERIMENTS.md`); if the quarter model barely
forgot, a larger recovery is expected for that reason alone.

Extract and attack are inference. Their default `--batch-size` is **16**, matching
`config/eval_everything.yaml` and `config/eval_multilingual.yaml`. There is no
gradient accumulation at inference.

## 1. Materialise English TOFU splits on disk

English forget/retain currently live on the Hub. The steering scripts only call
`load_from_disk`, so write them once:

```bash
python steering/make_en_datasets.py
python steering/check_data.py
python steering/check_data.py --tokenizer --model-family aya-expanse-8B
```

French, Japanese, and Arabic data are already at `dataset/forget01_{fr,ja,ar}` and
`dataset/retain99_{fr,ja,ar}`.

## 2. Extract a steering vector

One model at a time (8B bf16 is ~16 GB). Default family is `aya-expanse-8B`.
Extract in the language you will attack; for an interleaved checkpoint that is each
of its trained languages in turn.

```bash
BASE=outputs/tofu_finetuned_5epoch_aya_10_lang_2e5
CKPT=$BASE/grad_diff_2e-05_forget01_5_en

python steering/extract_steering_vector.py \
  --base-model "$BASE" \
  --unlearned-model "$CKPT" \
  --aux-dataset ./dataset/retain99_en --language en \
  --model-family aya-expanse-8B \
  --out steering/vectors/aya_grad_diff_en.pt
```

Half (English side of `en+fr_half`):

```bash
python steering/extract_steering_vector.py \
  --base-model "$BASE" \
  --unlearned-model $BASE/grad_diff_2e-05_forget01_5_en+fr_half \
  --aux-dataset ./dataset/retain99_en --language en \
  --model-family aya-expanse-8B \
  --out steering/vectors/aya_grad_diff_en+fr_half_en.pt
```

Repeat with `--aux-dataset ./dataset/retain99_fr --language fr` for the French
vector, and likewise for the third and quarter checkpoints in each of their
languages (fr, ja, ar as needed).

## 3. Attack

```bash
python steering/attack_generate.py \
  --model "$CKPT" \
  --vector steering/vectors/aya_grad_diff_en.pt \
  --forget-dataset ./dataset/forget01_en --language en \
  --model-family aya-expanse-8B \
  --alphas 0 0.5 1 2 4 --layer-set late \
  --out steering/results/aya_grad_diff_en_late.json

python steering/attack_generate.py \
  --model "$CKPT" \
  --vector steering/vectors/aya_grad_diff_en.pt \
  --forget-dataset ./dataset/forget01_en --language en \
  --model-family aya-expanse-8B \
  --alphas 0 0.5 1 2 4 --layer-set late --random-vector \
  --out steering/results/aya_grad_diff_en_late_random.json
```

Same pattern on an interleaved checkpoint, matching vector, forget split, and
language:

```bash
python steering/attack_generate.py \
  --model $BASE/grad_diff_2e-05_forget01_5_en+fr_half \
  --vector steering/vectors/aya_grad_diff_en+fr_half_en.pt \
  --forget-dataset ./dataset/forget01_en --language en \
  --model-family aya-expanse-8B \
  --alphas 0 0.5 1 2 4 --layer-set late \
  --out steering/results/aya_grad_diff_en+fr_half_en_late.json
```

Sweep `late` / `middle` / `all` if you want a layer-set grid. Score English (and
French) with ROUGE-L recall and Arabic/Japanese with chrF (`steering/metrics.py`
keeps non-Latin scripts; stock `rouge_score` does not).

## 4. Summarise

```bash
python steering/summarize_results.py forget-quality \
  --checkpoints \
    outputs/tofu_finetuned_5epoch_aya_10_lang_2e5/grad_diff_2e-05_forget01_5_en \
    outputs/tofu_finetuned_5epoch_aya_10_lang_2e5/grad_diff_2e-05_forget01_5_en+fr_half \
    outputs/tofu_finetuned_5epoch_aya_10_lang_2e5/grad_diff_2e-05_forget01_5_en+fr+ja_third \
    outputs/tofu_finetuned_5epoch_aya_10_lang_2e5/grad_diff_2e-05_forget01_5_en+fr+ar_third \
    outputs/tofu_finetuned_5epoch_aya_10_lang_2e5/grad_diff_2e-05_forget01_5_en+fr+ja+ar_quarter \
  --csv results/aya_forget_quality.csv

python steering/summarize_results.py attacks \
  --results-dir steering/results \
  --csv results/aya_attack_recovery.csv

python steering/summarize_results.py comparison \
  --results-dir steering/results \
  --csv results/steering_comparison.csv

python steering/summarize_results.py interleave-split \
  --results-dir steering/results \
  --csv results/steering_interleave_split.csv
```

Or run the whole grid with `./run_steering_recovery.sh`.

| Pattern | Conclusion |
|---|---|
| EN recovers, interleaved arms do not, and forget metrics are similar | evidence that splitting the forget set across languages removes knowledge more thoroughly |
| recovery falls from half to third to quarter | spreading thinner makes the attack harder |
| EN recovers, interleaved does not, but the interleaved arm forgot much less (or much more) | the mix changed forgetting depth; do not attribute the gap to multilinguality alone |
| every arm recovers | interleaved unlearning does not stop the attack |
| no arm recovers | the attack failed; check sign, ceiling, and alpha range |
| random control recovers similarly | no directional steering result |

Always report unattacked score, best attacked score, recovery delta, and random-control delta.

## Notes

- `max_length` is 500, matching `forget.py` / `finetune.py`.
- Unlearning: `batch_size=2`, `gradient_accumulation_steps=3` (already used).
- Extract/attack: `--batch-size 16`, matching the eval configs.

## Current pipeline (Batch 6)

Everything below supersedes sections 1-4. Results go to `steering/results_v2/`, leaving the
Batch 4 JSONs intact for comparison. See `EXPERIMENTS.md` Batch 6 for the five divergences
this fixes and why each one mattered.

### 0. Check the arithmetic (no GPU, seconds)

```bash
python steering/check_steering_math.py
```

Asserts on a toy model that `alpha=0` is a no-op, that only the last position is edited,
that `‖delta‖ == alpha*‖h‖`, that the edit is antiparallel to `g`, that the hook fires under
KV caching, and that subtracting the direction moves activations toward `f_ft`. Run it after
touching the hook — it catches the Batch 4 bugs in seconds instead of GPU-hours.

### 1. Auxiliary forget set and auxiliary model

The vector must not come from the model being attacked, or it can smuggle in the very
answers the attack claims to recover. Lift two retain authors out of `retain99_en` and
unlearn *those* from the shared parent:

```bash
python steering/make_aux_forget.py          # seed 42 -> author blocks 17 and 153
python forget.py --config-name forget_aux_en
```

Gives `dataset/forget01_aux_en` (40 rows), `dataset/retain99_aux_en` (3920 rows) and
`.../grad_diff_2e-05_forget01aux_5_en`. The seed is load-bearing: the vector is reproducible
only if the auxiliary authors are.

### 2. One vector for every arm

All seven checkpoints share one finetuned parent, and the vector never touches the target,
so a single English vector serves every arm in every language — which also makes applying it
to fr/ar/ja/iw a direct test of the paper's language-agnosticism claim.

```bash
BASE=outputs/tofu_finetuned_5epoch_aya_10_lang_2e5
python steering/extract_steering_vector.py \
  --base-model "$BASE" \
  --unlearned-model $BASE/grad_diff_2e-05_forget01aux_5_en \
  --aux-dataset ./dataset/forget01_aux_en --language en \
  --out steering/vectors/aya_aux_en.pt
```

The printed per-layer *displacement* (`‖mean per-sample difference‖`, max 2.0) says where the
unlearning shift concentrates. A near-zero row means the samples disagree and that layer
carries no consistent direction.

### 3. Attack

```bash
CKPT=$BASE/grad_diff_2e-05_forget01_5_en

# Algorithm 2: 3-layer window, start layer swept over all 30 positions
python steering/attack_generate.py \
  --model "$CKPT" --vector steering/vectors/aya_aux_en.pt \
  --forget-dataset ./dataset/forget01_en --language en \
  --mode sweep --alphas 0.5 --window 2 \
  --out steering/results_v2/aya_aux_en_en_sweep.json

# Appendix K.3 variant: all layers at once, which needs a much smaller alpha
python steering/attack_generate.py \
  --model "$CKPT" --vector steering/vectors/aya_aux_en.pt \
  --forget-dataset ./dataset/forget01_en --language en \
  --mode all-layers --alphas 0.05 0.1 0.2 \
  --out steering/results_v2/aya_aux_en_en_alllayers.json
```

Add `--random-vector` for the control: a unit-norm Gaussian at the same alpha with the same
`‖h‖` rescaling. Alpha is 0.5 for English and 0.8 elsewhere, per the paper.

`--vector` may be omitted when every alpha is 0, which is how the `f_ft` reference passes are
produced — those give the "before unlearning" ceiling that turns a raw recovery delta into a
fraction of what unlearning removed.

### 4. Score and summarise

```bash
python steering/nli_score.py steering/results_v2/*.json

python steering/summarize_results.py comparison \
  --results-dir steering/results_v2 --csv results/steering_comparison_v2.csv
python steering/summarize_results.py layer-profile --results-dir steering/results_v2
```

`nli_score.py` implements Appendix E.1 Eq. 4 and is a separate pass, so generations can be
re-scored without regenerating them. It flags Hebrew and Japanese as zero-shot: the XNLI
checkpoint was not finetuned on either, so keep chrF beside those numbers.

`layer-profile` is the diagnostic the fixed-`late` design could not produce. A peak means
the suppression signal is localised; a flat profile at zero across all 30 start layers is a
much stronger negative than one failed window.

Or run the lot with `./run_steering_replication.sh` (phased, skip-if-done, waits for GPU).
