# Unlearning experiments

Log of the cross-lingual unlearning experiments run on top of the 10-language TOFU
finetune, what each one measures, and where its outputs live.

## Common setup

Unless an experiment says otherwise, every run shares these settings.

- **Base model**: `outputs/tofu_finetuned_5epoch_aya_10_lang_2e5`, i.e. aya-expanse-8b
  finetuned for 5 epochs at lr 2e-5 on `dataset/full_merged_all_10_lang`. The fictitious
  TOFU authors exist in this model in ten languages only: en, fr, ar, ja, ru, fa, ko, hi,
  iw, id. No other language can be unlearned meaningfully, because the facts were never
  taught in it.
- **Unlearning method**: `grad_diff` (negated loss on the forget set plus ordinary loss on
  the retain set), lr 2e-5, weight decay 0.01, seed 42.
- **Effective batch 6**, as `batch_size: 2` with `gradient_accumulation_steps: 3`. A plain
  batch of 6 runs out of memory on the 46 GB L40 (see the OOM at the top of
  `logs/forget_en_iw.txt`).
- **Forget set**: TOFU `forget01`, 40 question/answer pairs covering exactly two authors,
  contiguous and 20 rows each: Basil Mahfouz Al-Kuwaiti in rows 0-19, Nikolai Abilov in
  rows 20-39. **Retain set**: `retain99`, 3960 pairs per language.
- **Evaluation**: `evaluate_util.py` over the four standard tasks (retain, real authors,
  world facts, forget), full datasets (`ds_size: null`), batch 16. English reads the
  perturbed splits from the hub via `config/eval_everything.yaml`; every other language
  reads `dataset/<split>_perturbed_<lang>` via `config/eval_multilingual.yaml`.
- **Aggregation**: `aggregate_eval_stat.py` turns the raw eval logs into a one-row CSV of
  per-task probabilities, truth ratios and Model Utility.

### How the languages are combined

`config/forget_joint.yaml` takes a list of languages and a `language_mix`:

- `concat` - every language contributes the whole forget split. Two languages give 80
  forget examples, so each of the 40 items is unlearned twice, once per language.
- `interleave` - each item is unlearned in one language only, alternating by row. English
  takes rows 0, 2, 4, ..., the partner language takes rows 1, 3, 5, ... Two languages give
  40 forget examples, 20 per language, and the union still covers the split exactly once.

The retain set is always the full union of both languages (7920 rows for a pair),
independent of the mix, so retention pressure is identical across settings.

Step counts follow from the forget-set size at effective batch 6: a `concat` pair is 13
steps per epoch (66 at 5 epochs, 39 at 3, 26 at 2), while `interleave` and monolingual runs
are 6 steps per epoch (33 at 5 epochs).

## Naming and outputs

Every experiment has an id of the form `<language tag>_<N>ep`.

- model: `outputs/tofu_finetuned_5epoch_aya_10_lang_2e5/grad_diff_2e-05_forget01_<N>_<tag>`
- logs: `logs/forget_<exp_id>.txt` and `logs/eval_<exp_id>_on_<lang>.txt`
- results: `results/<exp_id>/eval_<lang>.csv`

The language tag is the pair joined with `+`, with `_half` appended for the interleaved
runs, e.g. `en+fr` and `en+fr_half`. Because both the tag and the epoch count are part of
the model directory, no experiment can overwrite another.

## Batch 1: English/Hebrew transfer (completed 2026-08-18)

Driven by `run_en_iw_experiments.sh`, results in `results/en_iw_transfer/`, timeline in
`logs/en_iw_experiments.txt`. Three models: English only, Hebrew only, and joint en+iw,
all at 5 epochs, each evaluated in English and Hebrew.

Headline numbers, as `Prob. Forget` (lower means more forgotten) and `Model Utility`:

| model | eval en | eval iw |
| --- | --- | --- |
| unlearned on en | 0.0100 forget, 0.505 MU | 0.5865 forget, 0.400 MU |
| unlearned on iw | 0.4834 forget, 0.524 MU | 0.0129 forget, 0.401 MU |
| unlearned on en+iw | 1.3e-05 forget, 0.397 MU | 2.5e-04 forget, 0.311 MU |

Monolingual unlearning is almost entirely language-local: each model drops its own
language to ~0.01 while the other language stays at 0.48-0.59. Joint unlearning erases
both but over-forgets, with retain probability collapsing from ~0.5-0.6 to 0.13-0.18.

The English-only model's English evaluation predates this batch and lives in
`results/aya_graddiff_10lang/finetune_result01_unlearn_grad_diff_2e-05_en.csv`.

## Batch 2: epoch sweep, more language pairs, and split forget sets

Driven by `run_transfer_experiments.sh`. Ten runs, each evaluated in both of its languages.

### Experiments 1 and 2: how many epochs does joint unlearning need?

- `en+iw_2ep` - joint en+iw, `concat`, 2 epochs, 26 steps
- `en+iw_3ep` - joint en+iw, `concat`, 3 epochs, 39 steps

Identical to the completed joint run except for the epoch count, so the three points
(2, 3, 5 epochs) trace how forgetting and collateral damage develop over training. The
5-epoch point is the batch 1 joint model, whose results are in `results/en_iw_transfer/`.

### Experiments 3, 4 and 5: does the pattern hold for other language pairs?

- `en+fr_5ep`, `en+fr_3ep` - joint English/French
- `en+fa_5ep`, `en+fa_3ep` - joint English/Farsi
- `en+ja_5ep`, `en+ja_3ep` - joint English/Japanese

Same recipe as the en+iw joint run, at 5 and 3 epochs. French, Farsi and Japanese stand in
for the originally requested German, Turkish and Chinese, which have no TOFU translations
in `dataset/` and were never part of the base finetune. The three substitutes span
different distances from English: French is closely related, Farsi shares Hebrew's
right-to-left script family, and Japanese is typologically distant with a different
writing system.

### Experiments 6 and 7: what if each fact is unlearned in only one language?

- `en+iw_half_5ep` - `interleave`, en/iw, 5 epochs, 33 steps
- `en+fr_half_5ep` - `interleave`, en/fr, 5 epochs, 33 steps

Half the forget items are unlearned in English and the other half in the partner language,
alternating by row, so no item is ever seen in both. Evaluation still runs over the full
40-item forget set in each language, which means every evaluation mixes 20 items that were
unlearned in the language being tested with 20 that were unlearned only in the other one.
Comparing those two halves isolates cross-lingual transfer within a single model, and the
run uses the same 40 forget examples and 33 steps as a monolingual baseline, unlike the
80-example `concat` runs.

The per-item split can be recovered after the fact: `eval_log_forget.json` is keyed by data
index, so even indices are the English-unlearned items and odd indices the partner-language
ones.

## Running

```bash
./run_transfer_experiments.sh --list          # show the experiment table
DRY_RUN=1 ./run_transfer_experiments.sh       # print every command, run nothing
./run_transfer_experiments.sh                 # run everything, ~5-6 hours
./run_transfer_experiments.sh en+fr_5ep       # run one experiment
FORCE=1 ./run_transfer_experiments.sh en+fr_5ep   # redo it from scratch
```

Every step waits for 40 GB of free GPU before starting, so the script can be launched while
another job is running. Completed steps are skipped, so the script is safe to re-run after
an interruption. Under tmux:

```bash
tmux new -s exp2 -d './run_transfer_experiments.sh 2>&1 | tee logs/transfer_experiments.txt'
tmux attach -t exp2
```

Note that a session created this way ends as soon as the script finishes; the `tee` file is
what survives.

## Caveats

- **No Forget Quality.** `aggregate_eval_stat.py` only computes the KS-test forget quality
  when `retain_result` points at an evaluated retain model, and that entry is commented out
  in `config/aggregate_eval_stat.yaml`. Every CSV therefore reports Model Utility and the
  per-task probabilities and truth ratios only.
- **No per-epoch checkpoints in batch 2.** These runs set `save_epoch_checkpoints: false`,
  so only the final model is written. It saves about 90 GB and 15 minutes per run, but a
  crash mid-run loses that run. Set the flag to `true` to restore the old behaviour.
- **Missing monolingual baselines.** French, Farsi and Japanese have no single-language
  unlearned model, so the new pairs cannot yet be read the way en+iw is read against its en
  and iw baselines. Running them would be
  `python forget.py --config-name forget_fr forget_loss=grad_diff batch_size=2 gradient_accumulation_steps=3`
  plus the two evaluations.
- **Cross-pair evaluation is not included.** Each model is evaluated only in the two
  languages it was trained on. Evaluating, say, the en+ja model on Hebrew would measure
  transfer to an untouched language; add the language to the last field of the experiment
  table to do so.
