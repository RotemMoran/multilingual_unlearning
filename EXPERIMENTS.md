# Unlearning experiments

Log of the cross-lingual unlearning experiments run on top of the 10-language TOFU
finetune: what each one measures, where its outputs live, and what came out.

Batches 1 to 5 are finished and their numbers are below. Batch 6 is running.

## Common setup

Unless an experiment says otherwise, every run shares these settings.

- **Base model**: `outputs/tofu_finetuned_5epoch_aya_10_lang_2e5`, i.e. aya-expanse-8b
finetuned for 5 epochs at lr 2e-5 on `dataset/full_merged_all_10_lang`. The fictitious
TOFU authors exist in this model in ten languages only: en, fr, ar, fa, hi, id, iw, ja,
ko, ru. No other language can be unlearned or evaluated meaningfully, because the facts
were never taught in it.
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

- `concat` - every language contributes the whole forget split, so each of the 40 items is
unlearned once per language. Two languages give 80 forget examples.
- `interleave` - each item is unlearned in exactly one language, assigned by row: item `i`
goes to language `i % N`. The forget set stays 40 examples however many languages are
used, split as evenly as 40 allows (20/20 for two languages, 14/13/13 for three, 10 each
for four), and the union still covers the split exactly once.

The retain set is always the full union of all the languages (3960 rows each, so 7920 for a
pair and 15840 for four), independent of the mix, so retention pressure per language is
identical across settings.

Step counts follow from the forget-set size at effective batch 6: a `concat` pair is 13
steps per epoch (66 at 5 epochs, 39 at 3, 26 at 2), while every `interleave` run is 7 steps
per epoch, 35 at 5 epochs, the same budget as a monolingual run.

## Naming and outputs

Every experiment has an id of the form `<language tag>_<N>ep`.

- model: `outputs/tofu_finetuned_5epoch_aya_10_lang_2e5/grad_diff_2e-05_forget01_<N>_<tag>`
- logs: `logs/forget_<exp_id>.txt` and `logs/eval_<exp_id>_on_<lang>.txt`
- results: `results/<exp_id>/eval_<lang>.csv`

The language tag joins the languages with `+` and adds how the forget set was divided:
nothing for `concat`, `_half`, `_third` or `_quarter` for `interleave` over two, three or
four languages, e.g. `en+fr`, `en+fr_half`, `en+fr+ja_third`, `en+fr+ja+ar_quarter`.
Because both the tag and the epoch count are part of the model directory, no experiment can
overwrite another.

## How to read the numbers

Three columns of the result CSVs carry most of the signal.

- **Prob. Forget** - mean ground-truth probability over the 40 forget items in the language
being evaluated. Lower means more forgotten. There is no evaluated retain model, so no
Forget Quality (see caveats); the practical reference for "not forgotten at all" is the
0.48-0.59 that batch 1 measured in a language the model was never unlearned in. The
`base` row of batch 3 is the proper per-language before-unlearning number, 0.90-0.98.
- **Prob. Retain** - the same quantity on the retain authors, the collateral damage
measure. The base finetune sits near 0.5-0.9 depending on language, and joint runs that
over-forget drive it to 0.13-0.18.
- **Model Utility** - harmonic mean of the six non-forget probabilities and truth ratios.
It moves much less than Prob. Retain, so read it as a coarse summary only.

Two helpers regenerate the tables in this file from whatever results exist:

```bash
python collect_results.py                                 # matrices for every experiment
python collect_results.py en+fr+ja_third_5ep              # one experiment
python forget_split_probs.py en+fr_half_5ep --langs en,fr # per-item transfer breakdown
```

`forget_split_probs.py` exists because Prob. Forget is a plain mean over the 40 items,
while an `interleave` run unlearned each item in only one language. Since item `i` was
assigned to language `i % N`, the mean can be split after the fact using the data indices in
`eval_log_forget.json`: the diagonal is direct unlearning, the off-diagonal cells are
direct measurements of cross-lingual transfer.

## Batch 1: English/Hebrew transfer (completed 2026-08-18)

Driven by `run_en_iw_experiments.sh`, results in `results/en_iw_transfer/`, timeline in
`logs/en_iw_experiments.txt`. Three models, all at 5 epochs: English only, Hebrew only, and
joint en+iw (`concat`, 80 forget examples), each evaluated in English and Hebrew.


| model           | forget en | forget iw | retain en | retain iw | MU en  | MU iw  |
| --------------- | --------- | --------- | --------- | --------- | ------ | ------ |
| unlearned on en | 0.0100    | 0.5865    | 0.5939    | 0.8595    | 0.5050 | 0.4003 |
| unlearned on iw | 0.4834    | 0.0129    | 0.8777    | 0.5140    | 0.5241 | 0.4006 |
| joint en+iw     | 1.30e-05  | 2.46e-04  | 0.1790    | 0.1351    | 0.3973 | 0.3110 |


Monolingual unlearning is almost entirely language-local: each model drops its own language
to ~0.01 while the other language stays at 0.48-0.59, essentially untouched. Joint
unlearning erases both, but over-forgets: retain probability collapses from ~0.5-0.6 to
0.13-0.18. These two rows are the reference points for everything that follows.

The English-only model's English evaluation predates this batch and lives in
`results/aya_graddiff_10lang/finetune_result01_unlearn_grad_diff_2e-05_en.csv`.

## Batch 2: epoch sweep, more language pairs, split forget sets (completed 2026-08-20)

Driven by `run_transfer_experiments.sh`, timeline in `logs/transfer_experiments.txt`. Ten
runs, each evaluated in both of its own languages. L2 is the non-English language of the
run.


| model                      | mix        | forget en | forget L2 | retain en | retain L2 | MU en  | MU L2  |
| -------------------------- | ---------- | --------- | --------- | --------- | --------- | ------ | ------ |
| `en+iw_2ep`                | concat     | 0.4680    | 0.1472    | 0.9136    | 0.7416    | 0.5283 | 0.3924 |
| `en+iw_3ep`                | concat     | 0.0522    | 0.0390    | 0.7589    | 0.5978    | 0.5090 | 0.3911 |
| joint en+iw, 5ep (batch 1) | concat     | 1.30e-05  | 2.46e-04  | 0.1790    | 0.1351    | 0.3973 | 0.3110 |
| `en+fr_3ep`                | concat     | 0.0015    | 0.0088    | 0.4329    | 0.6213    | 0.4699 | 0.4441 |
| `en+fr_5ep`                | concat     | 3.78e-07  | 1.18e-06  | 0.2784    | 0.1279    | 0.4461 | 0.3211 |
| `en+fa_3ep`                | concat     | 0.1238    | 0.0128    | 0.8423    | 0.3787    | 0.5250 | 0.3845 |
| `en+fa_5ep`                | concat     | 0.0034    | 1.34e-04  | 0.5405    | 0.2633    | 0.4836 | 0.3590 |
| `en+ja_3ep`                | concat     | 0.1567    | 0.0096    | 0.8541    | 0.3002    | 0.5352 | 0.4064 |
| `en+ja_5ep`                | concat     | 0.0041    | 9.99e-04  | 0.6619    | 0.3934    | 0.5177 | 0.4342 |
| `en+iw_half_5ep`           | interleave | 0.1070    | 0.0842    | 0.8296    | 0.6735    | 0.5205 | 0.3841 |
| `en+fr_half_5ep`           | interleave | 0.0105    | 0.0420    | 0.7138    | 0.7146    | 0.5029 | 0.4412 |




### Experiments 1 and 2: how many epochs does joint unlearning need?

`en+iw_2ep` and `en+iw_3ep` are identical to the batch 1 joint run except for the epoch
count, so with it they trace forgetting and collateral damage over training. Both move
together and steeply: 2 epochs barely touches English (0.468) while Hebrew is already at
0.147, 3 epochs reaches ~0.04-0.05 in both while retain probability is still 0.60-0.76, and
5 epochs drives forgetting to 1e-04 but takes retain down with it to 0.13-0.18. Three
epochs is the best of the three trade-offs; nothing here separates forgetting from damage.

### Experiments 3, 4 and 5: does the pattern hold for other language pairs?

`en+fr`, `en+fa` and `en+ja`, at 5 and 3 epochs. French, Farsi and Japanese stand in for
the originally requested German, Turkish and Chinese, which have no TOFU translations in
`dataset/` and were never part of the base finetune. The substitutes span different
distances from English: French is closely related, Farsi is a right-to-left script, and
Japanese is typologically distant with a different writing system.

The epoch trade-off reproduces everywhere, but the pairs are not equally aggressive.
English/French is by far the most destructive - at 5 epochs it reaches 1e-06 forgetting
with retain at 0.13-0.28, deeper than en+iw at the same budget - which fits the idea that a
closely related partner language reinforces the same update. The distant pairs are milder
and markedly asymmetric: at 3 epochs Farsi and Japanese are already at ~0.01 while English
lags at 0.12-0.16, and their retain probability (0.30-0.38) is far below the English side
(0.84-0.85). Forgetting and damage in the weaker language both come cheaper.

### Experiments 6 and 7: what if each fact is unlearned in only one language?

`en+iw_half_5ep` and `en+fr_half_5ep` use `interleave`: half the forget items are unlearned
in English and the other half in the partner language, no item in both, 40 examples and 35
steps in total, the same budget as a monolingual run. This is the most interesting result of
the batch. Both languages end up substantially forgotten (0.01-0.11) while retain
probability stays high (0.67-0.83), a far better trade-off than any `concat` run: en+fr at 5
epochs reaches comparable forgetting only at retain 0.13-0.28.

Splitting the aggregate per item shows this is genuine transfer, not an averaging artifact:

`en+fr_half_5ep`


| eval language | unlearned in en | unlearned in fr |
| ------------- | --------------- | --------------- |
| en            | **0.0084**      | 0.0125          |
| fr            | 0.0736          | **0.0105**      |


`en+iw_half_5ep`


| eval language | unlearned in en | unlearned in iw |
| ------------- | --------------- | --------------- |
| en            | **0.0279**      | 0.1861          |
| iw            | 0.1471          | **0.0213**      |


Bold is the diagonal, i.e. items unlearned in the language being evaluated. The items that
were never unlearned in the evaluation language still drop to 0.0125-0.19, against 0.48-0.59
for the monolingual models of batch 1 that had no in-language unlearning at all. Transfer is
near-total for French (0.0125 vs 0.0084 on the diagonal) and partial but large for Hebrew
(0.19 vs 0.028). The apparent difference from batch 1 is that here the model is
simultaneously unlearning *other* items in the evaluation language, which seems to carry the
rest of the forget set with it. That hypothesis is what batch 3 pushes on.

## Batch 3: how thin can the forget set be spread? (completed 2026-08-21)

Same driver, timeline in `logs/transfer_experiments_batch3.txt`. Every model is evaluated in
all ten languages of the base finetune, not just the ones it was trained on, so each run
reports both the languages it unlearned in and the languages it never saw.


| exp_id                    | unlearn languages | forget items per language | eval   |
| ------------------------- | ----------------- | ------------------------- | ------ |
| `en+fr_half_5ep`          | en, fr            | 20 / 20                   | all 10 |
| `en+fr+ja_third_5ep`      | en, fr, ja        | 14 / 13 / 13              | all 10 |
| `en+fr+ar_third_5ep`      | en, fr, ar        | 14 / 13 / 13              | all 10 |
| `en+fr+ja+ar_quarter_5ep` | en, fr, ja, ar    | 10 each                   | all 10 |


All four use `interleave` at 5 epochs, so the forget budget is fixed at 40 examples and 35
steps no matter how many languages share it; only the number of languages changes. The
retain set grows with the language count (11880 rows for three, 15840 for four), which keeps
retention pressure per language constant.

Two questions:

1. **Does spreading the same 40 items over more languages still erase the facts?** Batch 2
  showed that a 20/20 split leaves both languages forgotten. If the same holds at 13 and 10
   items per language, unlearning cost per language falls with every language added.
2. **How far does it reach into languages that were never touched?** The seven (or six)
  untouched languages are pure transfer: the model unlearned nothing in them, and batch 1
   says a monolingual run leaves such a language at 0.48-0.59. Whether the multi-language
   runs move them is the real test of the hypothesis above.

Experiment 1 reuses the batch 2 `en+fr_half` model, so it adds eight evaluations and no
training; its en and fr numbers are already in the batch 2 table.

A `base` row was added to the experiment table alongside them: mix `none` means there is
nothing to unlearn, so the model is the base finetune itself and only the ten evaluations
run. It finished after the four unlearning runs (`results/base/eval_<lang>.csv`) and is the
per-language "before unlearning" reference these tables were missing: Prob. Forget is
0.90-0.98 in every language.

### Results

Regenerate with `python collect_results.py en+fr_half_5ep en+fr+ja_third_5ep en+fr+ar_third_5ep en+fr+ja+ar_quarter_5ep` for the per-language matrices and
`forget_split_probs.py` for the per-item breakdown of each run.

#### Experiment 1: `en+fr_half_5ep` in all ten languages (completed)

Prob. Forget, unlearning languages in bold:


| model            | **en**     | **fr**     | ar     | fa     | hi     | id     | iw     | ja     | ko     | ru     |
| ---------------- | ---------- | ---------- | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ |
| `en+fr_half_5ep` | **0.0105** | **0.0420** | 0.4379 | 0.4345 | 0.6778 | 0.1017 | 0.3715 | 0.4328 | 0.2215 | 0.2856 |


Prob. Retain and Model Utility:


| metric        | en     | fr     | ar     | fa     | hi     | id     | iw     | ja     | ko     | ru     |
| ------------- | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ |
| Prob. Retain  | 0.7138 | 0.7146 | 0.7803 | 0.7819 | 0.8064 | 0.7660 | 0.7834 | 0.7568 | 0.7725 | 0.7495 |
| Model Utility | 0.5029 | 0.4412 | 0.4229 | 0.4278 | 0.4037 | 0.4645 | 0.3983 | 0.4811 | 0.4486 | 0.4205 |


Unlearning in English and French leaves the facts intact in most other languages. The two
trained languages drop to 0.01-0.04, while the eight untouched ones sit at 0.22-0.68, an
order of magnitude higher, and their retain probability (0.75-0.81) is if anything above the
trained pair's 0.71, so nothing was damaged in passing. The spread among the untouched
languages is wide and does not track script or family in any obvious way - Indonesian
(0.1017) is much lower than Hindi (0.6778), with Korean and Russian in between - and it is
not directly comparable to the batch 1 reference of 0.48-0.59 because no evaluation of the
base finetune exists per language.

Splitting the forget set per item shows the same wall:


| eval language | unlearned in en | unlearned in fr |
| ------------- | --------------- | --------------- |
| en            | **0.0084**      | 0.0125          |
| fr            | 0.0736          | **0.0105**      |
| ar            | 0.5136          | 0.3622          |
| fa            | 0.5049          | 0.3641          |
| hi            | 0.6972          | 0.6584          |
| id            | 0.1151          | 0.0884          |
| iw            | 0.4097          | 0.3333          |
| ja            | 0.4799          | 0.3857          |
| ko            | 0.2728          | 0.1702          |
| ru            | 0.3253          | 0.2458          |


Within en and fr, both halves are forgotten whichever language did the unlearning, i.e.
transfer between the two trained languages is near-total. In the other eight languages both
halves stay high and close to each other, which is the point: what transferred was not the
individual items but the behaviour in the languages that were trained. Batch 2's conclusion
was that unlearning other items in a language carries the rest of the forget set with it in
*that* language; experiment 1 shows the effect does not extend to languages that received no
unlearning at all.

#### Experiment 2: `en+fr+ja_third_5ep`, 13-14 items per language (completed)

Prob. Forget, unlearning languages in bold:


| model                | **en**     | **fr**     | ar     | fa     | hi     | id     | iw     | **ja**     | ko     | ru     |
| -------------------- | ---------- | ---------- | ------ | ------ | ------ | ------ | ------ | ---------- | ------ | ------ |
| `en+fr+ja_third_5ep` | **0.0412** | **0.0859** | 0.4694 | 0.4634 | 0.5735 | 0.2362 | 0.4894 | **0.1283** | 0.1861 | 0.3990 |


Prob. Retain and Model Utility:


| metric        | en     | fr     | ar     | fa     | hi     | id     | iw     | ja     | ko     | ru     |
| ------------- | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ |
| Prob. Retain  | 0.8360 | 0.7786 | 0.8072 | 0.7977 | 0.7792 | 0.8233 | 0.8163 | 0.6260 | 0.7310 | 0.8095 |
| Model Utility | 0.5156 | 0.4375 | 0.4162 | 0.4231 | 0.4097 | 0.4660 | 0.3944 | 0.4637 | 0.4492 | 0.4177 |


Spreading the same 40 items over three languages still forgets in all three, but less
sharply than over two: 0.04 / 0.09 / 0.13 here against 0.01 / 0.04 for the en+fr pair, with
Japanese, the language furthest from the other two, weakest. The trade is cheaper collateral
damage - retain probability rises to 0.78-0.84 (Japanese 0.63) from 0.71 for the pair. The
untouched languages do not budge in any consistent direction when a third language is added:
Hindi and Korean drop a little (0.68 to 0.57, 0.22 to 0.19), Indonesian, Hebrew and Russian
rise (0.10 to 0.24, 0.37 to 0.49, 0.29 to 0.40).


| eval language | unlearned in en | unlearned in fr | unlearned in ja |
| ------------- | --------------- | --------------- | --------------- |
| en            | **0.0101**      | 0.0260          | 0.0897          |
| fr            | 0.1407          | **0.0070**      | 0.1057          |
| ar            | 0.5643          | 0.2585          | 0.5780          |
| fa            | 0.5391          | 0.3395          | 0.5058          |
| hi            | 0.6116          | 0.5739          | 0.5321          |
| id            | 0.3035          | 0.1331          | 0.2669          |
| iw            | 0.5934          | 0.3654          | 0.5014          |
| ja            | 0.2138          | 0.1327          | **0.0318**      |
| ko            | 0.2739          | 0.1747          | 0.1028          |
| ru            | 0.4862          | 0.2108          | 0.4933          |


Transfer among the three trained languages is real but directional. English absorbs it
almost completely: every group, including the items only ever unlearned in Japanese, reads
below 0.09 there. Transfer into French and Japanese is partial, 0.11-0.21 for the groups
unlearned elsewhere against 0.007 and 0.03 on their diagonals. The untouched languages again
show no per-group structure, only the same flat high values.

#### Experiment 3: `en+fr+ar_third_5ep`, 13-14 items per language (completed)

Prob. Forget, unlearning languages in bold:


| model                | **en**     | **fr**     | **ar**     | fa     | hi     | id     | iw     | ja     | ko     | ru     |
| -------------------- | ---------- | ---------- | ---------- | ------ | ------ | ------ | ------ | ------ | ------ | ------ |
| `en+fr+ar_third_5ep` | **0.0347** | **0.0591** | **0.1123** | 0.3411 | 0.5849 | 0.1447 | 0.3123 | 0.4038 | 0.2702 | 0.3135 |


Prob. Retain and Model Utility:


| metric        | en     | fr     | ar     | fa     | hi     | id     | iw     | ja     | ko     | ru     |
| ------------- | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ |
| Prob. Retain  | 0.7970 | 0.7058 | 0.5420 | 0.7015 | 0.7703 | 0.7565 | 0.7497 | 0.7175 | 0.7546 | 0.7567 |
| Model Utility | 0.5122 | 0.4459 | 0.4103 | 0.4215 | 0.4064 | 0.4645 | 0.3990 | 0.4773 | 0.4484 | 0.4183 |


Swapping Japanese for Arabic barely changes the trained languages (0.03 / 0.06 / 0.11), so
the three-way result is not specific to one partner. Arabic pays more for it than Japanese
did in retain probability, 0.54 against 0.63, the lowest retain value in batch 3.


| eval language | unlearned in en | unlearned in fr | unlearned in ar |
| ------------- | --------------- | --------------- | --------------- |
| en            | **0.0096**      | 0.0168          | 0.0795          |
| fr            | 0.1081          | **0.0062**      | 0.0593          |
| ar            | 0.2183          | 0.0735          | **0.0371**      |
| fa            | 0.4214          | 0.2389          | 0.3568          |
| hi            | 0.6012          | 0.5608          | 0.5916          |
| id            | 0.2192          | 0.0709          | 0.1382          |
| iw            | 0.4497          | 0.2288          | 0.2479          |
| ja            | 0.4610          | 0.2840          | 0.4619          |
| ko            | 0.3156          | 0.1805          | 0.3111          |
| ru            | 0.4398          | 0.1428          | 0.3482          |


The same directional pattern holds, with English the best receiver (0.008-0.08 across all
three groups) and Arabic the worst (0.22 for English-unlearned items). One suggestive
difference from experiment 2: with Arabic in the mix the two languages closest to it,
Farsi (0.3411 against 0.4634 with Japanese) and Hebrew (0.3123 against 0.4894), come out
lower, while Japanese itself rises to 0.4038. That is the only hint in batch 3 of leakage
following script or family, and it rests on single runs, so it is a lead rather than a
finding.

#### Experiment 4: `en+fr+ja+ar_quarter_5ep`, 10 items per language (completed)

Prob. Forget, unlearning languages in bold:


| model                     | **en**     | **fr**     | **ar**     | fa     | hi     | id     | iw     | **ja**     | ko     | ru     |
| ------------------------- | ---------- | ---------- | ---------- | ------ | ------ | ------ | ------ | ---------- | ------ | ------ |
| `en+fr+ja+ar_quarter_5ep` | **0.0982** | **0.2196** | **0.2466** | 0.4049 | 0.5720 | 0.2837 | 0.4388 | **0.1587** | 0.2799 | 0.3478 |


Prob. Retain and Model Utility:


| metric        | en     | fr     | ar     | fa     | hi     | id     | iw     | ja     | ko     | ru     |
| ------------- | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ |
| Prob. Retain  | 0.8336 | 0.7881 | 0.6967 | 0.7357 | 0.7448 | 0.7972 | 0.7779 | 0.5960 | 0.6957 | 0.7654 |
| Model Utility | 0.5232 | 0.4527 | 0.4235 | 0.4222 | 0.4099 | 0.4607 | 0.4008 | 0.4697 | 0.4416 | 0.4219 |


Spreading the same 40 items over four languages still forgets in all four relative to the
base finetune (0.90-0.98), but the effect is thinner than at two or three: 0.10 / 0.22 /
0.16 / 0.25 against 0.01 / 0.04 for the en+fr pair and 0.03-0.13 for the triples. English
is still the strongest (0.0982). French is the language that suffers most from thinning,
rising from 0.0420 in the pair to 0.2196 here. Japanese, despite being the furthest of the
four, holds up better than French or Arabic (0.1587, close to its 0.1283 as a third).
Arabic at 0.2466 is more than twice its 0.1123 as a third.

Retain stays high, 0.70-0.83 except Japanese at 0.60, the same pattern as experiment 2.
Spreading thinner is cheaper for Arabic retain (0.70 against 0.54 when it was a third).
The six untouched languages sit at 0.28-0.57, still well above the trained four and close
to their values in the earlier runs, so adding a fourth language does not punch through to
the rest of the set. Against the base finetune those six have dropped from 0.90-0.94, so
some leakage exists, but they remain several times higher than the trained pair at 20/20
and do not move in a consistent direction as languages are added. Farsi (0.4049) and
Hebrew (0.4388) land between their values in the two triples, so the script/family hint
from experiment 3 does not get stronger when Arabic and Japanese are both in the mix.


| eval language | unlearned in en | unlearned in fr | unlearned in ja | unlearned in ar |
| ------------- | --------------- | --------------- | --------------- | --------------- |
| en            | **0.0211**      | 0.0513          | 0.1378          | 0.1825          |
| fr            | 0.2649          | **0.0187**      | 0.3520          | 0.2426          |
| ar            | 0.3082          | 0.2845          | 0.3474          | **0.0463**      |
| fa            | 0.4697          | 0.2667          | 0.5306          | 0.3524          |
| hi            | 0.5725          | 0.5813          | 0.5276          | 0.6068          |
| id            | 0.3619          | 0.1235          | 0.3925          | 0.2569          |
| iw            | 0.5381          | 0.4313          | 0.4492          | 0.3367          |
| ja            | 0.2540          | 0.1510          | **0.0551**      | 0.1747          |
| ko            | 0.4026          | 0.2369          | 0.1877          | 0.2924          |
| ru            | 0.3520          | 0.2492          | 0.3471          | 0.4427          |


The per-item split shows that thinning hurts transfer, not in-language unlearning. The
ten items actually unlearned in each language still drop to 0.02-0.06 on that language's
diagonal, as low as the triples. The higher aggregates come from the other 30 items:
French off-diagonal is 0.24-0.35 against 0.019 on the diagonal, Arabic 0.28-0.35 against
0.046, Japanese 0.15-0.25 against 0.055. English is again the best receiver (0.021-0.183)
but no longer absorbs the Japanese- and Arabic-unlearned groups the way it did at three
languages. The untouched languages show no per-group structure, only the same flat high
values as before.

#### Batch 3 summary

Prob. Forget on the four languages that were ever trained, against the base finetune.
Bold is a language the model unlearned in.


| model                     | **en**     | **fr**     | **ja**     | **ar**     |
| ------------------------- | ---------- | ---------- | ---------- | ---------- |
| `base`                    | 0.9797     | 0.9671     | 0.8951     | 0.9075     |
| `en+fr_half_5ep`          | **0.0105** | **0.0420** | 0.4328     | 0.4379     |
| `en+fr+ja_third_5ep`      | **0.0412** | **0.0859** | **0.1283** | 0.4694     |
| `en+fr+ar_third_5ep`      | **0.0347** | **0.0591** | 0.4038     | **0.1123** |
| `en+fr+ja+ar_quarter_5ep` | **0.0982** | **0.2196** | **0.1587** | **0.2466** |


On the two questions this batch was for:

1. **Spreading the same 40 items still erases the facts, but transfer thins with every
  language added.** Ten in-language items are enough to wipe those ten items (diagonals
   0.02-0.06). What fails at four languages is carrying the other 30 with them, especially
   into French and Arabic. Unlearning cost per language does fall - retain stays at
   0.60-0.83 throughout, far above the `concat` collapses of batches 1 and 2 - but
   forgetting quality falls with it, so the extra languages are not free.
2. **Languages that were never touched stay in a different regime.** They drop from the
  base 0.90-0.94 to 0.10-0.68 depending on the run, so there is leakage, but they remain
   well above the co-trained languages and do not collapse as more partners are added.
  Transfer among languages that share the forget set is a different phenomenon from
  transfer into a language that received no unlearning at all.



## Batch 4: steering-vector recovery (completed 2026-08-21)

Driven by `run_steering_recovery.sh`, timeline in `logs/steering_recovery.txt`,
tables from `steering/summarize_results.py comparison` and `interleave-split`.
Raw attacks live in `steering/results/`; the joined CSV is
`results/steering_comparison.csv`.

The question is whether unlearning only **suppresses** the forget answers in the
residual stream, so that subtracting

```text
v[i] = mean_unlearned(h_i) - mean_base(h_i)
h_i  = h_i - alpha * v[i]
```

at the late decoder layers (`24–31`) restores generation. Each vector is
extracted on `retain99_<lang>` and applied while generating `forget01_<lang>`.
Alphas are `0 0.5 1 2 4`. English is scored with ROUGE-L recall; every other
language with chrF. A norm-matched random vector is the control: if it recovers
about as much as the real vector, there is no directional result.

Seven checkpoints, 16 `(model, language)` pairs, all trained-language sides
attacked. Every interleaved arm has the same optimizer budget as a monolingual
run (40 forget rows, 5 epochs), so a recovery gap is not an extra-step confound.

Headline numbers, one row per attack. **recovery** is best-over-alpha minus
`alpha=0`. **random Δ** is the same quantity on the random vector.


| model                     | mix         | n/lang | lang | P.Forget | P.Retain | MU     | metric         | baseline | best α | best   | recovery | random Δ |
| ------------------------- | ----------- | ------ | ---- | -------- | -------- | ------ | -------------- | -------- | ------ | ------ | -------- | -------- |
| `en`                      | monolingual | 40     | en   | 0.0100   | 0.5939   | 0.5050 | rougeL_recall  | 0.4233   | 0.5    | 0.4708 | **+0.0475** | +0.0046 |
| `iw`                      | monolingual | 40     | iw   | 0.0129   | 0.5140   | 0.4006 | chrf           | 0.1697   | 0.5    | 0.2490 | **+0.0793** | +0.0071 |
| `en+fr_half_5ep`          | interleave  | 20     | en   | 0.0105   | 0.7138   | 0.5029 | rougeL_recall  | 0.3523   | 0.5    | 0.4169 | **+0.0646** | +0.0042 |
| `en+fr_half_5ep`          | interleave  | 20     | fr   | 0.0420   | 0.7146   | 0.4412 | chrf           | 0.3690   | 0.0    | 0.3690 | +0.0000  | +0.0000 |
| `en+iw_half_5ep`          | interleave  | 20     | en   | 0.1070   | 0.8296   | 0.5205 | rougeL_recall  | 0.4734   | 0.5    | 0.4928 | +0.0195  | +0.0212 |
| `en+iw_half_5ep`          | interleave  | 20     | iw   | 0.0842   | 0.6735   | 0.3841 | chrf           | 0.2818   | 0.5    | 0.3028 | +0.0210  | +0.0000 |
| `en+fr+ja_third_5ep`      | interleave  | 14     | en   | 0.0412   | 0.8360   | 0.5156 | rougeL_recall  | 0.4386   | 0.5    | 0.4902 | +0.0516  | +0.0423 |
| `en+fr+ja_third_5ep`      | interleave  | 13     | fr   | 0.0859   | 0.7786   | 0.4375 | chrf           | 0.3973   | 0.0    | 0.3973 | +0.0000  | +0.0000 |
| `en+fr+ja_third_5ep`      | interleave  | 13     | ja   | 0.1283   | 0.6260   | 0.4637 | chrf           | 0.1652   | 0.5    | 0.1864 | +0.0212  | +0.0086 |
| `en+fr+ar_third_5ep`      | interleave  | 14     | en   | 0.0347   | 0.7970   | 0.5122 | rougeL_recall  | 0.4582   | 0.0    | 0.4582 | +0.0000  | +0.0000 |
| `en+fr+ar_third_5ep`      | interleave  | 13     | fr   | 0.0591   | 0.7058   | 0.4459 | chrf           | 0.3942   | 0.5    | 0.4009 | +0.0067  | +0.0000 |
| `en+fr+ar_third_5ep`      | interleave  | 13     | ar   | 0.1123   | 0.5420   | 0.4103 | chrf           | 0.2818   | 0.0    | 0.2818 | +0.0000  | +0.0000 |
| `en+fr+ja+ar_quarter_5ep` | interleave  | 10     | en   | 0.0982   | 0.8336   | 0.5232 | rougeL_recall  | 0.4978   | 0.5    | 0.5114 | +0.0136  | +0.0135 |
| `en+fr+ja+ar_quarter_5ep` | interleave  | 10     | fr   | 0.2196   | 0.7881   | 0.4527 | chrf           | 0.4346   | 0.0    | 0.4346 | +0.0000  | +0.0075 |
| `en+fr+ja+ar_quarter_5ep` | interleave  | 10     | ja   | 0.1587   | 0.5960   | 0.4697 | chrf           | 0.1674   | 0.5    | 0.1858 | +0.0184  | +0.0000 |
| `en+fr+ja+ar_quarter_5ep` | interleave  | 10     | ar   | 0.2466   | 0.6967   | 0.4235 | chrf           | 0.2870   | 0.5    | 0.3119 | +0.0249  | +0.0117 |


Bold recoveries are the ones that beat the random control by a clear margin.
`alpha=1` and above collapse generation on every arm (scores go to ~0), so the
only working setting on this grid is `alpha=0.5`.

### What recovered

Monolingual unlearning is suppression, not erasure. English-only recovers
+0.0475 ROUGE-L (random +0.0046). Hebrew-only is the strongest result in the
batch: chrF 0.170 → 0.249 (+0.0793, random +0.0071), from the lowest unattacked
generation score of any arm.

`en+fr_half` still recovers in English (+0.0646, random +0.0042), at least as
much as the English-only model, with matched step count and the same Prob.
Forget (0.010). French on that same checkpoint does not move at all. Hebrew on
`en+iw_half` recovers a little (+0.0210, random 0); the English side of that
pair is not a result (real +0.0195, random +0.0212).

Spreading to three or four languages mostly removes a directional attack.
`en+fr+ar_third` is flat in every language. The English bumps on
`en+fr+ja_third` (+0.0516) and the quarter (+0.0136) are matched by the random
vector (+0.0423 and +0.0135). Japanese and Arabic on the quarter show small
real-over-random gaps (+0.018 and +0.013) from a weaker forgetting regime
(P.Forget 0.16–0.25), which is the confound the plan flagged: thinner
unlearning is easier to "recover" because less was gone.

French never recovers on any interleaved model. That is the most consistent
negative in the batch, and it is not a scoring artifact — chrF and ROUGE-L both
stay at or below the `alpha=0` baseline.

Generation baselines are high even when Prob. Forget is ~0.01 (English-only
still scores 0.42 ROUGE-L unattacked). The attack measures overlap of sampled
answers, not token probability, so a model can look only partly forgotten on
this metric while the eval CSVs say it is gone.

### Interleave split

Item `i` was unlearned in language `i % N`. Recovery on the diagonal (items
actually unlearned in the evaluation language) versus the off-diagonal is in
`results/steering_interleave_split.csv`. The two clean recoveries split the
same way as the aggregate:

`en+fr_half` English, `alpha=0.5`:


| eval | unlearned in en | unlearned in fr |
| ---- | --------------- | --------------- |
| en   | **+0.0760**     | +0.0533         |
| fr   | +0.0000         | +0.0000         |


Both English halves recover; neither French half does. Transfer of the
*suppression* (the thing the vector undoes) is English-internal.

`en+iw_half` Hebrew: the diagonal recovers more (+0.0303) than the
English-unlearned items (+0.0117), so the small Hebrew result is mostly the
items that were actually unlearned in Hebrew.

### Batch 4 summary

On the original question:

1. **Monolingual EN and IW leave recoverable residual-stream directions.** The
   random controls stay near zero, so this is the unlearning vector, not
   perturbation size. Hebrew is the cleaner of the two because its unattacked
   generations are actually bad.
2. **Interleaving with French does not block the English attack at 20/20**, and
   does not create a French attack at any mix. Adding a third or fourth language
   removes a *directional* English recovery (the random vector catches up),
   which is not the same as "the knowledge is gone."
3. **Do not read the quarter's small Japanese/Arabic bumps as deeper
   unlearning.** Those languages were the least forgotten (P.Forget 0.16–0.25).
   The arms that forgot as hard as monolingual English (`en`, `en+fr_half` EN)
   are the ones whose recovery, or lack of it, can be compared.

Regenerate with:

```bash
python steering/summarize_results.py comparison \
    --results-dir steering/results --csv results/steering_comparison.csv
python steering/summarize_results.py interleave-split \
    --results-dir steering/results --csv results/steering_interleave_split.csv
```



## Batch 5: English paired with every other language (completed 2026-08-24)

Same driver, `run_transfer_experiments.sh`, timeline in `logs/transfer_experiments_batch5.txt`.
Batch 3 held the pair at en+fr and varied how many languages shared the forget set; this
batch does the opposite. It fixes the mix at half/half - 20 forget items in English, 20 in
the partner, `interleave` at 5 epochs - and varies *which* language English is paired with,
over all nine partners, then evaluates every model in all ten languages. The output is a
nine-row transfer matrix with English held constant.

Nine models, 90 evaluations, no failures. Seven models were trained here; `en+fr_half_5ep`
was already complete from batches 2 and 3, and `en+iw_half_5ep` reused its batch 2 model
(trained with these exact settings) and only needed its eight missing languages, which is
why its eval field in the driver is now `all`.

**The headline: which language English is paired with decides how much English forgets.**
Prob. Forget in English ranges from 0.0105 to 0.2882 across the nine pairs, on an identical
20 English items and an identical 35-step budget. Everything else in this batch follows from
that.

### Why this is comparable to what came before

English is listed first in every pair, so `interleave` sends the even-indexed items to
English and the odd-indexed ones to the partner, exactly as in `en+fr_half` and
`en+iw_half`. Because `forget01` is two contiguous authors (rows 0-19 and 20-39), the
even/odd split gives each language ten items from each author, so no pair is accidentally
unlearning one author more than the other.

The forget set stays at 40 examples and 7 steps per epoch, 35 steps at 5 epochs, the same
optimizer budget as a monolingual run and as every interleaved arm of batches 2, 3 and 4.
The retain set is the union of both languages, 7920 rows, identical to the two pairs already
run. Nothing but the partner language changes, so a difference in the results is a
difference in the partner. `en+ja_half_5ep` and `en+fa_half_5ep` are new models, not the
`concat` `en+ja_5ep` and `en+fa_5ep` of batch 2; the `_half` tag keeps the directories apart.

### Questions

1. **Is the near-total en/fr transfer a property of French or of the half/half setting?**
   The per-item splits so far give two points: items never unlearned in the evaluation
   language read 0.0125 in French but 0.19 in Hebrew. Nine partners turn that into a range,
   and it is the cleanest available measure of how much a partner language does the
   forgetting for you.
2. **Does leakage into the untouched languages track the partner?** Batch 3's only hint of
   script or family structure was that adding Arabic lowered Farsi and Hebrew while
   Japanese rose. This batch tests it directly and repeatedly: `en+ar_half` should pull
   Farsi and Hebrew down further than `en+ja_half` does, `en+ko_half` should pull Japanese
   down, `en+hi_half` and `en+ru_half` have no close relative in the set and should pull
   nothing in particular. Eight untouched columns per model, nine models, one prediction
   per cell.
3. **Which partners are cheap in collateral damage?** Prob. Retain in batch 3 was 0.71-0.84
   for English and French but 0.63 for Japanese and 0.54 for Arabic. Whether that is the
   partner or the three-language setting is currently unidentifiable; here every partner is
   measured in the same two-language setting.
4. **Is English always the strong receiver?** English absorbed transfer almost completely in
   both triples. If it does so with all nine partners, that is a property of English in this
   model rather than of any pair.

### Results

Prob. Forget, with each model's two unlearning languages in bold. Rows are ordered by how
much English forgot, which is the axis the rest of this section is about.


| model            | en         | fr         | ar         | fa         | hi         | id         | iw         | ja         | ko         | ru         |
| ---------------- | ---------- | ---------- | ---------- | ---------- | ---------- | ---------- | ---------- | ---------- | ---------- | ---------- |
| `base`           | 0.9797     | 0.9671     | 0.9075     | 0.9063     | 0.9340     | 0.9436     | 0.9163     | 0.8951     | 0.9155     | 0.9037     |
| `en`             | **0.0100** | -          | -          | -          | -          | -          | 0.5865     | -          | -          | -          |
| `en+fr_half_5ep` | **0.0105** | **0.0420** | 0.4379     | 0.4345     | 0.6778     | 0.1017     | 0.3715     | 0.4328     | 0.2215     | 0.2856     |
| `en+id_half_5ep` | **0.0111** | 0.0729     | 0.3838     | 0.4122     | 0.5336     | **0.0411** | 0.2967     | 0.3619     | 0.2164     | 0.2401     |
| `en+ru_half_5ep` | **0.0269** | 0.1926     | 0.4417     | 0.4137     | 0.6013     | 0.2617     | 0.4290     | 0.4451     | 0.3318     | **0.0689** |
| `en+iw_half_5ep` | **0.1070** | 0.1865     | 0.4717     | 0.4879     | 0.5854     | 0.2473     | **0.0842** | 0.5061     | 0.3878     | 0.3369     |
| `en+ko_half_5ep` | **0.1224** | 0.4282     | 0.6182     | 0.5950     | 0.5856     | 0.5210     | 0.6150     | 0.3105     | **0.1227** | 0.5372     |
| `en+ar_half_5ep` | **0.2086** | 0.3762     | **0.0779** | 0.3992     | 0.6868     | 0.4033     | 0.4806     | 0.5278     | 0.5624     | 0.4758     |
| `en+hi_half_5ep` | **0.2684** | 0.5156     | 0.6399     | 0.5662     | **0.1637** | 0.5760     | 0.6254     | 0.4625     | 0.4547     | 0.5896     |
| `en+ja_half_5ep` | **0.2718** | 0.5409     | 0.6213     | 0.5885     | 0.5250     | 0.6689     | 0.6097     | **0.0277** | 0.3097     | 0.4818     |
| `en+fa_half_5ep` | **0.2882** | 0.5456     | 0.4205     | **0.0995** | 0.6730     | 0.5100     | 0.6095     | 0.5607     | 0.5816     | 0.5515     |


Every pair forgets in both of its own languages relative to the base finetune, but by
wildly different amounts, and the ordering is almost entirely about English. Pairing English
with French, Indonesian or Russian leaves English at 0.011-0.027, as low as any run in the
project. Pairing it with Hindi, Japanese or Farsi leaves English at 0.27-0.29, twenty times
higher, while those partners themselves forget perfectly well (Japanese reaches 0.0277, the
deepest single-language number in any batch). The partner is never the language that
suffers: in all nine pairs the partner's own Prob. Forget (0.028-0.164) is at or below
English's.

#### The monolingual English row, and why it matters here

The `en` row is the batch 1 English-only model (`grad_diff_2e-05_forget01_5_en`), added as
the reference these tables were read against without having it in view. It is budget-matched
to every pair: 40 forget examples and 35 steps at effective batch 6, the difference being
that all 40 items are in English instead of 20 in English and 20 in a partner. It was only
ever evaluated in English and Hebrew, hence the eight dashes.

Two things follow from the two cells that exist.

**Splitting the forget set across two languages dominates keeping it in one.** Monolingual
English reaches Prob. Forget 0.0100 at retain 0.5939; `en+fr_half_5ep` reaches 0.0105 at
retain 0.7138, on the same 40 items and the same 35 steps, and forgets in French as well.
Identical forgetting, 0.12 more retain, one extra language covered. Batch 2 established that
`interleave` beats `concat`; this says it also beats monolingual unlearning at equal budget,
which is the stronger claim and the one that makes half/half the default worth building on.

**English unlearning on its own does not travel.** With all 40 items in English, Hebrew sits
at 0.5865 - the batch 1 result that monolingual unlearning is language-local. In
`en+iw_half_5ep`, the twenty items unlearned in English read 0.1471 in Hebrew. Same
direction of transfer, four times deeper, and the only difference is that Hebrew was
concurrently being unlearned on the *other* twenty items. Read the donor numbers below with
that in mind.

#### English is a reliable donor into a co-unlearned language

The per-item split separates the two directions. Since item `i` was unlearned in language
`i % 2`, "into en" is the twenty items unlearned only in the partner, evaluated in English,
and "into partner" is the twenty English items evaluated in the partner language. Both are
pure transfer. The last column is the mean aggregate over the eight languages the model
never touched.


| pair  | en diagonal | partner diagonal | into en | into partner | untouched mean |
| ----- | ----------- | ---------------- | ------- | ------------ | -------------- |
| en+fr | 0.0084      | 0.0105           | 0.0125  | 0.0736       | 0.3704         |
| en+id | 0.0013      | 0.0050           | 0.0209  | 0.0771       | 0.3147         |
| en+ru | 0.0161      | 0.0157           | 0.0376  | 0.1221       | 0.3896         |
| en+iw | 0.0279      | 0.0213           | 0.1861  | 0.1471       | 0.4012         |
| en+ko | 0.0187      | 0.0247           | 0.2261  | 0.2208       | 0.5263         |
| en+ar | 0.1103      | 0.0124           | 0.3068  | 0.1435       | 0.4890         |
| en+hi | 0.0756      | 0.0533           | 0.4611  | 0.2741       | 0.5537         |
| en+ja | 0.1061      | 0.0030           | 0.4376  | 0.0524       | 0.5432         |
| en+fa | 0.1246      | 0.0171           | 0.4519  | 0.1819       | 0.5565         |


Read the two middle columns against each other. **Into the partner** is low everywhere,
0.05-0.27 against a base of 0.90-0.94: items unlearned in English are forgotten in every one
of the nine partner languages, including Japanese at 0.0524 and Farsi at 0.1819. **Into
English** spans 0.0125 to 0.4611, a 37-fold range on the same budget. French, Indonesian and
Russian hand their unlearning to English almost completely; Arabic, Japanese, Farsi and
Hindi barely hand over anything, even though each of them forgot its own twenty items down
to 0.003-0.05.

So transfer is strongly directional, and the direction that works is English outward. Batch
3's conclusion that English is the best receiver was measured on triples that all contained
French - the single best donor in the set - and does not survive contact with the other
eight partners.

The "into partner" column is not English acting alone, and the monolingual row above is what
pins that down: English carries its items into a partner language that is simultaneously
being unlearned on different items (0.05-0.27), but into a language receiving no unlearning
at all it carries almost nothing (Hebrew 0.5865). Batch 3 saw the same wall from the other
side, with `en+fr_half`'s eight untouched languages sitting at 0.22-0.68. Co-unlearning is
the enabling condition for the donor effect, not a detail of it.

The diagonals carry a second, smaller surprise. English unlearning its own twenty items
should not depend on the partner at all, yet the English diagonal runs from 0.0013 with
Indonesian to 0.1246 with Farsi, two orders of magnitude, while the partner diagonals stay
flat at 0.003-0.05. A distant partner appears to dilute the English update itself, not just
fail to receive it. Nothing here explains why, and it is a single run per pair.

#### Leakage into untouched languages has cluster structure

The untouched-mean column above spans 0.31-0.56, so runs differ in how much they leak
overall, and any claim about one language pulling another has to be read against its own
run. Taking each run's untouched mean as that reference, the candidate relative pairs from
batch 3's script/family hint come out like this.


| relative pair | measured as         | cell    | run's untouched mean | deviation |
| ------------- | ------------------- | ------- | -------------------- | --------- |
| fr - id       | `en+fr_half` on id  | 0.1017  | 0.3704               | -0.27     |
| fr - id       | `en+id_half` on fr  | 0.0729  | 0.3147               | -0.24     |
| ja - ko       | `en+ko_half` on ja  | 0.3105  | 0.5263               | -0.22     |
| ja - ko       | `en+ja_half` on ko  | 0.3097  | 0.5432               | -0.23     |
| fr - ru       | `en+ru_half` on fr  | 0.1926  | 0.3896               | -0.20     |
| ar - fa       | `en+fa_half` on ar  | 0.4205  | 0.5565               | -0.14     |
| ar - fa       | `en+ar_half` on fa  | 0.3992  | 0.4890               | -0.09     |
| ar - iw       | `en+ar_half` on iw  | 0.4806  | 0.4890               | -0.01     |
| ar - iw       | `en+iw_half` on ar  | 0.4717  | 0.4012               | +0.07     |
| fa - hi       | `en+fa_half` on hi  | 0.6730  | 0.5565               | +0.12     |
| fa - hi       | `en+hi_half` on fa  | 0.5662  | 0.5537               | +0.01     |


Two strong symmetric clusters, French-Indonesian and Japanese-Korean, both around -0.22 to
-0.27 in both directions. One weaker but consistent cluster, Arabic-Farsi. And two pairs
that are genuinely related but show nothing at all: Arabic-Hebrew, which are both Semitic,
and Farsi-Hindi, which are both Indo-Iranian. The reference mean includes the cell being
tested, which shrinks these deviations slightly, so they are if anything understated.

Batch 3's hint was therefore real but its explanation was wrong. Family does not predict
leakage: the two strongest clusters share no family (French-Indonesian) or no writing system
(Japanese-Korean), while the two clearest family relations in the set show no effect. What
the working clusters have in common is closeness in script or in typology rather than
descent, which this batch can identify but not explain.

Scored against the predictions registered above, question 2 went two and a half for four:
Arabic did pull Farsi down (-0.09) but not Hebrew (-0.01), Korean did pull Japanese down
(-0.22), Hindi did behave as a language with no relative in the set, and Russian did not -
it pulled French (-0.20) and Indonesian (-0.13), so it has relatives after all.

#### Collateral damage

Prob. Retain, same row order, unlearning languages in bold.


| model            | en         | fr         | ar         | fa         | hi         | id         | iw         | ja         | ko         | ru         |
| ---------------- | ---------- | ---------- | ---------- | ---------- | ---------- | ---------- | ---------- | ---------- | ---------- | ---------- |
| `base`           | 0.9786     | 0.9702     | 0.9367     | 0.9303     | 0.9293     | 0.9574     | 0.9260     | 0.9179     | 0.9182     | 0.9330     |
| `en`             | **0.5939** | -          | -          | -          | -          | -          | 0.8595     | -          | -          | -          |
| `en+fr_half_5ep` | **0.7138** | **0.7146** | 0.7803     | 0.7819     | 0.8064     | 0.7660     | 0.7834     | 0.7568     | 0.7725     | 0.7495     |
| `en+id_half_5ep` | **0.7146** | 0.7837     | 0.7642     | 0.7476     | 0.7743     | **0.5761** | 0.7435     | 0.7089     | 0.7321     | 0.7422     |
| `en+ru_half_5ep` | **0.8033** | 0.8300     | 0.7932     | 0.7786     | 0.7866     | 0.8154     | 0.7955     | 0.7408     | 0.7473     | **0.6994** |
| `en+iw_half_5ep` | **0.8296** | 0.8454     | 0.8012     | 0.7940     | 0.7862     | 0.7966     | **0.6735** | 0.7735     | 0.7861     | 0.7975     |
| `en+ko_half_5ep` | **0.8465** | 0.8959     | 0.8450     | 0.8284     | 0.7892     | 0.8760     | 0.8452     | 0.7093     | **0.6333** | 0.8531     |
| `en+ar_half_5ep` | **0.8419** | 0.8391     | **0.5657** | 0.7278     | 0.7916     | 0.8209     | 0.7845     | 0.7574     | 0.7858     | 0.8111     |
| `en+hi_half_5ep` | **0.8850** | 0.9061     | 0.8453     | 0.8245     | **0.6641** | 0.8836     | 0.8453     | 0.7779     | 0.7867     | 0.8578     |
| `en+ja_half_5ep` | **0.8914** | 0.8984     | 0.8378     | 0.8262     | 0.7895     | 0.8823     | 0.8413     | **0.4000** | 0.7213     | 0.8455     |
| `en+fa_half_5ep` | **0.8872** | 0.8936     | 0.8107     | **0.6272** | 0.8036     | 0.8511     | 0.8318     | 0.7734     | 0.8143     | 0.8237     |


For English, damage tracks forgetting almost monotonically across the nine pairs: retain
0.71 where Prob. Forget is 0.011, 0.80 at 0.027, 0.84-0.85 at 0.12-0.21, 0.89 at 0.27-0.29.
No partner buys English forgetting cheaply; the pairs that leave English intact are the
pairs that left English remembering.

Partners differ more. French forgets to 0.0420 at retain 0.7146, while Arabic forgets to
0.0779 for retain 0.5657 and Japanese, the deepest forgetter at 0.0277, pays 0.4000, the
worst retain value anywhere in the project.

Model Utility, same row order:


| model            | en         | fr         | ar         | fa         | hi         | id         | iw         | ja         | ko         | ru         |
| ---------------- | ---------- | ---------- | ---------- | ---------- | ---------- | ---------- | ---------- | ---------- | ---------- | ---------- |
| `base`           | 0.5627     | 0.4941     | 0.4197     | 0.4293     | 0.3803     | 0.4950     | 0.4105     | 0.4887     | 0.4635     | 0.4320     |
| `en`             | **0.5050** | -          | -          | -          | -          | -          | 0.4003     | -          | -          | -          |
| `en+fr_half_5ep` | **0.5029** | **0.4412** | 0.4229     | 0.4278     | 0.4037     | 0.4645     | 0.3983     | 0.4811     | 0.4486     | 0.4205     |
| `en+id_half_5ep` | **0.5111** | 0.4482     | 0.4151     | 0.4219     | 0.4019     | **0.4459** | 0.3878     | 0.4695     | 0.4438     | 0.4184     |
| `en+ru_half_5ep` | **0.5200** | 0.4608     | 0.4122     | 0.4184     | 0.4024     | 0.4595     | 0.3966     | 0.4760     | 0.4385     | **0.4199** |
| `en+iw_half_5ep` | **0.5205** | 0.4554     | 0.4161     | 0.4177     | 0.4067     | 0.4616     | **0.3841** | 0.4711     | 0.4434     | 0.4174     |
| `en+ko_half_5ep` | **0.5153** | 0.4566     | 0.4116     | 0.4234     | 0.4037     | 0.4571     | 0.3965     | 0.4789     | **0.4300** | 0.4151     |
| `en+ar_half_5ep` | **0.5291** | 0.4599     | **0.4044** | 0.4185     | 0.4013     | 0.4635     | 0.4012     | 0.4714     | 0.4387     | 0.4243     |
| `en+hi_half_5ep` | **0.5346** | 0.4692     | 0.4163     | 0.4222     | **0.4145** | 0.4731     | 0.4033     | 0.4815     | 0.4516     | 0.4159     |
| `en+ja_half_5ep` | **0.5294** | 0.4652     | 0.4116     | 0.4277     | 0.4083     | 0.4727     | 0.4015     | **0.4361** | 0.4357     | 0.4262     |
| `en+fa_half_5ep` | **0.5308** | 0.4681     | 0.4153     | **0.4219** | 0.4071     | 0.4646     | 0.4034     | 0.4726     | 0.4477     | 0.4238     |


Model Utility earns its billing as a coarse summary here. Across ninety cells it stays
within 0.38-0.56 and no cell moves more than 0.06 from the base finetune. The consistent
declines are in the two languages that started highest, English (-0.028 to -0.060 in every
pair) and French (-0.025 to -0.053); the largest single drops are English and French on
`en+fr_half_5ep` (-0.0597, -0.0529) and Japanese on `en+ja_half_5ep` (-0.0526). Hindi goes
the other way and rises above base in all nine runs (+0.021 to +0.034).

What it does not do is track collateral damage. Japanese on `en+ja_half_5ep` lost 0.518 of
Prob. Retain and only 0.053 of utility; English on `en+fr_half_5ep` lost half as much retain
(0.265) for a slightly *larger* utility drop; Arabic on `en+ar_half_5ep` lost 0.371 of retain
for 0.015 of utility. It is a harmonic mean over six quantities that unlearning barely
touches, so read Prob. Retain for damage and treat this table as completeness.

One comparison is worth pulling out of it: monolingual English sits at 0.5050, below eight of
the nine pairs (0.5029-0.5346). The same ordering as Prob. Retain, and the same conclusion -
splitting the forget set costs English less than concentrating it there.

`en+fr_half_5ep` remains the best overall trade-off in the project: 0.0105 / 0.0420 forgotten
at 0.7138 / 0.7146 retained, with both languages symmetric. `en+id_half_5ep` has the lowest
diagonals ever measured here (0.0013 and 0.0050) and matches it in English, but pays for it
in Indonesian retain (0.5761).

#### On the questions

1. **Near-total en/fr transfer is a property of the partner, not of the half/half setting.**
   Partner-to-English transfer spans 0.0125 to 0.4611 with the budget fixed. French is at
   the extreme end of a wide range, and Indonesian and Russian are the only partners that
   come close to it.
2. **Leakage does track the partner, in clusters that follow script and typology rather
   than family.** French-Indonesian and Japanese-Korean are strong and symmetric,
   Arabic-Farsi is weaker, and the two related pairs Arabic-Hebrew and Farsi-Hindi show
   nothing. Batch 3's single Arabic hint survives; its family explanation does not.
3. **No partner is cheap in English, and partners differ in what they cost themselves.**
   English damage is a function of English forgetting. French is the cheapest partner,
   Japanese by far the most expensive.
4. **English is not always the strong receiver.** It is always the strong *donor*, but only
   into a language that is itself being unlearned: items unlearned in English are forgotten
   in all nine partner languages (0.05-0.27), while the monolingual English model leaves
   Hebrew at 0.5865. Whether the reverse direction holds depends entirely on the partner, and
   batch 3 could not see this because French was in all of its triples.
5. **Half/half beats monolingual at equal budget** - not one of the original four questions,
   but the monolingual row answers it in passing. Same 40 items and 35 steps: English alone
   gives Prob. Forget 0.0100 at retain 0.5939, `en+fr_half_5ep` gives 0.0105 at 0.7138 and
   forgets in French too. There is no reason left to run a monolingual English arm except as
   a reference.

Hindi is worth one separate note. It is the most resistant language in the set from every
direction: highest as an untouched language in every other run (0.52-0.69), the weakest donor
into English (0.4611), and the only partner that does not reach below 0.1 on its own diagonal
when trained (0.1637, at the highest partner retain of 0.6641). Whatever makes a language
easy to unlearn through English, Hindi has least of it.

### Regenerate

```bash
python collect_results.py base en en+fr_half_5ep en+id_half_5ep en+ru_half_5ep \
    en+iw_half_5ep en+ko_half_5ep en+ar_half_5ep en+hi_half_5ep en+ja_half_5ep \
    en+fa_half_5ep
for l in fr ar fa hi id iw ja ko ru; do
    python forget_split_probs.py "en+${l}_half_5ep" --langs "en,$l"
done
```

The `en` row comes from the two batch 1 CSVs, which `collect_results.py` maps onto the
experiment id `en` through its legacy layout patterns: English from
`results/aya_graddiff_10lang/finetune_result01_unlearn_grad_diff_2e-05_en.csv` and Hebrew
from `results/en_iw_transfer/unlearn_en_eval_iw.csv`.

### Cost

11 hours 20 minutes on one L40, 2026-08-23 16:44 to 2026-08-24 04:04, against an estimate of
12: seven unlearning runs at ~13 minutes and 78 evaluations at ~8 minutes, no failures and
nothing to re-run. 112 GB of new checkpoints.

### Caveats specific to this batch

- **One run per pair, one seed.** Batch 3 showed untouched-language values swinging by 0.1
and more between runs that differed only in their third language. The claims above that
clear that bar are the English range (0.011-0.288), the donor asymmetry (0.0125-0.4611 into
English against 0.05-0.27 outward), and the two -0.22 clusters. The Arabic-Farsi cluster at
-0.09 to -0.14 does not, and should be treated as a lead.
- **Leakage was compared across runs of unequal strength.** The cluster table controls for
this with each run's own untouched mean, which is crude: it assumes a run's leakage scales
uniformly across the eight languages. A partner-by-language model fit over all nine runs
would test that assumption, and nine runs is thin for it.
- **The monolingual English row is two cells wide.** `en` and `iw` are the only
single-language models, and the English one was only evaluated in English and Hebrew, so
"English unlearning alone does not travel" rests on a single language. Evaluating that
existing checkpoint in the other eight is eight evaluations and about an hour of GPU, with
no training, and it would turn the donor claim into a full nine-language comparison against
the "into partner" column. It is the cheapest high-value run left in the project.
- **Still no monolingual partner baselines.** Nothing here separates "the partner language
did the work" from "the pair did the work" for the other eight partners: nine monolingual
runs would say whether Japanese unlearns to 0.0277 because of the pairing or on its own.
- **The English diagonal moving with the partner is unexplained.** 0.0013 with Indonesian
against 0.1246 with Farsi, on identical English items and an identical step count. The
retain union differs between pairs and so does the partner's forget gradient, but this
batch cannot say which, and a wrong answer here would change how the donor asymmetry is
read.
- Batch 6 supersedes the idea of extending batch 4 to these nine checkpoints: its single
auxiliary vector applies to any checkpoint, so the partner-independence question can be
asked there for far less compute.



## Batch 6: steering recovery, redone to the paper's specification (in progress)

Batch 4 was our own reading of the steering attack. Xiang et al., *Multilingual Unlearning
in LLMs: Transfer, Dynamics, and Reversibility* (arXiv `2606.03291`, PDF in the repo root)
is the paper that introduced it, and comparing Section 4.4 and Algorithms 1-2 against
`run_steering_recovery.sh` turned up **five divergences, four of them substantive**. This
batch re-runs the attack with those fixed, writing to `steering/results_v2/` so the Batch 4
JSONs survive for a before/after comparison.

Driver `run_steering_replication.sh`; log `logs/steering_replication_stage1.txt`.

### What was wrong, and what replaces it

| # | Batch 4 | Batch 6 (paper) | Why it matters |
|---|---|---|---|
| 1 | Raw mean difference, no normalisation, `--normalize` never passed | Per-sample L2 normalisation before differencing; injection is `h - alpha*‖h‖*g` | Per-layer `‖v‖` ran 0.15 (L0) to 76.5 (L30) in `aya_grad_diff_en_en_summary.json`. One alpha was a rounding error early and catastrophic late, so the alpha grid never tested a comparable intervention |
| 2 | Vector from the **target** unlearned model, measured on `retain99` | Vector from **one auxiliary** unlearned model, measured on the set it actually unlearned | Neither model was trained to suppress retain data, so the Batch 4 difference was collateral parameter drift, not a suppression direction |
| 3 | Mean over all answer tokens; hook broadcast over the whole sequence | Last prompt token only; last position only at each decoding step | Algorithm 1 line 6 and Algorithm 2 line 14. Averaging mixes suppression with per-token content; broadcasting rewrote the prompt on the prefill pass |
| 4 | Fixed `late` = L24-31, 8 layers at once | 3-layer window (`N=2`), start layer swept over all 30 positions | Suppression is localised. One fixed block can miss it entirely, and "the attack failed" is only meaningful after the sweep |
| 5 | ROUGE-L / chrF | NLI equivalence score (Appendix E.1 Eq. 4), lexical metrics kept alongside | Their Table 8: ROUGE-L agrees with human judgement 62-68% of the time, NLI 89% |

### The key simplification

All seven checkpoints share one finetuned parent, `tofu_finetuned_5epoch_aya_10_lang_2e5`.
Because the paper's vector comes from `f_ft` and an auxiliary model and never touches the
target, **one** auxiliary model and **one** English vector serve all seven checkpoints in
every language. Batch 4 extracted 16 per-language vectors; this batch extracts one, and
gets a stronger test for less compute — applying an English-derived direction to fr/ar/ja/iw
is a direct test of the paper's language-agnosticism claim, which the per-language design
could not make.

`steering/make_aux_forget.py` lifts two whole author blocks out of `retain99_en` under a
fixed seed (seed 42 gives blocks 17 and 153: Asha Majaliwa and Femke Van der Veen) into
`dataset/forget01_aux_en` (40 rows) and `dataset/retain99_aux_en` (3920 rows), mirroring
`forget01`'s two-authors-by-20-pairs shape. `config/forget_aux_en.yaml` then unlearns them
from `f_ft` with the same `grad_diff` recipe as every other arm, landing at
`grad_diff_2e-05_forget01aux_5_en` (~33 optimizer steps). The seed is load-bearing: the
vector is only reproducible if the auxiliary authors are.

Before the vector is trusted, the driver runs a **sanity gate** — `f_ft` and `f_un_aux`
both generate on `forget01_aux_en`, and the auxiliary model has to actually suppress its
own authors. If it does not, the `f_ft -> f_un_aux` difference is not an unlearning
direction and nothing downstream means anything.

### Verified before spending GPU time

`steering/check_steering_math.py` asserts the hook arithmetic on CPU against a 4-layer
randomly-initialised Cohere model, in seconds. All twelve checks pass:

- `alpha=0` is bit-identical to no intervention (max abs diff `0.00e+00`)
- positions `0..T-2` come out untouched — the Batch 4 broadcast bug cannot recur silently
- `‖delta‖` equals `alpha * ‖h‖` exactly, so alpha is a fraction of the activation norm
- the edit is antiparallel to `g` (cosine `-1.0`): the attack subtracts, it does not add
- the hook fires once per decoding step under KV caching, not just on one forward pass
- the sweep covers `c = 0..29` with every 3-layer window inside a 32-layer model
- extraction returns unit rows indexed by decoder layer
- subtracting `(f_un - f_ft)` provably moves activations back toward `f_ft` (0.8479 → 0.5475)

The last two are the sign and indexing checks that a wrong result would otherwise be
indistinguishable from.

### Run plan

Phases are skip-if-done, so an interrupted run resumes:

| phase | what | cost |
|---|---|---|
| `prep` | auxiliary splits, auxiliary model, sanity gate, the one English vector | ~15 min |
| `reference` | `f_ft` generations on `forget01_{en,fr,iw,ja,ar}` | ~15 min |
| `quick` | Appendix K.3 all-layer variant, alpha `0.05 0.1 0.2`, real + random | ~1.5 h |
| `sweep` | Algorithm 2 start-layer sweep, alpha 0.5 (en) / 0.8 (other), real + random | ~13 h |
| `nli` | Eq. 4 scores over every result file | ~20 min |
| `summarize` | `comparison`, `layer-profile`, `interleave-split` | seconds |

`quick` runs first on purpose: if an all-layer intervention at small alpha does nothing
anywhere, the 13-hour sweep is not worth starting.

**Status:** relaunched 2026-08-25 20:05 and running from `prep`, now that Batch 5 has
finished and the card is free; `reference` was already done for `ja` and `ar`. The wait for
the GPU was the reason for the gap: the Batch 5 eval loop held 35 GB of the 46 GB card and
an 8B inference pass needs 20 GB, so the two could not share it. The first launch attempt
proved that the hard way: it passed a free-memory check in the gap between two evals,
collided with the next one, and OOMed along with two of the neighbour's evals. `wait_for_gpu`
now waits for an **idle** card and re-checks after a settle window instead of racing into a
gap, so the run is safe to leave unattended. Relaunch with:

```bash
tmux new -s steer-repl -d './run_steering_replication.sh 2>&1 | tee logs/steering_replication.txt'
```

### The reference column Batch 4 lacked

`results/steering_comparison.csv` has no `f_ft` column, so a `+0.0475` recovery could not
be read as a share of what unlearning removed. The `reference` phase adds it, and
`summarize_results.py comparison` now reports

```text
recovery_frac = (best - unlearned) / (f_ft - unlearned)
```

which is the quantity behind the paper's "over half" (Qwen) and "90%" (Gemma). A recovery
of +0.05 against a 0.30 unlearning drop is a very different claim from the same +0.05
against a 0.06 drop, and Batch 4 could not tell them apart.

**First result, already in.** The Japanese and Arabic reference passes completed, so the two
Batch 4 recoveries that were hardest to interpret can now be put on that scale:

| arm | lang | f_ft | unlearned | best | recovery | share of what unlearning removed |
|---|---|---|---|---|---|---|
| `en+fr+ja_third` | ja | 0.3779 | 0.1652 | 0.1864 | +0.0212 | 10.0% |
| `en+fr+ja+ar_quarter` | ja | 0.3779 | 0.1674 | 0.1858 | +0.0184 | 8.7% |
| `en+fr+ja+ar_quarter` | ar | 0.5246 | 0.2870 | 0.3119 | +0.0249 | 10.5% |
| `en+fr+ar_third` | ar | 0.5246 | 0.2818 | 0.2818 | +0.0000 | 0.0% |

Batch 4 could only say these bumps were "small, and possibly just shallower forgetting."
The denominator makes the size explicit: even at their best they win back roughly a tenth
of what unlearning removed, against the paper's "over half" and "90%". Whether that gap is
the implementation (which Batch 6 fixes) or the `grad_diff` objective (which it cannot) is
exactly what the rest of this batch is for.

### Which Batch 4 conclusions are at stake

Batch 4 concluded (1) monolingual EN and IW leave recoverable directions, with Hebrew the
strongest result at chrF +0.0793; (2) interleaving with French does not block the English
attack at 20/20; (3) the quarter's Japanese/Arabic bumps reflect shallower forgetting, not
recoverability. Each rests on a `late`-layer, un-normalised, target-derived vector, so all
three are re-opened rather than refined:

- **If Batch 6 recovers more**, the Batch 4 negatives were measurement failures. "French
  never recovers on any interleaved model" — the most consistent negative in that batch —
  would be the first claim to fall, since a fixed late-layer window is exactly what a
  localised non-English suppression signal would slip past.
- **If Batch 6 recovers about the same**, the Batch 4 numbers stand on a correct
  implementation and the interpretation gets the `recovery_frac` denominator it was missing.
- **If Batch 6 recovers nothing anywhere**, a flat 30-start-layer profile is a far stronger
  negative than one failed window, and the honest reading is that `grad_diff` on this model
  does not leave a linearly recoverable direction — see the caveat below.

Results and the resolved verdict go here once the sweep finishes.

### Caveats specific to this batch

- **The paper's headline uses DPO with IDK refusals; we use `grad_diff`.** DPO trains a
  clean uniform refusal, which is a highly learnable direction. `grad_diff` produces
  degenerate output instead, so even a faithful implementation may recover less than their
  numbers. Their Appendix K.2 shows the effect surviving GA and NPO, but `grad_diff`
  specifically is untested there. A weak result here is therefore ambiguous between "our
  implementation" and "this objective", and only the former has now been ruled out.
- **NLI coverage.** `joeddav/xlm-roberta-large-xnli` is finetuned on XNLI's 15 languages,
  which include en/fr/ar but **not Hebrew or Japanese**. Spot checks on iw/ja entailment
  pairs behave sensibly (0.997 and 0.996 entailment on true pairs, 1.0 contradiction on
  false ones), but three examples are not validation: those scores are zero-shot
  cross-lingual transfer. Every score carries a `validated` flag and chrF stays reported
  alongside for those two languages.
- **One auxiliary model, one seed.** A different pair of auxiliary authors gives a
  different vector. The paper does not report sensitivity to this choice and neither can we
  from a single draw.
- **Out of scope:** cross-source-language extraction (Appendix K.1), and re-running the
  Batch 4 retain-set vector as a formal ablation.

### Regenerate

```bash
./run_steering_replication.sh --list                  # arms, vector, phases
DRY_RUN=1 ./run_steering_replication.sh               # print every command
PHASES="prep reference quick" ./run_steering_replication.sh
./run_steering_replication.sh                         # everything, in order

python steering/check_steering_math.py                # CPU, no GPU needed
python steering/summarize_results.py comparison \
    --results-dir steering/results_v2 --csv results/steering_comparison_v2.csv
python steering/summarize_results.py layer-profile --results-dir steering/results_v2
```

## Running

```bash
./run_transfer_experiments.sh --list          # show the experiment table
DRY_RUN=1 ./run_transfer_experiments.sh       # print every command, run nothing
./run_transfer_experiments.sh                 # run everything
./run_transfer_experiments.sh en+fr_5ep       # run one experiment
FORCE=1 ./run_transfer_experiments.sh en+fr_5ep   # redo it from scratch
```

Every step waits for 40 GB of free GPU before starting, so the script can be launched while
another job is running. Completed steps are skipped, so it is safe to re-run after an
interruption. Batch 3 was launched as:

```bash
tmux new -s exp3 -d './run_transfer_experiments.sh en+fr_half_5ep en+fr+ja_third_5ep \
    en+fr+ar_third_5ep en+fr+ja+ar_quarter_5ep 2>&1 | tee logs/transfer_experiments_batch3.txt'
tmux attach -t exp3
```

Note that a session created this way ends as soon as the script finishes; the `tee` file is
what survives. Budget roughly 7 hours for batch 3: three unlearning runs at ~13 minutes and
38 evaluations at ~8-11 minutes each.

Batch 4 (steering recovery on the seven checkpoints above) is a separate driver.
It finished in about four hours on one L40:

```bash
./run_steering_recovery.sh --list
DRY_RUN=1 ./run_steering_recovery.sh
tmux new -s steering-recovery -d './run_steering_recovery.sh 2>&1 | tee logs/steering_recovery.txt'
```

Completed extract/attack files are skipped, so a re-run only fills gaps. Override
`FORCE=1` to redo them. The GPU wait default is 20 GB free (`GPU_FREE_MIB`).

Batch 5 used the same driver as batches 2 and 3, and was launched as:

```bash
tmux new -s exp5 -d './run_transfer_experiments.sh en+iw_half_5ep en+ar_half_5ep \
    en+ko_half_5ep en+ja_half_5ep en+ru_half_5ep en+hi_half_5ep en+id_half_5ep \
    en+fa_half_5ep 2>&1 | tee logs/transfer_experiments_batch5.txt'
```

The run order comes from the experiment table in the driver, not from the command line, and
was set so the questions were answered as early as possible: `en+iw_half_5ep` first because
it needed no training, then Arabic and Korean, which carried the family predictions of
question 2 against Farsi, Hebrew and Japanese.

## Caveats

- **No Forget Quality.** `aggregate_eval_stat.py` only computes the KS-test forget quality
when `retain_result` points at an evaluated retain model, and that entry is commented out
in `config/aggregate_eval_stat.yaml`. Every CSV therefore reports Model Utility and the
per-task probabilities and truth ratios only.
- **The base model baseline now exists** in `results/base/`. Prob. Forget before unlearning
is 0.90-0.98 in every language. The batch 1 and 2 tables above were written against the
older 0.48-0.59 untouched-language reference from batch 1; batch 3 uses `base`.
- **No per-epoch checkpoints in batches 2 and 3.** These runs set
`save_epoch_checkpoints: false`, so only the final model is written. It saves about 90 GB
and 15 minutes per run, but a crash mid-run loses that run. Set the flag to `true` to
restore the old behaviour.
- **Missing monolingual baselines.** English and Hebrew are the only languages with a
single-language unlearned model, so no other pair can be read the way en+iw is read against
its en and iw baselines. Batch 5 made this the main open gap: it compared nine partners
without knowing what any of them does alone, and its central finding (that the partner
decides how much English forgets) is exactly what a monolingual row would calibrate.
Running one would be
`python forget.py --config-name forget_fr forget_loss=grad_diff batch_size=2 gradient_accumulation_steps=3`
plus the evaluations.
- **Batch 2 evaluated only the trained languages.** Its models were never evaluated in the
other eight languages; add languages to the last field of the experiment table (or set it
to `all`) to fill that in. Batch 5 did this for `en+iw_half_5ep`; the `concat` runs remain
two-language only.

