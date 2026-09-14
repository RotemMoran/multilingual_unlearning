# Split and Forget: Joint Multilingual Unlearning in Large Language Models

Code and experiment log for **Split and Forget: Joint Multilingual Unlearning in Large
Language Models**.

When a fact has to be removed from a multilingual LLM, is it better to unlearn it in every
language at once, or to **split** the forget set so that each item is unlearned in exactly
one language? This repository runs both settings on Aya-Expanse 8B over a ten-language
extension of TOFU, evaluates every resulting model in all ten languages, and then probes
whether the forgotten answers can be brought back by an inference-time activation-steering
attack.

---

## 🌍 Data and base model

Experiments use **TOFU** (Task of Fictitious Unlearning), a benchmark of QA pairs derived
from synthetic autobiographies of fictitious authors, extended by translation to ten
languages — English, French, Arabic, Japanese, Russian, Farsi, Korean, Hindi, Hebrew,
Indonesian — spanning five language families and varying resource levels. All ten are in
`dataset/`.

The base model throughout is **Aya-Expanse 8B** finetuned for 5 epochs on all ten languages
(`outputs/tofu_finetuned_5epoch_aya_10_lang_2e5`). The fictitious authors exist in that model
in these ten languages only, so no other language can be meaningfully unlearned or evaluated.

TOFU: [website](https://locuslab.github.io/tofu) ·
[paper](http://arxiv.org/abs/2401.06121) ·
[code](https://github.com/locuslab/tofu)

---

## ⚙️ Installation

```bash
conda create -n tofu python=3.10
conda activate tofu
conda install pytorch pytorch-cuda=11.8 -c pytorch -c nvidia
conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
```

---

## 📂 Repository Structure

```
.
├── config/
│   ├── finetune.yaml
│   ├── forget.yaml                  # single-language unlearning
│   ├── forget_<lang>.yaml           # one per language
│   ├── forget_joint.yaml            # several languages in one run
│   ├── forget_aux_en.yaml           # auxiliary run used by the steering attack
│   ├── eval_everything.yaml         # English evaluation
│   ├── eval_multilingual.yaml       # non-English evaluation
│   └── aggregate_eval_stat.yaml
├── dataset/
│   └── ...
├── finetune.py
├── forget.py
├── evaluate_util.py
├── aggregate_eval_stat.py
├── collect_results.py               # eval CSVs -> markdown matrices
├── utils.py
├── run_transfer_experiments.sh      # joint-unlearning batch driver
├── run_steering_replication.sh      # steering batch driver
├── steering/                        # activation-steering recovery attack
│   ├── make_aux_forget.py
│   ├── extract_steering_vector.py
│   ├── attack_generate.py
│   ├── nli_score.py
│   ├── summarize_results.py
│   └── README.md
└── EXPERIMENTS.md                   # full experiment log with results
```

---

## 🚀 Workflow

### 1️⃣ Finetuning

First, fine-tune the model on the **full dataset**. Hyperparameters, model selection and
data paths are in `config/finetune.yaml`:

```bash
python finetune.py
```

---

### 2️⃣ Unlearning (single language)

To forget one dataset or language subset, set which one in `config/forget.yaml` and run:

```bash
python forget.py
```

There is also one ready-made config per language, e.g.:

```bash
python forget.py --config-name forget_iw forget_loss=grad_diff
```

The per-language configs do not all carry the same `forget_loss`, so set it explicitly when
comparing arms — every experiment in `EXPERIMENTS.md` uses `grad_diff`.

---

### 3️⃣ Joint (multilingual) unlearning

`config/forget_joint.yaml` unlearns the **same forget split in several languages in one
run**. The language list and the tag that names the output directory are passed on the
command line:

```bash
python forget.py --config-name forget_joint \
  language=[en,fr] lang_tag=en+fr language_mix=concat num_epochs=5
```

`language_mix` decides how the per-language forget sets are combined:

| `language_mix` | Forget set | Forget examples (2 languages) |
|---|---|---|
| `concat` | every language contributes the whole split, so each item is unlearned once **per language** | 80 |
| `interleave` | item `i` is unlearned in language `i % N` only, so the split stays one copy of itself however many languages share it | 40 (20 per language) |

`interleave` is what makes a multilingual arm comparable with a monolingual one: the forget
set stays at 40 rows, so the optimizer budget is identical (7 steps per epoch at effective
batch 6, 35 at 5 epochs). With `concat` the budget grows with the number of languages.
The **retain** set is always the full union of the trained languages in both cases, so
retention pressure per language does not change with the mix.

```bash
# 20 forget items in English, 20 in Hebrew
python forget.py --config-name forget_joint \
  language=[en,iw] lang_tag=en+iw_half language_mix=interleave num_epochs=5

# 10 items each in English, French, Japanese, Arabic
python forget.py --config-name forget_joint \
  language=[en,fr,ja,ar] lang_tag=en+fr+ja+ar_quarter language_mix=interleave num_epochs=5
```

`lang_tag` is naming only; the convention is the `+`-joined language codes, plus
`_half` / `_third` / `_quarter` when interleaving over two, three or four languages. Both
the tag and the epoch count go into the output path, so no run can overwrite another:

```
outputs/tofu_finetuned_5epoch_aya_10_lang_2e5/grad_diff_2e-05_forget01_<num_epochs>_<lang_tag>
```

Any language in the ten-language finetune can be used. English is read from the Hub, the
other nine from `dataset/forget01_<lang>` and `dataset/retain99_<lang>`, as listed under
`data_path` in the config.

#### Running the whole batch

`run_transfer_experiments.sh` drives the joint-unlearning experiments end to end
(unlearn, then evaluate each model in every language, then aggregate). Each step waits for
a free GPU and is skipped when its output already exists, so an interrupted run resumes:

```bash
./run_transfer_experiments.sh --list             # show the experiment table and exit
./run_transfer_experiments.sh                    # run everything
./run_transfer_experiments.sh en+fr_half_5ep     # run only that experiment
DRY_RUN=1 ./run_transfer_experiments.sh          # print the commands, run nothing
FORCE=1 ./run_transfer_experiments.sh            # redo completed steps
```

Outputs are keyed by experiment id (`<lang_tag>_<N>ep`): logs in
`logs/forget_<exp_id>.txt` and `logs/eval_<exp_id>_on_<lang>.txt`, per-language CSVs in
`results/<exp_id>/eval_<lang>.csv`. `python collect_results.py` turns those CSVs into the
model × language markdown matrices used in `EXPERIMENTS.md`.

---

### 4️⃣ Evaluation

`evaluate_util.py` scores a checkpoint on the four standard TOFU tasks (retain, real
authors, world facts, forget). English reads the perturbed splits from the Hub via
`config/eval_everything.yaml`; the other nine languages read
`dataset/<split>_perturbed_<lang>` from disk via `config/eval_multilingual.yaml`. Both take
the checkpoint and the language on the command line:

```bash
CKPT=outputs/tofu_finetuned_5epoch_aya_10_lang_2e5/grad_diff_2e-05_forget01_5_en+fr_half

python evaluate_util.py --config-name eval_everything   model_path=$CKPT language=en
python evaluate_util.py --config-name eval_multilingual model_path=$CKPT language=fr
```

Results land in `$CKPT/eval_results/ds_sizeNone_<lang>/`.

---

### 5️⃣ Aggregating Evaluation Results

`aggregate_eval_stat.py` turns the raw eval logs into a one-row CSV of per-task
probabilities, truth ratios, **Model Utility** and **Forget Quality**. The retain-model and
unlearned-model paths can be set in `config/aggregate_eval_stat.yaml`, or per language on
the command line, which is what the batch drivers do:

```bash
python aggregate_eval_stat.py lang=fr \
  ckpt_result=$CKPT/eval_results/ds_sizeNone_fr/eval_log_aggregated.json \
  method_name=grad_diff save_file=results/en+fr_half_5ep/eval_fr.csv
```

---

### 6️⃣ Steering-vector recovery attack

Does unlearning **remove** the knowledge, or only suppress it? The attack in `steering/`
answers that by recovering forgotten answers at inference time, with no weight updates. It
follows Xiang et al., [arXiv 2606.03291](https://arxiv.org/abs/2606.03291), Section 4.4 and
Algorithms 1-2: a direction is read off the difference between the finetuned parent and an
unlearned model, then subtracted from the residual stream during generation.

```text
s[l] = sum_x ( h_un(x)/‖h_un(x)‖ - h_ft(x)/‖h_ft(x)‖ )   at the last prompt token
g[l] = s[l] / ‖s[l]‖
h    = h - alpha * ‖h‖ * g[l]        last position only, over a swept 3-layer window
```

#### Check the arithmetic (no GPU, seconds)

```bash
python steering/check_steering_math.py
```

Asserts on a toy model that `alpha=0` is a no-op, that only the last position is edited,
that `‖delta‖ == alpha*‖h‖`, and that the hook fires under KV caching. Run it after
touching the hook.

#### 1. Data and the auxiliary unlearned model

The English splits live on the Hub but the steering scripts only call `load_from_disk`, so
materialise them once. The vector must **not** come from the model being attacked, or it can
smuggle in the very answers the attack claims to recover — so it is read off an auxiliary
model that unlearned two *retain* authors instead:

```bash
python steering/make_en_datasets.py
python steering/check_data.py
python steering/make_aux_forget.py           # seed 42 -> author blocks 17 and 153
python forget.py --config-name forget_aux_en
```

The seed is load-bearing: the vector is reproducible only if the auxiliary authors are.

#### 2. Extract the vector

All the unlearning arms share one finetuned parent and the vector never touches the target,
so a **single English vector serves every arm in every language** — which also makes
applying it to fr/ar/ja/iw a direct test of the language-agnosticism claim.

```bash
BASE=outputs/tofu_finetuned_5epoch_aya_10_lang_2e5

python steering/extract_steering_vector.py \
  --base-model "$BASE" \
  --unlearned-model $BASE/grad_diff_2e-05_forget01aux_5_en \
  --aux-dataset ./dataset/forget01_aux_en --language en \
  --out steering/vectors/aya_aux_en.pt
```

`steering/vectors/aya_aux_en.pt` is committed, so this step can be skipped.

#### 3. Attack

```bash
CKPT=$BASE/grad_diff_2e-05_forget01_5_en+fr_half

# Algorithm 2: 3-layer window, start layer swept over all 30 positions
python steering/attack_generate.py \
  --model "$CKPT" --vector steering/vectors/aya_aux_en.pt \
  --forget-dataset ./dataset/forget01_en --language en \
  --mode sweep --alphas 0.5 --window 2 \
  --out steering/results_v2/aya_aux_en+fr_half_en_sweep.json

# the same run with a norm-matched random direction: the control
python steering/attack_generate.py \
  --model "$CKPT" --vector steering/vectors/aya_aux_en.pt \
  --forget-dataset ./dataset/forget01_en --language en \
  --mode sweep --alphas 0.5 --window 2 --random-vector \
  --out steering/results_v2/aya_aux_en+fr_half_en_sweep_random.json
```

`--mode all-layers` is the Appendix K.3 variant, which hits every layer at once and needs a
much smaller alpha (`--alphas 0.05 0.1 0.2`). Alpha is 0.5 for English and 0.8 elsewhere,
per the paper. `--vector` may be omitted when every alpha is 0, which is how the
"before unlearning" reference passes on the parent model are produced.

The **random control is not optional.** A random direction at the same alpha can knock a
degenerate model out of a repetition loop and so raise the score on its own; without the
control four of the arms here read as successes when they are not.

#### 4. Score and summarise

```bash
python steering/nli_score.py steering/results_v2/*.json

python steering/summarize_results.py comparison \
  --results-dir steering/results_v2 --csv results/steering_comparison_v2.csv
python steering/summarize_results.py layer-profile --results-dir steering/results_v2
```

Scoring is a separate pass, so generations can be re-scored without regenerating them.
`nli_score.py` implements the Appendix E.1 equivalence score; lexical metrics are ROUGE-L
recall for English and chrF elsewhere (`steering/metrics.py` keeps non-Latin scripts, which
stock `rouge_score` does not). `layer-profile` shows recovery against injection start layer,
which is the diagnostic that distinguishes a localised suppression signal from a flat null.

#### Running the whole batch

```bash
./run_steering_replication.sh --list                   # show the arms and exit
./run_steering_replication.sh                          # every phase, in order
PHASES="prep reference quick" ./run_steering_replication.sh
./run_steering_replication.sh en iw                    # restrict to those arms
DRY_RUN=1 ./run_steering_replication.sh                # print commands, run nothing
```

Phases (`prep reference quick sweep nli summarize`) run in order and each is skip-if-done.
The sweep phase is an overnight job; run it under tmux.

#### What came out

On Aya with `grad_diff`, the direction is real but small: monolingual English recovers
about **8%** of the suppressed performance, and only in the **early** layers, peaking at a
window over layers 6-8. Injecting past layer ~8 collapses generation, which is why a fixed
late-layer injection (the earlier `run_steering_recovery.sh` design) found nothing. Only 2
of 16 `(model, language)` pairs beat their random control, and the English vector does not
transfer to French, Arabic, Japanese or Hebrew. Full method notes, every command, the
column-by-column reading of the comparison table and the complete results are in
[`steering/README.md`](steering/README.md).

---

## 📝 Notes

- Every logged experiment unlearns with `grad_diff` at lr 2e-5, seed 42, and effective batch 6
  (`batch_size: 2` with `gradient_accumulation_steps: 3`), which is what fits an 8B model on
  one 46 GB card. `max_length` is 500 in finetuning, unlearning and steering alike.
- The forget split is TOFU `forget01`: 40 QA pairs covering exactly two authors, 20 rows each.
- [`EXPERIMENTS.md`](EXPERIMENTS.md) is the running log of every experiment batch — what each
  one measures, where its outputs live, and what came out.
- [`steering/README.md`](steering/README.md) documents the recovery attack in full.

---

## 📚 Built on

The ten-language TOFU translations and the finetuning, unlearning and evaluation code come
from *Multilingual Amnesia: On the Transferability of Unlearning in Multilingual LLMs*
(EACL 2026):

```bibtex
@misc{farashah2026multilingualamnesiatransferabilityunlearning,
  title         = {Multilingual Amnesia: On the Transferability of Unlearning in Multilingual LLMs},
  author        = {Dehghanpour Farashah, Alireza and Khandelwal, Aditi and Fauchard, Marylou and Shi, Zhuan and Rostamzadeh, Negar and Farnadi, Golnoosh},
  year          = {2026},
  eprint        = {2601.05641},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  url           = {https://arxiv.org/abs/2601.05641}
}
```
