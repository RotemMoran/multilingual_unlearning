# Steering-recovery experiment (Aya-Expanse 8B)

Does monolingual unlearning only suppress knowledge, leaving it recoverable by an
inference-time activation intervention? Does joint English–Hebrew unlearning make that
recovery harder?

```text
v[i] = mean_unlearned(h_i) - mean_base(h_i)
h_i  = h_i - alpha * v[i]
```

The model is **Aya-Expanse 8B** throughout (`config/model_config.yaml`,
`CohereForAI/aya-expanse-8b`). The common base is the 10-language TOFU finetune:

```text
outputs/tofu_finetuned_5epoch_aya_10_lang_2e5
```

## Unlearning arms already on disk

These were trained with the same hyperparameters as `config/forget_en.yaml`,
`config/forget_iw.yaml`, `config/forget_en_iw.yaml`, and `config/forget_joint.yaml`:
`batch_size: 2`, `gradient_accumulation_steps: 3` (effective batch 6), 5 epochs.

| Arm | Forget data | Rows/epoch | Checkpoint |
|---|---|---:|---|
| A | English | 40 | `.../grad_diff_2e-05_forget01_5_en` |
| B | English + Hebrew | 80 | `.../grad_diff_2e-05_forget01_5_en+iw` |
| IW | Hebrew (`forget01_iw`) | 40 | `.../grad_diff_2e-05_forget01_5_iw` |
| NPO-A | English (NPO) | 40 | `.../npo_2e-05_forget01_5_en` |

Hebrew unlearning and the steering attack both use `dataset/forget01_iw`, not
`forget01_perturbed_iw`.

Aya decoder blocks are at `model.layers`. Extraction and attack hook that same
tensor. Means are taken over answer tokens only (`labels != -100`).

### Why A and B are not automatically comparable

Arm B concatenates English and Hebrew, so each epoch has **80** forget rows instead
of **40**. With `batch_size=2` and `gradient_accumulation_steps=3`, that is about
33 optimizer steps for A/IW and 66 for B.

If the attack recovers A but not B, two explanations are possible:

1. Joint EN+IW unlearning actually removed the knowledge more thoroughly.
2. B simply trained twice as long, so it forgot harder for that reason alone.

That is all the “10-epoch English control” meant: an English-only run with 10
epochs would match B’s optimizer-step count while staying monolingual. It is
**not required** for the steering pipeline. Before attributing a B-vs-A recovery
gap to multilinguality, check that the two checkpoints are similar on forget
probability / truth ratio / utility (`summarize_results.py forget-quality`). If B
forgot much more, the extra steps are a likely confound.

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

Hebrew data is already at `dataset/forget01_iw` and `dataset/retain99_iw`.

## 2. Extract a steering vector

One model at a time (8B bf16 is ~16 GB). Default family is `aya-expanse-8B`.

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

Hebrew (same forget file the IW/joint runs trained on):

```bash
python steering/extract_steering_vector.py \
  --base-model "$BASE" \
  --unlearned-model $BASE/grad_diff_2e-05_forget01_5_iw \
  --aux-dataset ./dataset/retain99_iw --language iw \
  --model-family aya-expanse-8B \
  --out steering/vectors/aya_grad_diff_iw.pt
```

Repeat for the joint checkpoint.

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

Hebrew attack on `forget01_iw`:

```bash
python steering/attack_generate.py \
  --model $BASE/grad_diff_2e-05_forget01_5_iw \
  --vector steering/vectors/aya_grad_diff_iw.pt \
  --forget-dataset ./dataset/forget01_iw --language iw \
  --model-family aya-expanse-8B \
  --alphas 0 0.5 1 2 4 --layer-set late \
  --out steering/results/aya_grad_diff_iw_late.json
```

Sweep `late` / `middle` / `all` if you want a layer-set grid. Score English with
ROUGE-L recall and Hebrew with chrF (`steering/metrics.py` keeps non-Latin
scripts; stock `rouge_score` does not).

## 4. Summarise

```bash
python steering/summarize_results.py forget-quality \
  --checkpoints \
    outputs/tofu_finetuned_5epoch_aya_10_lang_2e5/grad_diff_2e-05_forget01_5_en \
    outputs/tofu_finetuned_5epoch_aya_10_lang_2e5/grad_diff_2e-05_forget01_5_en+iw \
    outputs/tofu_finetuned_5epoch_aya_10_lang_2e5/grad_diff_2e-05_forget01_5_iw \
  --csv results/aya_forget_quality.csv

python steering/summarize_results.py attacks \
  --results-dir steering/results \
  --csv results/aya_attack_recovery.csv
```

| Pattern | Conclusion |
|---|---|
| A recovers, B does not, and forget metrics are similar | evidence for the joint-language treatment |
| A recovers, B does not, but B forgot much more | extra optimizer steps are the likely confound |
| A and B both recover | joint unlearning does not stop the attack |
| no arm recovers | the attack failed; check sign, ceiling, and alpha range |
| random control recovers similarly | no directional steering result |

Always report unattacked score, best attacked score, recovery delta, and random-control delta.

## Notes

- `max_length` is 500, matching `forget.py` / `finetune.py`.
- Unlearning: `batch_size=2`, `gradient_accumulation_steps=3` (already used).
- Extract/attack: `--batch-size 16`, matching the eval configs.
