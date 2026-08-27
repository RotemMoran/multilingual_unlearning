#!/usr/bin/env python
"""Split the forget-set probability of an interleaved run by the language each item
was unlearned in.

`Prob. Forget` in the result CSVs is the mean ground-truth probability over all 40
forget items, which for an interleaved run mixes items unlearned in the evaluation
language with items unlearned only in another language. Because `interleave` assigns
item i to language i % len(langs), the two groups can be separated after the fact from
`eval_log_forget.json`, which is keyed by data index. The off-diagonal cells are direct
measurements of cross-lingual transfer.

    python forget_split_probs.py en+fr_half_5ep --langs en,fr
    python forget_split_probs.py en+fr+ja+ar_quarter_5ep --langs en,fr,ja,ar
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
BASE = ROOT / "outputs/tofu_finetuned_5epoch_aya_10_lang_2e5"
LANGUAGES = ["en", "fr", "ar", "fa", "hi", "id", "iw", "ja", "ko", "ru"]


def model_dir(exp_id, forget_loss="grad_diff"):
    tag, _, epochs = exp_id.rpartition("_")
    epochs = epochs.removesuffix("ep")
    return BASE / f"{forget_loss}_2e-05_forget01_{epochs}_{tag}"


def group_means(aggregated, langs):
    """Mean ground-truth probability per unlearning language."""
    losses = json.load(open(aggregated))["eval_log_forget.json"]["avg_gt_loss"]
    groups = {lang: [] for lang in langs}
    for index, loss in losses.items():
        groups[langs[int(index) % len(langs)]].append(np.exp(-loss))
    return {lang: float(np.mean(values)) for lang, values in groups.items()}, len(losses)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("exp_id")
    parser.add_argument("--langs", required=True,
                        help="unlearning languages in table order, e.g. en,fr,ja")
    parser.add_argument("--forget-loss", default="grad_diff")
    args = parser.parse_args()

    langs = args.langs.split(",")
    model = model_dir(args.exp_id, args.forget_loss)
    if not model.is_dir():
        sys.exit(f"no model at {model}")

    rows = []
    for lang in LANGUAGES:
        aggregated = model / f"eval_results/ds_sizeNone_{lang}/eval_log_aggregated.json"
        if aggregated.is_file():
            means, total = group_means(aggregated, langs)
            rows.append((lang, means, total))
    if not rows:
        sys.exit(f"no evaluations under {model}/eval_results")

    print(f"{args.exp_id}: Prob. Forget over {rows[0][2]} items, "
          f"split by the language each item was unlearned in\n")
    print("| eval language | " + " | ".join(f"unlearned in {l}" for l in langs) + " |")
    print("| --- |" + " --- |" * len(langs))
    for lang, means, _ in rows:
        cells = []
        for unlearned_in in langs:
            value = means[unlearned_in]
            marker = "**" if unlearned_in == lang else ""
            cells.append(f"{marker}{value:.4f}{marker}")
        print(f"| {lang} | " + " | ".join(cells) + " |")
    print("\nBold is the diagonal: items unlearned in the language being evaluated.")


if __name__ == "__main__":
    main()
