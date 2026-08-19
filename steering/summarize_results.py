"""Collate results into the two tables the experiment is actually judged on.

    # Table 1 -- are the arms comparable? Run this FIRST.
    python steering/summarize_results.py forget-quality \
        --checkpoints \
          outputs/tofu_finetuned_5epoch_aya_10_lang_2e5/grad_diff_2e-05_forget01_5_en \
          outputs/tofu_finetuned_5epoch_aya_10_lang_2e5/grad_diff_2e-05_forget01_5_en+iw \
          outputs/tofu_finetuned_5epoch_aya_10_lang_2e5/grad_diff_2e-05_forget01_5_iw

    # Table 2 -- did the attack recover anything?
    python steering/summarize_results.py attacks --results-dir steering/results

Table 1 must be read before Table 2 means anything: if arm B unlearned harder than arm A,
its lower recovery is explained by that alone. See steering/README.md.
"""

import argparse
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _fmt_table(headers, rows):
    widths = [max(len(str(h)), *(len(str(r[i])) for r in rows)) if rows else len(str(h))
              for i, h in enumerate(headers)]
    line = "  ".join(str(h).ljust(w) for h, w in zip(headers, widths))
    out = [line, "  ".join("-" * w for w in widths)]
    for r in rows:
        out.append("  ".join(str(c).ljust(w) for c, w in zip(r, widths)))
    return "\n".join(out)


def arm_label(path):
    """Turn a save_dir into a short arm label.

    save_dir is `${forget_loss}_${lr}_${split}_${num_epochs}_${language}`. The loss
    itself can contain underscores (`grad_diff`, `grad_diff_KL`), as can the joint
    language tag (`en+iw` is fine; older joint tags used `joint_en_iw`).
    """
    name = os.path.basename(os.path.normpath(path))
    match = re.match(r"^(.+)_([0-9.eE+-]+)_(forget\d+)_(\d+)_(.+)$", name)
    if match:
        loss, _lr, _split, epochs, lang = match.groups()
        return f"{loss} {lang} {epochs}ep"
    return name


def cmd_forget_quality(args):
    """Forget-side TOFU metrics per checkpoint, for both eval languages.

    Uses aggregate_eval_stat.get_model_utility, which needs no retain model. Only the
    KS-test 'Forget Quality' p-value requires one, and that is not what the arms are
    matched on.
    """
    # imported here, not at module scope, so the `attacks` subcommand does not need hydra
    from aggregate_eval_stat import get_model_utility

    rows = []
    for ckpt in args.checkpoints:
        for lang in args.languages:
            path = os.path.join(ckpt, "eval_results", f"ds_size{args.ds_size}_{lang}",
                                "eval_log_aggregated.json")
            if not os.path.exists(path):
                rows.append([arm_label(ckpt), lang, "MISSING", "-", "-", "-"])
                continue
            util = get_model_utility(json.load(open(path)))
            rows.append([
                arm_label(ckpt), lang,
                f"{util['Prob. Forget']:.4f}",
                f"{util['Truth Ratio Forget']:.4f}",
                f"{util['Prob. Retain']:.4f}",
                f"{util['Model Utility']:.4f}",
            ])

    print("FORGET QUALITY / UTILITY  (arms must be comparable before attacks mean anything)")
    print()
    print(_fmt_table(
        ["arm", "eval lang", "Prob. Forget", "TruthRatio Forget", "Prob. Retain", "Model Utility"],
        rows))
    print()
    print("Lower 'Prob. Forget' = more forgetting. If the arms differ a lot here, fix that")
    print("(tune epochs or lr per arm) before comparing attack recovery.")
    print("Note: ROUGE is deliberately absent -- rouge_score scores 0 for Hebrew. See README.")
    if args.csv:
        _write_csv(args.csv, ["arm", "eval_lang", "prob_forget", "truth_ratio_forget",
                              "prob_retain", "model_utility"], rows)


def cmd_attacks(args):
    paths = sorted(args.files or glob.glob(os.path.join(args.results_dir, "*.json")))
    if not paths:
        print(f"no result files found in {args.results_dir}")
        return

    rows = []
    for p in paths:
        d = json.load(open(p))
        res = d["results"]
        primary = d.get("primary_metric", "rougeL_recall")
        base = res.get("0.0", res.get("0"))
        base_score = base[primary] if base else float("nan")
        best_score = d["best_score"]
        rows.append([
            os.path.basename(p).replace(".json", ""),
            d.get("language", "?"),
            d.get("layer_set", "?"),
            "RANDOM" if d.get("random_vector_control") else "real",
            primary,
            f"{base_score:.4f}",
            str(d["best_alpha"]),
            f"{best_score:.4f}",
            f"{best_score - base_score:+.4f}",
        ])

    print("ATTACK RECOVERY  (best-over-alpha; 'recovery' is best minus the alpha=0 baseline)")
    print()
    print(_fmt_table(
        ["result", "lang", "layers", "vector", "metric", "baseline", "best a", "best", "recovery"],
        rows))
    print()
    print("How to read this:")
    print("  * recovery ~= 0 on a 'real' row      -> the attack failed against that model")
    print("  * recovery > 0 on a 'RANDOM' row too -> the attack is measuring perturbation")
    print("    magnitude, not the unlearning direction. No result; do not report it.")
    print("  * compare English-only (A) vs joint EN+IW (B). If B resists only because it")
    print("    took more optimizer steps (80 rows/epoch vs 40), that is not a language effect.")
    if args.csv:
        _write_csv(args.csv, ["result", "lang", "layers", "vector", "metric", "baseline",
                              "best_alpha", "best", "recovery"], rows)


def _write_csv(path, headers, rows):
    import csv
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(headers)
        w.writerows(rows)
    print(f"\nwrote {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    fq = sub.add_parser("forget-quality", help="per-checkpoint TOFU forget metrics")
    fq.add_argument("--checkpoints", nargs="+", required=True)
    fq.add_argument("--languages", nargs="+", default=["en", "iw"])
    fq.add_argument("--ds-size", default="None",
                    help="the ds_size in the eval save_dir name (default: None)")
    fq.add_argument("--csv")
    fq.set_defaults(func=cmd_forget_quality)

    at = sub.add_parser("attacks", help="attack recovery table")
    at.add_argument("--results-dir", default="steering/results")
    at.add_argument("--files", nargs="*")
    at.add_argument("--csv")
    at.set_defaults(func=cmd_attacks)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
