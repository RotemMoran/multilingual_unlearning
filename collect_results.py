#!/usr/bin/env python
"""Turn the per-experiment evaluation CSVs into markdown matrices for EXPERIMENTS.md.

    python collect_results.py                       # every experiment found under results/
    python collect_results.py en+fr+ja_third_5ep    # only these experiments
    python collect_results.py --metric "Model Utility"

Each row is a model, each column an evaluation language. Without --metric the three
metrics that matter for cross-lingual unlearning are printed in turn: how much of the
forget set survives, how much of the retain set survives, and overall Model Utility.
"""
import argparse
import csv
import re
import sys
from pathlib import Path

RESULT_ROOT = Path(__file__).resolve().parent / "results"
LANGUAGES = ["en", "fr", "ar", "fa", "hi", "id", "iw", "ja", "ko", "ru"]
DEFAULT_METRICS = ["Prob. Forget", "Prob. Retain", "Model Utility"]

# results/<exp_id>/eval_<lang>.csv is the current layout; the two others are the
# hand-named files from the first English/Hebrew batch
LAYOUTS = [
    (re.compile(r"^eval_(?P<lang>[a-z]{2})$"), lambda d, m: (d, m["lang"])),
    (re.compile(r"^unlearn_(?P<tag>.+)_eval_(?P<lang>[a-z]{2})$"),
     lambda d, m: (m["tag"], m["lang"])),
    (re.compile(r"^finetune_result01_unlearn_grad_diff_2e-05_(?P<lang>[a-z]{2})$"),
     lambda d, m: ("en", m["lang"])),
]


def read_row(path):
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    return rows[0] if rows else None


def collect():
    """Return {experiment: {language: {metric: value}}}."""
    table = {}
    for path in sorted(RESULT_ROOT.rglob("*.csv")):
        directory = path.parent.name if path.parent != RESULT_ROOT else ""
        for pattern, name in LAYOUTS:
            match = pattern.match(path.stem)
            if not match:
                continue
            experiment, lang = name(directory, match)
            row = read_row(path)
            if row is not None:
                table.setdefault(experiment, {})[lang] = row
            break
    return table


def fmt(value):
    if value is None:
        return "-"
    value = float(value)
    return f"{value:.2e}" if 0 < value < 0.001 else f"{value:.4f}"


def print_matrix(table, experiments, metric):
    languages = [lang for lang in LANGUAGES
                 if any(lang in table[e] for e in experiments)]
    print(f"\n#### {metric}\n")
    print("| model | " + " | ".join(languages) + " |")
    print("| --- |" + " --- |" * len(languages))
    for experiment in experiments:
        cells = [fmt(table[experiment].get(lang, {}).get(metric)) for lang in languages]
        print(f"| `{experiment}` | " + " | ".join(cells) + " |")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("experiments", nargs="*", help="experiment ids, default all")
    parser.add_argument("--metric", action="append", dest="metrics",
                        help="metric column to print, repeatable")
    args = parser.parse_args()

    table = collect()
    if not table:
        sys.exit(f"no result CSVs under {RESULT_ROOT}")

    experiments = args.experiments or sorted(table)
    missing = [e for e in experiments if e not in table]
    if missing:
        sys.exit(f"no results for {', '.join(missing)}; have {', '.join(sorted(table))}")

    for metric in args.metrics or DEFAULT_METRICS:
        print_matrix(table, experiments, metric)


if __name__ == "__main__":
    main()
