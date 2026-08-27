"""Build the auxiliary forget/retain pair used to derive the steering vector.

Xiang et al. (arXiv 2606.03291, Section 4.4) never derive the steering direction from the
model under attack. They take the shared finetuned model f_ft, unlearn a *different*
forget set from it to get an auxiliary model f_un_aux, and read the direction off the
per-layer difference between the two. The real forget set is reserved for evaluation, so
the attack never touches the facts it is trying to recover.

That auxiliary forget set is drawn from the retain authors ("randomly shuffling retain set
authors"). This script materialises it:

    dataset/forget01_aux_en    40 rows  -- 2 whole authors lifted out of retain99
    dataset/retain99_aux_en  3920 rows  -- everything else, the retain signal for f_un_aux

Author identity is not a column anywhere in this repo, but TOFU keeps each author's 20 QA
pairs contiguous, so an author is exactly one 20-row block. Lifting whole blocks (rather
than 40 loose rows) is what makes the auxiliary task the same *shape* as the real one:
forget01 is likewise two whole authors.

The seed is load-bearing. The vector is only reproducible if the auxiliary authors are, so
rerunning this must pick the same two blocks; changing --seed invalidates every vector
extracted from the resulting model.

Run once (CPU, seconds):

    python steering/make_aux_forget.py
    python steering/make_aux_forget.py --seed 7 --force    # a different draw
"""

import argparse
import os
import shutil
import sys

import datasets
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PAIRS_PER_AUTHOR = 20          # TOFU: 20 QA pairs per author, contiguous
AUTHORS_TO_LIFT = 2            # matches forget01, which is 2 authors / 40 rows


def block_rows(block_ids):
    """Row indices covered by these 20-row author blocks, ascending."""
    return [b * PAIRS_PER_AUTHOR + off
            for b in sorted(block_ids)
            for off in range(PAIRS_PER_AUTHOR)]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="dataset/retain99_en",
                    help="retain split to carve the auxiliary authors out of")
    ap.add_argument("--forget-out", default="dataset/forget01_aux_en")
    ap.add_argument("--retain-out", default="dataset/retain99_aux_en")
    ap.add_argument("--authors", type=int, default=AUTHORS_TO_LIFT)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--force", action="store_true", help="overwrite existing output dirs")
    args = ap.parse_args()

    source = os.path.join(REPO, args.source)
    forget_out = os.path.join(REPO, args.forget_out)
    retain_out = os.path.join(REPO, args.retain_out)

    if not os.path.isdir(source):
        sys.exit(f"missing {args.source} -- run `python steering/make_en_datasets.py` first")

    for out in (forget_out, retain_out):
        if os.path.isdir(out):
            if not args.force:
                existing = datasets.load_from_disk(out)["train"]
                print(f"{os.path.relpath(out, REPO)}: already present "
                      f"({len(existing)} rows), skipping. Use --force to redraw.")
                return
            shutil.rmtree(out)

    retain = datasets.load_from_disk(source)["train"]
    if len(retain) % PAIRS_PER_AUTHOR:
        sys.exit(f"{args.source} has {len(retain)} rows, not a multiple of "
                 f"{PAIRS_PER_AUTHOR} -- the author blocks are not intact")
    n_authors = len(retain) // PAIRS_PER_AUTHOR

    # Choose author blocks, not rows: sampling loose rows would give the auxiliary model a
    # partially-forgotten author, which is a different unlearning problem than forget01.
    rng = np.random.default_rng(args.seed)
    chosen = sorted(rng.choice(n_authors, size=args.authors, replace=False).tolist())
    forget_idx = block_rows(chosen)
    retain_idx = [i for i in range(len(retain)) if i not in set(forget_idx)]

    forget = retain.select(forget_idx)
    remainder = retain.select(retain_idx)

    # f_un_aux must not see the auxiliary authors in its retain signal, or the forget and
    # retain terms of grad_diff fight over the same facts and the difference we measure
    # afterwards is not a suppression direction.
    overlap = set(forget["question"]) & set(remainder["question"])
    if overlap:
        sys.exit(f"{len(overlap)} question(s) appear in both splits, e.g. "
                 f"{min(overlap)!r}")

    datasets.DatasetDict({"train": forget}).save_to_disk(forget_out)
    datasets.DatasetDict({"train": remainder}).save_to_disk(retain_out)

    print(f"source:      {args.source} ({len(retain)} rows, {n_authors} authors)")
    print(f"seed:        {args.seed}")
    print(f"author blocks lifted: {chosen} -> rows "
          f"{forget_idx[0]}-{forget_idx[len(forget_idx) // args.authors - 1]}, ...")
    print(f"\n{args.forget_out}: {len(forget)} rows")
    for row in forget.select(range(min(3, len(forget)))):
        print(f"    Q: {row['question'][:88]}")
    print(f"\n{args.retain_out}: {len(remainder)} rows")
    print("\nNext: train the auxiliary model with")
    print("    python forget.py --config-name forget_aux_en")


if __name__ == "__main__":
    main()
