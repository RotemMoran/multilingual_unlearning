"""Materialise the two English TOFU splits that the steering scripts read from disk.

Run once (needs Hub access):

    python steering/make_en_datasets.py

Why this exists: every non-English split in this repo is a translated copy saved with
`save_to_disk`. English is the one language the original unlearning code special-cased
to the Hub (`load_forget_retain` with a string path, `TextDatasetQAStat` when
`language == 'en'`), so `dataset/forget01_en` and `dataset/retain99_en` never existed.

steering/extract_steering_vector.py (`--aux-dataset`), steering/attack_generate.py
(`--forget-dataset`) and check_data.py's token-parity audit all call
`datasets.load_from_disk` with no Hub fallback. Writing the English splits once keeps
a single code path for both languages.
"""

import os
import sys

import datasets

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# (destination directory, TOFU config name, expected rows)
SPLITS = [
    ("dataset/forget01_en", "forget01", 40),
    ("dataset/retain99_en", "retain99", 3960),
]


def main():
    failures = []
    for rel_path, config_name, expected_rows in SPLITS:
        out = os.path.join(REPO, rel_path)
        if os.path.isdir(out):
            existing = datasets.load_from_disk(out)["train"]
            print(f"{rel_path}: already present ({len(existing)} rows), skipping")
            continue

        # load_dataset returns a DatasetDict keyed "train"; saving it whole is what makes
        # load_from_disk(path)["train"] work, exactly like the translated directories.
        dsd = datasets.load_dataset("locuslab/TOFU", config_name)
        rows = len(dsd["train"])
        if rows != expected_rows:
            failures.append(f"{config_name}: got {rows} rows, expected {expected_rows}")
        missing = {"question", "answer"} - set(dsd["train"].column_names)
        if missing:
            failures.append(f"{config_name}: missing column(s) {sorted(missing)}")

        dsd.save_to_disk(out)
        print(f"{rel_path}: wrote {rows} rows, columns={dsd['train'].column_names}")

    if failures:
        print("\nPROBLEMS:", *failures, sep="\n  ")
        sys.exit(1)
    print("\nDone. Re-run `python steering/check_data.py` to verify the arms.")


if __name__ == "__main__":
    main()
