"""Pre-flight data checks. CPU-only, seconds to run, no model weights.

Run this from the repository root before spending GPU time:

    python steering/check_data.py
    python steering/check_data.py --tokenizer --model-family aya-expanse-8B

Checks:
  1. The Aya forget configs load the expected number of rows, and mixed-language
     batches survive the real collator.
  2. Hebrew unlearning and steering both target dataset/forget01_iw (not the
     perturbed eval file).
  3. Full 10-language fine-tuning sequence lengths and truncation counts.
  4. Answer-token length parity between English and Hebrew. NPO's compute_batch_nll
     sums NLL over tokens, so if one language tokenizes much longer it gets more
     weight in a mixed batch.
"""

import argparse
import ast
from collections import Counter
import os
import statistics
import sys

import datasets
import torch
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_module import TextForgetDatasetQA
from steering.model_utils import DEFAULT_MODEL_FAMILY, REPO_ROOT, ensure_repo_cwd
from utils import get_model_identifiers_from_yaml

REPO = REPO_ROOT
# forget.py / finetune.py hardcode this; it is not a Hydra field.
MAX_LENGTH = 500


def load_cfg(config_name):
    text = open(os.path.join(REPO, "config", config_name)).read()
    text = text.replace("${hydra:runtime.cwd}", REPO)
    return OmegaConf.create(text)


def _real_collator():
    """The actual custom_data_collator_forget, without importing deepspeed."""
    src = open(os.path.join(REPO, "dataloader.py")).read()
    fn = next(n for n in ast.parse(src).body
              if getattr(n, "name", None) == "custom_data_collator_forget")
    ns = {"torch": torch}
    exec(compile(ast.Module([fn], []), "dataloader.py", "exec"), ns)
    return ns["custom_data_collator_forget"]


class _FakeTokenizer:
    """Enough of a tokenizer to exercise shapes without downloading one."""
    eos_token_id = 0

    def tokenize(self, s, **kw):
        return s.split()

    def __call__(self, text, max_length=None, truncation=None, **kw):
        ids = [abs(hash(w)) % 1000 + 1 for w in text.split()]
        if max_length is not None:
            ids = ids[:max_length]
        d = {"input_ids": ids, "attention_mask": [1] * len(ids)}
        return type("E", (), {
            "input_ids": ids,
            "attention_mask": d["attention_mask"],
            "__getitem__": staticmethod(lambda k: d[k]),
        })()


def check_structure(config_name, expected_len):
    cfg = load_cfg(config_name)
    ds = TextForgetDatasetQA(
        cfg.data_path, tokenizer=_FakeTokenizer(), model_family=cfg.model_family,
        max_length=MAX_LENGTH, split=cfg.split, loss_type="npo",
        language=cfg.language, language_mix=cfg.get("language_mix", "concat"),
    )
    if hasattr(ds.forget_data, "column_names") and "language" in ds.forget_data.column_names:
        langs = [ds.forget_data[i]["language"] for i in range(len(ds))]
        mix = {l: langs.count(l) for l in dict.fromkeys(langs)}
    else:
        mix = {str(cfg.language): len(ds)}
    assert len(ds) == expected_len, f"{config_name}: len {len(ds)} != {expected_len}"

    for item in (ds[0], ds[len(ds) - 1]):
        for tensors in item:
            for t in tensors:
                assert t.shape == (MAX_LENGTH,), (
                    f"expected [{MAX_LENGTH}], got {tuple(t.shape)}"
                )

    batch = _real_collator()([ds[0], ds[len(ds) - 1]])
    shapes = []
    for stream in batch:
        seq = stream[0].shape[1]
        assert all(t.shape == (2, seq) for t in stream), (
            f"{config_name}: collated stream shapes disagree: {[tuple(t.shape) for t in stream]}"
        )
        assert seq == MAX_LENGTH, f"{config_name}: seq {seq} != {MAX_LENGTH}"
        shapes.append(tuple(stream[0].shape))
    print(f"  {config_name:28s} len={len(ds):3d}  mix={mix}  "
          f"collates to forget={shapes[0]} retain={shapes[1]} (max_length={MAX_LENGTH})")
    return ds


def check_finetune_dataset(path):
    data = datasets.load_from_disk(os.path.join(REPO, path))["train"]
    mix = Counter(data["language"])
    expected = {lang: 4000 for lang in ("ar", "en", "fa", "fr", "hi", "id", "iw", "ja", "ko", "ru")}
    assert len(data) == 40000, f"fine-tune dataset has {len(data)} rows, expected 40000"
    assert dict(mix) == expected, f"unexpected fine-tune language mix: {dict(mix)}"
    print(f"  {path}: {len(data)} rows, {dict(mix)}")


def check_forget_source(joint_config):
    joint = load_cfg(joint_config)
    path = str(joint.data_path.iw.forget)
    assert path.rstrip("/").endswith("forget01_iw"), (
        f"joint config unlearns on {path!r}, expected dataset/forget01_iw"
    )
    data = datasets.load_from_disk(os.path.join(REPO, "dataset/forget01_iw"))["train"]
    print(f"  joint config forget source: {path}  ({len(data)} rows)  OK")

    iw = load_cfg("forget_iw.yaml")
    iw_path = str(iw.data_path.forget)
    assert iw_path.rstrip("/").endswith("forget01_iw"), (
        f"forget_iw.yaml unlearns on {iw_path!r}, expected dataset/forget01_iw"
    )
    print(f"  forget_iw.yaml forget source: {iw_path}  OK")


def _percentile(values, fraction):
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(fraction * (len(ordered) - 1)))]


def check_finetune_lengths(tokenizer, model_cfg, data_path, candidates):
    data = datasets.load_from_disk(os.path.join(REPO, data_path))["train"]
    all_lengths = []
    by_language = {lang: [] for lang in sorted(set(data["language"]))}

    for start in range(0, len(data), 512):
        rows = data.select(range(start, min(start + 512, len(data))))
        texts = []
        languages = []
        for row in rows:
            lang = row["language"]
            texts.append(
                model_cfg["question_start_tag"][lang]
                + row["question"]
                + model_cfg["question_end_tag"]
                + model_cfg["answer_tag"][lang]
                + row["answer"]
            )
            languages.append(lang)
        encoded = tokenizer(texts, add_special_tokens=True, truncation=False, padding=False)
        for lang, input_ids in zip(languages, encoded["input_ids"]):
            length = len(input_ids)
            all_lengths.append(length)
            by_language[lang].append(length)

    print("  language      mean   p95   p99   max")
    for lang, lengths in by_language.items():
        print(
            f"  {lang:8s} {statistics.mean(lengths):8.1f} "
            f"{_percentile(lengths, .95):5d} {_percentile(lengths, .99):5d} {max(lengths):5d}"
        )
    print(
        f"  {'ALL':8s} {statistics.mean(all_lengths):8.1f} "
        f"{_percentile(all_lengths, .95):5d} {_percentile(all_lengths, .99):5d} "
        f"{max(all_lengths):5d}"
    )
    print("\n  candidate max_length -> rows truncated (length > candidate)")
    safe = []
    for candidate in candidates:
        count = sum(length > candidate for length in all_lengths)
        print(f"  {candidate:4d} -> {count:5d}/{len(all_lengths)} ({100 * count / len(all_lengths):6.2f}%)")
        if count == 0:
            safe.append(candidate)
    if safe:
        print(f"  smallest zero-truncation candidate: {min(safe)}")
    else:
        print("  no candidate has zero truncation; use 500 first or consciously accept/report truncation")


def check_token_parity(tokenizer, model_cfg):
    en_path = os.path.join(REPO, "dataset/forget01_en")
    if not os.path.isdir(en_path):
        print("  skip: dataset/forget01_en is missing. Run `python steering/make_en_datasets.py`.")
        return

    stats = {}
    for lang, path in (("en", "dataset/forget01_en"), ("iw", "dataset/forget01_iw")):
        data = datasets.load_from_disk(os.path.join(REPO, path))["train"]
        lens = [len(tokenizer.tokenize(model_cfg["answer_tag"][lang] + row["answer"])) for row in data]
        stats[lang] = sum(lens) / len(lens)
        print(f"  {lang}: mean answer length {stats[lang]:.1f} tokens over {len(lens)} rows")

    ratio = max(stats.values()) / min(stats.values())
    print(f"  ratio = {ratio:.2f}x")
    if ratio > 1.15:
        print("  WARNING: >15% divergence. NPO's compute_batch_nll sums NLL over tokens,")
        print("           so the longer-tokenizing language dominates mixed batches.")
    else:
        print("  -> within 15%; no length normalization needed. Record this number.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", action="store_true",
                    help="audit Aya token lengths (requires the tokenizer to be cached)")
    ap.add_argument("--model-family", default=DEFAULT_MODEL_FAMILY)
    ap.add_argument("--full-data", default="dataset/full_merged_all_10_lang")
    ap.add_argument("--candidates", type=int, nargs="+", default=[192, 256, 320, 384, 500, 512])
    args = ap.parse_args()

    ensure_repo_cwd()

    configs = (
        ("forget_joint.yaml", 80),
        ("forget_en.yaml", 40),
        ("forget_iw.yaml", 40),
    )

    print("CHECK 1  dataset structure and mixed-language collation")
    for config_name, expected_len in configs:
        check_structure(config_name, expected_len)
    check_finetune_dataset(args.full_data)

    print("\nCHECK 2  forget-source")
    check_forget_source(configs[0][0])

    if args.tokenizer:
        from transformers import AutoTokenizer
        model_cfg = get_model_identifiers_from_yaml(args.model_family)
        tokenizer = AutoTokenizer.from_pretrained(model_cfg["hf_key"])

        print("\nCHECK 3  full fine-tuning sequence-length audit")
        check_finetune_lengths(tokenizer, model_cfg, args.full_data, args.candidates)
        print("\nCHECK 4  answer-token length parity")
        check_token_parity(tokenizer, model_cfg)
    else:
        print("\nCHECKS 3-4 skipped (pass --tokenizer to run them)")

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
