"""Inference-time steering-vector recovery attack.

Generates on the TOFU forget set while subtracting the unlearning steering vector from
the residual stream at chosen decoder layers, to test whether knowledge suppressed by
unlearning can be recovered without touching the weights:

    h[layer] <- h[layer] - alpha * v[layer]

PyTorch forward hooks are used because the perturbation has to be applied *during*
autoregressive decoding, not just on a single forward pass. The hook works unchanged
under KV caching: after the first step it simply sees T == 1 and the subtraction
broadcasts.

Attack success is ROUGE-L recall of the generation against the gold forget answer -- the
same quantity `get_model_utility()` reports as Forget ROUGE, so results are directly
comparable to the repo's existing eval numbers.

Usage:
    python steering/attack_generate.py \
        --model ./outputs/tofu_finetuned_5epoch_aya_10_lang_2e5/grad_diff_2e-05_forget01_5_en \
        --vector steering/vectors/aya_grad_diff_en.pt \
        --forget-dataset ./dataset/forget01_en --language en \
        --alphas 0 0.5 1 2 4 --layer-set late \
        --out steering/results/aya_grad_diff_en_attack.json

    python steering/attack_generate.py \
        --model ./outputs/tofu_finetuned_5epoch_aya_10_lang_2e5/grad_diff_2e-05_forget01_5_iw \
        --vector steering/vectors/aya_grad_diff_iw.pt \
        --forget-dataset ./dataset/forget01_iw --language iw \
        --alphas 0 0.5 1 2 4 --layer-set late \
        --out steering/results/aya_grad_diff_iw_attack.json

Controls (each answers a specific "but maybe..." objection):
    --alphas 0            no-op; must reproduce the unattacked model
    --random-vector       norm-matched random direction; must NOT recover
    --model <base model>  vector applied to the base model; must degrade it
"""

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import datasets

from steering.metrics import chrf_scores, rouge_recall
from steering.model_utils import (
    DEFAULT_MODEL_FAMILY,
    INFER_BATCH_SIZE,
    LAYER_SETS,
    ensure_repo_cwd,
    get_decoder_layers,
    load_model,
    load_tokenizer,
)
from utils import get_model_identifiers_from_yaml


def make_hook(vec, alpha, normalize):
    """Subtract alpha * vec from this layer's residual-stream output."""
    def hook(module, args, output):
        is_tuple = isinstance(output, tuple)
        hs = output[0] if is_tuple else output              # [B, T, d]
        delta = vec.to(device=hs.device, dtype=hs.dtype)
        if normalize:
            # Scale the direction to each position's own activation norm, so one alpha
            # means the same relative push at every layer and every token.
            delta = delta / delta.norm().clamp_min(1e-9) * hs.norm(dim=-1, keepdim=True)
        hs = hs - alpha * delta
        return (hs,) + output[1:] if is_tuple else hs
    return hook


def build_prompts(data, model_configs, language):
    """Prompt exactly as the model was trained to see it.

    Built from the tags in model_config.yaml rather than by decoding input_ids and
    splitting on "Answer: " the way run_generation does -- that round-trip raises
    IndexError whenever the split symbol and the language disagree.
    """
    q_start = model_configs["question_start_tag"][language]
    q_end = model_configs["question_end_tag"]
    a_tag = model_configs["answer_tag"][language]
    return [q_start + row["question"] + q_end + a_tag for row in data]


@torch.no_grad()
def generate(model, tokenizer, prompts, batch_size, max_new_tokens):
    tokenizer.padding_side = "left"          # required for batched decoder-only generation
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    outputs = []
    for i in range(0, len(prompts), batch_size):
        chunk = prompts[i:i + batch_size]
        enc = tokenizer(chunk, add_special_tokens=True, return_tensors="pt",
                        padding=True).to(model.device)
        out = model.generate(
            enc.input_ids, attention_mask=enc.attention_mask,
            max_new_tokens=max_new_tokens, do_sample=False, use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
        )
        outputs.extend(tokenizer.batch_decode(out[:, enc.input_ids.shape[-1]:],
                                              skip_special_tokens=True))
    return outputs


def run_condition(model, tokenizer, prompts, answers, vector, layers, alpha, normalize,
                  batch_size, max_new_tokens):
    """Generate under one (alpha, layers) setting and score it."""
    decoder_layers = get_decoder_layers(model)
    handles = []
    try:
        if alpha != 0 and vector is not None:
            for i in layers:
                # vector[i] is the mean output of layers[i], captured by the same
                # read-only hook point in extract_steering_vector.py -- so the direction
                # is subtracted in exactly the space it was measured in.
                handles.append(decoder_layers[i].register_forward_hook(
                    make_hook(vector[i], alpha, normalize)))
        gen = generate(model, tokenizer, prompts, batch_size, max_new_tokens)
    finally:
        for h in handles:
            h.remove()

    scores = rouge_recall(gen, answers)
    rouge = sum(scores["rougeL_recall"].values()) / len(gen)
    chrf = sum(chrf_scores(gen, answers).values()) / len(gen)
    return gen, rouge, chrf


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="model under attack (usually unlearned)")
    ap.add_argument("--vector", required=True, help=".pt from extract_steering_vector.py")
    ap.add_argument("--forget-dataset", required=True, help="load_from_disk path")
    ap.add_argument("--language", default="en")
    ap.add_argument("--model-family", default=DEFAULT_MODEL_FAMILY)
    ap.add_argument("--out", required=True)
    ap.add_argument("--alphas", type=float, nargs="+", default=[0, 0.5, 1, 2, 4])
    ap.add_argument("--layer-set", default="late", choices=sorted(LAYER_SETS))
    ap.add_argument("--normalize", action="store_true",
                    help="scale the direction to each position's activation norm")
    ap.add_argument("--random-vector", action="store_true",
                    help="CONTROL: replace v with a norm-matched random direction")
    ap.add_argument("--batch-size", type=int, default=INFER_BATCH_SIZE)
    ap.add_argument("--max-new-tokens", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    ensure_repo_cwd()
    torch.manual_seed(args.seed)
    model_cfg = get_model_identifiers_from_yaml(args.model_family)
    tokenizer = load_tokenizer(model_cfg, checkpoint=args.model)

    payload = torch.load(args.vector, map_location="cpu", weights_only=False)
    vector_family = payload.get("model_family")
    if vector_family is not None and vector_family != args.model_family:
        raise ValueError(
            f"vector was extracted with model_family={vector_family!r}, but the attack "
            f"was launched with {args.model_family!r}"
        )
    vector = payload["vector"]
    if args.random_vector:
        # Same per-layer norm, random direction: if this recovers the answers too, the
        # attack is measuring "perturbation of this size" rather than the unlearning
        # direction, and there is no result.
        rand = torch.randn_like(vector)
        rand = rand / rand.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        vector = rand * payload["per_layer_norm"].unsqueeze(-1)
        print("CONTROL: using a norm-matched RANDOM vector")

    data = datasets.load_from_disk(args.forget_dataset)["train"]
    prompts = build_prompts(data, model_cfg, args.language)
    answers = [row["answer"] for row in data]
    print(f"forget set: {args.forget_dataset} ({len(prompts)} questions, language={args.language})")

    model = load_model(args.model, model_cfg, args.device)

    n_layers = len(get_decoder_layers(model))
    layers = LAYER_SETS[args.layer_set](n_layers)
    if vector.shape[0] != n_layers:
        raise ValueError(
            f"vector has {vector.shape[0]} rows but the model has {n_layers} decoder "
            f"layers; the vector was extracted from a different model, or with an older "
            f"version of extract_steering_vector.py (expected layer_indexing="
            f"'decoder_layer_output', got {payload.get('layer_indexing', 'unknown')!r})"
        )
    print(f"model has {n_layers} layers; injecting at '{args.layer_set}' = {layers}")

    # chrF is the headline for non-Latin scripts: it is character-based, so unlike
    # word-level ROUGE it stays meaningful under Hebrew morphology.
    primary = "rougeL_recall" if args.language == "en" else "chrf"
    print(f"scoring on '{primary}' (language={args.language})")

    results = {}
    for alpha in args.alphas:
        gen, rouge, chrf = run_condition(model, tokenizer, prompts, answers, vector,
                                         layers, alpha, args.normalize, args.batch_size,
                                         args.max_new_tokens)
        results[str(alpha)] = {"rougeL_recall": rouge, "chrf": chrf, "generations": gen}
        print(f"  alpha={alpha:<5} ROUGE-L recall = {rouge:.4f}   chrF = {chrf:.4f}")

    best = max(results, key=lambda a: results[a][primary])
    payload_out = {
        "model": args.model,
        "vector": args.vector,
        "vector_source": {k: payload.get(k) for k in
                          ("base_model", "unlearned_model", "aux_dataset", "language")},
        "forget_dataset": args.forget_dataset,
        "language": args.language,
        "layer_set": args.layer_set,
        "layers": layers,
        "normalize": args.normalize,
        "random_vector_control": args.random_vector,
        "questions": [row["question"] for row in data],
        "gold_answers": answers,
        "results": results,
        "primary_metric": primary,
        # The attacker-optimal number: a model must not look robust merely because its
        # best alpha sits elsewhere on the grid.
        "best_alpha": float(best),
        "best_score": results[best][primary],
        "best_rougeL_recall": results[best]["rougeL_recall"],
        "best_chrf": results[best]["chrf"],
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload_out, f, indent=2, ensure_ascii=False)

    baseline = results.get("0.0", results.get("0"))
    print(f"\nbest:     alpha={best} {primary}={results[best][primary]:.4f}")
    if baseline:
        print(f"baseline: alpha=0 {primary}={baseline[primary]:.4f}  "
              f"(recovery = {results[best][primary] - baseline[primary]:+.4f})")
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
