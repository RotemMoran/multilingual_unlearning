"""Inference-time steering-vector recovery attack (Xiang et al. 2606.03291, Algorithm 2).

At every decoding step, for each layer l in a window [c, c+N]:

    h <- h - alpha * ||h||_2 * g[l]          applied to the last position only

Generates on the TOFU forget set under that intervention to test whether knowledge
suppressed by unlearning can be recovered without touching the weights.

Three details that distinguish this from a naive activation edit, all of which the first
attempt at this experiment got wrong:

* The direction is rescaled to the *current activation norm* at every position and layer.
  g[l] is a unit vector (see extract_steering_vector.py), so alpha is a fraction of ||h||
  and means the same thing at layer 3 and layer 30. Subtracting a raw difference vector
  instead makes small alphas no-ops early and catastrophic late.

* Only the last position is perturbed (Algorithm 2 line 14 writes H(l)[t], where t is the
  position being generated). Broadcasting the subtraction across the whole sequence also
  rewrites the prompt on the prefill pass, which is a different and much blunter edit.

* The injection window is 3 layers (N=2) and its start layer is SWEPT, not fixed. The
  suppression signal is localised; hitting a fixed block of late layers can miss it
  entirely. `--mode sweep` reproduces the c = 1..L-N loop of Algorithm 2.

Forward hooks are used because the perturbation has to be applied *during* autoregressive
decoding, not just on one forward pass. The hook works unchanged under KV caching: after
the first step it sees T == 1 and "the last position" is the token being generated.

Scoring here is lexical (ROUGE-L recall for English, chrF elsewhere). The paper's headline
metric is an NLI equivalence score; add it afterwards with

    python steering/nli_score.py steering/results_v2/<file>.json

which is a separate pass so generations can be re-scored without regenerating them.

Usage:
    # Algorithm 2: sweep the window start at the paper's alpha
    python steering/attack_generate.py \
        --model ./outputs/.../grad_diff_2e-05_forget01_5_en \
        --vector steering/vectors/aya_grad_diff_aux_en.pt \
        --forget-dataset ./dataset/forget01_en --language en \
        --mode sweep --alphas 0.5 \
        --out steering/results_v2/en_en_sweep.json

    # Appendix K.3: all layers at once, small alpha, no window hyperparameter
    python steering/attack_generate.py ... --mode all-layers --alphas 0.05 0.1 0.2

Controls (each answers a specific "but maybe..." objection):
    alpha 0 is always run    no-op; reproduces the unattacked model
    --random-vector          unit-norm Gaussian, same alpha and same ||h|| scaling
    --model <base model>     vector applied to f_ft; should degrade it, not help
"""

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import datasets

from steering.metrics import chrf_scores, rouge_recall
from steering.model_utils import (
    DEFAULT_MODEL_FAMILY,
    DEFAULT_WINDOW,
    INFER_BATCH_SIZE,
    build_prompts,
    ensure_repo_cwd,
    get_decoder_layers,
    load_model,
    load_tokenizer,
    sweep_starts,
    window_layers,
)
from utils import get_model_identifiers_from_yaml

EXPECTED_LAYER_INDEXING = "decoder_layer_output_lastpos_l2norm"


def make_hook(unit_direction, alpha):
    """Subtract alpha * ||h|| * g from the last position of this layer's output.

    On the prefill pass T is the prompt length and the last position is the final prompt
    token; on every subsequent step KV caching makes T == 1 and it is the token being
    generated. Both are "H(l)[t]" in Algorithm 2.
    """
    def hook(module, args, output):
        is_tuple = isinstance(output, tuple)
        hs = output[0] if is_tuple else output                  # [B, T, d]
        g = unit_direction.to(device=hs.device, dtype=hs.dtype)
        hs = hs.clone()                                         # generate() reuses buffers
        last = hs[:, -1:, :]                                    # [B, 1, d]
        hs[:, -1:, :] = last - alpha * last.norm(dim=-1, keepdim=True) * g
        return (hs,) + output[1:] if is_tuple else hs
    return hook


@torch.no_grad()
def generate(model, tokenizer, prompts, batch_size, max_new_tokens):
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


def run_condition(model, tokenizer, prompts, answers, vector, layers, alpha,
                  batch_size, max_new_tokens):
    """Generate under one (alpha, layers) setting and score it lexically."""
    decoder_layers = get_decoder_layers(model)
    handles = []
    try:
        if alpha != 0 and layers:
            for i in layers:
                # vector[i] is the direction measured at the output of layers[i] by the
                # same hook point in extract_steering_vector.py, so it is subtracted in
                # exactly the space it was measured in.
                handles.append(decoder_layers[i].register_forward_hook(
                    make_hook(vector[i], alpha)))
        gen = generate(model, tokenizer, prompts, batch_size, max_new_tokens)
    finally:
        for h in handles:
            h.remove()

    rouge = sum(rouge_recall(gen, answers)["rougeL_recall"].values()) / len(gen)
    chrf = sum(chrf_scores(gen, answers).values()) / len(gen)
    return gen, rouge, chrf


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="model under attack (usually unlearned)")
    ap.add_argument("--vector", help=".pt from extract_steering_vector.py; optional when "
                                    "every alpha is 0 (a plain reference generation pass)")
    ap.add_argument("--forget-dataset", required=True, help="load_from_disk path")
    ap.add_argument("--language", default="en")
    ap.add_argument("--model-family", default=DEFAULT_MODEL_FAMILY)
    ap.add_argument("--out", required=True)
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.5],
                    help="steering strengths; alpha=0 is always added as the baseline")
    ap.add_argument("--mode", default="sweep", choices=("sweep", "window", "all-layers"),
                    help="sweep: Algorithm 2 start-layer loop; window: one fixed window; "
                         "all-layers: the Appendix K.3 variant (use a small alpha)")
    ap.add_argument("--start-layer", type=int, default=None,
                    help="window start for --mode window")
    ap.add_argument("--window", type=int, default=DEFAULT_WINDOW,
                    help="N in [c, c+N]; the paper uses 2, i.e. 3 layers")
    ap.add_argument("--stride", type=int, default=1, help="start-layer stride for sweep")
    ap.add_argument("--random-vector", action="store_true",
                    help="CONTROL: unit-norm Gaussian instead of the extracted direction")
    ap.add_argument("--batch-size", type=int, default=INFER_BATCH_SIZE)
    ap.add_argument("--max-new-tokens", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    ensure_repo_cwd()
    torch.manual_seed(args.seed)
    model_cfg = get_model_identifiers_from_yaml(args.model_family)
    tokenizer = load_tokenizer(model_cfg, checkpoint=args.model)
    # Left padding: required for batched decoder-only generation, and it puts the final
    # prompt token at index -1, which is where the vector was measured.
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    alphas = [a for a in args.alphas if a != 0]
    if args.vector is None:
        if alphas:
            raise SystemExit("--vector is required unless every --alphas entry is 0")
        payload, vector = {}, None
        print("no vector: reference generation pass (alpha=0 only)")
    else:
        payload = torch.load(args.vector, map_location="cpu", weights_only=False)
        vector_family = payload.get("model_family")
        if vector_family is not None and vector_family != args.model_family:
            raise ValueError(
                f"vector was extracted with model_family={vector_family!r}, but the attack "
                f"was launched with {args.model_family!r}"
            )
        indexing = payload.get("layer_indexing")
        if indexing != EXPECTED_LAYER_INDEXING:
            raise ValueError(
                f"vector has layer_indexing={indexing!r}, expected "
                f"{EXPECTED_LAYER_INDEXING!r}. This attack rescales a UNIT direction to "
                f"||h||; a vector saved under the old un-normalised, answer-token-averaged "
                f"convention would make alpha meaningless. Re-extract it."
            )
        vector = payload["vector"]
        vector = vector / vector.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        if args.random_vector:
            # Matched norm and scale: a unit Gaussian direction injected with the same
            # alpha and the same ||h|| rescaling. If this recovers the answers too, the
            # attack is measuring "a perturbation of this size" and there is no result.
            rand = torch.randn_like(vector)
            vector = rand / rand.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            print("CONTROL: using a unit-norm RANDOM vector")

    data = datasets.load_from_disk(args.forget_dataset)["train"]
    prompts = build_prompts(data, model_cfg, args.language)
    answers = [row["answer"] for row in data]
    print(f"forget set: {args.forget_dataset} ({len(prompts)} questions, "
          f"language={args.language})")

    model = load_model(args.model, model_cfg, args.device)
    n_layers = len(get_decoder_layers(model))
    if vector is not None and vector.shape[0] != n_layers:
        raise ValueError(
            f"vector has {vector.shape[0]} rows but the model has {n_layers} decoder "
            f"layers; it was extracted from a different model"
        )

    if args.mode == "all-layers":
        windows = [("all", list(range(n_layers)))]
    elif args.mode == "window":
        if args.start_layer is None:
            raise SystemExit("--mode window requires --start-layer")
        windows = [(args.start_layer, window_layers(args.start_layer, args.window, n_layers))]
    else:
        windows = [(c, window_layers(c, args.window, n_layers))
                   for c in sweep_starts(args.window, n_layers, args.stride)]

    print(f"model has {n_layers} layers; mode={args.mode}, window=N+1="
          f"{args.window + 1}, {len(windows)} window(s) x {len(alphas)} alpha(s) "
          f"+ 1 baseline = {len(windows) * len(alphas) + 1} generation passes")

    # chrF is the headline for non-Latin scripts: it is character-based, so unlike
    # word-level ROUGE it stays meaningful under Hebrew morphology.
    primary = "rougeL_recall" if args.language == "en" else "chrf"
    print(f"scoring on '{primary}' (language={args.language})\n")

    started = time.time()
    # alpha=0 needs no hook, so the baseline is independent of window and alpha.
    gen, rouge, chrf = run_condition(model, tokenizer, prompts, answers, vector, [], 0,
                                     args.batch_size, args.max_new_tokens)
    baseline = {"alpha": 0.0, "start_layer": None, "layers": [], "rougeL_recall": rouge,
                "chrf": chrf, "generations": gen}
    print(f"  baseline (alpha=0)          ROUGE-L={rouge:.4f}  chrF={chrf:.4f}")

    conditions = [baseline]
    for alpha in alphas:
        for start, layers in windows:
            gen, rouge, chrf = run_condition(model, tokenizer, prompts, answers, vector,
                                             layers, alpha, args.batch_size,
                                             args.max_new_tokens)
            conditions.append({"alpha": alpha, "start_layer": start, "layers": layers,
                               "rougeL_recall": rouge, "chrf": chrf, "generations": gen})
            delta = rouge - baseline["rougeL_recall"] if primary == "rougeL_recall" \
                else chrf - baseline["chrf"]
            print(f"  alpha={alpha:<5} start={start!s:<4} layers={layers[0]}-"
                  f"{layers[-1]:<3} ROUGE-L={rouge:.4f}  chrF={chrf:.4f}  "
                  f"({primary} {delta:+.4f})")

    steered = [c for c in conditions if c["alpha"] != 0]
    best = max(steered, key=lambda c: c[primary]) if steered else baseline

    out_payload = {
        "model": args.model,
        "vector": args.vector,
        "vector_source": {k: payload.get(k) for k in
                          ("base_model", "unlearned_model", "aux_dataset", "language")},
        "forget_dataset": args.forget_dataset,
        "language": args.language,
        "mode": args.mode,
        "window": args.window,
        "n_layers": n_layers,
        "random_vector_control": args.random_vector,
        "questions": [row["question"] for row in data],
        "gold_answers": answers,
        "primary_metric": primary,
        "baseline": baseline[primary],
        "conditions": conditions,
        # The attacker-optimal setting: a model must not look robust merely because its
        # best (alpha, layer) sits elsewhere on the grid.
        "best_alpha": best["alpha"],
        "best_start_layer": best["start_layer"],
        "best_layers": best["layers"],
        "best_score": best[primary],
        "recovery": best[primary] - baseline[primary],
        "elapsed_sec": round(time.time() - started, 1),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out_payload, f, indent=2, ensure_ascii=False)

    print(f"\nbaseline: {primary}={baseline[primary]:.4f}")
    print(f"best:     {primary}={best[primary]:.4f} at alpha={best['alpha']} "
          f"start={best['start_layer']} (recovery {out_payload['recovery']:+.4f})")
    print(f"saved -> {args.out}  ({out_payload['elapsed_sec']}s)")


if __name__ == "__main__":
    main()
