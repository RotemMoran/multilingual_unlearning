"""Extract per-layer unlearning steering vectors (Xiang et al. 2606.03291, Algorithm 1).

    s[l] = sum_x  ( h_un(x)/||h_un(x)||  -  h_ft(x)/||h_ft(x)|| )      at the last prompt token
    g[l] = s[l] / ||s[l]||

The two models are f_ft (the shared finetuned parent) and f_un_aux (an *auxiliary* model
that unlearned a different, disjoint forget set). Neither has seen the forget01 authors
being attacked, so the direction cannot smuggle the answers in: it encodes the unlearning
transformation itself, not the content that was unlearned. attack_generate.py subtracts
alpha * ||h|| * g[l] at inference time.

Three properties of Algorithm 1 that this file is careful to reproduce, because the first
attempt at this experiment got all three wrong and recovered nothing:

* Hidden states are L2-normalised *per sample, per layer, before differencing*. Raw mean
  differences grow ~500x from layer 0 to layer 30 on Aya, so an un-normalised vector makes
  a single alpha mean "imperceptible" early and "destroy the model" late. Normalising here
  and rescaling to ||h|| at injection time is what makes one alpha meaningful everywhere.

* Only the LAST token of the prompt is read (Algorithm 1 line 6; Section 4.3 "the final
  token of the full prompt"). Averaging over answer tokens mixes the suppression signal
  with per-token content and washes out the direction.

* The dataset is the set the auxiliary model actually unlearned. A difference measured on
  data that neither model was trained to suppress captures collateral parameter drift, not
  suppression.

Two implementation notes carried over from the first version, both still load-bearing:

* Activations come from read-only forward hooks on the decoder layers, not from
  `output_hidden_states=True`. The latter is off by one (index 0 is the embedding output)
  and its last entry has `model.norm` applied, so it is NOT the raw output of the last
  decoder block. Injecting a post-norm direction back into pre-norm space would be wrong
  for exactly the late layers this attack targets.

* Models are loaded ONE AT A TIME. Per-sample normalisation happens inside each pass, and
  summing the normalised states separately then subtracting is identical to summing the
  per-sample differences, because both passes walk the same prompts in the same order.
  The run asserts the sample counts match.

Usage:
    python steering/extract_steering_vector.py \
        --base-model ./outputs/tofu_finetuned_5epoch_aya_10_lang_2e5 \
        --unlearned-model ./outputs/tofu_finetuned_5epoch_aya_10_lang_2e5/grad_diff_2e-05_forget01aux_5_en \
        --aux-dataset ./dataset/forget01_aux_en --language en \
        --model-family aya-expanse-8B --out steering/vectors/aya_grad_diff_aux_en.pt
"""

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import datasets

from steering.model_utils import (
    DEFAULT_MODEL_FAMILY,
    INFER_BATCH_SIZE,
    build_prompts,
    ensure_repo_cwd,
    get_decoder_layers,
    load_model,
    load_tokenizer,
)
from utils import get_model_identifiers_from_yaml

# Bumped whenever the extraction convention changes, so attack_generate.py can refuse a
# vector built under the old (un-normalised, answer-token-averaged) scheme instead of
# silently mixing conventions.
LAYER_INDEXING = "decoder_layer_output_lastpos_l2norm"


@torch.no_grad()
def accumulate_normalised_lastpos(model, tokenizer, prompts, batch_size, device):
    """Sum of L2-normalised last-token activations per decoder layer.

    Returns (sums [n_layers, d] in float64, n_samples). sums[i] corresponds to the output
    of decoder layers[i] -- no offset, no final-norm special case -- which is precisely
    the tensor attack_generate.py perturbs.
    """
    layers = get_decoder_layers(model)
    sums = [None] * len(layers)

    def make_capture(i):
        def capture(module, args, output):
            h = output[0] if isinstance(output, tuple) else output      # [B, T, d]
            # Left padding, so -1 is the final real prompt token for every row.
            last = h[:, -1, :].float()
            last = last / last.norm(dim=-1, keepdim=True).clamp_min(1e-9)
            # float64 to accumulate: summing bf16-derived values over thousands of
            # samples loses several significant digits.
            contrib = last.sum(dim=0).double()
            sums[i] = contrib if sums[i] is None else sums[i] + contrib
        return capture

    handles = [layer.register_forward_hook(make_capture(i)) for i, layer in enumerate(layers)]
    n_samples = 0
    try:
        for i in range(0, len(prompts), batch_size):
            chunk = prompts[i:i + batch_size]
            enc = tokenizer(chunk, add_special_tokens=True, return_tensors="pt",
                            padding=True).to(device)
            model(input_ids=enc.input_ids, attention_mask=enc.attention_mask,
                  use_cache=False)
            n_samples += len(chunk)
    finally:
        for h in handles:
            h.remove()

    if sums[0] is None:
        raise RuntimeError("no prompts were processed")
    return torch.stack(sums).cpu(), n_samples


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-model", required=True, help="the finetuned parent, f_ft")
    ap.add_argument("--unlearned-model", required=True,
                    help="auxiliary unlearned checkpoint, f_un_aux")
    ap.add_argument("--aux-dataset", required=True,
                    help="the set f_un_aux unlearned, e.g. ./dataset/forget01_aux_en")
    ap.add_argument("--language", default="en",
                    help="selects the QA tags from config/model_config.yaml")
    ap.add_argument("--model-family", default=DEFAULT_MODEL_FAMILY)
    ap.add_argument("--out", required=True, help="destination .pt file")
    ap.add_argument("--batch-size", type=int, default=INFER_BATCH_SIZE)
    ap.add_argument("--limit", type=int, default=None,
                    help="cap on auxiliary examples (default: all)")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    ensure_repo_cwd()
    model_cfg = get_model_identifiers_from_yaml(args.model_family)
    tokenizer = load_tokenizer(model_cfg, checkpoint=args.base_model)
    # Left padding puts the final prompt token at index -1 for every row in the batch.
    tokenizer.padding_side = "left"

    data = datasets.load_from_disk(args.aux_dataset)["train"]
    if args.limit:
        data = data.select(range(min(args.limit, len(data))))
    prompts = build_prompts(data, model_cfg, args.language)
    print(f"aux dataset: {args.aux_dataset} ({len(prompts)} prompts, language={args.language})")

    print(f"[1/2] base model (f_ft): {args.base_model}")
    base = load_model(args.base_model, model_cfg, args.device)
    sum_base, n_base = accumulate_normalised_lastpos(
        base, tokenizer, prompts, args.batch_size, args.device)
    del base
    torch.cuda.empty_cache()

    print(f"[2/2] auxiliary unlearned model (f_un_aux): {args.unlearned_model}")
    unlearned = load_model(args.unlearned_model, model_cfg, args.device)
    sum_unlearned, n_unlearned = accumulate_normalised_lastpos(
        unlearned, tokenizer, prompts, args.batch_size, args.device)
    del unlearned
    torch.cuda.empty_cache()

    # If these differ the two sums cover different prompts and their difference is noise.
    assert n_base == n_unlearned, (
        f"sample counts diverged ({n_base} vs {n_unlearned}) -- the tokenizer differs "
        "between models, or the prompt list changed between passes"
    )

    s = sum_unlearned - sum_base                                  # [L, d], float64
    # ||mean per-sample difference||. Bounded by 2 since both operands are unit vectors.
    # High values mean the samples agree on a direction; near-zero means they cancel and
    # there is no consistent suppression signal at that layer.
    displacement = (s / n_base).norm(dim=-1)
    vector = (s / s.norm(dim=-1, keepdim=True).clamp_min(1e-12)).float()

    payload = {
        "vector": vector,
        "displacement": displacement.float(),
        "base_model": args.base_model,
        "unlearned_model": args.unlearned_model,
        "aux_dataset": args.aux_dataset,
        "language": args.language,
        "model_family": args.model_family,
        "n_examples": n_base,
        "layer_indexing": LAYER_INDEXING,
        "n_layers": int(vector.shape[0]),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save(payload, args.out)

    print(f"\nsaved {tuple(vector.shape)} to {args.out}  ({n_base} prompts)")
    print("every row is a unit vector; per-layer mean displacement (max 2.0):")
    peak = displacement.max().item()
    for i in range(vector.shape[0]):
        bar = "#" * int(40 * displacement[i].item() / max(peak, 1e-9))
        print(f"  {i:3d}  {displacement[i].item():8.5f}  {bar}")
    print(f"peak displacement at layer {int(displacement.argmax())} ({peak:.5f})")

    summary = args.out.rsplit(".", 1)[0] + "_summary.json"
    with open(summary, "w") as f:
        json.dump({k: v for k, v in payload.items() if not torch.is_tensor(v)} | {
            "displacement": displacement.tolist(),
            "peak_displacement_layer": int(displacement.argmax()),
        }, f, indent=2)
    print(f"summary: {summary}")


if __name__ == "__main__":
    main()
