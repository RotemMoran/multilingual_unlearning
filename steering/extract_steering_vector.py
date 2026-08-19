"""Extract an unlearning steering vector from a pair of models.

The vector is the per-layer difference in mean residual-stream activation between an
unlearned model and the base finetuned model it came from, measured over an auxiliary
dataset:

    v[layer] = mean_unlearned(h[layer]) - mean_base(h[layer])

Subtracting alpha * v from the unlearned model at inference time pushes its activations
back toward the base model -- that is the recovery attack in attack_generate.py.

Three implementation notes that matter:

* No TransformerLens is required. Core PyTorch forward hooks work for Aya/Cohere
  checkpoints (decoder blocks at `model.layers`).

* Activations are captured with read-only hooks on the decoder layers rather than with
  `output_hidden_states=True`, so that vector[i] is *exactly* the output of layers[i] --
  the same tensor attack_generate.py perturbs. `output_hidden_states` is off by one
  (hidden_states[0] is the embedding output) and, worse, its last entry has the final
  `model.norm` applied, so hidden_states[-1] is NOT the raw output of the last decoder
  layer. Injecting that post-norm direction back into pre-norm space would be wrong for
  exactly the late layers this attack targets. Verified empirically against a real HF
  model; see steering/README.md.

* The models are loaded ONE AT A TIME. We only need means, and
  mean(unlearned) - mean(base) == mean(unlearned - base) as long as both are averaged over
  the same token set, so there is no reason to hold two models in memory at once. The
  token sets are identical because both passes use the same tokenizer and an unshuffled
  loader; the run asserts the counts match.

Usage:
    python steering/extract_steering_vector.py \
        --base-model ./outputs/tofu_finetuned_5epoch_aya_10_lang_2e5 \
        --unlearned-model ./outputs/tofu_finetuned_5epoch_aya_10_lang_2e5/grad_diff_2e-05_forget01_5_en \
        --aux-dataset ./dataset/retain99_en --language en \
        --model-family aya-expanse-8B --out steering/vectors/aya_grad_diff_en.pt

    python steering/extract_steering_vector.py \
        --base-model ./outputs/tofu_finetuned_5epoch_aya_10_lang_2e5 \
        --unlearned-model ./outputs/tofu_finetuned_5epoch_aya_10_lang_2e5/grad_diff_2e-05_forget01_5_iw \
        --aux-dataset ./dataset/retain99_iw --language iw \
        --model-family aya-expanse-8B --out steering/vectors/aya_grad_diff_iw.pt
"""

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_module import TextDatasetQAEval, custom_data_collator_with_indices
from steering.model_utils import (
    DEFAULT_MODEL_FAMILY,
    INFER_BATCH_SIZE,
    ensure_repo_cwd,
    get_decoder_layers,
    load_model,
    load_tokenizer,
)
from utils import get_model_identifiers_from_yaml


def build_loader(aux_dataset, tokenizer, model_family, language, max_length, batch_size,
                 limit=None):
    """Auxiliary-data loader. Unshuffled: both models must see identical batches."""
    ds = TextDatasetQAEval(
        aux_dataset, tokenizer=tokenizer, model_family=model_family,
        max_length=max_length, question_key="question", answer_key="answer",
        language=language,
    )
    if limit:
        ds.data = ds.data.select(range(min(limit, len(ds.data))))
    return torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        collate_fn=custom_data_collator_with_indices,
    )


@torch.no_grad()
def accumulate_layer_means(model, loader, device):
    """Mean output of each decoder layer over answer tokens.

    Returns (means [n_layers, d], token_count), indexed identically to the decoder
    ModuleList: means[i] is the output of layers[i], which is precisely the tensor
    attack_generate.py subtracts from.
    """
    layers = get_decoder_layers(model)
    sums = [None] * len(layers)
    state = {"mask": None, "count": 0}

    def make_capture(i):
        def capture(module, args, output):
            h = output[0] if isinstance(output, tuple) else output   # [B, T, d]
            # fp32 for the masked product, float64 to accumulate across batches:
            # summing bf16 over ~10^5 tokens loses several significant digits.
            contrib = (h.float() * state["mask"]).sum(dim=(0, 1)).double()
            sums[i] = contrib if sums[i] is None else sums[i] + contrib
        return capture

    handles = [layer.register_forward_hook(make_capture(i)) for i, layer in enumerate(layers)]
    try:
        for input_ids, labels, attention_mask, _ in loader:
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            attention_mask = attention_mask.to(device)

            # `labels != -100` is exactly the answer tokens:
            # convert_raw_data_to_model_format sets question tokens and padding to -100.
            # Masking here also avoids the flash-attention hazard where padded positions
            # hold undefined values (the same reason dataloader.py masks before its KL terms).
            state["mask"] = (labels != -100).unsqueeze(-1)
            model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
            state["count"] += int(state["mask"].squeeze(-1).sum().item())
    finally:
        for h in handles:
            h.remove()

    if sums[0] is None:
        raise RuntimeError("auxiliary loader produced no batches")
    if state["count"] == 0:
        raise RuntimeError("no answer tokens found (labels were all -100)")
    return torch.stack([s / state["count"] for s in sums]).float().cpu(), state["count"]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-model", required=True, help="base finetuned model")
    ap.add_argument("--unlearned-model", required=True, help="unlearned checkpoint")
    ap.add_argument("--aux-dataset", required=True,
                    help="load_from_disk path, e.g. ./dataset/retain99_en")
    ap.add_argument("--language", default="en", help="selects the QA tags from model_config.yaml")
    ap.add_argument("--model-family", default=DEFAULT_MODEL_FAMILY)
    ap.add_argument("--out", required=True, help="destination .pt file")
    ap.add_argument("--batch-size", type=int, default=INFER_BATCH_SIZE)
    ap.add_argument("--max-length", type=int, default=500)
    ap.add_argument("--limit", type=int, default=None,
                    help="cap on auxiliary examples (default: all)")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    ensure_repo_cwd()
    model_cfg = get_model_identifiers_from_yaml(args.model_family)
    tokenizer = load_tokenizer(model_cfg, checkpoint=args.base_model)

    loader = build_loader(args.aux_dataset, tokenizer, args.model_family, args.language,
                          args.max_length, args.batch_size, args.limit)
    print(f"aux dataset: {args.aux_dataset} ({len(loader.dataset)} examples, language={args.language})")

    print(f"[1/2] base model: {args.base_model}")
    base = load_model(args.base_model, model_cfg, args.device)
    mu_base, n_base = accumulate_layer_means(base, loader, args.device)
    del base
    torch.cuda.empty_cache()

    print(f"[2/2] unlearned model: {args.unlearned_model}")
    unlearned = load_model(args.unlearned_model, model_cfg, args.device)
    mu_unlearned, n_unlearned = accumulate_layer_means(unlearned, loader, args.device)
    del unlearned
    torch.cuda.empty_cache()

    # If these differ the two means were taken over different token sets and their
    # difference is meaningless.
    assert n_base == n_unlearned, (
        f"token counts diverged ({n_base} vs {n_unlearned}) -- the tokenizer differs "
        "between models or the loader was shuffled"
    )

    vector = mu_unlearned - mu_base                      # [L, d]
    per_layer_norm = vector.norm(dim=-1)

    payload = {
        "vector": vector,
        "per_layer_norm": per_layer_norm,
        "base_mean_norm": mu_base.norm(dim=-1),
        "base_model": args.base_model,
        "unlearned_model": args.unlearned_model,
        "aux_dataset": args.aux_dataset,
        "language": args.language,
        "model_family": args.model_family,
        "n_tokens": n_base,
        "n_examples": len(loader.dataset),
        # vector[i] is the output of decoder layers[i]. No offset, no final-norm special
        # case: attack_generate.py applies vector[i] at layers[i] directly.
        "layer_indexing": "decoder_layer_output",
        "n_layers": int(vector.shape[0]),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save(payload, args.out)

    print(f"\nsaved {tuple(vector.shape)} to {args.out}  ({n_base} answer tokens)")
    print("per-layer ||v|| (layer: norm, relative to base activation norm):")
    rel = (per_layer_norm / payload["base_mean_norm"].clamp_min(1e-9))
    for i in range(vector.shape[0]):
        bar = "#" * int(40 * rel[i] / max(rel.max().item(), 1e-9))
        print(f"  {i:3d}  {per_layer_norm[i]:10.4f}  {rel[i]:7.4f}  {bar}")

    summary = args.out.rsplit(".", 1)[0] + "_summary.json"
    with open(summary, "w") as f:
        json.dump({k: v for k, v in payload.items()
                   if not torch.is_tensor(v)} | {
                       "per_layer_norm": per_layer_norm.tolist(),
                       "relative_norm": rel.tolist(),
                   }, f, indent=2)
    print(f"summary: {summary}")


if __name__ == "__main__":
    main()
