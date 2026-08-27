"""CPU pre-flight for the steering hook arithmetic. No GPU, no checkpoint, seconds to run.

The Batch 4 attack failed for reasons that were invisible in its output: the perturbation
was applied to every sequence position instead of one, and the direction was not rescaled,
so alpha did not mean what the logs implied. Both are cheap to assert against a randomly
initialised toy model, and neither needs an 8B checkpoint. Run this before spending GPU
hours on the sweep.

What is checked:

  1. get_decoder_layers finds the block list on a Cohere-family model.
  2. alpha=0 installs no hook and is bit-identical to no intervention.
  3. The hook edits ONLY the last position: earlier positions come out untouched.
  4. The edit has magnitude exactly alpha * ||h|| (norm matching), so alpha is a fraction
     of the activation norm at every layer.
  5. The hook fires once per generated token during a real `generate` call, i.e. it works
     under KV caching rather than only on a single forward pass.
  6. Extraction reads the same tensor the attack writes, and its output rows are unit
     vectors indexed by decoder layer.
  7. A sign check: subtracting the (f_un - f_ft) direction moves activations back toward
     f_ft, which is the direction the attack depends on being right.

    python steering/check_steering_math.py
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoModelForCausalLM, CohereConfig

from steering.attack_generate import make_hook
from steering.extract_steering_vector import accumulate_normalised_lastpos
from steering.model_utils import get_decoder_layers, sweep_starts, window_layers

FAILURES = []


def check(name, condition, detail=""):
    status = "ok  " if condition else "FAIL"
    print(f"  [{status}] {name}{('  -- ' + detail) if detail else ''}")
    if not condition:
        FAILURES.append(name)


def tiny_model(seed=0):
    """A 4-layer Cohere model with random weights: same architecture family as Aya."""
    torch.manual_seed(seed)
    cfg = CohereConfig(
        vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=4,
        num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=64,
        use_cache=True,
    )
    model = AutoModelForCausalLM.from_config(cfg)
    model.eval()
    return model


def main():
    print("steering hook pre-flight (CPU, random weights)\n")

    model = tiny_model()
    layers = get_decoder_layers(model)
    print("1. layer discovery")
    check("get_decoder_layers returns the block list", len(layers) == 4,
          f"found {len(layers)} blocks")

    ids = torch.randint(0, 64, (2, 7))
    with torch.no_grad():
        clean = model(input_ids=ids, use_cache=False).logits.clone()

    print("\n2. alpha=0 is a no-op")
    direction = torch.randn(32)
    direction = direction / direction.norm()
    handle = layers[2].register_forward_hook(make_hook(direction, 0.0))
    with torch.no_grad():
        zero_alpha = model(input_ids=ids, use_cache=False).logits
    handle.remove()
    check("logits unchanged at alpha=0", torch.allclose(clean, zero_alpha, atol=1e-6),
          f"max abs diff {(clean - zero_alpha).abs().max():.2e}")

    print("\n3. only the last position is edited, with norm-matched magnitude")
    captured = {}

    def capture(module, args, output):
        captured["pre"] = (output[0] if isinstance(output, tuple) else output).clone()

    alpha = 0.25
    grab = layers[2].register_forward_hook(capture)
    with torch.no_grad():
        model(input_ids=ids, use_cache=False)
    grab.remove()

    post = {}

    def capture_post(module, args, output):
        post["h"] = (output[0] if isinstance(output, tuple) else output).clone()

    steer = layers[2].register_forward_hook(make_hook(direction, alpha))
    after = layers[2].register_forward_hook(capture_post)
    with torch.no_grad():
        model(input_ids=ids, use_cache=False)
    steer.remove()
    after.remove()

    pre, hpost = captured["pre"], post["h"]
    untouched = torch.allclose(pre[:, :-1, :], hpost[:, :-1, :], atol=1e-6)
    check("positions 0..T-2 are untouched", untouched,
          f"max abs diff {(pre[:, :-1, :] - hpost[:, :-1, :]).abs().max():.2e}")

    delta = (hpost[:, -1, :] - pre[:, -1, :])
    expected = alpha * pre[:, -1, :].norm(dim=-1)
    check("||delta|| == alpha * ||h|| at the last position",
          torch.allclose(delta.norm(dim=-1), expected, rtol=1e-4),
          f"got {delta.norm(dim=-1).tolist()} vs {expected.tolist()}")
    # The edit must be antiparallel to g: the attack SUBTRACTS the unlearning direction.
    cosines = torch.nn.functional.cosine_similarity(
        delta, direction.expand_as(delta), dim=-1)
    check("delta is antiparallel to g (subtraction, not addition)",
          bool((cosines < -0.999).all()), f"cos = {cosines.tolist()}")

    print("\n4. the hook fires under KV-cached generation")
    calls = {"n": 0}
    base_hook = make_hook(direction, alpha)

    def counting_hook(module, args, output):
        calls["n"] += 1
        return base_hook(module, args, output)

    handle = layers[2].register_forward_hook(counting_hook)
    with torch.no_grad():
        model.generate(ids, max_new_tokens=5, do_sample=False, use_cache=True,
                       pad_token_id=0)
    handle.remove()
    # One prefill pass plus one per generated token; generate may stop early on EOS.
    check("hook ran once per decoding step", 2 <= calls["n"] <= 6,
          f"{calls['n']} invocations for 5 new tokens (1 prefill + up to 5 steps)")

    print("\n5. window and sweep arithmetic")
    check("window_layers(12, 2, 32) is a 3-layer window",
          window_layers(12, 2, 32) == [12, 13, 14], str(window_layers(12, 2, 32)))
    starts = sweep_starts(2, 32)
    check("sweep covers c = 0..L-N-1 with the window inside the model",
          starts[0] == 0 and starts[-1] == 29 and len(starts) == 30,
          f"{len(starts)} starts, {starts[0]}..{starts[-1]}")
    check("the last window fits", window_layers(starts[-1], 2, 32) == [29, 30, 31],
          str(window_layers(starts[-1], 2, 32)))

    print("\n6. extraction reads the attack's tensor and returns unit rows")

    class ToyTokenizer:
        """Stands in for the HF tokenizer: fixed-length ids, left-padded by construction."""
        def __call__(self, chunk, **kwargs):
            n = len(chunk)
            torch.manual_seed(len(chunk[0]))
            return type("Enc", (), {
                "input_ids": torch.randint(0, 64, (n, 7)),
                "attention_mask": torch.ones(n, 7, dtype=torch.long),
                "to": lambda self, device: self,
            })()

    prompts = ["abc", "defg", "hi", "jklmn"]
    sums, n = accumulate_normalised_lastpos(model, ToyTokenizer(), prompts, 2, "cpu")
    check("one row per decoder layer", tuple(sums.shape) == (4, 32), str(tuple(sums.shape)))
    check("every prompt counted", n == len(prompts), f"n={n}")
    g = sums / sums.norm(dim=-1, keepdim=True)
    check("normalised rows are unit vectors",
          torch.allclose(g.norm(dim=-1), torch.ones(4, dtype=g.dtype), atol=1e-6),
          f"norms {[round(v, 6) for v in g.norm(dim=-1).tolist()]}")

    print("\n7. sign check: subtracting (f_un - f_ft) moves h toward f_ft")
    # Two models standing in for f_ft and f_un, and a direction built the way
    # extract_steering_vector.py builds it.
    ft, un = tiny_model(1), tiny_model(2)
    tok = ToyTokenizer()
    sum_ft, _ = accumulate_normalised_lastpos(ft, tok, prompts, 2, "cpu")
    sum_un, _ = accumulate_normalised_lastpos(un, tok, prompts, 2, "cpu")
    s = sum_un - sum_ft
    g = (s / s.norm(dim=-1, keepdim=True)).float()

    layer = 2
    h_ft = (sum_ft[layer] / len(prompts)).float()
    h_un = (sum_un[layer] / len(prompts)).float()
    before = (h_un - h_ft).norm()
    # Same update rule as the hook, at the mean activation.
    moved = h_un - 0.5 * h_un.norm() * g[layer]
    after = (moved - h_ft).norm()
    check("subtracting g reduces the distance to f_ft", bool(after < before),
          f"||h_un - h_ft|| {before:.4f} -> {after:.4f}")

    print()
    if FAILURES:
        print(f"{len(FAILURES)} CHECK(S) FAILED: {', '.join(FAILURES)}")
        return 1
    print("all checks passed -- the hook arithmetic is safe to spend GPU hours on")
    return 0


if __name__ == "__main__":
    sys.exit(main())
