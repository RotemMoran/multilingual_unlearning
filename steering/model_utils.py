"""Shared model plumbing for the steering scripts.

The extraction and attack scripts MUST agree on what "layer i" means. Keeping the layer
lookup in one place is what guarantees that.

Model loading matches finetune.py / forget.py / evaluate_util.py: bf16 and optional
flash-attention-2 from config/model_config.yaml. There is no resolve_precision helper
in this repo.
"""

import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MODEL_FAMILY = "aya-expanse-8B"

# Paths used by the Aya runs already in this repo.
DEFAULT_BASE_MODEL = os.path.join(REPO_ROOT, "outputs", "tofu_finetuned_5epoch_aya_10_lang_2e5")

# Match config/forget_en.yaml, forget_iw.yaml, forget_en_iw.yaml, forget_joint.yaml.
UNLEARN_BATCH_SIZE = 2
UNLEARN_GRAD_ACCUM = 3
# Match config/eval_everything.yaml and eval_multilingual.yaml (inference, no accum).
INFER_BATCH_SIZE = 16


def ensure_repo_cwd():
    """utils.get_model_identifiers_from_yaml opens config/model_config.yaml relatively."""
    os.chdir(REPO_ROOT)


def get_decoder_layers(model):
    """The ModuleList of decoder blocks (LLaMA/Cohere/Mistral/GPT-style)."""
    for attr in ("model.layers", "transformer.h", "model.decoder.layers"):
        obj = model
        for part in attr.split("."):
            obj = getattr(obj, part, None)
            if obj is None:
                break
        if obj is not None:
            return obj
    raise AttributeError(f"could not locate decoder layers on {type(model).__name__}")


def attn_implementation(model_cfg):
    return "flash_attention_2" if model_cfg.get("flash_attention2") == "true" else None


def load_model(path, model_cfg, device, dtype=None):
    model = AutoModelForCausalLM.from_pretrained(
        path,
        attn_implementation=attn_implementation(model_cfg),
        torch_dtype=dtype if dtype is not None else torch.bfloat16,
        trust_remote_code=True,
    )
    model.eval()
    return model.to(device)


def load_tokenizer(model_cfg, checkpoint=None):
    """Prefer a saved checkpoint tokenizer so this works offline; else the HF id."""
    src = model_cfg["hf_key"]
    if checkpoint and os.path.isdir(checkpoint) and os.path.exists(
            os.path.join(checkpoint, "tokenizer.json")):
        src = checkpoint
    tokenizer = AutoTokenizer.from_pretrained(src)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def build_prompts(data, model_configs, language):
    """Prompt exactly as the model was trained to see it, question only.

    Shared by extract_steering_vector.py and attack_generate.py so that "the last prompt
    token" means the same position in both -- the direction has to be injected in the same
    place it was measured.

    Built from the tags in config/model_config.yaml rather than by decoding input_ids and
    splitting on "Answer: " the way run_generation does: that round-trip raises IndexError
    whenever the split symbol and the language disagree.
    """
    q_start = model_configs["question_start_tag"][language]
    q_end = model_configs["question_end_tag"]
    a_tag = model_configs["answer_tag"][language]
    return [q_start + row["question"] + q_end + a_tag for row in data]


# Xiang et al. inject over a window [c, c+N] and sweep the start layer c, rather than
# fixing a block of layers. N=2 (a 3-layer window) is their reported setting.
DEFAULT_WINDOW = 2


def window_layers(start, window, n_layers):
    """Layers [start, start+window], clipped to the model."""
    if not 0 <= start < n_layers:
        raise ValueError(f"start layer {start} outside 0..{n_layers - 1}")
    return list(range(start, min(start + window + 1, n_layers)))


def sweep_starts(window, n_layers, stride=1):
    """Every start layer whose full window fits, as in Algorithm 2 (c = 1..L-N)."""
    return list(range(0, n_layers - window, stride))


LAYER_SETS = {
    "late": lambda n: list(range(int(n * 0.75), n)),
    "middle": lambda n: list(range(int(n * 0.375), int(n * 0.75))),
    "early": lambda n: list(range(0, int(n * 0.375))),
    "all": lambda n: list(range(n)),
}
