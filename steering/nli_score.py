"""NLI equivalence scoring (Xiang et al. 2606.03291, Appendix E.1, Eq. 4).

    S(x, y) = (P_E(x,y) + P_E(y,x)) / 2  *  (1 - P_C(x,y))  *  (1 - P_N(x,y))

for a model prediction x against a reference answer y. The symmetric entailment term
rewards mutual implication; the contradiction and neutral terms are vetoes -- if x is
confidently contradictory or merely unrelated to y, the corresponding factor collapses to
near zero and the score dies regardless of entailment probability.

That veto structure is the whole reason this metric is used for unlearning. Post-unlearning
generations are dominated by refusals and hallucinations, which land in "neutral" with high
confidence (measured on this checkpoint: an "I don't have that information" refusal against
a real answer scores P_N = 0.987, so the neutral term contributes 0.013). Lexical overlap
metrics cannot make that distinction: the paper's Table 8 reports ROUGE-L agreeing with
human judgement 62-68% of the time against 89% for this NLI score.

Language coverage caveat. `joeddav/xlm-roberta-large-xnli` is finetuned on XNLI, which
covers 15 languages including en/fr/ar but NOT Hebrew or Japanese. XLM-R's pretraining
covers both, and spot checks on iw/ja entailment pairs look sensible, but those scores are
unvalidated zero-shot cross-lingual transfer. Every score this module emits carries a
`validated` flag, and reports must keep chrF alongside NLI for iw/ja.

Used as a post-pass over attack_generate.py output so generations can be re-scored without
regenerating them:

    python steering/nli_score.py steering/results_v2/*.json

or as a library:

    from steering.nli_score import NliScorer
    scorer = NliScorer()
    scores = scorer.score(predictions, references)
"""

import argparse
import json
import os
import sys

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_ID = "joeddav/xlm-roberta-large-xnli"

# The 15 languages this checkpoint was actually finetuned on (XNLI). Anything else is
# zero-shot and must be labelled as such.
XNLI_LANGUAGES = frozenset(
    ("en", "fr", "es", "de", "el", "bg", "ru", "tr", "ar", "vi", "th", "zh",
     "hi", "sw", "ur")
)


class NliScorer:
    """Eq. 4, batched. Two directed passes per pair: (x,y) and (y,x)."""

    def __init__(self, model_id=MODEL_ID, device="cuda:0", batch_size=32, max_length=256):
        self.batch_size = batch_size
        self.max_length = max_length
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_id)
        self.model.eval().to(device)

        # Read the label order off the config instead of assuming it. This checkpoint is
        # (contradiction, neutral, entailment), but that ordering is not universal across
        # NLI checkpoints and getting it wrong silently inverts the metric.
        id2label = {int(k): str(v).lower() for k, v in self.model.config.id2label.items()}
        try:
            self.i_entail = next(i for i, l in id2label.items() if "entail" in l)
            self.i_contra = next(i for i, l in id2label.items() if "contradict" in l)
            self.i_neutral = next(i for i, l in id2label.items() if "neutral" in l)
        except StopIteration:
            raise ValueError(
                f"cannot locate entailment/contradiction/neutral in id2label={id2label}"
            )

    @torch.no_grad()
    def _probs(self, premises, hypotheses):
        """Softmax probabilities for each (premise, hypothesis) pair."""
        out = []
        for i in range(0, len(premises), self.batch_size):
            enc = self.tokenizer(
                premises[i:i + self.batch_size], hypotheses[i:i + self.batch_size],
                return_tensors="pt", padding=True, truncation=True,
                max_length=self.max_length,
            ).to(self.device)
            out.append(self.model(**enc).logits.float().softmax(-1).cpu())
        return torch.cat(out) if out else torch.zeros(0, 3)

    def score(self, predictions, references):
        """Eq. 4 per pair, in [0, 1]. Empty predictions score 0 without a forward pass."""
        if len(predictions) != len(references):
            raise ValueError(f"{len(predictions)} predictions vs {len(references)} references")

        # A blank generation is not "neutral", it is a non-answer; the NLI model would give
        # it an arbitrary label, so short-circuit it.
        live = [i for i, p in enumerate(predictions) if p and p.strip()]
        scores = [0.0] * len(predictions)
        if not live:
            return scores

        preds = [predictions[i].strip() for i in live]
        refs = [references[i].strip() for i in live]

        forward = self._probs(preds, refs)          # P(.|x -> y)
        backward = self._probs(refs, preds)         # P(.|y -> x), for the symmetric term

        symmetric = (forward[:, self.i_entail] + backward[:, self.i_entail]) / 2
        veto = (1 - forward[:, self.i_contra]) * (1 - forward[:, self.i_neutral])
        for pos, idx in enumerate(live):
            scores[idx] = float(symmetric[pos] * veto[pos])
        return scores


def annotate(path, scorer):
    """Add NLI scores to one attack_generate.py result file, in place."""
    with open(path) as f:
        payload = json.load(f)

    gold = payload["gold_answers"]
    conditions = payload["conditions"]
    for cond in conditions:
        per_item = scorer.score(cond["generations"], gold)
        cond["nli_per_item"] = [round(s, 6) for s in per_item]
        cond["nli"] = sum(per_item) / len(per_item) if per_item else 0.0

    baseline = next(c for c in conditions if c["alpha"] == 0)
    steered = [c for c in conditions if c["alpha"] != 0]
    best = max(steered, key=lambda c: c["nli"]) if steered else baseline

    language = payload.get("language", "en")
    payload["nli"] = {
        "model": MODEL_ID,
        "language": language,
        # False means the score is zero-shot cross-lingual and must be reported with that
        # caveat attached; it does not mean the score is wrong.
        "validated": language in XNLI_LANGUAGES,
        "baseline": baseline["nli"],
        "best": best["nli"],
        "best_alpha": best["alpha"],
        "best_start_layer": best["start_layer"],
        "best_layers": best["layers"],
        "recovery": best["nli"] - baseline["nli"],
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return payload["nli"]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+", help="attack_generate.py result JSONs")
    ap.add_argument("--model-id", default=MODEL_ID)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--force", action="store_true",
                    help="re-score files that already carry NLI scores")
    args = ap.parse_args()

    todo = []
    for path in args.files:
        if not os.path.isfile(path):
            print(f"skip (missing): {path}")
            continue
        if not args.force:
            with open(path) as f:
                if "nli" in json.load(f):
                    print(f"skip (already scored): {path}")
                    continue
        todo.append(path)

    if not todo:
        print("nothing to do")
        return

    print(f"loading {args.model_id}")
    scorer = NliScorer(args.model_id, device=args.device, batch_size=args.batch_size)

    unvalidated = set()
    for path in todo:
        summary = annotate(path, scorer)
        if not summary["validated"]:
            unvalidated.add(summary["language"])
        print(f"{os.path.basename(path)}: baseline={summary['baseline']:.4f} "
              f"best={summary['best']:.4f} (alpha={summary['best_alpha']}, "
              f"start={summary['best_start_layer']}) "
              f"recovery={summary['recovery']:+.4f}"
              f"{'' if summary['validated'] else '  [zero-shot NLI]'}")

    if unvalidated:
        print(f"\nNOTE: {', '.join(sorted(unvalidated))} are outside XNLI's 15 finetuning "
              f"languages. Those NLI scores are zero-shot cross-lingual transfer; report "
              f"chrF alongside them.")


if __name__ == "__main__":
    sys.exit(main())
