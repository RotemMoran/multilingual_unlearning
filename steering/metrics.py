"""Script-agnostic generation metrics.

Why this file exists instead of `from evaluate_util import eval_rouge_recall`:

1. rouge_score's default tokenizer normalises with `re.sub(r"[^a-z0-9]+", " ", text)`,
   which deletes every non-ASCII character. Verified directly:

       >>> from rouge_score import tokenize
       >>> tokenize.tokenize("שמו המלא של המחבר", None)
       []

   So ROUGE-L recall is identically 0.0 for Hebrew, Arabic, Farsi, Hindi, Japanese,
   Korean and Russian -- it measures nothing at all. Passing a Unicode-aware tokenizer
   to RougeScorer fixes it.

2. evaluate_util's eval_bleu/eval_chrf call `evaluate.load(...)`, which downloads the
   metric script from the Hub at call time. That fails on an offline compute node.

chrF is included because it is character-based and therefore the more trustworthy
headline number for Hebrew, where word-level overlap is sensitive to morphology.
"""

import re
from collections import Counter

from rouge_score import rouge_scorer

_WORD = re.compile(r"\w+", re.UNICODE)


class UnicodeTokenizer:
    """Tokenizer for rouge_scorer that keeps non-Latin scripts.

    No stemming: rouge_score's Porter stemmer is English-only, and applying it across
    languages would be worse than not stemming at all.
    """

    def tokenize(self, text):
        return _WORD.findall(text.lower())


def rouge_recall(gen_outputs, ground_truths, indices=None):
    """ROUGE-1/ROUGE-L recall, keyed by index. Same return shape as
    evaluate_util.eval_rouge_recall, so it is a drop-in replacement."""
    if indices is None:
        indices = range(len(gen_outputs))
    scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], tokenizer=UnicodeTokenizer())
    rouge1, rougeL = {}, {}
    for gen, gt, idx in zip(gen_outputs, ground_truths, indices):
        scores = scorer.score(gt, gen)
        rouge1[idx] = scores["rouge1"].recall
        rougeL[idx] = scores["rougeL"].recall
    return {"rouge1_recall": rouge1, "rougeL_recall": rougeL}


def _char_ngrams(text, n):
    text = re.sub(r"\s+", "", text)
    return Counter(text[i:i + n] for i in range(len(text) - n + 1))


def chrf(hypothesis, reference, max_n=6, beta=2.0):
    """chrF: character n-gram F-score. Language- and script-agnostic."""
    precisions, recalls = [], []
    for n in range(1, max_n + 1):
        hyp_ng, ref_ng = _char_ngrams(hypothesis, n), _char_ngrams(reference, n)
        overlap = sum((hyp_ng & ref_ng).values())
        hyp_total, ref_total = sum(hyp_ng.values()), sum(ref_ng.values())
        if hyp_total:
            precisions.append(overlap / hyp_total)
        if ref_total:
            recalls.append(overlap / ref_total)
    if not precisions or not recalls:
        return 0.0
    p = sum(precisions) / len(precisions)
    r = sum(recalls) / len(recalls)
    if p + r == 0:
        return 0.0
    b2 = beta ** 2
    return (1 + b2) * p * r / (b2 * p + r)


def chrf_scores(gen_outputs, ground_truths, indices=None):
    if indices is None:
        indices = range(len(gen_outputs))
    return {idx: chrf(gen, gt) for gen, gt, idx in zip(gen_outputs, ground_truths, indices)}
