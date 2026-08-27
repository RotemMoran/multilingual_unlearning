"""Collate results into the tables the experiment is actually judged on.

    # Table 1 -- are the arms comparable? Run this FIRST.
    python steering/summarize_results.py forget-quality \
        --checkpoints \
          outputs/tofu_finetuned_5epoch_aya_10_lang_2e5/grad_diff_2e-05_forget01_5_en \
          outputs/tofu_finetuned_5epoch_aya_10_lang_2e5/grad_diff_2e-05_forget01_5_en+iw \
          outputs/tofu_finetuned_5epoch_aya_10_lang_2e5/grad_diff_2e-05_forget01_5_iw

    # Table 2 -- did the attack recover anything?
    python steering/summarize_results.py attacks --results-dir steering/results

    # Table 3 -- join forget-quality CSVs with attack JSONs (one row per model x lang).
    python steering/summarize_results.py comparison \
        --results-dir steering/results --csv results/steering_comparison.csv

    # Interleave only: split recovery by the language each forget item was unlearned in.
    python steering/summarize_results.py interleave-split --results-dir steering/results

    # Batch 5 (paper-faithful) runs: recovery as a fraction of what unlearning removed.
    python steering/summarize_results.py comparison \
        --results-dir steering/results_v2 --csv results/steering_comparison_v2.csv

    # Did the start-layer sweep find a localised suppression signal, or nothing anywhere?
    python steering/summarize_results.py layer-profile --results-dir steering/results_v2

Table 1 must be read before Table 2 means anything: if arm B unlearned harder than arm A,
its lower recovery is explained by that alone. See steering/README.md.

Both result formats are readable here. Batch 4 files store a `results` dict keyed by alpha
with a single fixed layer set; Batch 5 files store a `conditions` list over
(alpha, start_layer) and may carry NLI scores. Keeping one set of subcommands across both
is what makes the before/after comparison possible.
"""

import argparse
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

LANG_ALT = "en|fr|ar|fa|hi|id|iw|ja|ko|ru"
# Batch 4: aya_grad_diff_<tag>_<lang>_late[_random]
ATTACK_NAME = re.compile(
    rf"^aya_grad_diff_(.+)_({LANG_ALT})_late(?P<random>_random)?$"
)
# Batch 5: aya_aux_<tag>_<lang>_<mode>[_random]
ATTACK_NAME_V2 = re.compile(
    rf"^aya_aux_(.+)_({LANG_ALT})_(?:sweep|alllayers|window\d+)(?P<random>_random)?$"
)
# f_ft reference generations: ft_baseline_<lang>.json
FT_BASELINE_NAME = re.compile(rf"^ft_baseline_({LANG_ALT})$")
MIX_SUFFIXES = ("_half", "_third", "_quarter")
KNOWN_LANGS = ("en", "fr", "ar", "fa", "hi", "id", "iw", "ja", "ko", "ru")


def _fmt_table(headers, rows):
    widths = [max(len(str(h)), *(len(str(r[i])) for r in rows)) if rows else len(str(h))
              for i, h in enumerate(headers)]
    line = "  ".join(str(h).ljust(w) for h, w in zip(headers, widths))
    out = [line, "  ".join("-" * w for w in widths)]
    for r in rows:
        out.append("  ".join(str(c).ljust(w) for c, w in zip(r, widths)))
    return "\n".join(out)


def arm_label(path):
    """Turn a save_dir into a short arm label.

    save_dir is `${forget_loss}_${lr}_${split}_${num_epochs}_${language}`. The loss
    itself can contain underscores (`grad_diff`, `grad_diff_KL`), as can the joint
    language tag (`en+iw` is fine; older joint tags used `joint_en_iw`).
    """
    name = os.path.basename(os.path.normpath(path))
    match = re.match(r"^(.+)_([0-9.eE+-]+)_(forget\d+)_(\d+)_(.+)$", name)
    if match:
        loss, _lr, _split, epochs, lang = match.groups()
        return f"{loss} {lang} {epochs}ep"
    return name


def cmd_forget_quality(args):
    """Forget-side TOFU metrics per checkpoint, for both eval languages.

    Uses aggregate_eval_stat.get_model_utility, which needs no retain model. Only the
    KS-test 'Forget Quality' p-value requires one, and that is not what the arms are
    matched on.
    """
    # imported here, not at module scope, so the `attacks` subcommand does not need hydra
    from aggregate_eval_stat import get_model_utility

    rows = []
    for ckpt in args.checkpoints:
        for lang in args.languages:
            path = os.path.join(ckpt, "eval_results", f"ds_size{args.ds_size}_{lang}",
                                "eval_log_aggregated.json")
            if not os.path.exists(path):
                rows.append([arm_label(ckpt), lang, "MISSING", "-", "-", "-"])
                continue
            util = get_model_utility(json.load(open(path)))
            rows.append([
                arm_label(ckpt), lang,
                f"{util['Prob. Forget']:.4f}",
                f"{util['Truth Ratio Forget']:.4f}",
                f"{util['Prob. Retain']:.4f}",
                f"{util['Model Utility']:.4f}",
            ])

    print("FORGET QUALITY / UTILITY  (arms must be comparable before attacks mean anything)")
    print()
    print(_fmt_table(
        ["arm", "eval lang", "Prob. Forget", "TruthRatio Forget", "Prob. Retain", "Model Utility"],
        rows))
    print()
    print("Lower 'Prob. Forget' = more forgetting. If the arms differ a lot here, fix that")
    print("(tune epochs or lr per arm) before comparing attack recovery.")
    print("Note: ROUGE is deliberately absent -- rouge_score scores 0 for Hebrew. See README.")
    if args.csv:
        _write_csv(args.csv, ["arm", "eval_lang", "prob_forget", "truth_ratio_forget",
                              "prob_retain", "model_utility"], rows)


def cmd_attacks(args):
    paths = sorted(args.files or glob.glob(os.path.join(args.results_dir, "*.json")))
    if not paths:
        print(f"no result files found in {args.results_dir}")
        return

    rows = []
    for p in paths:
        d = json.load(open(p))
        res = d["results"]
        primary = d.get("primary_metric", "rougeL_recall")
        base = res.get("0.0", res.get("0"))
        base_score = base[primary] if base else float("nan")
        best_score = d["best_score"]
        rows.append([
            os.path.basename(p).replace(".json", ""),
            d.get("language", "?"),
            d.get("layer_set", "?"),
            "RANDOM" if d.get("random_vector_control") else "real",
            primary,
            f"{base_score:.4f}",
            str(d["best_alpha"]),
            f"{best_score:.4f}",
            f"{best_score - base_score:+.4f}",
        ])

    print("ATTACK RECOVERY  (best-over-alpha; 'recovery' is best minus the alpha=0 baseline)")
    print()
    print(_fmt_table(
        ["result", "lang", "layers", "vector", "metric", "baseline", "best a", "best", "recovery"],
        rows))
    print()
    print("How to read this:")
    print("  * recovery ~= 0 on a 'real' row      -> the attack failed against that model")
    print("  * recovery > 0 on a 'RANDOM' row too -> the attack is measuring perturbation")
    print("    magnitude, not the unlearning direction. No result; do not report it.")
    print("  * compare English-only (A) vs joint EN+IW (B). If B resists only because it")
    print("    took more optimizer steps (80 rows/epoch vs 40), that is not a language effect.")
    if args.csv:
        _write_csv(args.csv, ["result", "lang", "layers", "vector", "metric", "baseline",
                              "best_alpha", "best", "recovery"], rows)


def _write_csv(path, headers, rows):
    import csv
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(headers)
        w.writerows(rows)
    print(f"\nwrote {path}")


def _fmt4(value):
    if value is None or value == "":
        return "-"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{value:.2e}" if 0 < abs(value) < 0.001 else f"{value:.4f}"


def parse_attack_stem(stem):
    """Return (tag, lang, is_random) or None, for either result-file generation."""
    match = ATTACK_NAME_V2.match(stem) or ATTACK_NAME.match(stem)
    if not match:
        return None
    return match.group(1), match.group(2), bool(match.group("random"))


def unlearn_langs(tag):
    core = tag
    for suffix in MIX_SUFFIXES:
        if core.endswith(suffix):
            core = core[: -len(suffix)]
            break
    return core.split("+")


def mix_of(tag):
    for suffix in MIX_SUFFIXES:
        if tag.endswith(suffix):
            return "interleave"
    return "concat" if "+" in tag else "monolingual"


def items_per_lang(tag, lang):
    langs = unlearn_langs(tag)
    if tag.endswith("_half"):
        return 20
    if tag.endswith("_third"):
        return 14 if langs.index(lang) == 0 else 13
    if tag.endswith("_quarter"):
        return 10
    return 40


def tag_to_exp_id(tag):
    if tag in ("en", "iw") or tag.endswith("ep"):
        return tag
    return f"{tag}_5ep"


def _alpha_key(results, alpha):
    if str(alpha) in results:
        return str(alpha)
    as_float = f"{float(alpha):.1f}"
    if as_float in results:
        return as_float
    if str(int(float(alpha))) in results:
        return str(int(float(alpha)))
    return None


def is_v2(payload):
    """Batch 5 files carry a `conditions` list over (alpha, start_layer)."""
    return "conditions" in payload


def _baseline_and_best(payload):
    """(primary_metric, baseline, best_alpha, best_score), either format."""
    primary = payload.get("primary_metric", "rougeL_recall")
    if is_v2(payload):
        conditions = payload["conditions"]
        base = next((c[primary] for c in conditions if c["alpha"] == 0), float("nan"))
        return primary, base, float(payload["best_alpha"]), float(payload["best_score"])
    results = payload["results"]
    base_key = _alpha_key(results, 0)
    base = results[base_key][primary] if base_key else float("nan")
    return primary, base, float(payload["best_alpha"]), float(payload["best_score"])


def condition_generations(payload, alpha):
    """Generations for one alpha, whichever format the file is in.

    For v2 the best condition is identified by (alpha, start_layer), so this returns the
    highest-scoring condition at that alpha rather than the only one.
    """
    primary = payload.get("primary_metric", "rougeL_recall")
    if is_v2(payload):
        matching = [c for c in payload["conditions"] if float(c["alpha"]) == float(alpha)]
        if not matching:
            return None
        return max(matching, key=lambda c: c[primary])["generations"]
    key = _alpha_key(payload["results"], alpha)
    return payload["results"][key]["generations"] if key else None


def load_ft_baselines(results_dir):
    """f_ft reference score per language, keyed by lang.

    Without this, a recovery of +0.05 cannot be read as a share of what unlearning removed,
    which is the quantity behind the paper's "over half" and "90%" headline numbers.
    """
    out = {}
    for path in sorted(glob.glob(os.path.join(results_dir, "ft_baseline_*.json"))):
        match = FT_BASELINE_NAME.match(os.path.splitext(os.path.basename(path))[0])
        if not match:
            continue
        payload = json.load(open(path))
        primary, base, _, _ = _baseline_and_best(payload)
        entry = {"primary": primary, "score": base}
        nli = payload.get("nli")
        if nli:
            entry["nli"] = nli["baseline"]
        out[match.group(1)] = entry
    return out


def load_attack_files(results_dir, files=None):
    """Group real/random attack JSONs by (tag, lang)."""
    paths = files or sorted(glob.glob(os.path.join(results_dir, "*.json")))
    grouped = {}
    for path in paths:
        parsed = parse_attack_stem(os.path.splitext(os.path.basename(path))[0])
        if parsed is None:
            continue
        tag, lang, is_random = parsed
        slot = grouped.setdefault((tag, lang), {"real": None, "random": None, "path": None})
        payload = json.load(open(path))
        if is_random:
            slot["random"] = payload
        else:
            slot["real"] = payload
            slot["path"] = path
    return grouped


def cmd_comparison(args):
    """Join existing eval CSVs with attack JSONs into one comparison table."""
    from collect_results import collect

    evals = collect()
    grouped = load_attack_files(args.results_dir, args.files)
    if not grouped:
        print(f"no attack result files found in {args.results_dir}")
        return

    ft = load_ft_baselines(args.results_dir)
    headers = [
        "model", "mix", "items_per_lang", "eval_lang",
        "prob_forget", "prob_retain", "model_utility",
        "metric", "ft_baseline", "baseline", "best_alpha", "best_start_layer",
        "best", "recovery", "recovery_frac", "random_delta",
        "nli_ft", "nli_baseline", "nli_best", "nli_recovery", "nli_recovery_frac",
        "nli_random_delta", "nli_validated",
    ]
    rows = []
    display = []
    def _order(key):
        tag, lang = key
        lang_i = KNOWN_LANGS.index(lang) if lang in KNOWN_LANGS else 99
        return (tag, lang_i)

    def _frac(best, baseline, reference):
        """Share of the unlearning drop that the attack recovered.

        Undefined when f_ft is missing or when unlearning did not actually lower the score,
        in which case there is no "lost performance" for the attack to win back.
        """
        if reference is None or reference - baseline <= 1e-9:
            return ""
        return (best - baseline) / (reference - baseline)

    have_nli = False
    for tag, lang in sorted(grouped, key=_order):
        slot = grouped[(tag, lang)]
        real = slot["real"]
        if real is None:
            continue
        primary, baseline, best_alpha, best = _baseline_and_best(real)
        recovery = best - baseline
        best_start = real.get("best_start_layer", "")
        random_delta = ""
        if slot["random"] is not None:
            _, rand_base, _, rand_best = _baseline_and_best(slot["random"])
            random_delta = rand_best - rand_base

        ft_entry = ft.get(lang)
        ft_score = ft_entry["score"] if ft_entry else None
        recovery_frac = _frac(best, baseline, ft_score)

        nli = real.get("nli") or {}
        nli_random = (slot["random"] or {}).get("nli") or {}
        if nli:
            have_nli = True
        nli_ft = (ft_entry or {}).get("nli")
        nli_frac = _frac(nli["best"], nli["baseline"], nli_ft) if nli and nli_ft is not None \
            else ""

        exp_id = tag_to_exp_id(tag)
        metrics = evals.get(exp_id, {}).get(lang, {})
        mix = mix_of(tag)
        n_items = items_per_lang(tag, lang)
        rows.append([
            tag, mix, n_items, lang,
            metrics.get("Prob. Forget", ""),
            metrics.get("Prob. Retain", ""),
            metrics.get("Model Utility", ""),
            primary,
            "" if ft_score is None else ft_score,
            baseline, best_alpha, best_start, best, recovery, recovery_frac, random_delta,
            "" if nli_ft is None else nli_ft,
            nli.get("baseline", ""), nli.get("best", ""), nli.get("recovery", ""),
            nli_frac,
            (nli_random.get("recovery", "") if nli_random else ""),
            nli.get("validated", ""),
        ])
        display.append([
            tag, mix, str(n_items), lang,
            _fmt4(metrics.get("Prob. Forget")),
            _fmt4(metrics.get("Model Utility")),
            primary,
            _fmt4(ft_score), f"{baseline:.4f}", str(best_alpha), str(best_start),
            f"{best:.4f}", f"{recovery:+.4f}",
            f"{recovery_frac:.1%}" if recovery_frac != "" else "-",
            f"{random_delta:+.4f}" if random_delta != "" else "-",
            f"{nli['baseline']:.4f}" if nli else "-",
            f"{nli['best']:.4f}" if nli else "-",
            f"{nli['recovery']:+.4f}" if nli else "-",
        ])

    print("COMPARISON  (forget quality from results/*.csv, recovery from "
          f"{args.results_dir})")
    print()
    print(_fmt_table(
        ["model", "mix", "n/lang", "lang", "P.Forget", "MU", "metric",
         "f_ft", "unlearned", "best a", "start", "best", "recovery", "of lost",
         "random Δ", "NLI base", "NLI best", "NLI rec"],
        display))
    print()
    print("f_ft       = the finetuned parent's score, i.e. before any unlearning.")
    print("unlearned  = the attacked model at alpha=0.")
    print("'of lost'  = recovery / (f_ft - unlearned): the share of what unlearning")
    print("             removed that the attack won back. This is the quantity the paper")
    print("             reports as 'over half' (Qwen) and '90%' (Gemma).")
    print("random Δ   = the same recovery using a unit-norm Gaussian direction. If it")
    print("             matches 'recovery', the attack measures perturbation size and")
    print("             there is no result.")
    if not have_nli:
        print("\nNLI columns are empty: run `python steering/nli_score.py "
              f"{args.results_dir}/*.json` to fill them.")
    if not ft:
        print(f"\nNo ft_baseline_*.json in {args.results_dir}, so 'of lost' is blank. "
              "The driver generates these.")
    if args.csv:
        _write_csv(args.csv, headers, rows)


def _per_item_scores(generations, gold, metric):
    from steering.metrics import chrf_scores, rouge_recall
    if metric == "chrf":
        scores = chrf_scores(generations, gold)
        return [scores[i] for i in range(len(generations))]
    rouge = rouge_recall(generations, gold)
    return [rouge["rougeL_recall"][i] for i in range(len(generations))]


def cmd_interleave_split(args):
    """Split interleaved attack recovery by the language each item was unlearned in."""
    grouped = load_attack_files(args.results_dir, args.files)
    rows = []
    display = []
    for tag, lang in sorted(grouped, key=lambda k: (k[0], k[1])):
        if mix_of(tag) != "interleave":
            continue
        real = grouped[(tag, lang)]["real"]
        if real is None:
            continue
        langs = unlearn_langs(tag)
        primary, _, best_alpha, _ = _baseline_and_best(real)
        gold = real["gold_answers"]
        base_gen = condition_generations(real, 0)
        best_gen = condition_generations(real, best_alpha)
        if base_gen is None or best_gen is None:
            continue
        base_scores = _per_item_scores(base_gen, gold, primary)
        best_scores = _per_item_scores(best_gen, gold, primary)
        groups = {g: [] for g in langs}
        for i, (b, a) in enumerate(zip(base_scores, best_scores)):
            groups[langs[i % len(langs)]].append((b, a))
        for unlearned_in, pairs in groups.items():
            if not pairs:
                continue
            base_mean = sum(p[0] for p in pairs) / len(pairs)
            best_mean = sum(p[1] for p in pairs) / len(pairs)
            rows.append([
                tag, lang, unlearned_in, len(pairs), primary,
                base_mean, best_alpha, best_mean, best_mean - base_mean,
            ])
            marker = "*" if unlearned_in == lang else ""
            display.append([
                tag, lang, f"{marker}{unlearned_in}{marker}", str(len(pairs)),
                primary, f"{base_mean:.4f}", str(best_alpha),
                f"{best_mean:.4f}", f"{best_mean - base_mean:+.4f}",
            ])

    if not display:
        print("no interleaved attack results found")
        return
    print("INTERLEAVE SPLIT  (item i was unlearned in language i % N)")
    print()
    print(_fmt_table(
        ["model", "eval", "unlearned in", "n", "metric",
         "baseline", "best a", "best", "recovery"],
        display))
    print()
    print("* marks the diagonal: items unlearned in the language being evaluated.")
    if args.csv:
        _write_csv(args.csv, ["model", "eval_lang", "unlearned_in", "n", "metric",
                              "baseline", "best_alpha", "best", "recovery"], rows)


def cmd_layer_profile(args):
    """Recovery as a function of the injection window's start layer.

    This is the diagnostic the fixed-`late` design could not produce. A suppression signal
    localised to a few layers shows up as a peak; a flat profile at zero means the vector
    does nothing anywhere, which is a much stronger negative result than one failed window.
    """
    grouped = load_attack_files(args.results_dir, args.files)
    printed = False
    for tag, lang in sorted(grouped, key=lambda k: (k[0], k[1])):
        real = grouped[(tag, lang)]["real"]
        if real is None or not is_v2(real):
            continue
        primary = real.get("primary_metric", "rougeL_recall")
        metric = args.metric if args.metric != "auto" else primary
        conditions = [c for c in real["conditions"] if c["alpha"] != 0 and metric in c]
        if not conditions:
            continue
        baseline = next((c[metric] for c in real["conditions"] if c["alpha"] == 0), 0.0)

        rows = []
        for cond in sorted(conditions, key=lambda c: (c["alpha"], str(c["start_layer"]))):
            delta = cond[metric] - baseline
            span = 40
            # Centre the bar on zero so degradation reads as clearly as recovery.
            scale = max(abs(c[metric] - baseline) for c in conditions) or 1e-9
            filled = int(span * abs(delta) / scale)
            bar = ("-" * filled).rjust(span) if delta < 0 else " " * span + "+" * filled
            rows.append([str(cond["alpha"]), str(cond["start_layer"]),
                         f"{cond[metric]:.4f}", f"{delta:+.4f}", bar])

        print(f"\n{tag} / {lang}   metric={metric}  baseline={baseline:.4f}"
              f"{'  [RANDOM CONTROL]' if real.get('random_vector_control') else ''}")
        print(_fmt_table(["alpha", "start", metric, "delta", "  <- worse | better ->"], rows))
        printed = True

    if not printed:
        print(f"no swept (Batch 5) attack results found in {args.results_dir}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    fq = sub.add_parser("forget-quality", help="per-checkpoint TOFU forget metrics")
    fq.add_argument("--checkpoints", nargs="+", required=True)
    fq.add_argument("--languages", nargs="+", default=["en", "iw"])
    fq.add_argument("--ds-size", default="None",
                    help="the ds_size in the eval save_dir name (default: None)")
    fq.add_argument("--csv")
    fq.set_defaults(func=cmd_forget_quality)

    at = sub.add_parser("attacks", help="attack recovery table")
    at.add_argument("--results-dir", default="steering/results")
    at.add_argument("--files", nargs="*")
    at.add_argument("--csv")
    at.set_defaults(func=cmd_attacks)

    cmp_ = sub.add_parser("comparison", help="join eval CSVs with attack recovery")
    cmp_.add_argument("--results-dir", default="steering/results")
    cmp_.add_argument("--files", nargs="*")
    cmp_.add_argument("--csv")
    cmp_.set_defaults(func=cmd_comparison)

    split = sub.add_parser("interleave-split",
                           help="split interleaved recovery by unlearn language")
    split.add_argument("--results-dir", default="steering/results")
    split.add_argument("--files", nargs="*")
    split.add_argument("--csv")
    split.set_defaults(func=cmd_interleave_split)

    prof = sub.add_parser("layer-profile", help="recovery vs injection start layer")
    prof.add_argument("--results-dir", default="steering/results_v2")
    prof.add_argument("--files", nargs="*")
    prof.add_argument("--metric", default="auto",
                      choices=("auto", "rougeL_recall", "chrf", "nli"))
    prof.set_defaults(func=cmd_layer_profile)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
