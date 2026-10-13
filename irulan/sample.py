"""Pick a fixed, stratified sample of clean messages for model comparisons, and
render side-by-side reviews of the pairs each model generated for it.

    python3 -I pipeline.py sample [--seed S] [--out data/samples/sample50.jsonl]
    python3 -I pipeline.py review --sample data/samples/sample50.jsonl \
        --runs data/pairs/a.jsonl data/pairs/b.jsonl [--out data/review/sample50.md]

The sample file has one {id, bucket, words, variant} per line. Variants are
split evenly between plain and formatted with the same seed, so every model
sees the same assignment (generate --variants assigned).
"""

import argparse
import collections
import json
import os
import random

from . import check_pairs

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLACEHOLDER_RE = check_pairs.PLACEHOLDER_RE
# (label, min words, max words, how many)
STRATA = (("15-29", 15, 29, 15), ("30-59", 30, 59, 20), ("60+", 60, 10 ** 9, 15))


def mostly_placeholders(text, max_share=0.2, min_prose=10):
    toks = text.split()
    ph = sum(len(PLACEHOLDER_RE.findall(t)) for t in toks)
    return ph / max(1, len(toks)) > max_share or len(toks) - ph < min_prose


def select(records, seed, strata=STRATA):
    rng = random.Random(seed)
    picked = []
    for label, lo, hi, n in strata:
        pool = [r for r in records if lo <= len(r["text"].split()) <= hi and not mostly_placeholders(r["text"])]
        pool.sort(key=lambda r: r["id"])
        for r in rng.sample(pool, min(n, len(pool))):
            picked.append({"id": r["id"], "bucket": label, "words": len(r["text"].split())})
    order = list(range(len(picked)))
    rng.shuffle(order)
    half = len(picked) // 2
    for rank, i in enumerate(order):
        picked[i]["variant"] = "plain" if rank < half else "formatted"
    return picked


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def fence(text):
    return "~~~~text\n%s\n~~~~" % text.rstrip()


def model_label(pair):
    return "%s/%s" % (pair["backend"], pair["model"])


def run_failures(run_path):
    """Jobs whose last logged attempt failed and that have no pair."""
    log = run_path[: -len(".jsonl")] + ".log.jsonl"
    last = {}
    if os.path.exists(log):
        for e in read_jsonl(log):
            last[(e["id"], e["variant"])] = e
    return [e for e in last.values() if not e.get("ok")]


def pct(values, q):
    if not values:
        return 0.0
    v = sorted(values)
    return v[min(len(v) - 1, int(q * len(v)))]


def review(sample, runs, records, out_md, out_summary, skipped=()):
    by_id = {r["id"]: r for r in records}
    per_run = []
    for path in runs:
        pairs = read_jsonl(path) if os.path.exists(path) else []
        run = os.path.basename(path)[: -len(".jsonl")]
        label = "%s (%s)" % (model_label(pairs[0]), run) if pairs else run
        per_run.append((label, path, {(p["id"], p["variant"]): p for p in pairs}))

    lines = ["# Sample review", "",
             "%d messages; models: %s." % (len(sample), ", ".join(l for l, _, _ in per_run))]
    for s in skipped:
        lines.append("Skipped backend: %s." % s)
    lines.append("")
    for n, s in enumerate(sample, 1):
        rec = by_id.get(s["id"])
        if not rec:
            continue
        lines += ["## %d. `%s` (%s words, bucket %s, variant %s)" % (n, s["id"], s["words"], s["bucket"], s["variant"]),
                  "", "**Original (target)**", "", fence(rec["text"]), ""]
        for label, _, pairs in per_run:
            p = pairs.get((s["id"], s["variant"]))
            if not p:
                lines += ["**%s**: missing (failed or not run)" % label, ""]
                continue
            flags = check_pairs.check(p)
            lines += ["**%s** (ratio %.2f, %.1fs, flags: %s)" % (
                label, check_pairs.ratio(p), p["meta"].get("latency_s", 0), ", ".join(flags) or "none"),
                "", fence(p["input"]), ""]
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    sm = ["# Sample summary", "", "%d messages per model." % len(sample), ""]
    for s in skipped:
        sm.append("- skipped: %s" % s)
    sm += ["", "| model | ok | failed | ratio mean | ratio median | latency mean s | latency p90 s | out tokens mean | any flag |"
           + "".join(" %s |" % k for k in check_pairs.FLAGS),
           "|---" * (9 + len(check_pairs.FLAGS)) + "|"]
    wanted = {(s["id"], s["variant"]) for s in sample}
    for label, path, pairs in per_run:
        ps = [p for k, p in pairs.items() if k in wanted]
        summ = check_pairs.summarise(ps)
        lat = [p["meta"].get("latency_s", 0) for p in ps]
        outs = [(p["meta"].get("usage") or {}).get("output_tokens",
                (p["meta"].get("usage") or {}).get("completion_tokens")) for p in ps]
        outs = [o for o in outs if isinstance(o, (int, float))]
        fails = [e for e in run_failures(path) if (e["id"], e["variant"]) in wanted and (e["id"], e["variant"]) not in pairs]
        n = len(ps) or 1
        sm.append("| %s | %d | %d | %.2f | %.2f | %.1f | %.1f | %s | %.0f%% |" % (
            label, len(ps), len(fails), summ["ratio_mean"], summ["ratio_median"],
            sum(lat) / n, pct(lat, 0.9), ("%.0f" % (sum(outs) / len(outs))) if outs else "-",
            100.0 * summ["any_flag"] / n) + "".join(" %.0f%% |" % (100.0 * summ["flags"][k] / n) for k in check_pairs.FLAGS))
    by_variant = collections.defaultdict(list)
    for label, _, pairs in per_run:
        for k, p in pairs.items():
            if k in wanted:
                by_variant[(label, p["variant"])].append(p)
    sm += ["", "Length ratio by variant:", ""]
    for (label, v), ps in sorted(by_variant.items()):
        sm.append("- %s %s: %.2f (n=%d)" % (label, v, check_pairs.summarise(ps)["ratio_mean"], len(ps)))
    with open(out_summary, "w", encoding="utf-8") as f:
        f.write("\n".join(sm) + "\n")
    return "\n".join(sm)


def main_sample(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=os.path.join(ROOT, "data/clean/cc_messages.jsonl"))
    ap.add_argument("--seed", default="irulan-sample50")
    ap.add_argument("--out", default=os.path.join(ROOT, "data/samples/sample50.jsonl"))
    args = ap.parse_args(argv)
    picked = select(read_jsonl(args.input), args.seed)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for p in picked:
            f.write(json.dumps(p) + "\n")
    c = collections.Counter((p["bucket"], p["variant"]) for p in picked)
    print("%d picked -> %s; %s" % (len(picked), args.out, dict(sorted(c.items()))))


def main_review(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=os.path.join(ROOT, "data/clean/cc_messages.jsonl"))
    ap.add_argument("--sample", default=os.path.join(ROOT, "data/samples/sample50.jsonl"))
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--skipped", nargs="*", default=[], help="notes for backends that were skipped")
    ap.add_argument("--out", default=os.path.join(ROOT, "data/review/sample50.md"))
    args = ap.parse_args(argv)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    summary = args.out[: -len(".md")] + "_summary.md"
    print(review(read_jsonl(args.sample), args.runs, read_jsonl(args.input), args.out, summary, args.skipped))
    print("-> %s, %s" % (args.out, summary))
