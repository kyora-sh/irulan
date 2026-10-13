"""Flag generated inputs that break the back-translation contract.

    python3 -I pipeline.py check-pairs data/pairs/<run>.jsonl [...]

Flags per pair:
    placeholder_added    input has a placeholder (or more copies) the target lacks
    placeholder_dropped  input lost a placeholder the target has
    shorter              input has fewer words than the target
    too_long             input has more than 4x the target's words
    preamble             input opens with "Here's the rewrite", "Sure", ...
    answered             heuristic: the input answers the target's question instead of asking it
    markdown_in_plain    a plain-variant input (or a short formatted one) uses markdown
    tool_use             the backend ran tools/commands while generating
"""

import collections
import json
import re
import sys

PLACEHOLDER_RE = re.compile(r"<(?:SECRET|URL|EMAIL|IP|HOST|PATH|ID|NAME)>")
PREAMBLE_RE = re.compile(
    r"(?i)^\s*(?:\**\s*)?(?:here(?:'s| is| are)\b|sure\b|certainly\b|of course\b|absolutely\b|okay,? here|"
    r"rewritten(?: message| version)?\s*:|rewrite\s*:|revised(?: message| version)?\s*:|"
    r"below is\b|i(?:'ve| have) rewritten\b|the rewritten\b|polished(?: version)?\s*:)"
)
YES_NO_RE = re.compile(r"(?im)(?:^|\n|[.!?]\s+)(?:yes|no)[,.!]\s")
ANSWER_RE = re.compile(
    r"(?i)\bthe answer is\b|\bto answer your question\b|"
    r"\bhere(?:'s| is) how (?:you|to)\b|\bi(?:'ve| have) (?:fixed|updated|checked|reviewed|implemented|created|merged)\b|"
    r"\bi recommend\b|\bmy recommendation\b"
)
# "yeah" -> "Yes," is a faithful rewrite, not an answer.
TARGET_YES_NO_RE = re.compile(r"(?i)\b(?:ye|yes|yeah|yea|yep|yup|ya|no|nah|nope|kk|ok|okay)\b")
MARKDOWN_RE = re.compile(r"(?m)^\s*(?:[-*+]\s|\d+[.)]\s|#{1,6}\s)|\*\*[^*\n]+\*\*|^```")
MAX_RATIO = 4.0
FLAGS = ("placeholder_added", "placeholder_dropped", "shorter", "too_long", "preamble",
         "answered", "markdown_in_plain", "tool_use")


def words(text):
    return len(text.split())


def check(pair):
    inp, tgt = pair.get("input", ""), pair.get("target", "")
    flags = []
    ci = collections.Counter(PLACEHOLDER_RE.findall(inp))
    ct = collections.Counter(PLACEHOLDER_RE.findall(tgt))
    if ci - ct:
        flags.append("placeholder_added")
    if ct - ci:
        flags.append("placeholder_dropped")
    wi, wt = words(inp), words(tgt)
    if wi < wt:
        flags.append("shorter")
    if wt and wi > MAX_RATIO * wt:
        flags.append("too_long")
    if PREAMBLE_RE.match(inp):
        flags.append("preamble")
    # An answer is suspect when the target asks something and the input either
    # stops asking (no question mark left) or contains answer-shaped phrasing
    # that the target does not.
    if "?" in tgt:
        answer_shaped = (ANSWER_RE.search(inp) and not ANSWER_RE.search(tgt)) or (
            YES_NO_RE.search(inp) and not TARGET_YES_NO_RE.search(tgt))
        if "?" not in inp or answer_shaped:
            flags.append("answered")
    md_allowed = (pair.get("meta") or {}).get("markdown_allowed", pair.get("variant") == "formatted")
    if not md_allowed and MARKDOWN_RE.search(inp) and not MARKDOWN_RE.search(tgt):
        flags.append("markdown_in_plain")
    if (pair.get("meta") or {}).get("tool_items"):
        flags.append("tool_use")
    return flags


def ratio(pair):
    wt = words(pair.get("target", ""))
    return words(pair.get("input", "")) / wt if wt else 0.0


def summarise(pairs):
    n = len(pairs)
    counts = collections.Counter()
    any_flag = 0
    for p in pairs:
        f = check(p)
        counts.update(f)
        any_flag += bool(f)
    ratios = sorted(ratio(p) for p in pairs)
    return {
        "pairs": n,
        "any_flag": any_flag,
        "flags": {k: counts.get(k, 0) for k in FLAGS},
        "ratio_mean": sum(ratios) / n if n else 0.0,
        "ratio_median": ratios[n // 2] if n else 0.0,
    }


def render(name, s):
    n = s["pairs"] or 1
    lines = ["%s: %d pairs, %.0f%% flagged, length ratio mean %.2f / median %.2f" % (
        name, s["pairs"], 100.0 * s["any_flag"] / n, s["ratio_mean"], s["ratio_median"])]
    for k in FLAGS:
        lines.append("  %-20s %4d  %5.1f%%" % (k, s["flags"][k], 100.0 * s["flags"][k] / n))
    return "\n".join(lines)


def main(argv=None):
    paths = argv if argv is not None else sys.argv[1:]
    if not paths:
        sys.exit("usage: check-pairs <pairs.jsonl> [...]")
    for path in paths:
        with open(path, encoding="utf-8") as f:
            pairs = [json.loads(line) for line in f if line.strip()]
        print(render(path, summarise(pairs)))


if __name__ == "__main__":
    main()
