"""Extract the messages a human typed into Claude Code, then scrub them.

    python3 -I pipeline.py extract-cc   # walk transcripts, write raw + clean
    python3 -I pipeline.py rescrub      # reapply scrub to the raw extract only

Outputs (all under data/, which is git-ignored):
    data/raw/cc_messages.jsonl      filtered, deduped, NOT scrubbed
    data/clean/cc_messages.jsonl    scrubbed training targets
    data/entity_candidates.tsv      names to review into scrub/denylist.txt
    data/stats/cc_stats.json        counts only
    data/stats/cc_drops.jsonl       every authorship/language drop or strip, with its reason
"""

import argparse
import collections
import hashlib
import json
import os
import sys

from . import entities, filters, stats
from .scrub import Scrubber

DEFAULT_SRC = os.path.expanduser("~/.claude/projects")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHORT_WORDS = 3   # a message this short or shorter counts as a "short" message
SHORT_CAP = 3     # keep at most this many copies of each distinct short message
NON_HUMAN_SOURCES = {"system", "sdk"}


def iter_files(root):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        for fn in sorted(filenames):
            if fn.endswith(".jsonl"):
                yield dirpath, fn


def turn_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"]
        if parts:
            return "\n\n".join(parts)
    return None


def human_skip_reason(d):
    if d.get("type") != "user":
        return "not_user"
    if d.get("isSidechain"):
        return "sidechain"
    if d.get("isMeta"):
        return "meta"
    if d.get("isCompactSummary") or d.get("isVisibleInTranscriptOnly"):
        return "continuation"
    if d.get("entrypoint") == "sdk-cli":
        return "non_human_origin"
    origin = d.get("origin")
    if isinstance(origin, dict) and origin.get("kind") not in (None, "human"):
        return "non_human_origin"
    if d.get("promptSource") in NON_HUMAN_SOURCES:
        return "non_human_origin"
    if d.get("turnOrigin") not in (None, "human"):
        return "non_human_origin"
    return None


def walk(root):
    counts = collections.Counter()
    dropped = collections.Counter()
    out = []
    for dirpath, fn in iter_files(root):
        counts["files_read"] += 1
        rel = os.path.relpath(dirpath, root)
        project = rel.split(os.sep)[0]
        with open(os.path.join(dirpath, fn), encoding="utf-8", errors="replace") as f:
            for line in f:
                counts["lines"] += 1
                # cheap prefilter before parsing JSON
                if '"type":"user"' not in line:
                    continue
                if '"isSidechain":true' in line:
                    counts["sidechain_user_lines"] += 1
                    continue
                try:
                    d = json.loads(line)
                except ValueError:
                    counts["bad_json"] += 1
                    continue
                if d.get("type") != "user":
                    continue
                text = turn_text(d.get("message", {}).get("content"))
                if text is None:
                    continue  # tool results only
                counts["user_turns"] += 1
                reason = human_skip_reason(d)
                if reason:
                    dropped[reason] += 1
                    continue
                cleaned, reason = filters.clean_turn(text, counts)
                if reason:
                    dropped[reason] += 1
                    continue
                out.append({
                    "project": project,
                    "session_id": d.get("sessionId") or d.get("session_id"),
                    "timestamp": d.get("timestamp"),
                    "text": cleaned,
                })
    return out, counts, dropped


def dedupe(records, dropped, key_fn=filters.dedupe_key):
    """Exact dedupe after normalisation, plus a cap on repeated short messages."""
    records.sort(key=lambda r: (r.get("timestamp") or "", r.get("session_id") or ""))
    seen = set()
    short_seen = collections.Counter()
    out = []
    for r in records:
        words = len(r["text"].split())
        if words <= SHORT_WORDS:
            k = filters.short_key(r["text"])
            short_seen[k] += 1
            if short_seen[k] > SHORT_CAP:
                dropped["short_repeat_cap"] += 1
                continue
            # capped short messages are allowed to repeat exactly
            out.append(r)
            continue
        k = key_fn(r["text"])
        if k in seen:
            dropped["duplicate"] += 1
            continue
        seen.add(k)
        out.append(r)
    return out


def author_pass(records, cfg, keep_non_english, dropped, counts, log):
    """Corpus-level pass over cleaned turns, before dedupe: strip or drop
    text the writer did not type (see filters.author_check), then drop mainly
    non-English messages. Every drop and strip goes to `log`."""
    hot = frozenset()
    if cfg["enabled"]:
        hot = filters.template_lines(records, cfg["template_min_words"], cfg["template_min_sessions"])
    counts["template_lines"] = len(hot)
    out = []
    for r in records:
        base = {"session_id": r["session_id"], "timestamp": r["timestamp"], "project": r["project"],
                "words": len(r["text"].split())}
        text, reason, info = filters.author_check(r["text"], cfg, hot)
        if reason is None and cfg["language"] and not keep_non_english:
            en, de = filters.language_counts(text)
            text, reason, n_lang = filters.language_check(text, cfg)
            info = dict(info, en=en, de=de)
            if n_lang:
                info["stripped"] = info["stripped"] + ["non_english"] * n_lang
        if reason:
            dropped[reason] += 1
            log.append(dict(base, action="drop", reason=reason, scores=info.get("scores"),
                            en=info.get("en"), de=info.get("de"), text=r["text"]))
            continue
        if info["stripped"]:
            counts["author_messages_stripped"] += 1
            counts["author_paragraphs_stripped"] += len(info["stripped"])
            for cat in info["stripped"]:
                counts["stripped_" + cat] += 1
            log.append(dict(base, action="strip", reason="stripped_" + info["stripped"][0],
                            categories=info["stripped"], scores=info["scores"], text=r["text"], kept=text))
            r = dict(r, text=text)
        out.append(r)
    return out


def load_filter_config(path, overrides):
    cfg = dict(filters.AUTHOR_DEFAULTS)
    if path:
        with open(path, encoding="utf-8") as f:
            extra = json.load(f)
        unknown = set(extra) - set(cfg)
        if unknown:
            sys.exit("unknown filter config keys: %s" % ", ".join(sorted(unknown)))
        cfg.update(extra)
    cfg.update({k: v for k, v in overrides.items() if v is not None})
    return cfg


def with_meta(r, text):
    rec = dict(r)
    rec["text"] = text
    rec["words"] = len(text.split())
    rec["lines"] = text.count("\n") + 1
    return rec


def record_id(r):
    h = hashlib.sha256()
    h.update((r.get("session_id") or "").encode())
    h.update(b"\0")
    h.update((r.get("timestamp") or "").encode())
    h.update(b"\0")
    h.update(r["text"].encode())
    return h.hexdigest()[:16]


def write_jsonl(records, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def rescrub(args, extract_stats=None):
    raw = read_jsonl(args.raw)
    scrubber = Scrubber.from_files(args.denylist)
    totals = collections.Counter()
    dropped = collections.Counter()
    clean = []
    for r in raw:
        text, counts = scrubber.scrub(r["text"])
        totals.update(counts)
        if filters.prose_words(text) == 0:
            dropped["no_prose_after_scrub"] += 1
            continue
        rec = with_meta(r, text)
        clean.append(rec)
    # Two raw messages that differ only in a URL or ID are the same target.
    clean = dedupe(clean, dropped)
    write_jsonl(clean, args.clean)

    rows = entities.find_candidates((r["text"] for r in clean), min_count=args.min_entity_count)
    entities.write_tsv(rows, args.entities)

    prev = {}
    if extract_stats is None and os.path.exists(args.stats):
        with open(args.stats, encoding="utf-8") as f:
            prev = json.load(f).get("extract", {})
    out = {
        "extract": extract_stats or prev,
        "raw_messages": len(raw),
        "dropped_at_scrub": dict(dropped),
        "clean": stats.summarise(clean),
        "scrub": dict(totals),
        "denylist_terms": len(scrubber.denylist),
        "entity_candidates": len(rows),
    }
    stats.write(out, args.stats)
    print(stats.render(out))
    print("dropped at scrub: %s" % dict(dropped))
    print("entity candidates: %d -> %s" % (len(rows), args.entities))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default=DEFAULT_SRC, help="Claude Code projects dir")
    ap.add_argument("--raw", default=os.path.join(ROOT, "data/raw/cc_messages.jsonl"))
    ap.add_argument("--clean", default=os.path.join(ROOT, "data/clean/cc_messages.jsonl"))
    ap.add_argument("--entities", default=os.path.join(ROOT, "data/entity_candidates.tsv"))
    ap.add_argument("--stats", default=os.path.join(ROOT, "data/stats/cc_stats.json"))
    ap.add_argument("--denylist", default=os.path.join(ROOT, "scrub/denylist.txt"))
    ap.add_argument("--min-entity-count", type=int, default=2)
    ap.add_argument("--drop-log", default=os.path.join(ROOT, "data/stats/cc_drops.jsonl"),
                    help="jsonl with every authorship/language drop and strip and its reason")
    ap.add_argument("--filter-config", help="JSON file overriding filters.AUTHOR_DEFAULTS thresholds")
    ap.add_argument("--no-author-filter", action="store_true",
                    help="keep pasted transcripts, profiles, LLM output and templates")
    ap.add_argument("--keep-non-english", action="store_true", help="skip the English-only language filter")
    ap.add_argument("--par-threshold", type=int, help="paragraph paste score that strips it")
    ap.add_argument("--max-words", type=int, help="drop messages longer than this")
    ap.add_argument("--min-keep-words", type=int, help="prose words that must remain after a strip")
    ap.add_argument("--rescrub", action="store_true",
                    help="skip the transcript walk; re-read the raw extract and reapply scrubbing")
    args = ap.parse_args(argv)

    if args.rescrub:
        if not os.path.exists(args.raw):
            sys.exit("no raw extract at %s; run without --rescrub first" % args.raw)
        rescrub(args)
        return

    cfg = load_filter_config(args.filter_config, {
        "par_threshold": args.par_threshold, "max_words": args.max_words,
        "min_keep_words": args.min_keep_words, "enabled": False if args.no_author_filter else None})
    records, counts, dropped = walk(args.src)
    log = []
    records = author_pass(records, cfg, args.keep_non_english, dropped, counts, log)
    write_jsonl(log, args.drop_log)
    kept_before_dedupe = len(records)
    records = dedupe(records, dropped)
    raw = []
    for r in records:
        rec = with_meta(r, r["text"])
        rec = {"id": record_id(rec), **rec}
        raw.append(rec)
    write_jsonl(raw, args.raw)
    extract_stats = dict(counts)
    extract_stats["passed_filters"] = kept_before_dedupe
    extract_stats["kept"] = len(raw)
    extract_stats["dropped"] = dict(dropped)
    extract_stats["filter_config"] = cfg
    extract_stats["keep_non_english"] = args.keep_non_english
    extract_stats["raw"] = stats.summarise(raw)
    rescrub(args, extract_stats)


if __name__ == "__main__":
    main()
