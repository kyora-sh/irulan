"""Summary stats for an extract. Counts only, never message content."""

import collections
import json
import os

BUCKETS = ((0, 4, "<5"), (5, 14, "5-14"), (15, 29, "15-29"), (30, 59, "30-59"), (60, None, "60+"))


def bucket(words):
    for lo, hi, name in BUCKETS:
        if words >= lo and (hi is None or words <= hi):
            return name
    return BUCKETS[-1][2]


def summarise(records):
    hist = collections.Counter()
    words = 0
    for r in records:
        hist[bucket(r["words"])] += 1
        words += r["words"]
    return {
        "messages": len(records),
        "words": words,
        "histogram": {name: hist.get(name, 0) for _, _, name in BUCKETS},
    }


def write(stats, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, sort_keys=True)
        f.write("\n")


def render(stats):
    lines = []
    ex = stats.get("extract", {})
    if ex:
        lines.append("files read: %d (lines %d, user turns %d)" % (
            ex.get("files_read", 0), ex.get("lines", 0), ex.get("user_turns", 0)))
        lines.append("kept after filters: %d" % ex.get("kept", 0))
        lines.append("dropped:")
        for k, v in sorted(ex.get("dropped", {}).items(), key=lambda kv: -kv[1]):
            lines.append("  %-22s %d" % (k, v))
    cl = stats.get("clean", {})
    if cl:
        lines.append("clean: %d messages, %d words" % (cl["messages"], cl["words"]))
        lines.append("  histogram: " + ", ".join("%s=%d" % kv for kv in cl["histogram"].items()))
    sc = stats.get("scrub", {})
    if sc:
        lines.append("scrub replacements: " + ", ".join("%s=%d" % kv for kv in sorted(sc.items())))
    return "\n".join(lines)


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "data/stats/cc_stats.json"
    with open(path, encoding="utf-8") as f:
        print(render(json.load(f)))
