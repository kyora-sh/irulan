"""Build the training set from a pairs file.

    python3 -I pipeline.py prepare --pairs data/pairs/<run>.jsonl [--out data/train] ...

Per pair, in order:

1. skip lines that do not parse (a run still appending can leave a half line);
2. drop ids listed in data/exclude_ids.local.txt;
3. drop ids no longer in data/clean (the extract dropped them since);
4. rescrub input and target with the current denylist and scrub rules, so
   names added after generation are still removed;
5. drop targets under --min-target-words words, then pairs that check_pairs
   flags (on the rescrubbed text), except for flag types in --allow-flags;
   an input that talks about the rewrite itself ("Here's the rewritten
   message") is always dropped, even when `preamble` is allowed;
6. drop exact-duplicate inputs and pairs whose input and target disagree on
   a placeholder count;
7. replace each placeholder with an invented surrogate (the same one for the
   n-th <NAME> of the input and the n-th <NAME> of the target), seeded per
   pair id, so the model never learns to emit placeholder tokens.

The rest is split by session (train/val/test 90/5/5, fixed seed) and written
as chat-format jsonl with a versioned system prompt. stats.json holds counts
only, drops.jsonl holds ids and reasons only.
"""

import argparse
import collections
import hashlib
import json
import os
import random
import re

from . import check_pairs
from .scrub import PLACEHOLDERS, Scrubber

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SYSTEM_PROMPTS = {
    "system_v1": ("Rewrite the text in the writer's own voice: terse, direct, short lines. "
                  "Keep the meaning, asks, facts and numbers. Return only the rewrite."),
}
DEFAULT_SYSTEM = "system_v1"

# check_pairs flag types that do not by themselves make a pair unusable:
# `preamble` fires on assistant lead-ins that mirror the target ("Okay, here is
# what is left:"), and `answered` is a loose heuristic. The real meta preambles
# are caught by META_PREAMBLE_RE below.
DEFAULT_ALLOW_FLAGS = ("preamble", "answered")
META_PREAMBLE_RE = re.compile(
    r"(?i)^\s*(?:\**\s*)?(?:"
    r"(?:sure|certainly|of course|absolutely|okay|ok)\b[,.!]?\s*(?:here(?:'s| is)\s+)?(?:the|a|your)\s+"
    r"(?:rewrite|rewritten|revised|polished|reworded)"
    r"|here(?:'s| is| are)\s+(?:the|a|your)\s+(?:rewrite|rewritten|revised|polished|reworded|assistant)"
    r"|(?:rewritten|revised|polished|reworded)(?:\s+(?:message|version|text))?\s*:"
    r"|rewrite\s*:"
    r"|below is\s+(?:the|a|your)\s+(?:rewrite|rewritten|revised)"
    r"|i(?:'ve| have)\s+rewritten\b"
    r")"
)

SPLITS = ("train", "val", "test")
DEFAULT_RATIOS = (0.90, 0.05, 0.05)
DEFAULT_SEED = "irulan-split-v1"
CHARS_PER_TOKEN = 3.5
# <|im_start|>/<|im_end|> pairs, role names and the empty think block.
TEMPLATE_OVERHEAD_TOKENS = 16

PLACEHOLDER_RE = re.compile(r"<(SECRET|URL|EMAIL|IP|HOST|PATH|ID|NAME)>")
TYPES = tuple(p.strip("<>") for p in PLACEHOLDERS)

# Invented surrogate pools. Every value is fake on purpose: reserved example
# domains, RFC 5737 documentation IPs, *.internal hosts, made-up repos.
FAKE_NAMES = (
    "Maya", "Jonas", "Priya", "Tomas", "Lena", "Omar", "Ines", "Felix", "Nora", "Arjun",
    "Clara", "Mateo", "Sofia", "Liam", "Hana", "Ravi", "Greta", "Yusuf", "Elena", "Kofi",
    "Mila", "Theo", "Aiko", "Bruno", "Zara", "Niko", "Freya", "Dario", "Leila", "Oskar",
)
URL_HOSTS = ("example.com", "app.example.com", "docs.example.org", "staging.example.net",
             "dashboard.example.com", "status.example.org", "api.example.net")
URL_PATHS = ("", "/docs/setup", "/settings/billing", "/api/v2/status", "/blog/release-notes",
             "/projects/42/runs", "/admin/users", "/pricing", "/changelog", "/issues/118",
             "/reports/weekly", "/login")
EMAIL_DOMAINS = ("example.com", "example.org", "example.net")
IP_NETS = ("192.0.2", "198.51.100", "203.0.113")
HOST_KINDS = ("host", "node", "worker", "build", "db", "cache", "api", "runner")
FAKE_PATHS = (
    "~/code/demo-app/src/routes/users.ts", "/srv/app/config/settings.yaml",
    "~/work/sample-service/scripts/deploy.sh", "/home/dev/widget-shop/web/components/Header.tsx",
    "~/code/demo-app/notes/todo.md", "/var/log/demo-app/worker.log",
    "~/projects/toy-agent/agent/loop.py", "/opt/sample-api/.env.example",
    "~/code/widget-shop/packages/ui/src/Button.tsx", "/srv/sample-api/migrations/0042_add_index.sql",
    "~/projects/demo-cli/cmd/root.go", "/home/dev/toy-agent/tests/test_loop.py",
    # directories, for "clone it into <PATH>"
    "~/code/demo-app", "/srv/sample-api", "~/projects/toy-agent/agent", "/home/dev/widget-shop",
    "~/work/sample-service/infra", "/opt/demo-cli", "~/code/widget-shop/packages/ui",
)
SECRET_PREFIXES = ("sk-test-", "test_secret_", "fake-token-", "dummy-key-")


def words(text):
    return len(text.split())


def est_tokens(*texts):
    return int(round(sum(len(t) for t in texts) / CHARS_PER_TOKEN)) + TEMPLATE_OVERHEAD_TOKENS


# ---------------------------------------------------------------------------
# Inputs

def read_pairs(path):
    """Parsed records and the number of lines that did not parse."""
    out, bad = [], 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except ValueError:
                bad += 1
                continue
            if isinstance(rec, dict) and isinstance(rec.get("input"), str) and isinstance(rec.get("target"), str):
                out.append(rec)
            else:
                bad += 1
    return out, bad


def load_exclude_ids(path):
    ids = set()
    if path and os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.split("#", 1)[0].strip()
                if line:
                    ids.add(line)
    return ids


def load_sessions(path):
    """{id: session key} from the clean extract (project and session id)."""
    out = {}
    if not path or not os.path.exists(path):
        return out
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except ValueError:
                continue
            out[r["id"]] = "%s/%s" % (r.get("project", ""), r.get("session_id", ""))
    return out


# ---------------------------------------------------------------------------
# Surrogates

class PlaceholderMismatch(ValueError):
    pass


def _rng(seed, key):
    return random.Random(hashlib.sha256(("%s:%s" % (seed, key)).encode()).digest())


class Surrogates:
    def __init__(self, blocked=()):
        """`blocked` terms (the denylist) are never drawn as surrogates."""
        deny = {b.lower() for b in blocked}
        self.names = [n for n in FAKE_NAMES if n.lower() not in deny] or ["Robin"]

    def _one(self, kind, rng):
        if kind == "NAME":
            return rng.choice(self.names)
        if kind == "URL":
            return "https://%s%s" % (rng.choice(URL_HOSTS), rng.choice(URL_PATHS))
        if kind == "EMAIL":
            return "%s@%s" % (rng.choice(self.names).lower(), rng.choice(EMAIL_DOMAINS))
        if kind == "IP":
            return "%s.%d" % (rng.choice(IP_NETS), rng.randint(2, 254))
        if kind == "HOST":
            return "%s-%02d.internal" % (rng.choice(HOST_KINDS), rng.randint(1, 40))
        if kind == "PATH":
            return rng.choice(FAKE_PATHS)
        if kind == "ID":
            if rng.random() < 0.3:
                h = "%032x" % rng.getrandbits(128)
                return "%s-%s-%s-%s-%s" % (h[:8], h[8:12], h[12:16], h[16:20], h[20:])
            return "%0*x" % (rng.choice((7, 8, 12)), rng.getrandbits(48))
        if kind == "SECRET":
            return rng.choice(SECRET_PREFIXES) + "%016x" % rng.getrandbits(64)
        raise KeyError(kind)

    def draw(self, kind, n, rng):
        """n distinct surrogates of one type (repeats only once a pool runs out)."""
        out, seen = [], set()
        for _ in range(n):
            for _attempt in range(20):
                v = self._one(kind, rng)
                if v not in seen:
                    break
            seen.add(v)
            out.append(v)
        return out

    def fill(self, pair_id, inp, tgt, seed=DEFAULT_SEED):
        """Return (input, target, counts by type). Raises PlaceholderMismatch
        when the two sides disagree on how many of a placeholder they hold."""
        ci = collections.Counter(PLACEHOLDER_RE.findall(inp))
        ct = collections.Counter(PLACEHOLDER_RE.findall(tgt))
        if ci != ct:
            diff = sorted(set(ci) | set(ct))
            raise PlaceholderMismatch(",".join(k for k in diff if ci[k] != ct[k]))
        if not ct:
            return inp, tgt, collections.Counter()
        rng = _rng(seed, pair_id)
        values = {k: self.draw(k, ct[k], rng) for k in TYPES if ct[k]}

        def sub(text):
            seen = collections.Counter()

            def fn(m):
                k = m.group(1)
                v = values[k][seen[k]]
                seen[k] += 1
                return v
            return PLACEHOLDER_RE.sub(fn, text)
        return sub(inp), sub(tgt), ct


# ---------------------------------------------------------------------------
# Split

def assign_splits(session_sizes, seed=DEFAULT_SEED, ratios=DEFAULT_RATIOS):
    """{session: split}. Sessions are shuffled with a fixed seed, then val and
    test are filled up to their share of pairs (a session that would overshoot
    a split by more than half its target goes on to the next); the rest is
    train. Every session lands in exactly one split."""
    total = sum(session_sizes.values())
    targets = {"val": ratios[1] * total, "test": ratios[2] * total}
    filled = {"val": 0, "test": 0}
    order = sorted(session_sizes)
    random.Random(seed).shuffle(order)
    out = {}
    for s in order:
        n = session_sizes[s]
        dest = "train"
        for split in ("test", "val"):
            t = targets[split]
            if filled[split] < t and filled[split] + n <= t * 1.5:
                dest = split
                break
        if dest != "train":
            filled[dest] += n
        out[s] = dest
    return out


def chat_record(pair_id, system_version, inp, tgt, variant=None):
    rec = {"messages": [
        {"role": "system", "content": SYSTEM_PROMPTS[system_version]},
        {"role": "user", "content": inp},
        {"role": "assistant", "content": tgt},
    ], "id": pair_id, "system_version": system_version}
    if variant:
        rec["variant"] = variant
    return rec


# ---------------------------------------------------------------------------
# The pass

def _norm(text):
    return " ".join(text.split())


def _quantile(sorted_vals, q):
    if not sorted_vals:
        return 0
    return sorted_vals[min(len(sorted_vals) - 1, int(q * len(sorted_vals)))]


def prepare(pairs, scrubber, sessions=None, exclude=(), allow_flags=DEFAULT_ALLOW_FLAGS,
            min_target_words=2, seed=DEFAULT_SEED, ratios=DEFAULT_RATIOS,
            system_version=DEFAULT_SYSTEM, max_seq_len=1024, require_session=True):
    """Return ({split: [chat records]}, stats, drops). `sessions` maps id to a
    session key; with require_session, ids missing from it are dropped."""
    drops = []
    reasons = collections.Counter()
    flagged = collections.Counter()
    flagged_kept = collections.Counter()
    rescrub_counts = collections.Counter()
    surrogated = collections.Counter()
    pairs_with_surrogates = 0
    allow = set(allow_flags)
    seen_inputs = set()
    kept = []
    sessions = sessions if sessions is not None else {}
    surr = Surrogates(scrubber.denylist)

    def drop(p, reason):
        reasons[reason] += 1
        drops.append({"id": p.get("id"), "reason": reason})

    for p in pairs:
        pid = p.get("id")
        if pid in exclude:
            drop(p, "excluded")
            continue
        if require_session and pid not in sessions:
            drop(p, "not_in_clean")
            continue
        inp, ci = scrubber.scrub(p["input"])
        tgt, ct = scrubber.scrub(p["target"])
        rescrub_counts.update({"input_" + k: v for k, v in ci.items()})
        rescrub_counts.update({"target_" + k: v for k, v in ct.items()})
        if words(tgt) < min_target_words:
            drop(p, "short_target")
            continue
        # Flags are checked on the rescrubbed pair, so a raw URL the generator
        # wrote where the target has <URL> is not a placeholder mismatch.
        flags = check_pairs.check(dict(p, input=inp, target=tgt))
        flagged.update(flags)
        bad = [f for f in flags if f not in allow]
        if bad:
            drop(p, "flag_" + bad[0])
            continue
        if META_PREAMBLE_RE.match(inp):
            drop(p, "meta_preamble")
            continue
        key = _norm(inp)
        if key in seen_inputs:
            drop(p, "duplicate_input")
            continue
        try:
            inp_s, tgt_s, counts = surr.fill(pid, inp, tgt, seed)
        except PlaceholderMismatch:
            drop(p, "placeholder_mismatch")
            continue
        seen_inputs.add(key)
        flagged_kept.update(flags)
        if counts:
            pairs_with_surrogates += 1
            surrogated.update({k: 2 * v for k, v in counts.items()})  # both sides
        kept.append((p, sessions.get(pid, "id/" + str(pid)), inp_s, tgt_s))

    sizes = collections.Counter(s for _, s, _, _ in kept)
    where = assign_splits(sizes, seed, ratios)
    out = {k: [] for k in SPLITS}
    tok = {k: [] for k in SPLITS}
    tgt_tok = {k: [] for k in SPLITS}
    sess_per_split = collections.defaultdict(set)
    system = SYSTEM_PROMPTS[system_version]
    for p, s, inp, tgt in kept:
        split = where[s]
        out[split].append(chat_record(p.get("id"), system_version, inp, tgt, p.get("variant")))
        tok[split].append(est_tokens(system, inp, tgt))
        tgt_tok[split].append(est_tokens(tgt) - TEMPLATE_OVERHEAD_TOKENS + 2)
        sess_per_split[split].add(s)

    all_tok = sorted(t for k in SPLITS for t in tok[k])
    over = sum(1 for t in all_tok if t > max_seq_len)
    stats = {
        "pairs_read": len(pairs),
        "kept": len(kept),
        "dropped": sum(reasons.values()),
        "drop_reasons": dict(reasons.most_common()),
        "check_pairs_flags": dict(flagged.most_common()),
        "check_pairs_flags_kept": dict(flagged_kept.most_common()),
        "allow_flags": sorted(allow),
        "min_target_words": min_target_words,
        "rescrub_new_replacements": dict(sorted(rescrub_counts.items())),
        "surrogates": {"pairs": pairs_with_surrogates, "placeholders": sum(surrogated.values()),
                       "by_type": dict(sorted(surrogated.items()))},
        "splits": {k: {"pairs": len(out[k]), "sessions": len(sess_per_split[k]),
                       "est_tokens": sum(tok[k]), "est_target_tokens": sum(tgt_tok[k])} for k in SPLITS},
        "sessions": len(sizes),
        "system_version": system_version,
        "seed": seed,
        "tokens": {
            "method": "chars/%.1f + %d template tokens" % (CHARS_PER_TOKEN, TEMPLATE_OVERHEAD_TOKENS),
            "total": sum(all_tok),
            "mean": round(sum(all_tok) / len(all_tok), 1) if all_tok else 0,
            "p50": _quantile(all_tok, 0.5), "p95": _quantile(all_tok, 0.95),
            "p99": _quantile(all_tok, 0.99), "max": all_tok[-1] if all_tok else 0,
            "max_seq_len": max_seq_len, "over_max_seq_len": over,
            "share_over": round(over / len(all_tok), 4) if all_tok else 0.0,
        },
    }
    return out, stats, drops


def write_jsonl(records, path):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def render(stats):
    lines = ["read %d pairs (%d unparsable lines skipped): kept %d, dropped %d" % (
        stats["pairs_read"], stats.get("bad_lines", 0), stats["kept"], stats["dropped"])]
    for k, v in stats["drop_reasons"].items():
        lines.append("  drop %-22s %5d" % (k, v))
    if stats["check_pairs_flags_kept"]:
        lines.append("  kept despite allowed flags: %s" % json.dumps(stats["check_pairs_flags_kept"]))
    for k in SPLITS:
        s = stats["splits"][k]
        lines.append("  %-5s %5d pairs  %4d sessions  ~%d tokens" % (k, s["pairs"], s["sessions"], s["est_tokens"]))
    t = stats["tokens"]
    lines.append("  tokens (%s): mean %s, p50 %d, p95 %d, max %d; %d (%.1f%%) over %d" % (
        t["method"], t["mean"], t["p50"], t["p95"], t["max"], t["over_max_seq_len"],
        100 * t["share_over"], t["max_seq_len"]))
    su = stats["surrogates"]
    lines.append("  surrogates: %d placeholders in %d pairs %s" % (su["placeholders"], su["pairs"], json.dumps(su["by_type"])))
    if stats["rescrub_new_replacements"]:
        lines.append("  rescrub replaced: %s" % json.dumps(stats["rescrub_new_replacements"]))
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="prepare", description=__doc__.split("\n\n")[0])
    ap.add_argument("--pairs", required=True, nargs="+", help="pairs jsonl file(s)")
    ap.add_argument("--out", default=os.path.join(ROOT, "data/train"))
    ap.add_argument("--clean", default=os.path.join(ROOT, "data/clean/cc_messages.jsonl"),
                    help="clean extract, for session ids")
    ap.add_argument("--denylist", default=os.path.join(ROOT, "scrub/denylist.txt"))
    ap.add_argument("--exclude", default=os.path.join(ROOT, "data/exclude_ids.local.txt"))
    ap.add_argument("--allow-flags", default=",".join(DEFAULT_ALLOW_FLAGS),
                    help="comma-separated check_pairs flags that do not drop a pair ('' for none)")
    ap.add_argument("--min-target-words", type=int, default=2)
    ap.add_argument("--seed", default=DEFAULT_SEED)
    ap.add_argument("--system", default=DEFAULT_SYSTEM, choices=sorted(SYSTEM_PROMPTS))
    ap.add_argument("--max-seq-len", type=int, default=1024, help="only for the share-over report")
    ap.add_argument("--keep-missing-session", action="store_true",
                    help="keep pairs whose id is not in the clean extract (each is its own session)")
    args = ap.parse_args(argv)

    pairs, bad = [], 0
    for path in args.pairs:
        p, b = read_pairs(path)
        pairs.extend(p)
        bad += b
    scrubber = Scrubber.from_files(args.denylist)
    sessions = load_sessions(args.clean)
    exclude = load_exclude_ids(args.exclude)
    allow = [f for f in args.allow_flags.split(",") if f.strip()]
    unknown = [f for f in allow if f not in check_pairs.FLAGS]
    if unknown:
        raise SystemExit("unknown flag(s) in --allow-flags: %s" % ", ".join(unknown))
    splits, stats, drops = prepare(
        pairs, scrubber, sessions=sessions, exclude=exclude, allow_flags=allow,
        min_target_words=args.min_target_words, seed=args.seed, system_version=args.system,
        max_seq_len=args.max_seq_len, require_session=not args.keep_missing_session)
    stats["bad_lines"] = bad
    stats["sources"] = [os.path.basename(p) for p in args.pairs]
    stats["denylist_terms"] = len(scrubber.denylist)
    stats["exclude_ids"] = len(exclude)
    os.makedirs(args.out, exist_ok=True)
    for k in SPLITS:
        write_jsonl(splits[k], os.path.join(args.out, "%s.jsonl" % k))
    write_jsonl(drops, os.path.join(args.out, "drops.jsonl"))
    with open(os.path.join(args.out, "stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
        f.write("\n")
    print(render(stats))
    print("wrote %s/{train,val,test}.jsonl, stats.json, drops.jsonl" % os.path.relpath(args.out, ROOT))
    return stats


if __name__ == "__main__":
    main()
