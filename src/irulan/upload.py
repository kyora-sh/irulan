"""Turn a raw extract or the studio's pairs into the dataset upload payload, locally.

Raw extract (`irulan.raw.v1`): each line of data/raw/cc_messages.jsonl becomes

    {"id": "...", "text": "...", "meta": {"source": "claude-code", "session": "<hash>",
                                          "timestamp": "...", "words": n, "lines": n}}

`project` (the transcript directory name, which embeds your local path and
user name) is not uploaded; `session` is a hash of project and session id, so
the platform can still split train/val/test by session.

Handwritten pairs (`chat.v1`, for kyora's `sft` recipe): each line of
data/handwritten/pairs.jsonl becomes one conversation with the system_v1
instruction `irulan rewrite` sends, the assistant-style text as the user
turn and your version as the assistant turn:

    {"id": "hw_...", "messages": [{"role": "system", ...}, {"role": "user", "content": input},
                                  {"role": "assistant", "content": target}],
     "meta": {"source": "handwritten", "session": "handwritten-<day>", "timestamp": created_at}}
"""

import collections
import hashlib
import json
import re

from . import redact as _redact
from .rewrite import SYSTEM_PROMPT

FORMAT = "irulan.raw.v1"
PAIRS_FORMAT = "chat.v1"
PAIRS_RECIPE = "sft"
KINDS = ("raw", "pairs")

# kyora's data handling for uploads (docs/API.md, "Data handling").
POLICY_LINES = (
    "This upload contains your message text UNSCRUBBED: names, hosts, URLs, paths and IDs",
    "are not removed on this machine. Review the file before you send it.",
    "On upload, kyora first scrubs the data. Only the scrubbed version is stored, encrypted",
    "at rest. The original (unscrubbed) upload is not retained.",
)
SOURCE = "claude-code"


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def session_key(rec):
    h = hashlib.sha256(("%s/%s" % (rec.get("project") or "", rec.get("session_id") or "")).encode("utf-8"))
    return h.hexdigest()[:16]


def to_upload_record(rec):
    if not isinstance(rec.get("text"), str) or not rec["text"].strip():
        raise ValueError("record %r has no text" % rec.get("id"))
    text = rec["text"]
    rid = rec.get("id") or hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    meta = {"source": SOURCE, "session": session_key(rec), "timestamp": rec.get("timestamp"),
            "words": rec.get("words", len(text.split())), "lines": rec.get("lines", text.count("\n") + 1)}
    return {"id": rid, "text": text, "meta": meta}


def _text(rec, key, lineno):
    v = rec.get(key)
    if not isinstance(v, str) or not v.strip():
        raise ValueError("line %d: %s must be a non-empty string" % (lineno, key))
    return v.strip()


def to_pair_record(rec, lineno=0):
    if not isinstance(rec, dict):
        raise ValueError("line %d: not a pair object" % lineno)
    rid, inp, tgt = _text(rec, "id", lineno), _text(rec, "input", lineno), _text(rec, "target", lineno)
    created = rec.get("created_at") if isinstance(rec.get("created_at"), str) else None
    day = created[:10] if created and re.match(r"\d{4}-\d{2}-\d{2}", created) else None
    meta = {"source": "handwritten", "session": "handwritten-" + (day or "undated")}
    if created:
        meta["timestamp"] = created
    return {"id": rid,
            "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                         {"role": "user", "content": inp},
                         {"role": "assistant", "content": tgt}],
            "meta": meta}


def read_pairs(path):
    """Pair records from the studio's file. Fails on a line that is not a
    pair, rather than leaving it out."""
    out = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except ValueError:
                raise ValueError("line %d of %s is not JSON" % (lineno, path))
            out.append(to_pair_record(rec, lineno))
    return out


def _redact_record(r):
    counts = collections.Counter()
    if "messages" in r:
        for m in r["messages"][1:]:
            m["content"], c = _redact.redact(m["content"])
            counts.update(c)
    else:
        r["text"], c = _redact.redact(r["text"])
        counts.update(c)
        r["meta"]["words"] = len(r["text"].split())
        r["meta"]["lines"] = r["text"].count("\n") + 1
    return counts


def _words(r):
    if "messages" in r:
        return sum(len(m["content"].split()) for m in r["messages"][1:])
    return r["meta"]["words"]


def build(path, redact_secrets=False, kind="raw"):
    """Return (jsonl bytes, summary). Nothing leaves the machine here."""
    if kind not in KINDS:
        raise ValueError("kind must be one of %s" % ", ".join(KINDS))
    if kind == "pairs":
        records = read_pairs(path)
    else:
        records = [to_upload_record(r) for r in read_jsonl(path)]
    redacted = collections.Counter()
    touched = 0
    if redact_secrets:
        for r in records:
            counts = _redact_record(r)
            if counts:
                touched += 1
                redacted.update(counts)
    ids = collections.Counter(r["id"] for r in records)
    dupes = [i for i, n in ids.items() if n > 1]
    if dupes:
        raise ValueError("%d duplicate ids in %s (first: %s)" % (len(dupes), path, dupes[0]))
    data = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records).encode("utf-8")
    summary = {
        "path": path,
        "kind": kind,
        "format": PAIRS_FORMAT if kind == "pairs" else FORMAT,
        "recipe": PAIRS_RECIPE if kind == "pairs" else None,
        "records": len(records),
        "sessions": len({r["meta"]["session"] for r in records}),
        "words": sum(_words(r) for r in records),
        "bytes": len(data),
        "redacted_locally": bool(redact_secrets),
        "secrets_redacted": dict(sorted(redacted.items())),
        "records_with_secrets_redacted": touched,
    }
    return data, summary


def render_summary(s):
    unit = "pairs" if s.get("kind") == "pairs" else "messages"
    recipe = " for the %s recipe" % s["recipe"] if s.get("recipe") else ""
    lines = [
        "dataset file: %s" % s["path"],
        "  %d %s, %d sessions, %d words, %.1f KB as %s%s" % (
            s["records"], unit, s["sessions"], s["words"], s["bytes"] / 1024.0, s["format"], recipe),
    ]
    if s["redacted_locally"]:
        n = sum(s["secrets_redacted"].values())
        lines.append("  local secret redaction: %d replaced in %d %s %s" % (
            n, s["records_with_secrets_redacted"], unit, json.dumps(s["secrets_redacted"]) if n else ""))
    else:
        lines.append("  local secret redaction: off (pass --redact-secrets-locally to replace obvious credentials)")
    lines.append("")
    lines.extend(POLICY_LINES)
    return "\n".join(lines)
