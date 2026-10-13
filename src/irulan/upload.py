"""Turn a raw extract into the dataset upload payload, locally.

Each line of data/raw/cc_messages.jsonl becomes one upload record:

    {"id": "...", "text": "...", "meta": {"source": "claude-code", "session": "<hash>",
                                          "timestamp": "...", "words": n, "lines": n}}

`project` (the transcript directory name, which embeds your local path and
user name) is not uploaded; `session` is a hash of project and session id, so
the platform can still split train/val/test by session.
"""

import collections
import hashlib
import json

from . import redact as _redact

FORMAT = "irulan.raw.v1"

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


def build(path, redact_secrets=False):
    """Return (jsonl bytes, summary). Nothing leaves the machine here."""
    records = [to_upload_record(r) for r in read_jsonl(path)]
    redacted = collections.Counter()
    touched = 0
    if redact_secrets:
        for r in records:
            text, counts = _redact.redact(r["text"])
            if counts:
                touched += 1
                redacted.update(counts)
                r["text"] = text
                r["meta"]["words"] = len(text.split())
                r["meta"]["lines"] = text.count("\n") + 1
    ids = collections.Counter(r["id"] for r in records)
    dupes = [i for i, n in ids.items() if n > 1]
    if dupes:
        raise ValueError("%d duplicate ids in %s (first: %s)" % (len(dupes), path, dupes[0]))
    data = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records).encode("utf-8")
    summary = {
        "path": path,
        "format": FORMAT,
        "records": len(records),
        "sessions": len({r["meta"]["session"] for r in records}),
        "words": sum(r["meta"]["words"] for r in records),
        "bytes": len(data),
        "redacted_locally": bool(redact_secrets),
        "secrets_redacted": dict(sorted(redacted.items())),
        "records_with_secrets_redacted": touched,
    }
    return data, summary


def render_summary(s):
    lines = [
        "dataset file: %s" % s["path"],
        "  %d messages, %d sessions, %d words, %.1f KB as %s" % (
            s["records"], s["sessions"], s["words"], s["bytes"] / 1024.0, s["format"]),
    ]
    if s["redacted_locally"]:
        n = sum(s["secrets_redacted"].values())
        lines.append("  local secret redaction: %d replaced in %d messages %s" % (
            n, s["records_with_secrets_redacted"], json.dumps(s["secrets_redacted"]) if n else ""))
    else:
        lines.append("  local secret redaction: off (pass --redact-secrets-locally to replace obvious credentials)")
    lines.append("")
    lines.extend(POLICY_LINES)
    return "\n".join(lines)
