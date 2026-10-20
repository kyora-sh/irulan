"""Handwritten pairs: the record format, the pairs file and local checks.

One pair per line of the pairs file (default DATA_DIR/handwritten/pairs.jsonl):

    {"id": "hw_<16 hex>", "source": "handwritten", "shape", "length", "topic",
     "generator": "pasted" | {"backend", "model", "prompt_version"},
     "input", "target", "created_at"}

`input` is the assistant-style text and `target` the writer's own version.
The id is a hash of both sides, so the same pair is never saved twice.
"""

import collections
import datetime
import hashlib
import json
import os
import re
import threading

from .prompts import LENGTHS, SHAPES, StudioError

FIELDS = ("id", "source", "shape", "length", "topic", "generator", "input", "target", "created_at")
EDITABLE = {"input", "target", "topic", "shape", "length"}


def word_count(text):
    return len(text.split())


def pair_id(input_text, target_text):
    h = hashlib.sha256((input_text.strip() + "\n\0\n" + target_text.strip()).encode("utf-8")).hexdigest()
    return "hw_" + h[:16]


def now_iso():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def make_record(input_text, target_text, shape, length, topic="", generator="pasted", created_at=None):
    input_text, target_text = (input_text or "").strip(), (target_text or "").strip()
    if not input_text or not target_text:
        raise StudioError("both sides need text")
    if shape not in SHAPES:
        raise StudioError("unknown shape: %s" % shape)
    if length not in LENGTHS:
        raise StudioError("unknown length: %s" % length)
    if generator != "pasted" and not (isinstance(generator, dict)
                                      and set(generator) == {"backend", "model", "prompt_version"}):
        raise StudioError("generator must be 'pasted' or {backend, model, prompt_version}")
    return {
        "id": pair_id(input_text, target_text),
        "source": "handwritten",
        "shape": shape,
        "length": length,
        "topic": (topic or "").strip(),
        "generator": generator,
        "input": input_text,
        "target": target_text,
        "created_at": created_at or now_iso(),
    }


def counts(records):
    by_shape = collections.Counter(r.get("shape") for r in records)
    return {"total": len(records), "by_shape": {k: by_shape[k] for k in SHAPES if by_shape[k]}}


def repair_tail(path):
    """Drop a half-written last line (left by a hard kill) so appends stay valid JSONL."""
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return 0
    with open(path, "rb+") as f:
        f.seek(-1, os.SEEK_END)
        if f.read(1) == b"\n":
            return 0
        f.seek(0)
        data = f.read()
        keep = data.rfind(b"\n") + 1
        f.truncate(keep)
        return len(data) - keep


class PairStore:
    """The pairs file. Appends are one line each; edits and deletes rewrite
    the file through a temp file and os.replace. Lines that do not parse are
    kept verbatim on rewrite."""

    def __init__(self, path):
        self.path = path
        self.lock = threading.Lock()

    def _entries(self):
        out = []
        if not os.path.exists(self.path):
            return out
        with open(self.path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except ValueError:
                    out.append(line.rstrip("\n"))
                    continue
                out.append(rec if isinstance(rec, dict) else line.rstrip("\n"))
        return out

    def _rewrite(self, entries):
        d = os.path.dirname(self.path) or "."
        os.makedirs(d, exist_ok=True)
        tmp = "%s.tmp-%d" % (self.path, os.getpid())
        with open(tmp, "w", encoding="utf-8") as f:
            for e in entries:
                f.write((e if isinstance(e, str) else json.dumps(e, ensure_ascii=False)) + "\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self.path)

    def records(self):
        with self.lock:
            return [e for e in self._entries() if isinstance(e, dict)]

    def add(self, rec):
        """Append; a pair already saved (same id) is not written twice."""
        with self.lock:
            for e in self._entries():
                if isinstance(e, dict) and e.get("id") == rec["id"]:
                    return e, False
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
            repair_tail(self.path)
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                os.fsync(f.fileno())
            return rec, True

    def update(self, pid, changes):
        bad = set(changes) - EDITABLE
        if bad:
            raise StudioError("cannot edit: %s" % ", ".join(sorted(bad)))
        with self.lock:
            entries = self._entries()
            idx = next((i for i, e in enumerate(entries) if isinstance(e, dict) and e.get("id") == pid), None)
            if idx is None:
                raise StudioError("no pair %s" % pid, 404)
            old = entries[idx]
            merged = dict(old, **changes)
            generator = old.get("generator", "pasted")
            if "input" in changes and changes["input"].strip() != old.get("input", "").strip():
                generator = "pasted"
            new = make_record(merged["input"], merged["target"], merged["shape"], merged["length"],
                              merged.get("topic", ""), generator, old.get("created_at"))
            if any(isinstance(e, dict) and e.get("id") == new["id"] for i, e in enumerate(entries) if i != idx):
                raise StudioError("an identical pair is already saved", 409)
            entries[idx] = new
            self._rewrite(entries)
            return new

    def delete(self, pid):
        with self.lock:
            entries = self._entries()
            keep = [e for e in entries if not (isinstance(e, dict) and e.get("id") == pid)]
            if len(keep) == len(entries):
                raise StudioError("no pair %s" % pid, 404)
            self._rewrite(keep)


# ---------------------------------------------------------------------------
# Checks: specifics that only one side has. A rewrite keeps the facts, so a
# URL, number or code span on one side only is usually a slip.

URL_RE = re.compile(r"\b(?:https?://|www\.)[^\s<>()\"'`]+", re.I)
NUMBER_RE = re.compile(r"(?<![\d.])\d+(?:[.,:]\d+)*")
CODE_RE = re.compile(r"`([^`\n]+)`")
LIST_MAX = 4


def _urls(text):
    return {u.rstrip(".,;:!?") for u in URL_RE.findall(text)}


def _numbers(text):
    return set(NUMBER_RE.findall(URL_RE.sub(" ", text)))


def _codes(text):
    return {c.strip() for c in CODE_RE.findall(text) if c.strip()}


def _listing(items):
    items = sorted(items)
    more = len(items) - LIST_MAX
    return ", ".join(items[:LIST_MAX]) + (" and %d more" % more if more > 0 else "")


def check_pair(input_text, target_text):
    """Warnings for one pair. They never block a save."""
    inp, tgt = (input_text or "").strip(), (target_text or "").strip()
    if not tgt:
        return ["the target is empty"]
    if inp == tgt:
        return ["the target is identical to the input"]
    out = []
    for what, fn in (("URL", _urls), ("number", _numbers), ("code span", _codes)):
        a, b = fn(inp), fn(tgt)
        if what == "code span":  # the rewrite may drop the backticks
            only = ({x for x in a if x not in tgt}, {x for x in b if x not in inp})
        else:
            only = (a - b, b - a)
        for side, items in zip(("the input", "your version"), only):
            if items:
                out.append("%s%s only in %s: %s" % (what, "s" if len(items) > 1 else "", side, _listing(items)))
    return out


def stats(records):
    n = len(records)
    gen = collections.Counter("pasted" if r.get("generator") == "pasted" else
                              (r.get("generator") or {}).get("model", "?") for r in records)
    ratios = sorted(word_count(r["target"]) / max(1, word_count(r["input"])) for r in records)
    warned = sum(1 for r in records if check_pair(r["input"], r["target"]))
    return {
        "total": n,
        "by_shape": counts(records)["by_shape"],
        "by_length": dict(collections.Counter(r.get("length") for r in records).most_common()),
        "by_generator": dict(gen.most_common()),
        "input_words_mean": round(sum(word_count(r["input"]) for r in records) / n, 1) if n else 0,
        "target_words_mean": round(sum(word_count(r["target"]) for r in records) / n, 1) if n else 0,
        "target_to_input_ratio_median": round(ratios[n // 2], 2) if n else 0,
        "with_warnings": warned,
    }


def render_stats(s):
    lines = ["%d handwritten pairs (%d with check warnings)" % (s["total"], s["with_warnings"])]
    for k, v in s["by_shape"].items():
        lines.append("  %-22s %4d" % (k, v))
    lines.append("  length: %s" % json.dumps(s["by_length"]))
    lines.append("  generator: %s" % json.dumps(s["by_generator"]))
    lines.append("  words: input mean %s, target mean %s, target/input median %s" % (
        s["input_words_mean"], s["target_words_mean"], s["target_to_input_ratio_median"]))
    return "\n".join(lines)
