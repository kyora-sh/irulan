"""Studio: a local page for writing handwritten training pairs.

The page shows LLM-assistant-style text of a chosen shape (generated with
codex, or pasted) next to a textarea for the writer's own version. Each saved
pair is one line of data/handwritten/pairs.jsonl:

    {id, source, shape, length, topic, generator, input, target, created_at}

`input` is the assistant-style text and `target` the writer's version, so
`prepare --extra-pairs data/handwritten/pairs.jsonl` adds them to the training
set next to the back-translated pairs.

    python3 -I pipeline.py studio [--port 8765] [--data FILE]
    python3 -I pipeline.py studio --stats

The server binds 127.0.0.1 only and runs at most one codex call at a time.
"""

import argparse
import collections
import datetime
import hashlib
import http.server
import json
import os
import random
import re
import subprocess
import sys
import threading
import urllib.parse

from . import generate, prepare
from .scrub import Scrubber

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DATA = os.path.join(ROOT, "data/handwritten/pairs.jsonl")
HTML_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "studio.html")
PROMPT_VERSION = "studio_v1"
DEFAULT_MODEL = "gpt-6.1-sol"
DEFAULT_EFFORT = "low"
LENGTHS = ("short", "medium", "long")
FIELDS = ("id", "source", "shape", "length", "topic", "generator", "input", "target", "created_at")
MAX_BODY = 1 << 20

Shape = collections.namedtuple("Shape", "label words structure tone")

# words: (short, medium, long) word ranges.
SHAPES = collections.OrderedDict([
    ("pr_description", Shape(
        "PR description", ((80, 150), (150, 300), (300, 550)),
        "Markdown. A one-line summary, then `## Summary`, `## Changes` (a bulleted list), `## Testing` and "
        "optionally `## Notes` or `## Risks`. File paths, flags and config keys in inline code.",
        "confident, thorough, reviewer-friendly")),
    ("pr_review_comment", Shape(
        "PR review comment", ((30, 60), (60, 120), (120, 220)),
        "One inline review comment on a specific file and line: a brief acknowledgement, the concern, a "
        "suggested fix (optionally a short code block), and a closing question.",
        "collegial and hedged (\"nit:\", \"might be worth\", \"wdyt?\" written out in full)")),
    ("issue", Shape(
        "Issue", ((60, 120), (120, 250), (250, 450)),
        "Markdown with a title line, then `## Summary`, `## Steps to reproduce` (numbered), `## Expected "
        "behavior`, `## Actual behavior`, `## Environment` and optionally `## Possible cause`.",
        "precise, neutral, a little formal")),
    ("slack_status", Shape(
        "Slack status update", ((30, 60), (60, 120), (120, 200)),
        "A channel update with bolded labels (**Done:**, **In progress:**, **Blockers:**, **Next:**), short "
        "bullets under each, maybe one emoji code such as :white_check_mark:.",
        "upbeat, organised, reassuring")),
    ("slack_thread_reply", Shape(
        "Slack reply in a thread", ((20, 50), (50, 100), (100, 180)),
        "A reply to a teammate's question in a thread: an acknowledging opener (\"Great question!\"), the "
        "answer, a few bullets if there are several points, an offer to help further.",
        "friendly, helpful, slightly over-explained")),
    ("email_internal", Shape(
        "Email (internal)", ((80, 150), (150, 280), (280, 450)),
        "A `Subject:` line, a greeting, a context paragraph, a bulleted list of points or asks, next steps, "
        "and a sign-off with an invented first name.",
        "professional, warm, structured")),
    ("email_external", Shape(
        "Email (external / customer)", ((100, 180), (180, 300), (300, 450)),
        "A `Subject:` line, a greeting by an invented name, appreciation or an apology, a clear explanation "
        "without internal jargon, numbered next steps, an offer to set up a call, and a signature with an "
        "invented name and role.",
        "polite, empathetic, customer-facing")),
    ("design_doc_section", Shape(
        "Design doc section", ((150, 250), (250, 450), (450, 700)),
        "One section of a design doc (such as `## Proposed approach` or `## Alternatives considered`) with "
        "subheadings, trade-off bullets and optionally a small markdown table.",
        "analytical, balanced, thorough")),
    ("incident_summary", Shape(
        "Incident summary / postmortem", ((120, 220), (220, 400), (400, 650)),
        "Markdown: `## Summary`, `## Impact` (duration, share of requests, users affected), `## Timeline` "
        "(UTC times as bullets), `## Root cause`, `## Resolution`, `## Action items` (checkboxes with owners).",
        "blameless, calm, precise")),
    ("release_notes", Shape(
        "Release notes", ((60, 120), (120, 250), (250, 450)),
        "A version heading, then sections such as `### New`, `### Improvements`, `### Fixes` and "
        "`### Breaking changes`, bullets that start with a **bold feature name**.",
        "polished, enthusiastic, user-facing")),
    ("linkedin_post", Shape(
        "LinkedIn post", ((80, 150), (150, 250), (250, 400)),
        "A one-line hook, short paragraphs separated by blank lines, a short list of lessons, a reflective "
        "close, a question to readers and three to five hashtags.",
        "inspirational, humble-brag, reflective")),
    ("tweet_thread", Shape(
        "Tweet / thread", ((25, 45), (90, 160), (180, 300)),
        "Short: a single tweet under 280 characters. Medium and long: a thread of 3-5 or 6-9 tweets "
        "numbered `1/`, `2/`, ..., each under 280 characters, the first one a hook.",
        "punchy, engagement-seeking, a little breathless")),
    ("commit_message", Shape(
        "Commit message", ((15, 40), (40, 90), (90, 160)),
        "A conventional-commit subject line under 72 characters, a blank line, then a body wrapped at 72 "
        "columns with a short bulleted list of changes and a closing line on why.",
        "formal, complete, explanatory")),
    ("standup", Shape(
        "Standup update", ((25, 50), (50, 90), (90, 150)),
        "Bolded labels **Yesterday:**, **Today:** and **Blockers:**, each with one to three bullets.",
        "tidy, positive, complete sentences")),
    ("custom", Shape(
        "Custom", ((50, 100), (100, 250), (250, 450)),
        "", "")),
])

# Invented first names offered per call, so the model does not settle on one.
PEOPLE = (
    "Anouk", "Bilal", "Carmen", "Desmond", "Esra", "Farid", "Gwen", "Hiroshi", "Ilse", "Jarek",
    "Kemal", "Lucia", "Magnus", "Nadia", "Oren", "Paloma", "Quentin", "Rosa", "Stellan", "Tamsin",
    "Ugo", "Vera", "Wendell", "Ximena", "Yara", "Zoltan", "Amara", "Bastian", "Celeste", "Dov",
)

# Areas for an invented topic when none is given, so topics do not converge.
AREAS = (
    "payments and billing", "search indexing", "authentication and sessions", "CI pipelines",
    "mobile app releases", "data pipelines and ETL", "observability and alerting", "caching",
    "database migrations", "feature flags", "rate limiting", "background job queues",
    "frontend performance", "API versioning", "on-call tooling", "cloud cost reduction",
    "ML model serving", "access control and permissions", "file uploads and storage",
    "notifications and email delivery", "developer tooling", "dependency upgrades",
    "multi-region failover", "data retention and deletion", "load testing", "schema validation",
)

PROMPT_V1 = """You are helping build a style-transfer dataset. Write one realistic example of the text an LLM assistant such as Claude or ChatGPT produces when a software engineer asks it to draft the piece below. A person will later rewrite it in their own voice, so it must read unmistakably like assistant-drafted text.

Shape: {label}
Structure: {structure}
Tone: {tone}
Length: about {lo}-{hi} words.
{topic_line}

Write in the assistant register: polished, complete sentences; mild hedging ("It might be worth...", "I think we should consider..."); smooth transitions ("Additionally,", "That said,", "Overall,"); em dashes; bolded labels and tidy bullet lists wherever the shape allows them; a short summary or an offer to help at the end where it fits the shape.

Include concrete specifics that a rewrite has to keep: numbers (latencies, percentages, counts, dates, durations), names of invented services, components and people, relative file paths in a repository, config keys, version numbers and ticket numbers. Invent fresh, varied names and values for this piece; reuse nothing from these instructions. If people are mentioned, name them something like {people}.

Keep everything invented: no real companies, products or people, apart from widely used open-source tools and languages (Postgres, Kafka, Python and the like). Use no URLs, email addresses, IP addresses, absolute or home-directory paths, commit hashes, or anything that looks like a key or secret. Write as if everything were real: never call anything invented, fictional, hypothetical or an example in the text itself.

Do not use tools or run commands. {output_rule}"""

PROMPTS = {"studio_v1": PROMPT_V1}

OUTPUT_ONLY = "Output only the text itself: no preamble, no notes, no quotation marks around it."
OUTPUT_WITH_TOPIC = ("First write one line `TOPIC: <the topic you chose, under 15 words>`, then a line `---`, "
                     "then the text itself. Nothing else: no preamble, no notes.")
TOPIC_RE = re.compile(r"(?is)^\s*\**\s*topic\s*:\s*\**\s*(?P<topic>[^\n]*)\n(?:\s*-{3,}\s*\n)?(?P<text>.*)$")


def display_path(path):
    rel = os.path.relpath(os.path.abspath(path), ROOT)
    return path if rel.startswith("..") else rel


class StudioError(Exception):
    def __init__(self, msg, status=400):
        super().__init__(msg)
        self.status = status


def word_count(text):
    return len(text.split())


def build_prompt(shape, length, topic="", custom="", area=None, version=PROMPT_VERSION, rng=None):
    """Prompt for one piece of assistant-style text. With no topic, the model
    invents one in `area` and reports it on a TOPIC line."""
    rng = rng or random
    if shape not in SHAPES:
        raise StudioError("unknown shape: %s" % shape)
    if length not in LENGTHS:
        raise StudioError("unknown length: %s" % length)
    s = SHAPES[shape]
    lo, hi = s.words[LENGTHS.index(length)]
    label, structure, tone = s.label, s.structure, s.tone
    if shape == "custom":
        custom = (custom or "").strip()
        if not custom:
            raise StudioError("the custom shape needs a description")
        label = custom
        structure = "whatever is typical for this kind of text"
        tone = "the usual assistant register for it"
    topic = (topic or "").strip()
    if topic:
        topic_line = "Topic: %s" % topic
        output_rule = OUTPUT_ONLY
    else:
        area = area or rng.choice(AREAS)
        topic_line = ("Topic: invent a plausible, specific software-engineering situation in the area of %s. "
                      "It must be generic and invented, not about any real company." % area)
        output_rule = OUTPUT_WITH_TOPIC
    people = " or ".join(rng.sample(PEOPLE, 3))
    return PROMPTS[version].format(label=label, structure=structure, tone=tone, lo=lo, hi=hi,
                                   topic_line=topic_line, output_rule=output_rule, people=people)


def split_topic(text):
    """(topic, text) from a reply that may open with a TOPIC line."""
    t = generate.clean_output(text)
    m = TOPIC_RE.match(t)
    if not m:
        return "", t
    return m.group("topic").strip().strip("*").strip(), m.group("text").strip()


# ---------------------------------------------------------------------------
# Records

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
            generate.repair_tail(self.path)
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                os.fsync(f.fileno())
            return rec, True

    def update(self, pid, changes):
        allowed = {"input", "target", "topic", "shape", "length"}
        bad = set(changes) - allowed
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
# prepare preview: what the training-set pass would do with these pairs

def prepare_preview(records, scrubber):
    """{kept, drop_reasons, drops} for the handwritten pairs alone."""
    _, stats, drops = prepare.prepare(records, scrubber, sessions=prepare.handwritten_sessions(records))
    return {"kept": stats["kept"], "drop_reasons": stats["drop_reasons"], "drops": drops}


def placeholder_diff(rec, scrubber):
    """Placeholder types the two sides disagree on after scrubbing."""
    ci = collections.Counter(prepare.PLACEHOLDER_RE.findall(scrubber.scrub(rec["input"])[0]))
    ct = collections.Counter(prepare.PLACEHOLDER_RE.findall(scrubber.scrub(rec["target"])[0]))
    return sorted(k for k in set(ci) | set(ct) if ci[k] != ct[k])


def warnings_for(rec, scrubber):
    out = []
    pv = prepare_preview([rec], scrubber)
    for d in pv["drops"]:
        msg = "prepare would drop this pair: %s" % d["reason"]
        if d["reason"] in ("placeholder_mismatch", "flag_placeholder_added", "flag_placeholder_dropped"):
            msg += " (%s count differs after scrubbing; keep or drop it on both sides)" % ", ".join(
                placeholder_diff(rec, scrubber))
        out.append(msg)
    return out


def stats(records, scrubber):
    n = len(records)
    gen = collections.Counter("pasted" if r.get("generator") == "pasted" else
                              (r.get("generator") or {}).get("model", "?") for r in records)
    ratios = sorted(word_count(r["target"]) / max(1, word_count(r["input"])) for r in records)
    pv = prepare_preview(records, scrubber)
    return {
        "total": n,
        "by_shape": counts(records)["by_shape"],
        "by_length": dict(collections.Counter(r.get("length") for r in records).most_common()),
        "by_generator": dict(gen.most_common()),
        "input_words_mean": round(sum(word_count(r["input"]) for r in records) / n, 1) if n else 0,
        "target_words_mean": round(sum(word_count(r["target"]) for r in records) / n, 1) if n else 0,
        "target_to_input_ratio_median": round(ratios[n // 2], 2) if n else 0,
        "prepare": {"kept": pv["kept"], "drop_reasons": pv["drop_reasons"]},
    }


def render_stats(s):
    lines = ["%d handwritten pairs (prepare would keep %d)" % (s["total"], s["prepare"]["kept"])]
    for k, v in s["by_shape"].items():
        lines.append("  %-22s %4d" % (k, v))
    lines.append("  length: %s" % json.dumps(s["by_length"]))
    lines.append("  generator: %s" % json.dumps(s["by_generator"]))
    lines.append("  words: input mean %s, target mean %s, target/input median %s" % (
        s["input_words_mean"], s["target_words_mean"], s["target_to_input_ratio_median"]))
    if s["prepare"]["drop_reasons"]:
        lines.append("  prepare drops: %s" % json.dumps(s["prepare"]["drop_reasons"]))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Generators

class FakeGenerator:
    """Deterministic, no network; for tests."""

    def __init__(self):
        self.info = {"backend": "fake", "model": "fake-1", "prompt_version": PROMPT_VERSION}
        self.calls = 0

    def status(self):
        return True, ""

    def generate(self, prompt):
        self.calls += 1
        text = ("**Summary:** This change moves the `ledger-sync` retry loop into "
                "services/ledger/retry.py and cuts p95 latency from 840 ms to 310 ms. Additionally, "
                "it bumps the client to v2.3.1.\n\nThat said, it might be worth a second look.")
        if "TOPIC:" in prompt:
            return "TOPIC: retry loop for an invented ledger sync service\n---\n" + text
        return text


class CodexGenerator:
    """generate.CodexBackend (codex exec from an empty temp dir, read-only
    sandbox, prompt on stdin, last message captured)."""

    def __init__(self, model=DEFAULT_MODEL, effort=DEFAULT_EFFORT, timeout=240, tmp_root=None):
        self.backend = generate.make_backend("codex", model, effort=effort, timeout=timeout, tmp_root=tmp_root)
        self.info = {"backend": "codex", "model": self.backend.model, "prompt_version": PROMPT_VERSION}
        self._status = None

    def status(self):
        if self._status is None:
            ok, why = self.backend.available()
            if ok:
                try:
                    p = subprocess.run([self.backend.opts.get("codex_bin") or "codex", "login", "status"],
                                       capture_output=True, text=True, timeout=30)
                    out = (p.stdout + p.stderr).strip()
                    if p.returncode != 0 or re.search(r"(?i)not logged in", out):
                        ok, why = False, "codex is not logged in (run `codex login`): %s" % out[-200:]
                except (OSError, subprocess.TimeoutExpired) as e:
                    ok, why = False, "could not check codex login: %s" % e
            else:
                why = "codex CLI not found on PATH"
            self._status = (ok, why)
        return self._status

    def generate(self, prompt):
        try:
            return self.backend.call(prompt)["text"]
        except generate.BackendError as e:
            raise StudioError("codex failed: %s" % e, 502)


# ---------------------------------------------------------------------------
# HTTP

class StudioServer(http.server.ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, addr, store, generator, scrubber, quiet=False):
        super().__init__(addr, Handler)
        self.quiet = quiet
        self.store = store
        self.generator = generator
        self.scrubber = scrubber
        self.gen_lock = threading.Lock()
        self.generated = {}  # sha256 of a generated text -> generator info

    def remember(self, text):
        self.generated[hashlib.sha256(text.strip().encode("utf-8")).hexdigest()] = dict(self.generator.info)

    def generator_for(self, text):
        return self.generated.get(hashlib.sha256(text.strip().encode("utf-8")).hexdigest(), "pasted")


class Handler(http.server.BaseHTTPRequestHandler):
    server_version = "irulan-studio"

    def log_message(self, fmt, *args):
        if not self.server.quiet:
            sys.stderr.write("studio: %s\n" % (fmt % args))

    def _send(self, status, body, ctype="application/json; charset=utf-8"):
        data = body if isinstance(body, bytes) else json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(data)

    def _local_only(self):
        """Refuse other Host headers (DNS rebinding) and cross-origin writes."""
        port = self.server.server_address[1]
        hosts = {"127.0.0.1:%d" % port, "localhost:%d" % port}
        if self.headers.get("Host") not in hosts:
            self._send(403, {"error": "bad host"})
            return False
        origin = self.headers.get("Origin")
        if origin and urllib.parse.urlsplit(origin).netloc not in hosts:
            self._send(403, {"error": "bad origin"})
            return False
        return True

    def _body(self):
        if not (self.headers.get("Content-Type") or "").startswith("application/json"):
            raise StudioError("expected application/json", 415)
        n = int(self.headers.get("Content-Length") or 0)
        if n > MAX_BODY:
            raise StudioError("body too large", 413)
        try:
            body = json.loads(self.rfile.read(n) or b"{}")
        except ValueError:
            raise StudioError("invalid json")
        if not isinstance(body, dict):
            raise StudioError("expected a json object")
        return body

    def _dispatch(self, method):
        if not self._local_only():
            return
        path = urllib.parse.urlsplit(self.path).path
        try:
            if method == "GET" and path in ("/", "/index.html"):
                with open(HTML_PATH, "rb") as f:
                    return self._send(200, f.read(), "text/html; charset=utf-8")
            if method == "GET" and path == "/api/status":
                return self._send(200, self._status())
            if method == "GET" and path == "/api/pairs":
                recs = self.server.store.records()
                return self._send(200, {"pairs": list(reversed(recs)), "counts": counts(recs)})
            if method == "GET" and path == "/api/stats":
                s = stats(self.server.store.records(), self.server.scrubber)
                return self._send(200, dict(s, text=render_stats(s)))
            if method == "POST" and path == "/api/generate":
                return self._send(200, self._generate(self._body()))
            if method == "POST" and path == "/api/pairs":
                return self._send(200, self._save(self._body()))
            m = re.fullmatch(r"/api/pairs/(hw_[0-9a-f]{16})", path)
            if m and method == "PUT":
                body = self._body()
                rec = self.server.store.update(m.group(1), body)
                recs = self.server.store.records()
                return self._send(200, {"pair": rec, "counts": counts(recs),
                                        "warnings": warnings_for(rec, self.server.scrubber)})
            if m and method == "DELETE":
                self.server.store.delete(m.group(1))
                return self._send(200, {"deleted": m.group(1), "counts": counts(self.server.store.records())})
            self._send(404, {"error": "not found"})
        except StudioError as e:
            self._send(e.status, {"error": str(e)})

    def _status(self):
        ok, why = self.server.generator.status()
        return {
            "generator": dict(self.server.generator.info, available=ok, reason=why,
                              busy=self.server.gen_lock.locked()),
            "shapes": [{"id": k, "label": s.label, "words": s.words, "structure": s.structure, "tone": s.tone}
                       for k, s in SHAPES.items()],
            "lengths": list(LENGTHS),
            "counts": counts(self.server.store.records()),
            "data": display_path(self.server.store.path),
        }

    def _generate(self, body):
        ok, why = self.server.generator.status()
        if not ok:
            raise StudioError("generation is disabled: %s" % why, 503)
        prompt = build_prompt(body.get("shape", ""), body.get("length", ""), body.get("topic", ""),
                              body.get("custom", ""))
        if not self.server.gen_lock.acquire(blocking=False):
            raise StudioError("a generation is already running", 409)
        try:
            raw = self.server.generator.generate(prompt)
        finally:
            self.server.gen_lock.release()
        given = (body.get("topic") or "").strip()
        topic, text = split_topic(raw) if not given else ("", generate.clean_output(raw))
        if not text:
            raise StudioError("the model returned no text", 502)
        self.server.remember(text)
        return {"text": text, "topic": given or topic, "generator": self.server.generator.info}

    def _save(self, body):
        inp = body.get("input") or ""
        rec = make_record(inp, body.get("target") or "", body.get("shape", ""), body.get("length", ""),
                          body.get("topic", ""), self.server.generator_for(inp))
        rec, created = self.server.store.add(rec)
        return {"pair": rec, "created": created, "counts": counts(self.server.store.records()),
                "warnings": warnings_for(rec, self.server.scrubber)}

    def do_GET(self):
        self._dispatch("GET")

    def do_POST(self):
        self._dispatch("POST")

    def do_PUT(self):
        self._dispatch("PUT")

    def do_DELETE(self):
        self._dispatch("DELETE")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="studio", description=__doc__.split("\n\n")[0])
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--data", default=DEFAULT_DATA, help="pairs file (default data/handwritten/pairs.jsonl)")
    ap.add_argument("--denylist", default=os.path.join(ROOT, "scrub/denylist.txt"),
                    help="for the prepare preview on save")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--effort", default=DEFAULT_EFFORT)
    ap.add_argument("--timeout", type=int, default=240)
    ap.add_argument("--tmp-root", default=os.environ.get("IRULAN_TMP"),
                    help="where codex's empty per-call working dirs are created")
    ap.add_argument("--fake", action="store_true", help="fake generator, no codex (for trying the UI)")
    ap.add_argument("--stats", action="store_true", help="print stats for the pairs file and exit")
    args = ap.parse_args(argv)

    store = PairStore(args.data)
    scrubber = Scrubber.from_files(args.denylist)
    if args.stats:
        print(render_stats(stats(store.records(), scrubber)))
        return 0
    gen = FakeGenerator() if args.fake else CodexGenerator(
        generate.MODEL_ALIASES.get(args.model, args.model), args.effort or None, args.timeout, args.tmp_root)
    ok, why = gen.status()
    server = StudioServer(("127.0.0.1", args.port), store, gen, scrubber)
    url = "http://127.0.0.1:%d/" % server.server_address[1]
    print("irulan studio: %s" % url)
    print("  pairs file: %s (%d saved)" % (display_path(args.data), len(store.records())))
    if ok:
        print("  generator: %s %s, effort %s" % (gen.info["backend"], gen.info["model"], args.effort or "default"))
    else:
        print("  generation DISABLED: %s. Pasting your own text still works." % why)
    sys.stdout.flush()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
