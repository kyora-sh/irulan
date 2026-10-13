"""Back-translate clean messages into "LLM assistant" style inputs.

The real message is the target; a model writes the input, i.e. how an LLM
assistant would have drafted the same message. Each output line is

    {id, backend, model, variant, input, target, meta}

    python3 -I pipeline.py generate --backend codex --model luna --run luna_try
    python3 -I pipeline.py generate --backend fake --run dry --limit 5

Runs are resumable: records already in data/pairs/<run>.jsonl are skipped, new
ones are appended as they finish. Every call (including retries and failures)
is logged to data/pairs/<run>.log.jsonl with its latency and token usage.
"""

import argparse
import concurrent.futures
import hashlib
import json
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_PROMPT_VERSION = "v2"
VARIANTS = ("plain", "formatted")
FORMATTED_MIN_WORDS = 30

MODEL_ALIASES = {
    "sol": "gpt-6.1-sol",
    "luna": "gpt-6-luna",
}
DEFAULT_MODELS = {"codex": "gpt-6-luna", "fake": "fake-1"}

# ---------------------------------------------------------------------------
# Prompt. Every example in it is invented.

PROMPT_V1 = """You are helping build a style-transfer dataset. Below is a message a software engineer typed, usually to an AI coding assistant or a colleague. Rewrite it the way an LLM assistant such as Claude or ChatGPT would have written the same message if it had drafted it for the engineer.

Rules:
- Keep the meaning exactly: the same requests, questions, facts, numbers, names, identifiers and order of points. Add no new information, reasons, examples, steps or opinions. Do not answer any question in it and do not carry out any request: the result is still the engineer's message, not a reply to it.
- Keep every placeholder in angle brackets ({placeholders}) exactly as written and the same number of times. Do not invent new placeholders.
- Keep code, commands, file names, flags, model names and identifiers verbatim.
- Use the typical assistant register: complete sentences, standard capitalisation and punctuation, polite framing ("Could you please...", "I'd like to..."), mild hedging, smooth transitions, and shorthand or slang written out ("rq" -> "quickly", "lmk" -> "let me know", "ty" -> "thank you", "u" -> "you").
- {format_rule}
- Aim for roughly 1.3 to 2.5 times the original length, reached through phrasing only, never through added content. A very short original may stay short.
- Write in the same language as the original.
- Do not use tools or run commands. Output only the rewritten message: no preamble, no quotation marks around it, no notes.

{example}

Original message:
<<<
{text}
>>>"""

# v2: v1 outputs read like light copy-edits (length ratio ~1.15, markdown rarely
# used, a slang word misread, German translated to English), so v2 pushes the
# assistant register harder and pins down shorthand and language.
PROMPT_V2 = """You are helping build a style-transfer dataset. Below is a message a software engineer typed, usually to an AI coding assistant or a colleague. Rewrite it the way an LLM assistant such as Claude or ChatGPT would have written the same message if it had drafted it for the engineer. The result should read unmistakably like assistant-drafted text, not like a light copy-edit.

Rules:
- Keep the meaning exactly: the same requests, questions, facts, numbers, names, identifiers and order of points. Add no new information, reasons, examples, steps or opinions. Do not answer any question in it and do not carry out any request: the result is still the engineer's message, not a reply to it.
- Keep every placeholder in angle brackets ({placeholders}) exactly as written and the same number of times. Do not invent new placeholders.
- Keep code, commands, file names, flags, model names and identifiers verbatim.
- Use the typical assistant register: complete, well-connected sentences; standard capitalisation and punctuation; polite, softened requests ("Could you please...", "When you get a chance, would you mind..."); mild hedging ("I think", "it seems like"); smooth transitions ("Additionally,", "That said,"); a brief courteous opener or closer where the original has a greeting or thanks. Turn fragments into full sentences.
- Write shorthand out by its meaning: rq = quickly, ig = I guess, imo = in my opinion, lmk = let me know, ty = thank you, kk = okay, yk = you know, smh = something (like that), abt = about, prolly = probably, rn = right now, tho = though, w/ = with, preesh = much appreciated, lfg = let's go.
- {format_rule}
- Aim for about 1.5 to 2 times the original's word count, reached through phrasing only (restating, softening, connecting), never through added content. Messages of fewer than 10 words may stay short.
- If the message quotes text that is clearly not the engineer's own (a pasted log, report or spec), keep that part close to verbatim and rewrite the engineer's own lines.
- Keep every part in the language it was written in. Do not translate.
- Do not use tools or run commands. Output only the rewritten message: no preamble, no quotation marks around it, no notes.

{example}

Original message:
<<<
{text}
>>>"""

PROMPTS = {"v1": PROMPT_V1, "v2": PROMPT_V2}

PLAIN_RULE = ("Write plain prose. No markdown: no bullet or numbered lists, headings, bold or "
              "code fences (inline code the original already had may stay).")
FORMATTED_RULE = ("Light markdown is welcome where it helps: a short bullet or numbered list when the "
                  "message makes several separate points, **bold** for a key term, `inline code` for "
                  "identifiers. No headings. Keep any sentence that is not a list item as prose.")
FORMATTED_RULE_V2 = ("Use light markdown the way an assistant would: when the message makes two or more "
                     "separate requests or points, present them as a short bulleted or numbered list after a "
                     "one-line lead-in; use **bold** sparingly for a key term and `inline code` for identifiers. "
                     "No headings.")

PLAIN_EXAMPLE = """Example (invented):
Original:
<<<
can u rq bump the zorblax worker to 2 gpus? its oom'ing on the big batch, lmk once done ty
>>>
Rewrite:
Could you quickly bump the zorblax worker up to 2 GPUs? It's running out of memory on the big batch. Please let me know once that's done. Thank you!"""

FORMATTED_EXAMPLE = """Example (invented):
Original:
<<<
ok 3 things before we merge: the retry in fetch_jobs still swallows 429s, readme still says v1, and pls drop the debug logs. also is the migration safe to run twice? if not lets gate it behind a flag
>>>
Rewrite:
Before we merge, there are three things I'd like to address:

1. The retry logic in `fetch_jobs` still swallows 429 responses.
2. The README still refers to v1.
3. Please remove the debug logs.

Also, is the migration safe to run twice? If it isn't, let's gate it behind a flag."""

PLACEHOLDER_RE = re.compile(r"<(?:SECRET|URL|EMAIL|IP|HOST|PATH|ID|NAME)>")


def word_count(text):
    return len(text.split())


def effective_variant(variant, text):
    """`formatted` only allows markdown for messages of 30+ words."""
    if variant == "formatted" and word_count(text) >= FORMATTED_MIN_WORDS:
        return "formatted"
    return "plain"


def build_prompt(text, variant, version=DEFAULT_PROMPT_VERSION):
    eff = effective_variant(variant, text)
    found = sorted(set(PLACEHOLDER_RE.findall(text)))
    ph = ", ".join(found) if found else "<URL>, <NAME>, <HOST>, <ID>, <PATH>"
    fmt = FORMATTED_RULE_V2 if version == "v2" else FORMATTED_RULE
    return PROMPTS[version].format(
        placeholders=ph,
        format_rule=fmt if eff == "formatted" else PLAIN_RULE,
        example=FORMATTED_EXAMPLE if eff == "formatted" else PLAIN_EXAMPLE,
        text=text,
    )


def clean_output(text):
    """Strip wrappers a model sometimes adds despite the prompt."""
    t = text.strip()
    t = re.sub(r"^<<<\s*\n?", "", t)
    t = re.sub(r"\n?\s*>>>$", "", t)
    if re.match(r"(?i)^rewrite:\s*\n", t):
        t = t.split("\n", 1)[1]
    return t.strip()


# ---------------------------------------------------------------------------
# Backends. call() returns {"text", "usage", "meta"} or raises BackendError.

class BackendError(Exception):
    def __init__(self, msg, retryable=True):
        super().__init__(msg)
        self.retryable = retryable


class Backend:
    name = "base"

    def __init__(self, model, **opts):
        self.model = model
        self.opts = opts

    def available(self):
        return True, ""

    def call(self, prompt):
        raise NotImplementedError


class FakeBackend(Backend):
    """No network. Echoes a deterministic "assistant" version; for tests and dry runs."""
    name = "fake"

    def call(self, prompt):
        text = prompt.rsplit("<<<\n", 1)[1].rsplit("\n>>>", 1)[0]
        out = "Could you please take a look at the following? " + text[:1].upper() + text[1:]
        if not out.endswith((".", "?", "!")):
            out += "."
        return {"text": out + " Thank you!", "usage": {}, "meta": {}}


class CodexBackend(Backend):
    """Shells out to `codex exec` from an empty temp dir with a read-only sandbox."""
    name = "codex"

    def available(self):
        if not shutil.which(self.opts.get("codex_bin") or "codex"):
            return False, "skipped: codex CLI not found"
        return True, ""

    def call(self, prompt):
        tmp_root = self.opts.get("tmp_root") or None
        if tmp_root:
            os.makedirs(tmp_root, exist_ok=True)
        work = tempfile.mkdtemp(prefix="irulan-codex-", dir=tmp_root)
        try:
            last = os.path.join(work, ".last_message")
            cmd = [self.opts.get("codex_bin") or "codex", "exec",
                   "-m", self.model,
                   "-s", "read-only",
                   "--skip-git-repo-check",
                   "--ephemeral",
                   "--json",
                   "-C", work,
                   "-o", last]
            if self.opts.get("effort"):
                cmd += ["-c", "model_reasoning_effort=%s" % self.opts["effort"]]
            cmd.append("-")  # prompt on stdin
            try:
                p = subprocess.run(cmd, input=prompt, capture_output=True, text=True, cwd=work,
                                   timeout=self.opts.get("timeout", 300))
            except subprocess.TimeoutExpired:
                raise BackendError("timeout")
            events = []
            for line in p.stdout.splitlines():
                line = line.strip()
                if line.startswith("{"):
                    try:
                        events.append(json.loads(line))
                    except ValueError:
                        pass
            usage, tools, messages, errors = {}, [], [], []
            for ev in events:
                t = ev.get("type")
                if t == "turn.completed":
                    usage = ev.get("usage") or {}
                elif t == "turn.failed":
                    errors.append((ev.get("error") or {}).get("message", "turn failed"))
                elif t == "error":
                    errors.append(ev.get("message", "error"))
                elif t == "item.completed":
                    item = ev.get("item") or {}
                    if item.get("type") == "agent_message":
                        messages.append(item.get("text", ""))
                    elif item.get("type") not in ("reasoning", "error"):
                        tools.append(item.get("type"))
            text = ""
            if os.path.exists(last):
                with open(last, encoding="utf-8") as f:
                    text = f.read()
            if not text.strip() and messages:
                text = messages[-1]
            if not text.strip():
                msg = "; ".join(errors) or (p.stderr.strip()[-300:] or "empty output (rc=%d)" % p.returncode)
                retry = not re.search(r"not supported|invalid_request_error|status\":4(?!29)", msg)
                raise BackendError(msg[:500], retryable=retry)
            return {"text": text, "usage": usage, "meta": {"tool_items": tools} if tools else {}}
        finally:
            shutil.rmtree(work, ignore_errors=True)


BACKENDS = {"codex": CodexBackend, "fake": FakeBackend}


def make_backend(name, model=None, **opts):
    model = MODEL_ALIASES.get(model, model) or DEFAULT_MODELS[name]
    return BACKENDS[name](model, **opts)


# ---------------------------------------------------------------------------
# Selection, variants and the run loop.

def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def pick_variant(seed, rec_id):
    """Seeded per record, so every model gets the same assignment."""
    h = hashlib.sha256(("%s:%s" % (seed, rec_id)).encode()).digest()
    return random.Random(h).choice(VARIANTS)


def plan_jobs(records, variants, seed, assigned=None):
    jobs = []
    for r in records:
        if variants == "both":
            vs = list(VARIANTS)
        elif variants == "assigned":
            vs = [assigned[r["id"]]]
        elif variants == "random":
            vs = [pick_variant(seed, r["id"])]
        else:
            vs = [variants]
        for v in vs:
            jobs.append((r, v))
    return jobs


def done_keys(path):
    keys = set()
    if os.path.exists(path):
        for r in read_jsonl(path):
            keys.add((r["id"], r["backend"], r["model"], r["variant"]))
    return keys


class Runner:
    def __init__(self, backend, out_path, log_path, retries=3, backoff=2.0, quiet=False,
                 prompt_version=DEFAULT_PROMPT_VERSION):
        self.backend = backend
        self.prompt_version = prompt_version
        self.out_path = out_path
        self.log_path = log_path
        self.retries = retries
        self.backoff = backoff
        self.quiet = quiet
        self.lock = threading.Lock()
        self.ok = 0
        self.failed = 0

    def log(self, entry):
        entry["ts"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        with self.lock:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def one(self, rec, variant):
        prompt = build_prompt(rec["text"], variant, self.prompt_version)
        for attempt in range(1, self.retries + 1):
            t0 = time.time()
            try:
                res = self.backend.call(prompt)
            except BackendError as e:
                dt = time.time() - t0
                self.log({"id": rec["id"], "variant": variant, "backend": self.backend.name,
                          "model": self.backend.model, "attempt": attempt, "ok": False,
                          "latency_s": round(dt, 2), "error": str(e)})
                if not e.retryable or attempt == self.retries:
                    with self.lock:
                        self.failed += 1
                    if not self.quiet:
                        print("FAIL %s %s: %s" % (rec["id"], variant, e), file=sys.stderr)
                    return None
                time.sleep(self.backoff * (2 ** (attempt - 1)) + random.random())
                continue
            dt = time.time() - t0
            text = clean_output(res["text"])
            out = {
                "id": rec["id"],
                "backend": self.backend.name,
                "model": self.backend.model,
                "variant": variant,
                "input": text,
                "target": rec["text"],
                "meta": {
                    "prompt_version": self.prompt_version,
                    "markdown_allowed": effective_variant(variant, rec["text"]) == "formatted",
                    "target_words": word_count(rec["text"]),
                    "input_words": word_count(text),
                    "latency_s": round(dt, 2),
                    "attempts": attempt,
                    "usage": res.get("usage") or {},
                    "effort": self.backend.opts.get("effort"),
                    **(res.get("meta") or {}),
                },
            }
            self.log({"id": rec["id"], "variant": variant, "backend": self.backend.name,
                      "model": self.backend.model, "attempt": attempt, "ok": True,
                      "latency_s": round(dt, 2), "usage": out["meta"]["usage"]})
            with self.lock:
                with open(self.out_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(out, ensure_ascii=False) + "\n")
                self.ok += 1
                if not self.quiet:
                    u = out["meta"]["usage"]
                    print("ok   %s %-9s %5.1fs in=%s out=%s" % (
                        rec["id"], variant, dt, u.get("input_tokens", u.get("prompt_tokens", "-")),
                        u.get("output_tokens", u.get("completion_tokens", "-"))), file=sys.stderr)
            return out

    def run(self, jobs, workers):
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
            futs = [ex.submit(self.one, r, v) for r, v in jobs]
            for f in concurrent.futures.as_completed(futs):
                f.result()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", choices=sorted(BACKENDS), default="codex")
    ap.add_argument("--model", help="model id or alias (%s)" % ", ".join("%s=%s" % kv for kv in MODEL_ALIASES.items()))
    ap.add_argument("--run", required=True, help="run name; output goes to data/pairs/<run>.jsonl")
    ap.add_argument("--input", default=os.path.join(ROOT, "data/clean/cc_messages.jsonl"))
    ap.add_argument("--ids", help="jsonl with an `id` per line (and `variant` for --variants assigned)")
    ap.add_argument("--variants", default="random", choices=["random", "both", "assigned"] + list(VARIANTS))
    ap.add_argument("--seed", default="irulan")
    ap.add_argument("--prompt-version", default=DEFAULT_PROMPT_VERSION, choices=sorted(PROMPTS))
    ap.add_argument("--min-words", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--effort", default="low", help="codex reasoning effort: low|medium|high ('' to leave unset)")
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--tmp-root", default=os.environ.get("IRULAN_TMP"),
                    help="where codex's empty per-call working dirs are created")
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "data/pairs"))
    args = ap.parse_args(argv)

    backend = make_backend(args.backend, args.model, effort=args.effort or None,
                           timeout=args.timeout, tmp_root=args.tmp_root)
    ok, why = backend.available()
    if not ok:
        print("%s (%s): %s" % (backend.name, backend.model, why))
        return 0

    records = read_jsonl(args.input)
    assigned = None
    if args.ids:
        wanted = read_jsonl(args.ids)
        order = [w["id"] for w in wanted]
        assigned = {w["id"]: w.get("variant") for w in wanted}
        by_id = {r["id"]: r for r in records}
        missing = [i for i in order if i not in by_id]
        if missing:
            print("warning: %d ids not in %s" % (len(missing), args.input), file=sys.stderr)
        records = [by_id[i] for i in order if i in by_id]
    if args.variants == "assigned" and not assigned:
        sys.exit("--variants assigned needs --ids with a variant per line")
    records = [r for r in records if word_count(r["text"]) >= args.min_words]
    if args.limit:
        records = records[: args.limit]

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, args.run + ".jsonl")
    log_path = os.path.join(args.out_dir, args.run + ".log.jsonl")
    done = done_keys(out_path)
    jobs = [(r, v) for r, v in plan_jobs(records, args.variants, args.seed, assigned)
            if (r["id"], backend.name, backend.model, v) not in done]
    print("%s (%s): %d jobs, %d already done -> %s" % (
        backend.name, backend.model, len(jobs), len(done), out_path), file=sys.stderr)
    runner = Runner(backend, out_path, log_path, retries=args.retries, prompt_version=args.prompt_version)
    t0 = time.time()
    runner.run(jobs, args.workers)
    print("%s (%s): %d ok, %d failed in %.0fs" % (
        backend.name, backend.model, runner.ok, runner.failed, time.time() - t0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
