"""irulan studio: a local page for writing handwritten pairs.

The page shows LLM-assistant-style text of a chosen shape (generated with the
Codex CLI, or pasted) next to a textarea for your own version of it. Each
saved pair is one line of DATA_DIR/handwritten/pairs.jsonl; upload the file
with `irulan upload --kind pairs`.

The server binds 127.0.0.1 only, refuses other Host and Origin headers, takes
JSON bodies only and runs one codex call at a time.
"""

import hashlib
import http.server
import json
import os
import re
import sys
import threading
import urllib.parse

from . import generators, pairs
from .prompts import LENGTHS, SHAPES, StudioError, build_prompt, split_topic

PAGE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "page.html")
DEFAULT_PORT = 8765
MAX_BODY = 1 << 20


def default_data(data_dir=None):
    return os.path.join(data_dir or os.environ.get("IRULAN_DATA_DIR") or "data", "handwritten", "pairs.jsonl")


def display_path(path):
    rel = os.path.relpath(os.path.abspath(path))
    return path if rel.startswith("..") else rel


class StudioServer(http.server.ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, addr, store, generator, quiet=False):
        super().__init__(addr, Handler)
        self.quiet = quiet
        self.store = store
        self.generator = generator
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
        store = self.server.store
        try:
            if method == "GET" and path in ("/", "/index.html"):
                with open(PAGE, "rb") as f:
                    return self._send(200, f.read(), "text/html; charset=utf-8")
            if method == "GET" and path == "/api/status":
                return self._send(200, self._status())
            if method == "GET" and path == "/api/pairs":
                recs = store.records()
                return self._send(200, {"pairs": list(reversed(recs)), "counts": pairs.counts(recs)})
            if method == "GET" and path == "/api/stats":
                s = pairs.stats(store.records())
                return self._send(200, dict(s, text=pairs.render_stats(s)))
            if method == "POST" and path == "/api/generate":
                return self._send(200, self._generate(self._body()))
            if method == "POST" and path == "/api/pairs":
                return self._send(200, self._save(self._body()))
            m = re.fullmatch(r"/api/pairs/(hw_[0-9a-f]{16})", path)
            if m and method == "PUT":
                rec = store.update(m.group(1), self._body())
                return self._send(200, {"pair": rec, "counts": pairs.counts(store.records()),
                                        "warnings": pairs.check_pair(rec["input"], rec["target"])})
            if m and method == "DELETE":
                store.delete(m.group(1))
                return self._send(200, {"deleted": m.group(1), "counts": pairs.counts(store.records())})
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
            "counts": pairs.counts(self.server.store.records()),
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
        topic, text = split_topic(raw) if not given else ("", raw.strip())
        if not text:
            raise StudioError("the model returned no text", 502)
        self.server.remember(text)
        return {"text": text, "topic": given or topic, "generator": self.server.generator.info}

    def _save(self, body):
        inp = body.get("input") or ""
        rec = pairs.make_record(inp, body.get("target") or "", body.get("shape", ""), body.get("length", ""),
                                body.get("topic", ""), self.server.generator_for(inp))
        rec, created = self.server.store.add(rec)
        return {"pair": rec, "created": created, "counts": pairs.counts(self.server.store.records()),
                "warnings": pairs.check_pair(rec["input"], rec["target"])}

    def do_GET(self):
        self._dispatch("GET")

    def do_POST(self):
        self._dispatch("POST")

    def do_PUT(self):
        self._dispatch("PUT")

    def do_DELETE(self):
        self._dispatch("DELETE")


def add_arguments(ap):
    ap.add_argument("--port", type=int, default=DEFAULT_PORT, help="port on 127.0.0.1 (default: %(default)s)")
    ap.add_argument("--data", metavar="FILE",
                    help="pairs file (default: DATA_DIR/handwritten/pairs.jsonl, DATA_DIR is $IRULAN_DATA_DIR or data)")
    ap.add_argument("--fake", action="store_true", help="a fake generator instead of codex, for trying the page")
    ap.add_argument("--stats", action="store_true", help="print counts for the pairs file and exit")
    ap.add_argument("--model", default=generators.DEFAULT_MODEL, help="codex model (default: %(default)s)")
    ap.add_argument("--effort", default=generators.DEFAULT_EFFORT,
                    help="codex reasoning effort (default: %(default)s; empty: the codex default)")
    ap.add_argument("--timeout", type=int, default=generators.DEFAULT_TIMEOUT, help="seconds per codex call")
    ap.add_argument("--tmp-root", default=os.environ.get("IRULAN_TMP"),
                    help="where the empty per-call codex working dirs go (default: the system temp dir)")


def run(args, out=None):
    out = out or sys.stdout
    path = args.data or default_data()
    store = pairs.PairStore(path)
    if args.stats:
        print(pairs.render_stats(pairs.stats(store.records())), file=out)
        return 0
    gen = generators.FakeGenerator() if args.fake else generators.CodexGenerator(
        args.model, args.effort, args.timeout, args.tmp_root)
    ok, why = gen.status()
    server = StudioServer(("127.0.0.1", args.port), store, gen)
    print("irulan studio: http://127.0.0.1:%d/" % server.server_address[1], file=out)
    print("  pairs file: %s (%d saved)" % (display_path(path), len(store.records())), file=out)
    if ok:
        print("  generator: %s %s, effort %s" % (gen.info["backend"], gen.info["model"],
                                                 getattr(gen, "effort", None) or "default"), file=out)
    else:
        print("  generation DISABLED: %s. Pasting your own text still works." % why, file=out)
    out.flush()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0
