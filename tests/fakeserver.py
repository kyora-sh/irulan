"""An in-memory, stdlib-only server that implements docs/API.md, for tests.

    with FakeServer() as srv:
        client = irulan.Client(api_key=srv.api_key, base_url=srv.base_url)

Each GET of a dataset or fine-tune moves it one stage on, so wait() loops
finish in a few polls. Every request is recorded in `srv.requests`.
"""

import email.parser
import email.policy
import json
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

API_KEY = "test-key-not-real"
USER = "tester"
DATASET_FLOW = ("uploaded", "scrubbed", "pairs", "ready")
FT_FLOW = ("queued", "running", "running", "succeeded")


def error(status, message, type_="invalid_request_error", code=None, param=None):
    return status, {"error": {"message": message, "type": type_, "code": code, "param": param}}


class State:
    def __init__(self):
        self.lock = threading.Lock()
        self.datasets = {}
        self.uploads = {}       # dataset id -> list of records as received
        self.fine_tunes = {}
        self.requests = []
        self.n = 0
        self.fail_next = []     # statuses to return before handling (retry tests)
        self.ft_fail = False
        self.idempotency = {}

    def new_id(self, prefix):
        self.n += 1
        return "%s-%04d" % (prefix, self.n)


class Handler(BaseHTTPRequestHandler):
    server_version = "fake-irulan/1"

    def log_message(self, *args):
        pass

    @property
    def state(self):
        return self.server.state

    def _send(self, status, obj):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _body(self):
        n = int(self.headers.get("Content-Length") or 0)
        return self.rfile.read(n) if n else b""

    def _handle(self, method):
        raw = self._body()
        st = self.state
        with st.lock:
            st.requests.append({"method": method, "path": self.path, "headers": dict(self.headers), "body": raw})
            if st.fail_next:
                status = st.fail_next.pop(0)
                return self._send(*error(status, "injected failure", "server_error"))
        if self.headers.get("Authorization") != "Bearer %s" % API_KEY:
            return self._send(*error(401, "invalid api key", "authentication_error", "invalid_api_key"))
        path = self.path.split("?", 1)[0]
        if not path.startswith("/v1/"):
            return self._send(*error(404, "not found"))
        try:
            status, obj = self.route(method, path[len("/v1"):], raw)
        except Exception as e:  # noqa: BLE001 - a test server reports everything
            status, obj = error(500, "fake server error: %s" % e, "server_error")
        self._send(status, obj)

    do_GET = lambda self: self._handle("GET")  # noqa: E731
    do_POST = lambda self: self._handle("POST")  # noqa: E731
    do_DELETE = lambda self: self._handle("DELETE")  # noqa: E731

    # -- routes --------------------------------------------------------------

    def route(self, method, path, raw):
        st = self.state
        if path == "/models" and method == "GET":
            data = [{"id": "irulan-swe", "object": "model", "owned_by": "kyora", "system_version": "system_v1"}]
            data += [{"id": j["fine_tuned_model"], "object": "model", "owned_by": USER, "system_version": "system_v1"}
                     for j in st.fine_tunes.values() if j["status"] == "succeeded"]
            return 200, {"object": "list", "data": data}
        if path == "/chat/completions" and method == "POST":
            return self.chat(json.loads(raw))
        if path == "/datasets" and method == "POST":
            return self.upload(raw)
        if path == "/datasets" and method == "GET":
            return 200, {"object": "list", "data": list(st.datasets.values()), "has_more": False}
        m = re.match(r"^/datasets/([\w-]+)(/review)?$", path)
        if m:
            return self.dataset(method, m.group(1), bool(m.group(2)), raw)
        if path == "/fine_tunes" and method == "POST":
            return self.create_ft(json.loads(raw), self.headers.get("Idempotency-Key"))
        if path == "/fine_tunes" and method == "GET":
            return 200, {"object": "list", "data": list(st.fine_tunes.values()), "has_more": False}
        m = re.match(r"^/fine_tunes/([\w-]+)(/cancel)?$", path)
        if m:
            return self.ft(method, m.group(1), bool(m.group(2)))
        return error(404, "no route for %s %s" % (method, path))

    def chat(self, body):
        msgs = body.get("messages") or []
        if body.get("model") not in ("irulan-swe",) and not any(
                j.get("fine_tuned_model") == body.get("model") for j in self.state.fine_tunes.values()):
            return error(404, "model %s not found" % body.get("model"), code="model_not_found", param="model")
        user = [m["content"] for m in msgs if m.get("role") == "user"]
        system = [m["content"] for m in msgs if m.get("role") == "system"]
        text = "terse: " + user[-1].lower() if user else ""
        return 200, {
            "id": "chatcmpl-fake", "object": "chat.completion", "created": int(time.time()), "model": body["model"],
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            "x_seen_system": system,
        }

    def upload(self, raw):
        ctype = self.headers.get("Content-Type", "")
        if not ctype.startswith("multipart/form-data"):
            return error(400, "expected multipart/form-data", param="file")
        msg = email.parser.BytesParser(policy=email.policy.HTTP).parsebytes(
            b"Content-Type: " + ctype.encode() + b"\r\n\r\n" + raw)
        fields, data = {}, None
        for part in msg.iter_parts():
            name = part.get_param("name", header="content-disposition")
            if name == "file":
                data = part.get_payload(decode=True)
            else:
                fields[name] = part.get_content().strip()
        if data is None:
            return error(400, "missing file", param="file")
        records = []
        for i, line in enumerate(data.decode("utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except ValueError:
                return error(400, "line %d is not JSON" % i, code="invalid_jsonl", param="file")
            if not isinstance(rec.get("text"), str) or not rec["text"].strip():
                return error(400, "line %d has no text" % i, code="invalid_record", param="file")
            records.append(rec)
        st = self.state
        with st.lock:
            ds_id = st.new_id("ds")
            st.uploads[ds_id] = records
            st.datasets[ds_id] = {
                "id": ds_id, "object": "dataset", "name": fields.get("name"), "format": fields.get("format"),
                "status": "uploaded", "records": len(records), "bytes": len(data),
                "review": {"required": fields.get("review") != "skip", "status": "pending", "candidates": 0},
                "created_at": int(time.time()), "error": None,
            }
        return 200, st.datasets[ds_id]

    def dataset(self, method, ds_id, review, raw):
        st = self.state
        with st.lock:
            d = st.datasets.get(ds_id)
            if not d:
                return error(404, "dataset %s not found" % ds_id, code="dataset_not_found")
            if review and method == "GET":
                if d["status"] == "uploaded":
                    return error(409, "not scrubbed yet", code="dataset_not_scrubbed")
                return 200, {"object": "dataset.review", "dataset_id": ds_id, "status": d["review"]["status"],
                             "candidates": [{"term": "Zorblax", "count": 3, "example": "ask Zorblax about <URL>"}]}
            if review and method == "POST":
                body = json.loads(raw)
                d["review"]["status"] = "approved" if body.get("approve") else "pending"
                d["review"]["denylist_terms"] = len(body.get("denylist") or [])
                return 200, d
            if method == "DELETE":
                del st.datasets[ds_id]
                st.uploads.pop(ds_id, None)
                return 200, {"id": ds_id, "object": "dataset", "deleted": True}
            if method == "GET":
                i = DATASET_FLOW.index(d["status"])
                blocked = d["status"] == "scrubbed" and d["review"]["required"] and d["review"]["status"] != "approved"
                if i + 1 < len(DATASET_FLOW) and not blocked:
                    d["status"] = DATASET_FLOW[i + 1]
                return 200, dict(d)
        return error(405, "method not allowed")

    def create_ft(self, body, key):
        st = self.state
        with st.lock:
            if key and key in st.idempotency:
                return 200, st.fine_tunes[st.idempotency[key]]
            d = st.datasets.get(body.get("dataset_id"))
            if not d:
                return error(404, "dataset not found", code="dataset_not_found", param="dataset_id")
            if d["status"] != "ready":
                return error(409, "dataset is %s, not ready" % d["status"], code="dataset_not_ready", param="dataset_id")
            if not re.match(r"^[a-z0-9][a-z0-9-]{0,39}$", body.get("name") or ""):
                return error(400, "bad name", code="invalid_name", param="name")
            ft_id = st.new_id("ft")
            st.fine_tunes[ft_id] = {
                "id": ft_id, "object": "fine_tune", "dataset_id": d["id"], "base_model": body.get("base_model"),
                "name": body["name"], "status": "queued", "hyperparameters": body.get("hyperparameters") or {},
                "fine_tuned_model": None, "created_at": int(time.time()), "finished_at": None,
                "events": [{"created_at": int(time.time()), "level": "info", "message": "queued"}],
                "metrics": {}, "error": None,
            }
            if key:
                st.idempotency[key] = ft_id
            return 200, st.fine_tunes[ft_id]

    def ft(self, method, ft_id, cancel):
        st = self.state
        with st.lock:
            j = st.fine_tunes.get(ft_id)
            if not j:
                return error(404, "fine-tune %s not found" % ft_id, code="fine_tune_not_found")
            if cancel and method == "POST":
                if j["status"] in ("succeeded", "failed", "cancelled"):
                    return error(409, "already %s" % j["status"], code="fine_tune_finished")
                j["status"] = "cancelled"
                return 200, j
            if method == "GET":
                if j["status"] in ("queued", "running"):
                    j.setdefault("_step", 0)
                    j["_step"] += 1
                    nxt = FT_FLOW[min(j["_step"], len(FT_FLOW) - 1)]
                    if nxt == "succeeded" and st.ft_fail:
                        j["status"] = "failed"
                        j["error"] = {"code": "training_diverged", "message": "loss went to nan"}
                    else:
                        j["status"] = nxt
                    if j["status"] == "running":
                        j["metrics"] = {"step": j["_step"], "train_loss": 1.0 / j["_step"]}
                    if j["status"] == "succeeded":
                        j["fine_tuned_model"] = "irulan-%s-%s" % (USER, j["name"])
                    j["events"].append({"created_at": int(time.time()), "level": "info", "message": j["status"]})
                return 200, {k: v for k, v in j.items() if not k.startswith("_")}
        return error(405, "method not allowed")


class FakeServer:
    api_key = API_KEY

    def __init__(self):
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.httpd.state = State()
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    @property
    def state(self):
        return self.httpd.state

    @property
    def requests(self):
        return self.httpd.state.requests

    @property
    def base_url(self):
        return "http://127.0.0.1:%d/v1" % self.httpd.server_address[1]

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.httpd.shutdown()
        self.httpd.server_close()
