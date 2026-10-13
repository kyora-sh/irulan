"""Minimal JSON-over-HTTP transport on urllib. Stdlib only."""

import json
import random
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid

from . import errors

RETRY_STATUSES = (408, 429, 500, 502, 503, 504)
_STATUS_ERRORS = {
    401: errors.AuthenticationError,
    403: errors.PermissionDeniedError,
    404: errors.NotFoundError,
    409: errors.ConflictError,
    429: errors.RateLimitError,
}


def multipart(fields, files):
    """Encode form fields and files as multipart/form-data.
    `files` maps a field name to (filename, bytes, content type)."""
    boundary = "irulan-" + uuid.uuid4().hex
    out = []
    for name, value in fields.items():
        out.append(("--%s\r\nContent-Disposition: form-data; name=\"%s\"\r\n\r\n%s\r\n"
                    % (boundary, name, value)).encode("utf-8"))
    for name, (filename, data, ctype) in files.items():
        out.append(("--%s\r\nContent-Disposition: form-data; name=\"%s\"; filename=\"%s\"\r\n"
                    "Content-Type: %s\r\n\r\n" % (boundary, name, filename, ctype)).encode("utf-8"))
        out.append(data)
        out.append(b"\r\n")
    out.append(("--%s--\r\n" % boundary).encode("utf-8"))
    return b"".join(out), "multipart/form-data; boundary=%s" % boundary


class Transport:
    def __init__(self, base_url, api_key=None, timeout=60.0, max_retries=2, user_agent="irulan"):
        self.base_url = base_url.rstrip("/") if base_url else None
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        self.user_agent = user_agent

    def url(self, path, query=None):
        if not self.base_url:
            raise errors.NotAvailableYet(
                "kyora's hosted irulan API is not live yet (planned, see docs/API.md). "
                "Set IRULAN_BASE_URL (or Client(base_url=...)) to a server that implements it.")
        u = self.base_url + "/" + path.lstrip("/")
        q = {k: v for k, v in (query or {}).items() if v is not None}
        return u + ("?" + urllib.parse.urlencode(q) if q else "")

    def request(self, method, path, json_body=None, data=None, content_type=None, query=None, raw=False):
        url = self.url(path, query)
        headers = {"Accept": "application/json", "User-Agent": self.user_agent}
        if self.api_key:
            headers["Authorization"] = "Bearer %s" % self.api_key
        body = None
        if json_body is not None:
            body = json.dumps(json_body).encode("utf-8")
            headers["Content-Type"] = "application/json"
        elif data is not None:
            body = data
            headers["Content-Type"] = content_type or "application/octet-stream"
        if method == "POST":
            # Retries reuse the key, so a create is never applied twice.
            headers["Idempotency-Key"] = uuid.uuid4().hex
        attempt = 0
        while True:
            attempt += 1
            req = urllib.request.Request(url, data=body, headers=headers, method=method)
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    payload = resp.read()
                    if raw:
                        return payload
                    return json.loads(payload.decode("utf-8")) if payload.strip() else {}
            except urllib.error.HTTPError as e:
                payload = e.read()
                if e.code in RETRY_STATUSES and attempt <= self.max_retries:
                    self._sleep(attempt, e.headers.get("Retry-After"))
                    continue
                raise self._error(e.code, payload) from None
            except (urllib.error.URLError, OSError) as e:
                if attempt <= self.max_retries:
                    self._sleep(attempt, None)
                    continue
                raise errors.APIConnectionError("could not reach %s: %s" % (url, getattr(e, "reason", e))) from None

    @staticmethod
    def _sleep(attempt, retry_after):
        try:
            delay = float(retry_after)
        except (TypeError, ValueError):
            delay = 0.5 * (2 ** (attempt - 1)) + random.random() * 0.25
        time.sleep(min(delay, 30.0))

    @staticmethod
    def _error(status, payload):
        try:
            body = json.loads(payload.decode("utf-8"))
        except ValueError:
            body = None
        err = body.get("error") if isinstance(body, dict) and isinstance(body.get("error"), dict) else {}
        message = err.get("message") or (payload.decode("utf-8", "replace").strip()[:300] or "HTTP %d" % status)
        if status == 501:
            return errors.NotAvailableYet("not available on this server yet (501): %s" % message)
        cls = _STATUS_ERRORS.get(status, errors.APIError)
        return cls(message, status=status, type=err.get("type"), code=err.get("code"),
                   param=err.get("param"), body=body)
