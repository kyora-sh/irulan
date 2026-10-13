"""The irulan client: rewrite text with an irulan model, and manage datasets
and fine-tunes on kyora (planned, docs/API.md)."""

import os

from . import __version__, rewrite as _rw
from ._http import Transport
from .finetune import Datasets, FineTunes

# No hosted endpoint is live yet; until then IRULAN_BASE_URL must point at an
# implementation of docs/API.md (base URL includes /v1).
DEFAULT_BASE_URL = None


class Models:
    def __init__(self, client):
        self._c = client

    def list(self):
        resp = self._c._t.request("GET", "models")
        return resp.get("data", []) if isinstance(resp, dict) else resp


class Client:
    """Client(api_key=None, base_url=None)

    api_key defaults to $IRULAN_API_KEY, base_url to $IRULAN_BASE_URL (for
    example http://localhost:8000/v1). Both stay on this object; nothing is
    read from or written to disk."""

    def __init__(self, api_key=None, base_url=None, timeout=60.0, max_retries=2):
        self.api_key = api_key or os.environ.get("IRULAN_API_KEY") or None
        self.base_url = base_url or os.environ.get("IRULAN_BASE_URL") or DEFAULT_BASE_URL
        self._t = Transport(self.base_url, self.api_key, timeout=timeout, max_retries=max_retries,
                            user_agent="irulan-python/%s" % __version__)
        self.models = Models(self)
        self.datasets = Datasets(self)
        self.finetunes = FineTunes(self)

    def chat(self, messages, model=_rw.DEFAULT_MODEL, **params):
        """Raw /chat/completions call; returns the response dict."""
        body = {"model": model, "messages": messages}
        body.update(params)
        return self._t.request("POST", "chat/completions", json_body=body)

    def rewrite(self, text, model=_rw.DEFAULT_MODEL, temperature=None, max_tokens=None,
                system_version=_rw.SYSTEM_VERSION):
        """Rewrite `text` in the model's voice (irulan-swe, or your own
        irulan-<user>-<name>). Sends the same versioned system instruction the
        training data uses. Returns the rewritten text."""
        body = _rw.chat_body(text, model=model, system_version=system_version,
                             temperature=temperature, max_tokens=max_tokens)
        return _rw.completion_text(self._t.request("POST", "chat/completions", json_body=body))
