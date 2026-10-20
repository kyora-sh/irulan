"""Datasets and fine-tunes on kyora (planned API, see docs/API.md).

Every method raises errors.NotAvailableYet until a base URL is configured,
and when the server answers 501. Objects are returned as plain dicts in the
shape docs/API.md specifies.
"""

import os
import time

from . import errors, upload as _upload
from ._http import multipart

DATASET_STAGES = ("uploaded", "scrubbed", "pairs", "ready")
DATASET_TERMINAL = ("ready", "failed", "deleted")
FINETUNE_TERMINAL = ("succeeded", "failed", "cancelled")


def _data(resp):
    return resp.get("data", []) if isinstance(resp, dict) else resp


def _wait(get, obj_id, done, poll, timeout, on_update):
    start = time.monotonic()
    last = None
    while True:
        obj = get(obj_id)
        if on_update and obj.get("status") != last:
            on_update(obj)
        last = obj.get("status")
        if done(obj):
            return obj
        if timeout is not None and time.monotonic() - start > timeout:
            raise TimeoutError("%s still %s after %ss" % (obj_id, last, timeout))
        time.sleep(poll)


class Datasets:
    def __init__(self, client):
        self._c = client

    def upload(self, path, name=None, redact_secrets=False, review="required", kind="raw"):
        """Upload a raw extract (kind "raw", data/raw/cc_messages.jsonl) or
        the studio's handwritten pairs (kind "pairs", sent as chat.v1 for the
        sft recipe). The text goes up unscrubbed; the platform scrubs it.
        `redact_secrets=True` replaces obvious credentials locally first
        (irulan.redact). `review` is "required" (processing waits for
        POST .../review) or "skip"."""
        data, summary = _upload.build(path, redact_secrets=redact_secrets, kind=kind)
        return self.upload_bytes(data, name=name or os.path.basename(path), review=review,
                                 format=summary["format"], recipe=summary["recipe"])

    def upload_bytes(self, data, name="dataset.jsonl", review="required", format=_upload.FORMAT, recipe=None):
        if review not in ("required", "skip"):
            raise ValueError("review must be 'required' or 'skip'")
        self._c._t.url("datasets")  # NotAvailableYet before any encoding work
        fields = {"purpose": "fine-tune", "format": format, "name": name, "review": review}
        if recipe:
            fields["recipe"] = recipe
        body, ctype = multipart(fields, {"file": (name, data, "application/jsonl")})
        return self._c._t.request("POST", "datasets", data=body, content_type=ctype)

    def get(self, dataset_id):
        return self._c._t.request("GET", "datasets/%s" % dataset_id)

    def list(self, limit=None, after=None):
        return _data(self._c._t.request("GET", "datasets", query={"limit": limit, "after": after}))

    def delete(self, dataset_id):
        return self._c._t.request("DELETE", "datasets/%s" % dataset_id)

    def review(self, dataset_id):
        """The platform's name/entity review list for a scrubbed dataset:
        {"dataset_id", "status", "candidates": [{"term", "count", "example"}]}."""
        return self._c._t.request("GET", "datasets/%s/review" % dataset_id)

    def approve(self, dataset_id, denylist=()):
        """Approve the review, optionally with extra terms to scrub as <NAME>.
        The platform rescrubs with them, then generates pairs."""
        return self._c._t.request("POST", "datasets/%s/review" % dataset_id,
                                  json_body={"approve": True, "denylist": list(denylist)})

    def wait(self, dataset_id, until="ready", poll=10.0, timeout=None, on_update=None):
        """Poll until the dataset reaches `until` (a stage), or fails."""
        if until not in DATASET_STAGES:
            raise ValueError("until must be one of %s" % ", ".join(DATASET_STAGES))
        want = DATASET_STAGES.index(until)

        def done(d):
            st = d.get("status")
            if st in ("failed", "deleted"):
                raise errors.IrulanError("dataset %s is %s: %s" % (dataset_id, st, d.get("error")))
            return st in DATASET_STAGES and DATASET_STAGES.index(st) >= want
        return _wait(self.get, dataset_id, done, poll, timeout, on_update)


class FineTunes:
    def __init__(self, client):
        self._c = client

    def create(self, dataset_id, name, base="irulan-swe", hyperparameters=None):
        """Start a fine-tune of your own adapter on top of `base`. The
        resulting model id is irulan-<user>-<name>."""
        body = {"dataset_id": dataset_id, "base_model": base, "name": name}
        if hyperparameters:
            body["hyperparameters"] = dict(hyperparameters)
        return self._c._t.request("POST", "fine_tunes", json_body=body)

    def get(self, fine_tune_id):
        return self._c._t.request("GET", "fine_tunes/%s" % fine_tune_id)

    def list(self, limit=None, after=None):
        return _data(self._c._t.request("GET", "fine_tunes", query={"limit": limit, "after": after}))

    def cancel(self, fine_tune_id):
        return self._c._t.request("POST", "fine_tunes/%s/cancel" % fine_tune_id, json_body={})

    def wait(self, fine_tune_id, poll=15.0, timeout=None, on_update=None, raise_on_failure=True):
        """Poll until succeeded, failed or cancelled. Returns the final job;
        raises errors.FineTuneFailed on failed/cancelled unless told not to."""
        job = _wait(self.get, fine_tune_id, lambda j: j.get("status") in FINETUNE_TERMINAL,
                    poll, timeout, on_update)
        if raise_on_failure and job.get("status") != "succeeded":
            raise errors.FineTuneFailed(job)
        return job
