"""Client tests against the local fake server in tests/fakeserver.py."""

import json
import os
import tempfile
import unittest
from unittest import mock

import irulan
from irulan import errors, rewrite
from irulan._http import Transport

from .fakeserver import FakeServer

RAW = [
    {"id": "a1", "project": "-home-dev-demo", "session_id": "s1", "timestamp": "2026-01-01T00:00:00Z",
     "text": "can u rq bump the fake worker", "words": 6, "lines": 1},
    {"id": "a2", "project": "-home-dev-demo", "session_id": "s2", "timestamp": "2026-01-02T00:00:00Z",
     "text": "set password=Fake1234! then retry", "words": 4, "lines": 1},
]


def write_raw(dirpath, records=RAW):
    path = os.path.join(dirpath, "cc_messages.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    return path


class NoServerTests(unittest.TestCase):
    def test_not_available_without_base_url(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            c = irulan.Client()
            self.assertIsNone(c.base_url)
            for call in (lambda: c.rewrite("hi"), lambda: c.finetunes.list(), lambda: c.datasets.list(),
                         lambda: c.models.list()):
                with self.assertRaises(irulan.NotAvailableYet):
                    call()

    def test_env_config(self):
        with mock.patch.dict(os.environ, {"IRULAN_API_KEY": "k", "IRULAN_BASE_URL": "http://x/v1/"}, clear=True):
            c = irulan.Client()
            self.assertEqual((c.api_key, c._t.base_url), ("k", "http://x/v1"))
            self.assertEqual(irulan.Client(api_key="o", base_url="http://y/v1").api_key, "o")

    def test_system_prompt_is_versioned(self):
        self.assertEqual(irulan.SYSTEM_VERSION, "system_v1")
        msgs = rewrite.build_messages("x")
        self.assertEqual(msgs[0], {"role": "system", "content": rewrite.SYSTEM_PROMPTS["system_v1"]})
        with self.assertRaises(ValueError):
            rewrite.build_messages("x", "system_v0")


class ServerTests(unittest.TestCase):
    def setUp(self):
        self.srv = FakeServer().__enter__()
        self.addCleanup(self.srv.__exit__, None, None, None)
        self.c = irulan.Client(api_key=self.srv.api_key, base_url=self.srv.base_url)
        self.tmp = tempfile.mkdtemp()
        sleep = mock.patch("time.sleep", lambda s: None)
        sleep.start()
        self.addCleanup(sleep.stop)

    def last(self):
        return self.srv.requests[-1]

    def test_rewrite_sends_system_v1(self):
        self.assertEqual(self.c.rewrite("Could You Please Check"), "terse: could you please check")
        body = json.loads(self.last()["body"])
        self.assertEqual(body["model"], "irulan-swe")
        self.assertEqual(body["messages"][0]["content"], irulan.SYSTEM_PROMPT)
        self.assertNotIn("temperature", body)
        self.c.rewrite("x", temperature=0.2, max_tokens=50)
        body = json.loads(self.last()["body"])
        self.assertEqual((body["temperature"], body["max_tokens"]), (0.2, 50))
        self.assertEqual(self.last()["headers"]["Authorization"], "Bearer " + self.srv.api_key)

    def test_errors_use_openai_shape(self):
        with self.assertRaises(irulan.NotFoundError) as cm:
            self.c.rewrite("x", model="irulan-nobody-nothing")
        self.assertEqual((cm.exception.status, cm.exception.code, cm.exception.param), (404, "model_not_found", "model"))
        bad = irulan.Client(api_key="wrong", base_url=self.srv.base_url)
        with self.assertRaises(irulan.AuthenticationError):
            bad.models.list()

    def test_retries_on_503_and_gives_up(self):
        self.srv.state.fail_next = [503]
        self.assertEqual(len(self.c.models.list()), 1)
        self.srv.state.fail_next = [503, 503, 503]
        with self.assertRaises(irulan.APIError) as cm:
            self.c.models.list()
        self.assertEqual(cm.exception.status, 503)

    def test_501_is_not_available_yet(self):
        t = Transport(self.srv.base_url, self.srv.api_key)
        self.srv.state.fail_next = [501]
        with self.assertRaises(irulan.NotAvailableYet):
            t.request("GET", "fine_tunes")

    def test_connection_error(self):
        c = irulan.Client(api_key="k", base_url="http://127.0.0.1:9/v1", max_retries=0, timeout=2)
        with self.assertRaises(irulan.APIConnectionError):
            c.models.list()

    def test_dataset_upload_review_and_finetune(self):
        path = write_raw(self.tmp)
        ds = self.c.datasets.upload(path, name="mine", redact_secrets=True)
        self.assertEqual((ds["status"], ds["records"], ds["format"], ds["name"]), ("uploaded", 2, "irulan.raw.v1", "mine"))
        sent = self.srv.state.uploads[ds["id"]]
        self.assertEqual([s["id"] for s in sent], ["a1", "a2"])
        self.assertEqual(sent[1]["text"], "set password=<SECRET> then retry")
        self.assertNotIn("project", json.dumps(sent))           # local path stays local
        self.assertEqual(len({s["meta"]["session"] for s in sent}), 2)

        with self.assertRaises(irulan.ConflictError):
            self.c.finetunes.create(ds["id"], name="voice")
        d = self.c.datasets.wait(ds["id"], until="scrubbed", poll=0)
        self.assertEqual(d["status"], "scrubbed")
        self.assertEqual(self.c.datasets.get(ds["id"])["status"], "scrubbed")  # waits for review
        rev = self.c.datasets.review(ds["id"])
        self.assertEqual(rev["candidates"][0]["term"], "Zorblax")
        self.c.datasets.approve(ds["id"], denylist=["Zorblax"])
        self.assertEqual(self.c.datasets.wait(ds["id"], poll=0)["status"], "ready")

        job = self.c.finetunes.create(ds["id"], name="voice", hyperparameters={"epochs": 2})
        self.assertEqual(json.loads(self.last()["body"]),
                         {"dataset_id": ds["id"], "base_model": "irulan-swe", "name": "voice",
                          "hyperparameters": {"epochs": 2}})
        self.assertIn("Idempotency-Key", self.last()["headers"])
        seen = []
        done = self.c.finetunes.wait(job["id"], poll=0, on_update=lambda j: seen.append(j["status"]))
        self.assertEqual(done["fine_tuned_model"], "irulan-tester-voice")
        self.assertEqual(seen, ["running", "succeeded"])
        self.assertTrue(done["events"])
        self.assertEqual([j["id"] for j in self.c.finetunes.list()], [job["id"]])
        self.assertIn("irulan-tester-voice", [m["id"] for m in self.c.models.list()])
        self.assertEqual(self.c.rewrite("Hello", model="irulan-tester-voice"), "terse: hello")

        self.assertTrue(self.c.datasets.delete(ds["id"])["deleted"])
        with self.assertRaises(irulan.NotFoundError):
            self.c.datasets.get(ds["id"])

    def test_failed_and_cancelled_finetunes(self):
        ds = self.c.datasets.upload(write_raw(self.tmp), review="skip")
        self.c.datasets.wait(ds["id"], poll=0)
        self.srv.state.ft_fail = True
        job = self.c.finetunes.create(ds["id"], name="v2")
        with self.assertRaises(errors.FineTuneFailed) as cm:
            self.c.finetunes.wait(job["id"], poll=0)
        self.assertIn("loss went to nan", str(cm.exception))
        job = self.c.finetunes.create(ds["id"], name="v3")
        self.assertEqual(self.c.finetunes.cancel(job["id"])["status"], "cancelled")
        self.assertEqual(self.c.finetunes.wait(job["id"], poll=0, raise_on_failure=False)["status"], "cancelled")
        with self.assertRaises(irulan.ConflictError):
            self.c.finetunes.cancel(job["id"])

    def test_upload_validation_errors_surface(self):
        with self.assertRaises(ValueError):
            self.c.datasets.upload(write_raw(self.tmp, [RAW[0], RAW[0]]))  # duplicate ids, caught locally
        with self.assertRaises(irulan.APIError) as cm:
            self.c.datasets.upload_bytes(b"not json\n")
        self.assertEqual((cm.exception.status, cm.exception.code), (400, "invalid_jsonl"))


if __name__ == "__main__":
    unittest.main()
