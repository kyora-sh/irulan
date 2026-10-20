"""CLI smoke tests and an end-to-end run on invented transcripts:
extract -> upload -> datasets -> finetune -> rewrite, against the fake server."""

import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from irulan import cli

from .fakeserver import FakeServer

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def run(argv, env=None):
    out, err = io.StringIO(), io.StringIO()
    with mock.patch.dict(os.environ, env or {}, clear=True), \
            contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            code = cli.main(argv)
        except SystemExit as e:  # argparse
            code = e.code
    return code, out.getvalue(), err.getvalue()


def compact(d):
    return json.dumps(d, separators=(",", ":"))  # transcripts are compact JSON lines


def user(text, session, ts, **extra):
    d = {"type": "user", "sessionId": session, "timestamp": ts, "message": {"role": "user", "content": text}}
    d.update(extra)
    return compact(d)


def write_transcripts(root):
    """Two invented projects, three sessions, with typed turns and the kinds of
    turns the extract must drop."""
    proj_a = os.path.join(root, "-home-dev-demo-app")
    proj_b = os.path.join(root, "-home-dev-toy-agent")
    os.makedirs(proj_a)
    os.makedirs(proj_b)
    a = [
        user("can u rq bump the fake worker to 2 gpus? its oom'ing on the big batch", "s1", "2026-01-01T10:00:00Z"),
        user("<system-reminder>harness text</system-reminder>", "s1", "2026-01-01T10:01:00Z"),
        user("sidechain prompt nobody typed", "s1", "2026-01-01T10:02:00Z", isSidechain=True),
        user("meta caveat", "s1", "2026-01-01T10:03:00Z", isMeta=True),
        compact({"type": "user", "sessionId": "s1", "timestamp": "2026-01-01T10:04:00Z",
                    "message": {"content": [{"type": "tool_result", "content": "ok"}]}}),
        compact({"type": "assistant", "sessionId": "s1", "message": {"content": "Sure, on it."}}),
        user("[Request interrupted by user]", "s1", "2026-01-01T10:05:00Z"),
        user("ok lgtm, open the pr pls and use password=Fake1234! for the staging db", "s1", "2026-01-01T10:06:00Z"),
    ]
    b = [
        user("ich glaube das ist noch nicht fertig, aber wir können es morgen machen", "s2", "2026-01-02T09:00:00Z"),
        user("why does the fake build hang after step 3? feels wrong", "s2", "2026-01-02T09:01:00Z"),
        user("why does the fake build hang after step 3? feels wrong", "s3", "2026-01-03T09:01:00Z"),
        user("```\nTraceback (most recent call last):\n  boom\n```", "s3", "2026-01-03T09:02:00Z"),
        '{"type":"user","message":{"content":"half a li',  # a hard kill mid-write
    ]
    with open(os.path.join(proj_a, "s1.jsonl"), "w", encoding="utf-8") as f:
        f.write("\n".join(a) + "\n")
    with open(os.path.join(proj_b, "s2.jsonl"), "w", encoding="utf-8") as f:
        f.write("\n".join(b) + "\n")


class SmokeTests(unittest.TestCase):
    def test_help_for_every_command(self):
        code, out, _ = run(["--help"])
        self.assertEqual(code, 0)
        for cmd in ("extract", "stats", "studio", "upload", "datasets", "finetune", "models", "rewrite"):
            self.assertIn(cmd, out)
            self.assertEqual(run([cmd, "--help"])[0], 0, cmd)
        for sub in (["finetune", "create"], ["finetune", "status"], ["finetune", "list"], ["finetune", "cancel"],
                    ["datasets", "review"], ["datasets", "approve"], ["datasets", "delete"]):
            self.assertEqual(run(sub + ["--help"])[0], 0, sub)

    def test_no_command_prints_help(self):
        code, out, _ = run([])
        self.assertEqual(code, cli.EXIT_USAGE)
        self.assertIn("usage: irulan", out)

    def test_planned_commands_say_not_available(self):
        for argv in (["finetune", "list"], ["datasets", "list"], ["models"], ["rewrite", "hi"],
                     ["finetune", "create", "--dataset", "ds-1", "--name", "x"]):
            code, _, err = run(argv)
            self.assertEqual(code, cli.EXIT_NOT_AVAILABLE, argv)
            self.assertIn("not live yet", err)

    def test_stats_without_extract(self):
        with tempfile.TemporaryDirectory() as d:
            code, _, err = run(["stats"], env={"IRULAN_DATA_DIR": d})
        self.assertEqual(code, cli.EXIT_USAGE)
        self.assertIn("irulan extract", err)

    def test_pipeline_shim_and_module(self):
        for argv in ([sys.executable, "-I", os.path.join(ROOT, "pipeline.py"), "--help"],):
            p = subprocess.run(argv, capture_output=True, text=True, cwd=ROOT)
            self.assertEqual(p.returncode, 0, p.stderr)
            self.assertIn("usage: irulan", p.stdout)


class EndToEndTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.src = os.path.join(self.tmp, "projects")
        self.data = os.path.join(self.tmp, "data")
        write_transcripts(self.src)
        sleep = mock.patch("time.sleep", lambda s: None)
        sleep.start()
        self.addCleanup(sleep.stop)

    def extract(self):
        code, out, err = run(["extract", "--src", self.src, "--data-dir", self.data])
        self.assertEqual(code, 0, err)
        with open(os.path.join(self.data, "raw", "cc_messages.jsonl"), encoding="utf-8") as f:
            return [json.loads(line) for line in f], out

    def test_extract_layout_and_format(self):
        raw, out = self.extract()
        texts = [r["text"] for r in raw]
        self.assertEqual(texts, [
            "can u rq bump the fake worker to 2 gpus? its oom'ing on the big batch",
            "ok lgtm, open the pr pls and use password=Fake1234! for the staging db",  # not scrubbed locally
            "why does the fake build hang after step 3? feels wrong",                   # deduped across sessions
        ])
        for r in raw:
            self.assertEqual(sorted(r), ["id", "lines", "project", "session_id", "text", "timestamp", "words"])
            self.assertEqual(len(r["id"]), 16)
        self.assertEqual(raw[0]["project"], "-home-dev-demo-app")
        with open(os.path.join(self.data, "stats", "cc_stats.json"), encoding="utf-8") as f:
            st = json.load(f)
        self.assertEqual(st["raw_messages"], 3)
        self.assertEqual(st["extract"]["kept"], 3)
        self.assertEqual(st["extract"]["sidechain_user_lines"], 1)
        self.assertEqual(st["extract"]["dropped"]["meta"], 1)
        self.assertEqual(st["extract"]["dropped"]["interrupted"], 1)
        self.assertEqual(st["extract"]["dropped"]["non_english"], 1)
        self.assertEqual(st["extract"]["dropped"]["duplicate"], 1)
        self.assertEqual(st["extract"]["bad_json"], 1)
        self.assertTrue(os.path.exists(os.path.join(self.data, "stats", "cc_drops.jsonl")))
        self.assertFalse(os.path.exists(os.path.join(self.data, "clean")))
        self.assertIn("raw: 3 messages", out)
        code, out, _ = run(["stats"], env={"IRULAN_DATA_DIR": self.data})
        self.assertEqual(code, 0)
        self.assertIn("kept after filters: 3", out)

    def test_upload_needs_confirmation_and_shows_policy(self):
        self.extract()
        payload = os.path.join(self.tmp, "payload.jsonl")
        code, out, err = run(["upload", "--dry-run", "--redact-secrets-locally", "--payload-out", payload],
                             env={"IRULAN_DATA_DIR": self.data})
        self.assertEqual((code, out), (0, ""))
        self.assertIn("3 messages, 2 sessions", err)
        self.assertIn("UNSCRUBBED", err)
        self.assertIn("On upload, kyora first scrubs the data. Only the scrubbed version is stored, encrypted", err)
        self.assertIn("The original (unscrubbed) upload is not retained.", err)
        self.assertIn("dry run: nothing was uploaded.", err)
        with open(payload, encoding="utf-8") as f:
            sent = [json.loads(line) for line in f]
        self.assertEqual(sent[1]["text"], "ok lgtm, open the pr pls and use password=<SECRET> for the staging db")
        self.assertEqual(sorted(sent[0]), ["id", "meta", "text"])
        self.assertNotIn("home-dev", json.dumps(sent))
        with FakeServer() as srv:
            env = {"IRULAN_DATA_DIR": self.data, "IRULAN_BASE_URL": srv.base_url, "IRULAN_API_KEY": srv.api_key}
            with mock.patch("sys.stdin", io.StringIO("")):     # not a tty: no prompt, refuse
                code, _, err = run(["upload"], env=env)
            self.assertEqual(code, cli.EXIT_USAGE)
            self.assertIn("--yes", err)
            self.assertEqual(srv.requests, [])

    def test_full_flow(self):
        self.extract()
        with FakeServer() as srv:
            env = {"IRULAN_DATA_DIR": self.data, "IRULAN_BASE_URL": srv.base_url, "IRULAN_API_KEY": srv.api_key}
            code, out, err = run(["upload", "--yes", "--name", "demo"], env=env)
            self.assertEqual(code, 0, err)
            ds = json.loads(out)
            self.assertEqual(ds["records"], 3)
            # the unredacted run sends the text as extracted
            self.assertIn("password=Fake1234!", json.dumps(srv.state.uploads[ds["id"]]))

            code, out, _ = run(["datasets", "status", ds["id"], "--wait", "scrubbed", "--poll", "0"], env=env)
            self.assertEqual(json.loads(out)["status"], "scrubbed")
            tsv = os.path.join(self.tmp, "review.tsv")
            code, out, _ = run(["datasets", "review", ds["id"], "--out", tsv], env=env)
            self.assertEqual(code, 0)
            with open(tsv, encoding="utf-8") as f:
                self.assertEqual(f.read().splitlines()[1].split("\t")[1], "Zorblax")
            deny = os.path.join(self.tmp, "deny.txt")
            with open(deny, "w", encoding="utf-8") as f:
                f.write("# invented\nZorblax\n")
            code, out, _ = run(["datasets", "approve", ds["id"], "--denylist", deny], env=env)
            self.assertEqual(json.loads(out)["review"]["denylist_terms"], 1)
            code, out, _ = run(["datasets", "status", ds["id"], "--wait", "ready", "--poll", "0"], env=env)
            self.assertEqual(json.loads(out)["status"], "ready")

            code, out, err = run(["finetune", "create", "--dataset", ds["id"], "--name", "voice",
                                  "--epochs", "2", "--wait", "--poll", "0"], env=env)
            self.assertEqual(code, 0, err)
            job = json.loads(out)
            self.assertEqual((job["status"], job["fine_tuned_model"]), ("succeeded", "irulan-tester-voice"))
            self.assertEqual(job["hyperparameters"], {"epochs": 2})
            code, out, _ = run(["finetune", "list"], env=env)
            self.assertEqual(len(json.loads(out)), 1)
            code, out, _ = run(["finetune", "status", job["id"]], env=env)
            self.assertEqual(json.loads(out)["id"], job["id"])

            code, out, _ = run(["rewrite", "--model", "irulan-tester-voice", "Could", "You", "Check"], env=env)
            self.assertEqual((code, out), (0, "terse: could you check\n"))
            with mock.patch("sys.stdin", io.StringIO("From Stdin")):
                code, out, _ = run(["rewrite"], env=env)
            self.assertEqual(out, "terse: from stdin\n")
            code, _, err = run(["rewrite", "--model", "irulan-nobody-x", "hi"], env=env)
            self.assertEqual(code, cli.EXIT_API)
            self.assertIn("model_not_found", err)

            code, out, _ = run(["datasets", "delete", ds["id"], "--yes"], env=env)
            self.assertTrue(json.loads(out)["deleted"])

    def test_failed_finetune_exits_nonzero(self):
        self.extract()
        with FakeServer() as srv:
            srv.state.ft_fail = True
            env = {"IRULAN_DATA_DIR": self.data, "IRULAN_BASE_URL": srv.base_url, "IRULAN_API_KEY": srv.api_key}
            _, out, _ = run(["upload", "--yes", "--skip-review"], env=env)
            ds = json.loads(out)
            run(["datasets", "status", ds["id"], "--wait", "ready", "--poll", "0"], env=env)
            code, _, err = run(["finetune", "create", "--dataset", ds["id"], "--name", "v", "--wait", "--poll", "0"],
                               env=env)
            self.assertEqual(code, cli.EXIT_API)
            self.assertIn("failed", err)


if __name__ == "__main__":
    unittest.main()
