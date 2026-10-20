"""Studio tests: records, the pairs file, prompts, pair checks, the codex
command line and the HTTP API with the fake generator. Every text here is
made up; nothing calls codex."""

import http.client
import json
import os
import random
import tempfile
import threading
import unittest
from unittest import mock

from irulan.studio import generators, pairs, prompts, server as studio_server

SLOP = ("## Summary\n\nThis PR moves the retry loop for the **quillfeather** sync worker into "
        "services/quill/retry.py. Additionally, it lowers p95 latency from 840 ms to 310 ms and bumps "
        "the client to v2.3.1. That said, it might be worth a second look at the backoff cap.\n\n"
        "## Changes\n\n- Moved the retry loop\n- Added unit tests\n- Updated the config key `quill.max_retries`")
MINE = "move quillfeather retry loop to services/quill/retry.py, p95 840 -> 310ms, client v2.3.1, quill.max_retries"


def rec(inp=SLOP, tgt=MINE, shape="pr_description", length="medium", **kw):
    return pairs.make_record(inp, tgt, shape, length, **kw)


class RecordTests(unittest.TestCase):
    def test_fields_are_exactly_the_record_format(self):
        r = rec(topic="retry loop", generator={"backend": "codex", "model": "m", "prompt_version": "studio_v1"})
        self.assertEqual(tuple(r), pairs.FIELDS)
        self.assertEqual(pairs.FIELDS, ("id", "source", "shape", "length", "topic", "generator", "input",
                                        "target", "created_at"))
        self.assertEqual(r["source"], "handwritten")
        self.assertTrue(r["created_at"].endswith("Z"))

    def test_id_is_stable_and_depends_on_both_sides(self):
        a = rec(created_at="2026-01-02T03:04:05Z")
        b = rec(shape="issue", topic="other", created_at="2027-01-01T00:00:00Z")
        self.assertEqual(a["id"], b["id"])
        self.assertEqual(a["id"], pairs.pair_id(SLOP + "\n  ", "  " + MINE))
        self.assertRegex(a["id"], r"^hw_[0-9a-f]{16}$")
        self.assertNotEqual(a["id"], rec(tgt=MINE + " pls")["id"])
        self.assertNotEqual(a["id"], rec(inp=SLOP + " Overall, looks good.")["id"])
        self.assertNotEqual(pairs.pair_id("ab", "c"), pairs.pair_id("a", "bc"))

    def test_validation(self):
        for bad in (dict(inp=" "), dict(tgt=""), dict(shape="haiku"), dict(length="huge"),
                    dict(generator={"model": "x"})):
            with self.assertRaises(prompts.StudioError):
                rec(**bad)


class StoreTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.dir.name, "hand", "pairs.jsonl")
        self.store = pairs.PairStore(self.path)

    def tearDown(self):
        self.dir.cleanup()

    def lines(self):
        with open(self.path, encoding="utf-8") as f:
            return f.read().splitlines()

    def test_add_appends_once(self):
        r, created = self.store.add(rec())
        self.assertTrue(created)
        _, created = self.store.add(rec())
        self.assertFalse(created)
        self.store.add(rec(tgt="other text here"))
        self.assertEqual(len(self.lines()), 2)
        self.assertEqual(json.loads(self.lines()[0]), r)

    def test_add_repairs_a_half_written_tail(self):
        self.store.add(rec())
        with open(self.path, "a", encoding="utf-8") as f:
            f.write('{"id": "hw_x", "inp')
        self.store.add(rec(tgt="second one"))
        self.assertEqual([json.loads(x)["target"] for x in self.lines()], [MINE, "second one"])

    def test_edit_rewrites_atomically_and_keeps_order(self):
        a, _ = self.store.add(rec(created_at="2026-01-01T00:00:00Z",
                                  generator={"backend": "fake", "model": "f", "prompt_version": "studio_v1"}))
        b, _ = self.store.add(rec(tgt="second one"))
        new = self.store.update(a["id"], {"target": "edited text", "topic": "t2"})
        self.assertNotEqual(new["id"], a["id"])
        self.assertEqual(new["id"], pairs.pair_id(SLOP, "edited text"))
        self.assertEqual(new["created_at"], a["created_at"])
        self.assertEqual(new["generator"], a["generator"])
        self.assertEqual([r["id"] for r in self.store.records()], [new["id"], b["id"]])
        self.assertEqual(os.listdir(os.path.dirname(self.path)), ["pairs.jsonl"])
        edited_input = self.store.update(new["id"], {"input": "Could you please restart the worker?"})
        self.assertEqual(edited_input["generator"], "pasted")

    def test_edit_errors(self):
        a, _ = self.store.add(rec())
        b, _ = self.store.add(rec(tgt="second one"))
        with self.assertRaises(prompts.StudioError) as cm:
            self.store.update(b["id"], {"target": MINE})
        self.assertEqual(cm.exception.status, 409)
        with self.assertRaises(prompts.StudioError):
            self.store.update(a["id"], {"id": "hw_0000000000000000"})
        with self.assertRaises(prompts.StudioError) as cm:
            self.store.update("hw_ffffffffffffffff", {"target": "x"})
        self.assertEqual(cm.exception.status, 404)

    def test_delete_keeps_unparsable_lines(self):
        a, _ = self.store.add(rec())
        with open(self.path, "a", encoding="utf-8") as f:
            f.write("not json\n")
        b, _ = self.store.add(rec(tgt="second one"))
        self.store.delete(a["id"])
        self.assertEqual(self.lines()[0], "not json")
        self.assertEqual([r["id"] for r in self.store.records()], [b["id"]])
        with self.assertRaises(prompts.StudioError):
            self.store.delete(a["id"])

    def test_counts(self):
        self.store.add(rec())
        self.store.add(rec(tgt="second one", shape="issue"))
        self.store.add(rec(tgt="third one", shape="issue"))
        self.assertEqual(pairs.counts(self.store.records()),
                         {"total": 3, "by_shape": {"pr_description": 1, "issue": 2}})


class PromptTests(unittest.TestCase):
    def test_every_shape_and_length(self):
        for key, shape in prompts.SHAPES.items():
            for i, length in enumerate(prompts.LENGTHS):
                p = prompts.build_prompt(key, length, "a retry loop", custom="a changelog haiku")
                lo, hi = shape.words[i]
                self.assertIn("about %d-%d words" % (lo, hi), p)
                self.assertIn("Topic: a retry loop", p)
                self.assertIn(prompts.OUTPUT_ONLY, p)
                self.assertNotIn("TOPIC:", p)
                if key != "custom":
                    self.assertIn("Shape: %s" % shape.label, p)
                    self.assertIn(shape.structure, p)
                    self.assertIn(shape.tone, p)
                self.assertIn("Additionally", p)
                self.assertIn("relative file paths", p)
                self.assertIn("reuse nothing from these instructions", p)
                self.assertIn("never call anything invented", p)
                self.assertNotIn("{", p)
                self.assertEqual(sum(n in p for n in prompts.PEOPLE), 3)

    def test_seeded_rng_is_deterministic(self):
        a = prompts.build_prompt("issue", "short", "", rng=random.Random(7))
        self.assertEqual(a, prompts.build_prompt("issue", "short", "", rng=random.Random(7)))

    def test_specific_structures(self):
        self.assertIn("## Testing", prompts.build_prompt("pr_description", "long", "x"))
        self.assertIn("**Yesterday:**", prompts.build_prompt("standup", "short", "x"))
        self.assertIn("280 characters", prompts.build_prompt("tweet_thread", "short", "x"))
        self.assertIn("72 characters", prompts.build_prompt("commit_message", "short", "x"))

    def test_empty_topic_asks_the_model_to_pick_one(self):
        p = prompts.build_prompt("issue", "short", "  ", area="caching")
        self.assertIn("area of caching", p)
        self.assertIn("not about any real company", p)
        self.assertIn("TOPIC:", p)

    def test_custom_needs_a_description(self):
        self.assertIn("Shape: a changelog haiku", prompts.build_prompt("custom", "short", "x", "a changelog haiku"))
        for bad in (("custom", "short"), ("nope", "short"), ("issue", "tiny")):
            with self.assertRaises(prompts.StudioError):
                prompts.build_prompt(*bad, topic="x")

    def test_split_topic(self):
        self.assertEqual(prompts.split_topic("TOPIC: cache stampede on the menu service\n---\nHi team,\n\nBody."),
                         ("cache stampede on the menu service", "Hi team,\n\nBody."))
        self.assertEqual(prompts.split_topic("**Topic:** a thing\n\nBody"), ("a thing", "Body"))
        self.assertEqual(prompts.split_topic("Just the text."), ("", "Just the text."))


class CheckTests(unittest.TestCase):
    def test_a_faithful_rewrite_has_no_warnings(self):
        self.assertEqual(pairs.check_pair(SLOP, MINE), [])

    def test_empty_and_identical(self):
        self.assertEqual(pairs.check_pair(SLOP, "  "), ["the target is empty"])
        self.assertEqual(pairs.check_pair(SLOP, SLOP + "\n"), ["the target is identical to the input"])

    def test_one_sided_specifics(self):
        w = pairs.check_pair("Please check https://quill.example.dev/admin and the retry cap of 5 before Friday.",
                             "check retry cap before fri, cap 7")
        self.assertEqual(w, ["URL only in the input: https://quill.example.dev/admin",
                             "number only in the input: 5", "number only in your version: 7"])

    def test_code_spans_may_lose_their_backticks(self):
        self.assertEqual(pairs.check_pair("Set `quill.max_retries` to 3.", "set quill.max_retries to 3"), [])
        self.assertEqual(pairs.check_pair("Set `quill.max_retries` to 3.", "set the retry cap to 3"),
                         ["code span only in the input: quill.max_retries"])
        self.assertEqual(pairs.check_pair("Set the cap to 3.", "set `cap_max` to 3"),
                         ["code span only in your version: cap_max"])

    def test_numbers_in_urls_and_versions(self):
        self.assertEqual(pairs.check_pair("Bump to v2.3.1 at https://ex.dev/42.", "bump v2.3.1 https://ex.dev/42"), [])
        many = pairs.check_pair("1 2 3 4 5 6 words", "words")
        self.assertEqual(many, ["numbers only in the input: 1, 2, 3, 4 and 2 more"])

    def test_stats(self):
        rs = [rec(), rec(tgt="other one", shape="issue", generator={"backend": "codex", "model": "m",
                                                                  "prompt_version": "studio_v1"})]
        s = pairs.stats(rs)
        self.assertEqual(s["total"], 2)
        self.assertEqual(s["by_generator"], {"pasted": 1, "m": 1})
        self.assertEqual(s["with_warnings"], 1)
        self.assertIn("2 handwritten pairs (1 with check warnings)", pairs.render_stats(s))


class CodexCommandTests(unittest.TestCase):
    def test_command_line(self):
        g = generators.CodexGenerator()
        self.assertEqual(g.command("/w", "/w/.last"), [
            "codex", "exec", "-m", "gpt-6.1-sol", "-c", "model_reasoning_effort=low", "-s", "read-only",
            "--skip-git-repo-check", "--ephemeral", "-C", "/w", "-o", "/w/.last", "-"])
        self.assertNotIn("-c", generators.CodexGenerator(model="m2", effort="").command("/w", "/l"))
        self.assertEqual(g.info, {"backend": "codex", "model": "gpt-6.1-sol", "prompt_version": "studio_v1"})

    def test_generate_reads_the_last_message_from_an_empty_dir(self):
        seen = {}

        def fake_run(cmd, input, cwd, **kw):
            seen.update(cmd=cmd, input=input, cwd=cwd, listing=os.listdir(cwd))
            with open(cmd[cmd.index("-o") + 1], "w", encoding="utf-8") as f:
                f.write("  the text\n")
            return mock.Mock(returncode=0, stdout="noise", stderr="")

        with tempfile.TemporaryDirectory() as root, mock.patch("subprocess.run", fake_run):
            g = generators.CodexGenerator(tmp_root=root)
            self.assertEqual(g.generate("the prompt"), "  the text\n")
            self.assertEqual(seen["input"], "the prompt")
            self.assertEqual(seen["listing"], [])
            self.assertEqual(os.path.dirname(seen["cwd"]), root)
            self.assertEqual(os.listdir(root), [])

    def test_empty_output_is_an_error(self):
        with mock.patch("subprocess.run", return_value=mock.Mock(returncode=1, stdout="", stderr="boom")):
            with self.assertRaises(prompts.StudioError) as cm:
                generators.CodexGenerator().generate("p")
        self.assertEqual(cm.exception.status, 502)
        self.assertIn("boom", str(cm.exception))

    def test_status_without_codex(self):
        with mock.patch("shutil.which", return_value=None):
            self.assertEqual(generators.CodexGenerator().status(), (False, "codex CLI not found on PATH"))


class ServerTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.gen = generators.FakeGenerator()
        self.store = pairs.PairStore(os.path.join(self.dir.name, "pairs.jsonl"))
        self.server = studio_server.StudioServer(("127.0.0.1", 0), self.store, self.gen, quiet=True)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.dir.cleanup()

    def call(self, method, path, body=None, headers=None, ctype="application/json"):
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        h = {"Host": "127.0.0.1:%d" % self.port}
        data = None
        if body is not None:
            data = json.dumps(body).encode()
            h["Content-Type"] = ctype
        h.update(headers or {})
        c.request(method, path, body=data, headers=h)
        r = c.getresponse()
        raw = r.read()
        c.close()
        try:
            return r.status, json.loads(raw)
        except ValueError:
            return r.status, raw.decode()

    def test_round_trip(self):
        st, page = self.call("GET", "/")
        self.assertEqual(st, 200)
        self.assertIn("Irulan Studio", page)
        st, s = self.call("GET", "/api/status")
        self.assertTrue(s["generator"]["available"])
        self.assertEqual(len(s["shapes"]), len(prompts.SHAPES))

        st, g = self.call("POST", "/api/generate", {"shape": "pr_description", "length": "short", "topic": ""})
        self.assertEqual(st, 200)
        self.assertEqual(g["topic"], "retry loop for the ledger sync service")
        self.assertTrue(g["text"].startswith("**Summary:**"))

        st, saved = self.call("POST", "/api/pairs", {"input": g["text"], "target": "move retry loop, p95 840 -> 310",
                                                     "shape": "pr_description", "length": "short",
                                                     "topic": g["topic"]})
        self.assertEqual(st, 200)
        self.assertEqual(saved["pair"]["generator"], self.gen.info)
        self.assertEqual(saved["counts"], {"total": 1, "by_shape": {"pr_description": 1}})
        self.assertEqual(saved["warnings"], ["number only in the input: 2.3.1",
                                             "code span only in the input: ledger-sync"])

        st, pasted = self.call("POST", "/api/pairs", {"input": "Could you please restart the quill worker?",
                                                      "target": "restart quill worker pls", "shape": "slack_thread_reply",
                                                      "length": "short", "topic": ""})
        self.assertEqual(pasted["pair"]["generator"], "pasted")
        self.assertEqual(pasted["warnings"], [])

        st, lst = self.call("GET", "/api/pairs")
        self.assertEqual([p["id"] for p in lst["pairs"]], [pasted["pair"]["id"], saved["pair"]["id"]])

        st, ed = self.call("PUT", "/api/pairs/" + saved["pair"]["id"], {"target": g["text"]})
        self.assertEqual(st, 200)
        self.assertEqual(ed["warnings"], ["the target is identical to the input"])
        st, ed = self.call("PUT", "/api/pairs/" + ed["pair"]["id"], {"target": "moved the retry loop"})
        self.assertEqual(ed["pair"]["target"], "moved the retry loop")
        st, _ = self.call("DELETE", "/api/pairs/" + pasted["pair"]["id"])
        self.assertEqual(st, 200)
        self.assertEqual([r["id"] for r in self.store.records()], [ed["pair"]["id"]])
        st, stt = self.call("GET", "/api/stats")
        self.assertEqual(stt["total"], 1)
        self.assertEqual(self.gen.calls, 1)

    def test_local_only(self):
        st, _ = self.call("GET", "/api/status", headers={"Host": "evil.example.com"})
        self.assertEqual(st, 403)
        st, _ = self.call("POST", "/api/pairs", {"input": "a", "target": "b"},
                          headers={"Origin": "http://evil.example.com"})
        self.assertEqual(st, 403)
        st, _ = self.call("POST", "/api/pairs", {"input": "a", "target": "b"}, ctype="text/plain")
        self.assertEqual(st, 415)
        self.assertFalse(os.path.exists(self.store.path))
        self.assertEqual(self.server.server_address[0], "127.0.0.1")

    def test_one_generation_at_a_time_and_disabled(self):
        self.server.gen_lock.acquire()
        try:
            st, r = self.call("POST", "/api/generate", {"shape": "issue", "length": "short", "topic": "x"})
            self.assertEqual(st, 409)
        finally:
            self.server.gen_lock.release()
        self.gen.status = lambda: (False, "codex CLI not found on PATH")
        st, r = self.call("POST", "/api/generate", {"shape": "issue", "length": "short", "topic": "x"})
        self.assertEqual(st, 503)
        self.assertEqual(self.gen.calls, 0)
        st, s = self.call("GET", "/api/status")
        self.assertFalse(s["generator"]["available"])
        st, _ = self.call("POST", "/api/pairs", {"input": "Could you restart it?", "target": "restart it",
                                                 "shape": "issue", "length": "short"})
        self.assertEqual(st, 200)


class CliTests(unittest.TestCase):
    def test_stats_and_default_path(self):
        import contextlib
        import io
        from irulan import cli
        with tempfile.TemporaryDirectory() as d:
            pairs.PairStore(os.path.join(d, "handwritten", "pairs.jsonl")).add(rec())
            out = io.StringIO()
            with mock.patch.dict(os.environ, {"IRULAN_DATA_DIR": d}), contextlib.redirect_stdout(out):
                self.assertEqual(cli.main(["studio", "--stats"]), 0)
        self.assertIn("1 handwritten pairs", out.getvalue())


if __name__ == "__main__":
    unittest.main()
