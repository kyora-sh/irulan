"""Studio tests: records, the pairs file, prompts, prepare compatibility and
the HTTP API with a fake generator. Every text here is invented."""

import http.client
import json
import os
import sys
import tempfile
import threading
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from irulan import prepare, studio  # noqa: E402
from irulan.scrub import Scrubber  # noqa: E402

SLOP = ("## Summary\n\nThis PR moves the retry loop for the **quillfeather** sync worker into "
        "services/quill/retry.py. Additionally, it lowers p95 latency from 840 ms to 310 ms and bumps "
        "the client to v2.3.1. That said, it might be worth a second look at the backoff cap.\n\n"
        "## Changes\n\n- Moved the retry loop\n- Added 4 unit tests\n- Updated the config key `quill.max_retries`")
MINE = "move quillfeather retry loop to services/quill/retry.py, p95 840 -> 310ms, client v2.3.1"


def rec(inp=SLOP, tgt=MINE, shape="pr_description", length="medium", **kw):
    return studio.make_record(inp, tgt, shape, length, **kw)


class RecordTests(unittest.TestCase):
    def test_fields_are_exactly_the_record_format(self):
        r = rec(topic="retry loop", generator={"backend": "codex", "model": "m", "prompt_version": "studio_v1"})
        self.assertEqual(tuple(r), studio.FIELDS)
        self.assertEqual(r["source"], "handwritten")
        self.assertTrue(r["created_at"].endswith("Z"))

    def test_id_is_stable_and_depends_on_both_sides(self):
        a = rec(created_at="2026-01-02T03:04:05Z")
        b = rec(shape="issue", topic="other", created_at="2027-01-01T00:00:00Z")
        self.assertEqual(a["id"], b["id"])
        self.assertEqual(a["id"], studio.pair_id(SLOP + "\n  ", "  " + MINE))
        self.assertRegex(a["id"], r"^hw_[0-9a-f]{16}$")
        self.assertNotEqual(a["id"], rec(tgt=MINE + " pls")["id"])
        self.assertNotEqual(a["id"], rec(inp=SLOP + " Overall, looks good.")["id"])
        self.assertNotEqual(studio.pair_id("ab", "c"), studio.pair_id("a", "bc"))

    def test_validation(self):
        for bad in (dict(inp=" "), dict(tgt=""), dict(shape="haiku"), dict(length="huge"),
                    dict(generator={"model": "x"})):
            with self.assertRaises(studio.StudioError):
                rec(**bad)


class StoreTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.dir.name, "hand", "pairs.jsonl")
        self.store = studio.PairStore(self.path)

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
        self.assertEqual(new["id"], studio.pair_id(SLOP, "edited text"))
        self.assertEqual(new["created_at"], a["created_at"])
        self.assertEqual(new["generator"], a["generator"])
        self.assertEqual([r["id"] for r in self.store.records()], [new["id"], b["id"]])
        self.assertEqual(os.listdir(os.path.dirname(self.path)), ["pairs.jsonl"])
        edited_input = self.store.update(new["id"], {"input": "Could you please restart the worker?"})
        self.assertEqual(edited_input["generator"], "pasted")

    def test_edit_errors(self):
        a, _ = self.store.add(rec())
        b, _ = self.store.add(rec(tgt="second one"))
        with self.assertRaises(studio.StudioError) as cm:
            self.store.update(b["id"], {"target": MINE})
        self.assertEqual(cm.exception.status, 409)
        with self.assertRaises(studio.StudioError):
            self.store.update(a["id"], {"id": "hw_0000000000000000"})
        with self.assertRaises(studio.StudioError) as cm:
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
        with self.assertRaises(studio.StudioError):
            self.store.delete(a["id"])

    def test_counts(self):
        self.store.add(rec())
        self.store.add(rec(tgt="second one", shape="issue"))
        self.store.add(rec(tgt="third one", shape="issue"))
        self.assertEqual(studio.counts(self.store.records()),
                         {"total": 3, "by_shape": {"pr_description": 1, "issue": 2}})


class PromptTests(unittest.TestCase):
    def test_every_shape_and_length(self):
        for key, shape in studio.SHAPES.items():
            for i, length in enumerate(studio.LENGTHS):
                p = studio.build_prompt(key, length, "a retry loop", custom="a changelog haiku")
                lo, hi = shape.words[i]
                self.assertIn("about %d-%d words" % (lo, hi), p)
                self.assertIn("Topic: a retry loop", p)
                self.assertIn(studio.OUTPUT_ONLY, p)
                self.assertNotIn("TOPIC:", p)
                if key != "custom":
                    self.assertIn("Shape: %s" % shape.label, p)
                    self.assertIn(shape.structure, p)
                    self.assertIn(shape.tone, p)
                self.assertIn("Additionally", p)
                self.assertIn("relative file paths", p)
                self.assertNotIn("{", p)
                self.assertEqual(sum(n in p for n in studio.PEOPLE), 3)

    def test_seeded_rng_is_deterministic(self):
        import random
        a = studio.build_prompt("issue", "short", "", rng=random.Random(7))
        self.assertEqual(a, studio.build_prompt("issue", "short", "", rng=random.Random(7)))

    def test_specific_structures(self):
        self.assertIn("## Testing", studio.build_prompt("pr_description", "long", "x"))
        self.assertIn("**Yesterday:**", studio.build_prompt("standup", "short", "x"))
        self.assertIn("280 characters", studio.build_prompt("tweet_thread", "short", "x"))
        self.assertIn("72 characters", studio.build_prompt("commit_message", "short", "x"))

    def test_empty_topic_asks_for_an_invented_one(self):
        p = studio.build_prompt("issue", "short", "  ", area="caching")
        self.assertIn("area of caching", p)
        self.assertIn("not about any real company", p)
        self.assertIn("TOPIC:", p)

    def test_custom_needs_a_description(self):
        self.assertIn("Shape: a changelog haiku", studio.build_prompt("custom", "short", "x", "a changelog haiku"))
        for bad in (("custom", "short"), ("nope", "short"), ("issue", "tiny")):
            with self.assertRaises(studio.StudioError):
                studio.build_prompt(*bad, topic="x")

    def test_split_topic(self):
        self.assertEqual(studio.split_topic("TOPIC: cache stampede on the menu service\n---\nHi team,\n\nBody."),
                         ("cache stampede on the menu service", "Hi team,\n\nBody."))
        self.assertEqual(studio.split_topic("**Topic:** a thing\n\nBody"), ("a thing", "Body"))
        self.assertEqual(studio.split_topic("Just the text."), ("", "Just the text."))


class PrepareCompatTests(unittest.TestCase):
    def test_handwritten_pairs_survive_prepare(self):
        r = rec(created_at="2026-03-04T10:00:00Z")
        self.assertGreater(len(r["input"].split()), 4 * len(r["target"].split()))
        splits, stats, drops = prepare.prepare([r], Scrubber(), sessions=prepare.handwritten_sessions([r]))
        self.assertEqual(drops, [])
        out = [x for k in prepare.SPLITS for x in splits[k]][0]
        self.assertEqual(out["messages"][1]["content"], SLOP)
        self.assertEqual(out["messages"][2]["content"], MINE)
        self.assertEqual(stats["handwritten"], {"read": 1, "kept": 1})
        generated = dict(r, source=None)
        _, _, drops = prepare.prepare([generated], Scrubber(), sessions=prepare.handwritten_sessions([r]))
        self.assertTrue(drops[0]["reason"].startswith("flag_"))

    def test_sessions_are_per_day(self):
        rs = [rec(tgt="a b c", created_at="2026-03-04T10:00:00Z"),
              rec(tgt="d e f", created_at="2026-03-04T23:00:00Z"),
              rec(tgt="g h i", created_at="2026-03-05T01:00:00Z")]
        s = prepare.handwritten_sessions(rs)
        self.assertEqual(sorted(set(s.values())), ["handwritten/2026-03-04", "handwritten/2026-03-05"])

    def test_cli_extra_pairs(self):
        with tempfile.TemporaryDirectory() as d:
            store = studio.PairStore(os.path.join(d, "pairs.jsonl"))
            for i in range(6):
                store.add(rec(tgt="%s, take %d" % (MINE, i)))
            with open(store.path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"id": "g1", "input": "Could you restart it?", "target": "restart it"}) + "\n")
            out = os.path.join(d, "train")
            import contextlib
            import io
            with contextlib.redirect_stdout(io.StringIO()):
                stats = prepare.main(["--extra-pairs", store.path, "--out", out,
                                      "--clean", os.path.join(d, "none.jsonl"),
                                      "--denylist", os.path.join(d, "none.txt"),
                                      "--exclude", os.path.join(d, "none.txt")])
            self.assertEqual(stats["kept"], 1)  # the same input six times: duplicates dropped
            self.assertEqual(stats["drop_reasons"], {"duplicate_input": 5})
            self.assertEqual(stats["bad_lines"], 1)
            n = sum(1 for k in prepare.SPLITS for _ in open(os.path.join(out, "%s.jsonl" % k)))
            self.assertEqual(n, 1)
            with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
                prepare.main(["--out", out])

    def test_save_warns_about_placeholder_mismatch(self):
        r = rec(inp="Please check https://quill.example.dev/admin and the retry cap of 5 before Friday.",
                tgt="check retry cap (5) before fri")
        w = studio.warnings_for(r, Scrubber())
        self.assertEqual(len(w), 1)
        self.assertIn("URL", w[0])
        self.assertEqual(studio.warnings_for(rec(), Scrubber()), [])

    def test_stats(self):
        rs = [rec(), rec(tgt="other one", shape="issue", generator={"backend": "codex", "model": "m",
                                                                  "prompt_version": "studio_v1"})]
        s = studio.stats(rs, Scrubber())
        self.assertEqual(s["total"], 2)
        self.assertEqual(s["by_generator"], {"pasted": 1, "m": 1})
        self.assertEqual(s["prepare"]["kept"], 1)
        self.assertIn("2 handwritten pairs", studio.render_stats(s))


class ServerTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.gen = studio.FakeGenerator()
        self.store = studio.PairStore(os.path.join(self.dir.name, "pairs.jsonl"))
        self.server = studio.StudioServer(("127.0.0.1", 0), self.store, self.gen, Scrubber(), quiet=True)
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
        self.assertEqual(len(s["shapes"]), len(studio.SHAPES))

        st, g = self.call("POST", "/api/generate", {"shape": "pr_description", "length": "short", "topic": ""})
        self.assertEqual(st, 200)
        self.assertEqual(g["topic"], "retry loop for an invented ledger sync service")
        self.assertTrue(g["text"].startswith("**Summary:**"))

        st, saved = self.call("POST", "/api/pairs", {"input": g["text"], "target": "move retry loop, p95 840 -> 310",
                                                     "shape": "pr_description", "length": "short",
                                                     "topic": g["topic"]})
        self.assertEqual(st, 200)
        self.assertEqual(saved["pair"]["generator"], self.gen.info)
        self.assertEqual(saved["counts"], {"total": 1, "by_shape": {"pr_description": 1}})
        self.assertEqual(saved["warnings"], [])

        st, pasted = self.call("POST", "/api/pairs", {"input": "Could you please restart the quill worker?",
                                                      "target": "restart quill worker pls", "shape": "slack_thread_reply",
                                                      "length": "short", "topic": ""})
        self.assertEqual(pasted["pair"]["generator"], "pasted")

        st, lst = self.call("GET", "/api/pairs")
        self.assertEqual([p["id"] for p in lst["pairs"]], [pasted["pair"]["id"], saved["pair"]["id"]])

        st, ed = self.call("PUT", "/api/pairs/" + saved["pair"]["id"], {"target": "moved the retry loop"})
        self.assertEqual(st, 200)
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


if __name__ == "__main__":
    unittest.main()
