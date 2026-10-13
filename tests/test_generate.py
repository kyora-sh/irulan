"""Generator, pair-check and sampling tests. Every message here is invented."""

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from irulan import check_pairs, generate, sample  # noqa: E402


def pair(inp, tgt, variant="plain", **meta):
    return {"id": "x", "backend": "fake", "model": "fake-1", "variant": variant,
            "input": inp, "target": tgt, "meta": meta}


class PromptTests(unittest.TestCase):
    def test_formatted_needs_thirty_words(self):
        short = "can u rq restart the quux worker pls"
        long_ = " ".join(["word"] * 30)
        self.assertEqual(generate.effective_variant("formatted", short), "plain")
        self.assertEqual(generate.effective_variant("formatted", long_), "formatted")
        self.assertIn("No markdown", generate.build_prompt(short, "formatted"))
        self.assertIn("Use light markdown", generate.build_prompt(long_, "formatted"))
        self.assertIn("Light markdown", generate.build_prompt(long_, "formatted", "v1"))

    def test_prompt_lists_the_message_placeholders(self):
        p = generate.build_prompt("check <URL> and ping <NAME> about <URL>", "plain")
        self.assertIn("(<NAME>, <URL>)", p)
        self.assertTrue(p.rstrip().endswith("check <URL> and ping <NAME> about <URL>\n>>>"))

    def test_clean_output_strips_wrappers(self):
        self.assertEqual(generate.clean_output("<<<\nHello there.\n>>>"), "Hello there.")
        self.assertEqual(generate.clean_output("Rewrite:\nHello there."), "Hello there.")

    def test_seeded_variant_is_stable(self):
        a = [generate.pick_variant("s", "id%d" % i) for i in range(50)]
        b = [generate.pick_variant("s", "id%d" % i) for i in range(50)]
        self.assertEqual(a, b)
        self.assertEqual(set(a), {"plain", "formatted"})

    def test_aliases(self):
        self.assertEqual(generate.make_backend("codex", "sol").model, "gpt-6.1-sol")
        self.assertEqual(generate.make_backend("codex", "luna").model, "gpt-6-luna")


class RunnerTests(unittest.TestCase):
    def test_resumable_and_logged(self):
        recs = [{"id": "a1", "text": "can u rq check the zorblax build"},
                {"id": "b2", "text": "pls drop the debug flag ty"}]
        with tempfile.TemporaryDirectory() as d:
            src = os.path.join(d, "in.jsonl")
            with open(src, "w") as f:
                for r in recs:
                    f.write(json.dumps(r) + "\n")
            args = ["--backend", "fake", "--run", "t", "--input", src, "--out-dir", d, "--variants", "both"]
            generate.main(args)
            generate.main(args)  # second run must not duplicate
            with open(os.path.join(d, "t.jsonl")) as f:
                out = [json.loads(line) for line in f]
            self.assertEqual(len(out), 4)
            self.assertEqual({(o["id"], o["variant"]) for o in out},
                             {("a1", "plain"), ("a1", "formatted"), ("b2", "plain"), ("b2", "formatted")})
            self.assertTrue(all(o["target"] in (r["text"] for r in recs) for o in out))
            with open(os.path.join(d, "t.log.jsonl")) as f:
                self.assertEqual(len(f.readlines()), 4)


class ScriptedBackend(generate.Backend):
    """Raises the scripted errors in order, then succeeds."""
    name = "fake"

    def __init__(self, errors):
        super().__init__("fake-1")
        self.errors = list(errors)
        self.calls = 0

    def call(self, prompt):
        self.calls += 1
        if self.errors:
            e = self.errors.pop(0)
            if e is not None:
                raise e
        return {"text": "Could you please check the zorblax build?", "usage": {}, "meta": {}}


def rate_err():
    return generate.BackendError("stream error: 429 Too Many Requests",
                                 kind=generate.classify_error("429 Too Many Requests"))


class StopTests(unittest.TestCase):
    def runner(self, d, backend, **kw):
        kw.setdefault("rate_backoff", 0.0)
        kw.setdefault("backoff", 0.0)
        return generate.Runner(backend, os.path.join(d, "o.jsonl"), os.path.join(d, "o.log.jsonl"),
                               quiet=True, **kw)

    def jobs(self, n):
        return [({"id": "m%d" % i, "text": "pls check the zorblax build"}, "plain") for i in range(n)]

    def test_classify(self):
        self.assertEqual(generate.classify_error("You've hit your usage limit. Try again at 3:04 PM."), "usage_limit")
        self.assertEqual(generate.classify_error("unexpected status 429 Too Many Requests"), "rate_limit")
        self.assertEqual(generate.classify_error("Rate limit reached for requests"), "rate_limit")
        self.assertEqual(generate.classify_error("stream disconnected before completion"), "")

    def test_rate_limit_retries_past_the_normal_limit(self):
        with tempfile.TemporaryDirectory() as d:
            b = ScriptedBackend([rate_err(), rate_err(), rate_err(), rate_err()])
            r = self.runner(d, b, retries=2, rate_retries=6)
            r.run(self.jobs(1), 1)
            self.assertEqual((r.ok, r.failed, b.calls), (1, 0, 5))
            self.assertFalse(r.stop.is_set())

    def test_usage_limit_stops_the_run(self):
        with tempfile.TemporaryDirectory() as d:
            b = ScriptedBackend([generate.BackendError("You've hit your usage limit.", retryable=False,
                                                       kind="usage_limit")])
            r = self.runner(d, b)
            r.run(self.jobs(20), 1)
            self.assertTrue(r.stop.is_set())
            self.assertEqual((r.ok, r.failed, r.skipped, b.calls), (0, 1, 19, 1))

    def test_persistent_rate_limits_stop_the_run(self):
        with tempfile.TemporaryDirectory() as d:
            b = ScriptedBackend([rate_err() for _ in range(100)])
            r = self.runner(d, b, rate_retries=2, max_rate_failures=3)
            r.run(self.jobs(20), 1)
            self.assertTrue(r.stop.is_set())
            self.assertEqual((r.ok, r.failed, r.skipped, b.calls), (0, 3, 17, 6))

    def test_other_failures_in_a_row_stop_the_run(self):
        with tempfile.TemporaryDirectory() as d:
            b = ScriptedBackend([generate.BackendError("boom", retryable=False) for _ in range(100)])
            r = self.runner(d, b, max_consecutive_failures=4)
            r.run(self.jobs(10), 1)
            self.assertEqual((r.failed, r.skipped), (4, 6))

    def test_resume_drops_a_half_written_line(self):
        with tempfile.TemporaryDirectory() as d:
            src = os.path.join(d, "in.jsonl")
            with open(src, "w") as f:
                for i in range(3):
                    f.write(json.dumps({"id": "m%d" % i, "text": "pls check build %d" % i}) + "\n")
            args = ["--backend", "fake", "--run", "t", "--input", src, "--out-dir", d, "--variants", "plain"]
            out = os.path.join(d, "t.jsonl")
            generate.main(args + ["--limit", "2"])
            with open(out, "a") as f:
                f.write('{"id": "m2", "backend": "fa')
            self.assertEqual(generate.main(args), 0)
            with open(out) as f:
                rows = [json.loads(line) for line in f]
            self.assertEqual(sorted(r["id"] for r in rows), ["m0", "m1", "m2"])


class CheckTests(unittest.TestCase):
    def test_placeholders(self):
        self.assertIn("placeholder_dropped", check_pairs.check(pair("Please look at it, thanks a lot.", "look at <URL> pls")))
        self.assertIn("placeholder_added", check_pairs.check(pair("Please look at <URL> and <URL>.", "look at <URL> pls")))
        self.assertEqual(check_pairs.check(pair("Could you please look at <URL> for me?", "look at <URL> pls?")), [])

    def test_length_and_preamble(self):
        self.assertIn("shorter", check_pairs.check(pair("Fix it.", "fix the quux thing pls")))
        self.assertIn("too_long", check_pairs.check(pair(" ".join(["w"] * 21), "fix the quux thing pls")))
        self.assertIn("preamble", check_pairs.check(pair("Here's the rewritten message: Please fix the quux thing.", "fix quux pls")))
        self.assertIn("preamble", check_pairs.check(pair("Sure! Please fix the quux thing for me.", "fix quux pls")))

    def test_answered(self):
        self.assertIn("answered", check_pairs.check(pair(
            "Yes, the quux job is safe to rerun because it is idempotent.", "is the quux job safe to rerun?")))
        self.assertNotIn("answered", check_pairs.check(pair(
            "Could you confirm whether the quux job is safe to rerun?", "is the quux job safe to rerun?")))
        # "yeah" becoming "Yes," is a faithful rewrite
        self.assertNotIn("answered", check_pairs.check(pair(
            "Yes, that works. Is the quux job safe to rerun?", "yeah that works, quux job safe to rerun?")))

    def test_markdown_in_plain(self):
        self.assertIn("markdown_in_plain", check_pairs.check(pair(
            "Two things:\n- First item here.\n- Second one.", "2 things first item second one", markdown_allowed=False)))
        self.assertNotIn("markdown_in_plain", check_pairs.check(pair(
            "Two things:\n- First item here.\n- Second one.", "2 things first item second one", "formatted",
            markdown_allowed=True)))


class SampleTests(unittest.TestCase):
    def test_strata_and_variant_split(self):
        recs = []
        for i in range(200):
            n = 15 + (i % 70)
            recs.append({"id": "m%03d" % i, "text": " ".join(["quux"] * n)})
        recs.append({"id": "ph", "text": " ".join(["<URL>"] * 20)})
        a = sample.select(recs, "seed")
        b = sample.select(recs, "seed")
        self.assertEqual(a, b)
        self.assertEqual(len(a), 50)
        buckets = {}
        for p in a:
            buckets[p["bucket"]] = buckets.get(p["bucket"], 0) + 1
        self.assertEqual(buckets, {"15-29": 15, "30-59": 20, "60+": 15})
        self.assertEqual(sum(p["variant"] == "plain" for p in a), 25)
        self.assertNotIn("ph", [p["id"] for p in a])


if __name__ == "__main__":
    unittest.main()
