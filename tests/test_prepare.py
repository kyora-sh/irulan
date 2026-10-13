"""Training-set preparation tests. Every message, name and id here is invented."""

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from irulan import prepare  # noqa: E402
from irulan.scrub import Scrubber  # noqa: E402

PH = ("<NAME>", "<URL>", "<EMAIL>", "<HOST>", "<PATH>", "<ID>", "<IP>", "<SECRET>")


def pair(pid, inp, tgt, variant="plain"):
    return {"id": pid, "backend": "fake", "model": "fake-1", "variant": variant,
            "input": inp, "target": tgt, "meta": {"prompt_version": "v2"}}


def run(pairs, sessions=None, scrubber=None, **kw):
    sessions = sessions if sessions is not None else {p["id"]: "proj/s-" + p["id"] for p in pairs}
    return prepare.prepare(pairs, scrubber or Scrubber(), sessions=sessions, **kw)


def all_records(splits):
    return [r for k in prepare.SPLITS for r in splits[k]]


class SurrogateTests(unittest.TestCase):
    def setUp(self):
        self.s = prepare.Surrogates()

    def test_same_surrogate_on_both_sides_in_order(self):
        inp = "Could you ask <NAME> to look at <URL>, and then tell <NAME> about <HOST>?"
        tgt = "ask <NAME> to check <URL> then tell <NAME> abt <HOST>"
        i, t, counts = self.s.fill("p1", inp, tgt)
        self.assertEqual(counts["NAME"], 2)
        for text in (i, t):
            for ph in PH:
                self.assertNotIn(ph, text)
        names_i = [w for w in self.s.names if w in i]
        self.assertEqual(sorted(names_i), sorted(w for w in self.s.names if w in t))
        # first <NAME> maps to the first name on both sides
        first_i = min(names_i, key=i.index)
        first_t = min(names_i, key=t.index)
        self.assertEqual(first_i, first_t)
        host = [w for w in t.split() if w.endswith(".internal")][0]
        self.assertIn(host, i)

    def test_distinct_values_within_a_pair(self):
        inp = "Ping <NAME>, <NAME> and <NAME> about <IP> and <IP>."
        tgt = "ping <NAME> <NAME> <NAME> re <IP> <IP>"
        i, t, _ = self.s.fill("p2", inp, tgt)
        names = [w.strip(",.") for w in t.split()[1:4]]
        self.assertEqual(len(set(names)), 3)
        ips = t.split()[-2:]
        self.assertNotEqual(ips[0], ips[1])
        for ip in ips:
            self.assertTrue(ip.startswith(("192.0.2.", "198.51.100.", "203.0.113.")))

    def test_seeded_per_pair_id(self):
        inp, tgt = "Please email <EMAIL> with the <SECRET>.", "email <EMAIL> the <SECRET>"
        a = self.s.fill("same", inp, tgt)
        b = self.s.fill("same", inp, tgt)
        self.assertEqual(a, b)
        others = {self.s.fill("id%d" % n, inp, tgt)[1] for n in range(10)}
        self.assertGreater(len(others), 1)

    def test_surrogate_shapes(self):
        inp = "See <URL> at <PATH>, commit <ID>, key <SECRET>, mail <EMAIL>."
        _, t, _ = self.s.fill("p3", inp, inp)
        self.assertIn("https://", t)
        self.assertRegex(t, r"(?:example\.(?:com|org|net))")
        self.assertRegex(t, r"(?:sk-test-|test_secret_|fake-token-|dummy-key-)[0-9a-f]{16}")

    def test_count_mismatch_raises(self):
        with self.assertRaises(prepare.PlaceholderMismatch):
            self.s.fill("p4", "Ask <NAME> and <NAME>.", "ask <NAME>")
        with self.assertRaises(prepare.PlaceholderMismatch):
            self.s.fill("p5", "Open <URL>.", "open <PATH>")

    def test_denylisted_names_are_never_drawn(self):
        s = prepare.Surrogates(blocked=["maya", "jonas"])
        self.assertNotIn("Maya", s.names)
        self.assertNotIn("Jonas", s.names)


class PrepareTests(unittest.TestCase):
    def test_mismatch_pair_is_dropped(self):
        pairs = [pair("a1", "Could you ask <NAME> and <NAME> to review the zorp module?",
                      "ask <NAME> to review zorp"),
                 pair("a2", "Could you please restart the quux worker for me?", "restart quux worker pls")]
        splits, stats, drops = run(pairs)
        # check_pairs already flags it; with that flag allowed, the surrogate step catches it
        self.assertIn({"id": "a1", "reason": "flag_placeholder_added"}, drops)
        splits, stats, drops = run(pairs, allow_flags=("placeholder_added",))
        self.assertIn({"id": "a1", "reason": "placeholder_mismatch"}, drops)
        self.assertEqual([r["id"] for r in all_records(splits)], ["a2"])

    def test_rescrub_mismatch_after_new_denylist_term(self):
        # The input copied a name the target only had as plain text; once the
        # name is denylisted both sides are rescrubbed and still agree.
        pairs = [pair("b1", "Could you ask Blorvik to review the flaky zorp test?",
                      "ask blorvik to check flaky zorp test")]
        splits, stats, _ = run(pairs, scrubber=Scrubber(["blorvik"]))
        rec = all_records(splits)[0]
        text = json.dumps(rec)
        self.assertNotIn("lorvik", text.lower())
        self.assertEqual(stats["rescrub_new_replacements"], {"input_name": 1, "target_name": 1})
        self.assertEqual(stats["surrogates"]["by_type"], {"NAME": 2})

    def test_rescrub_catches_new_secret_and_url_in_input(self):
        pairs = [pair("b2", "Please check https://zorp-staging.quuxcorp.dev/admin and fix the banner.",
                      "check <URL> and fix the banner")]
        splits, stats, _ = run(pairs)
        rec = all_records(splits)[0]
        self.assertNotIn("quuxcorp", json.dumps(rec))
        self.assertEqual(stats["rescrub_new_replacements"], {"input_url": 1})

    def test_exclude_ids(self):
        pairs = [pair("c1", "Could you please restart the quux worker?", "restart quux worker pls"),
                 pair("c2", "Could you please bump the zorp version?", "bump zorp version")]
        splits, _, drops = run(pairs, exclude={"c1"})
        self.assertEqual([r["id"] for r in all_records(splits)], ["c2"])
        self.assertEqual(drops, [{"id": "c1", "reason": "excluded"}])

    def test_exclude_file_ignores_comments(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "x.local.txt")
            with open(p, "w") as f:
                f.write("# note\nabc123\n\n def456  # trailing\n")
            self.assertEqual(prepare.load_exclude_ids(p), {"abc123", "def456"})

    def test_short_targets_duplicates_and_flags(self):
        pairs = [
            pair("d1", "Fine, go ahead with the zorp change.", "ok"),                      # 1 word
            pair("d2", "Could you please run the tests again?", "rerun tests"),    # 2 words, kept
            pair("d3", "Could you please run the tests again?", "run tests again"),  # duplicate input
            pair("d4", "Here's the rewritten message: please deploy the zorp app.", "pls deploy zorp app"),
            pair("d5", "Okay, here is what is left: the quux banner and the footer.",
                 "ok what is left: quux banner and footer"),
            pair("d6", "No.", "can u deploy the zorp worker rq and then check the logs for errors"),
        ]
        splits, stats, drops = run(pairs)
        kept = {r["id"] for r in all_records(splits)}
        reasons = {d["id"]: d["reason"] for d in drops}
        self.assertEqual(kept, {"d2", "d5"})
        self.assertEqual(reasons["d1"], "short_target")
        self.assertEqual(reasons["d3"], "duplicate_input")
        self.assertEqual(reasons["d4"], "meta_preamble")
        self.assertEqual(reasons["d6"], "flag_shorter")
        self.assertEqual(stats["check_pairs_flags_kept"].get("preamble"), 1)
        splits, _, _ = run(pairs, min_target_words=3)
        self.assertNotIn("d2", {r["id"] for r in all_records(splits)})

    def test_no_session_drops_unless_allowed(self):
        pairs = [pair("e1", "Could you please restart the quux worker?", "restart quux worker pls")]
        _, _, drops = run(pairs, sessions={})
        self.assertEqual(drops, [{"id": "e1", "reason": "not_in_clean"}])
        splits, _, _ = run(pairs, sessions={}, require_session=False)
        self.assertEqual(len(all_records(splits)), 1)

    def test_chat_format(self):
        pairs = [pair("f1", "Could you please restart the quux worker?", "restart quux worker pls")]
        splits, stats, _ = run(pairs)
        rec = all_records(splits)[0]
        self.assertEqual([m["role"] for m in rec["messages"]], ["system", "user", "assistant"])
        self.assertEqual(rec["messages"][0]["content"], prepare.SYSTEM_PROMPTS["system_v1"])
        self.assertEqual(rec["messages"][2]["content"], "restart quux worker pls")
        self.assertEqual(rec["system_version"], "system_v1")
        self.assertGreater(stats["tokens"]["total"], 0)


class SplitTests(unittest.TestCase):
    def make(self, n_sessions=60):
        pairs, sessions = [], {}
        k = 0
        for s in range(n_sessions):
            for _ in range(1 + (s * 7) % 23):
                pid = "id%04d" % k
                pairs.append(pair(pid, "Could you please handle zorp task %d for the quux team?" % k,
                                  "handle zorp task %d pls" % k))
                sessions[pid] = "proj/sess-%02d" % s
                k += 1
        return pairs, sessions

    def test_sessions_never_cross_splits(self):
        pairs, sessions = self.make()
        splits, stats, _ = run(pairs, sessions=sessions)
        where = {}
        for k in prepare.SPLITS:
            for r in splits[k]:
                s = sessions[r["id"]]
                self.assertEqual(where.setdefault(s, k), k, "session %s leaked" % s)
        self.assertEqual(sum(len(splits[k]) for k in prepare.SPLITS), len(pairs))
        for k in ("val", "test"):
            share = len(splits[k]) / len(pairs)
            self.assertGreater(share, 0.02)
            self.assertLess(share, 0.09)
        self.assertGreater(len(splits["train"]) / len(pairs), 0.85)

    def test_split_is_deterministic(self):
        pairs, sessions = self.make()
        a, _, _ = run(pairs, sessions=sessions)
        b, _, _ = run(list(pairs), sessions=sessions)
        for k in prepare.SPLITS:
            self.assertEqual([r["id"] for r in a[k]], [r["id"] for r in b[k]])
        c, _, _ = run(pairs, sessions=sessions, seed="other-seed")
        self.assertNotEqual([r["id"] for r in a["test"]], [r["id"] for r in c["test"]])


class ReadPairsTests(unittest.TestCase):
    def test_partial_last_line_is_skipped(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "pairs.jsonl")
            with open(p, "w") as f:
                f.write(json.dumps(pair("g1", "Could you restart it?", "restart it")) + "\n")
                f.write('{"id": "g2", "input": "Could you')
            recs, bad = prepare.read_pairs(p)
            self.assertEqual([r["id"] for r in recs], ["g1"])
            self.assertEqual(bad, 1)


if __name__ == "__main__":
    unittest.main()
