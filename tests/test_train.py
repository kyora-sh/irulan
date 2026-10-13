"""Launcher-side tests for train/modal_train.py: data checks, the estimate and
the spend and profile guards. No Modal, no torch. Every value is invented."""

import importlib.util
import io
import json
import os
import random
import re
import sys
import tempfile
import unittest
from contextlib import redirect_stdout

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATH = os.path.join(HERE, "train", "modal_train.py")


def load(name="modal_train_under_test"):
    spec = importlib.util.spec_from_file_location(name, PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


mt = load()
SYSTEM = "Rewrite the text in the writer's own voice."


def rec(i, inp="Could you please restart the quux worker for me?", tgt="restart quux worker pls"):
    return {"id": "t%d" % i, "messages": [{"role": "system", "content": SYSTEM},
                                          {"role": "user", "content": inp},
                                          {"role": "assistant", "content": tgt}]}


def write_splits(d, train, val, test):
    for name, rows in (("train", train), ("val", val), ("test", test)):
        with open(os.path.join(d, name + ".jsonl"), "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")


class DataCheckTests(unittest.TestCase):
    def test_clean_splits_pass(self):
        with tempfile.TemporaryDirectory() as d:
            write_splits(d, [rec(i) for i in range(40)], [rec(100)], [rec(200)])
            stats, problems = mt.summarise_data(d, 1024, 8)
            self.assertEqual(problems, [])
            self.assertEqual(stats["train"]["n"], 40)
            self.assertGreater(stats["train"]["tokens"], 40 * 10)

    def test_schema_problems_are_reported(self):
        bad_roles = rec(1)
        bad_roles["messages"] = bad_roles["messages"][1:]
        leftover = rec(2, tgt="ping <NAME> about it")
        with tempfile.TemporaryDirectory() as d:
            write_splits(d, [rec(0), bad_roles, leftover], [rec(0)], [rec(3)])
            with open(os.path.join(d, "train.jsonl"), "a") as f:
                f.write('{"messages": [\n')
            _, problems = mt.summarise_data(d, 1024, 8)
        text = "\n".join(problems)
        self.assertIn("system, user, assistant", text)
        self.assertIn("placeholder token left", text)
        self.assertIn("not JSON", text)
        self.assertIn("id t0 is in both train and val", text)

    def test_over_length_examples_are_counted(self):
        long_ = rec(5, inp="word " * 2000, tgt="short reply here")
        with tempfile.TemporaryDirectory() as d:
            write_splits(d, [rec(0), long_], [rec(1)], [rec(2)])
            stats, _ = mt.summarise_data(d, 1024, 8)
        self.assertEqual(stats["train"]["over"], 1)
        self.assertEqual(stats["train"]["n"], 1)

    def test_dry_run_never_needs_modal(self):
        with tempfile.TemporaryDirectory() as d:
            write_splits(d, [rec(i) for i in range(64)], [rec(100)], [rec(200)])
            buf = io.StringIO()
            with redirect_stdout(buf):
                est = mt.main(["train", "--dry-run", "--data", d, "--scale-to", "1000"])
        self.assertIn("dry run: nothing was sent to Modal.", buf.getvalue())
        self.assertEqual(est["total_steps"], 4)  # 64 / 32 per step x 2 epochs


class EstimateTests(unittest.TestCase):
    def stats(self, n=9000, tokens=1_200_000):
        return {"train": {"n": n, "tokens": tokens, "pad": 1.0},
                "val": {"n": 500, "tokens": 60_000, "pad": 1.0},
                "test": {"n": 500, "tokens": 60_000, "pad": 1.0}}

    def test_steps_and_dollars(self):
        e = mt.estimate(self.stats(), "H100", epochs=2, effective_batch=32, micro_batch=8)
        self.assertEqual(e["grad_accum"], 4)
        self.assertEqual(e["steps_per_epoch"], 282)
        self.assertEqual(e["total_steps"], 564)
        self.assertAlmostEqual(e["usd"], e["hours"] * 3.95, delta=0.02)
        lo, hi = e["usd_range"]
        self.assertLess(lo, e["usd"])
        self.assertGreater(hi, e["usd"])

    def test_max_steps_shortens_the_estimate(self):
        full = mt.estimate(self.stats(), "H100")
        short = mt.estimate(self.stats(), "H100", max_steps=20, eval_every=10, val_limit=50)
        self.assertEqual(short["total_steps"], 20)
        self.assertEqual(short["evals"], 3)
        self.assertLess(short["work_tokens"], full["work_tokens"] / 20)
        self.assertLess(short["usd"], full["usd"])

    def test_scale(self):
        a = mt.estimate(self.stats(), "H100")
        b = mt.estimate(self.stats(), "H100", scale=2.0)
        self.assertGreater(b["work_tokens"], 1.9 * a["work_tokens"])

    def test_length_grouping_covers_every_example_once(self):
        lengths = [random.Random(i).randint(20, 900) for i in range(1000)]
        batches = mt.length_grouped_batches(lengths, 8, random.Random(0))
        flat = sorted(i for b in batches for i in b)
        self.assertEqual(flat, list(range(1000)))
        self.assertLess(mt.padding_factor(lengths, 8, grouped=True), mt.padding_factor(lengths, 8, grouped=False))


class GuardTests(unittest.TestCase):
    PROFILES = ["atira", "personal"]
    WS = {"atira": "atira", "personal": "someone-personal"}

    def check(self, profile, env=None, **kw):
        kw.setdefault("profiles", self.PROFILES)
        kw.setdefault("workspaces", self.WS)
        return mt.check_profile(profile, env=env or {}, **kw)

    def test_profile_must_be_explicit(self):
        with self.assertRaises(SystemExit) as cm:
            self.check(None)
        self.assertIn("no Modal profile selected", str(cm.exception))
        self.assertEqual(self.check(None, env={"MODAL_PROFILE": "personal"}), "personal")

    def test_atira_profile_and_workspace_are_refused(self):
        for name in ("atira", "Atira-dev"):
            with self.assertRaises(SystemExit) as cm:
                self.check(name, profiles=self.PROFILES + ["Atira-dev"])
            self.assertIn("--allow-atira-workspace", str(cm.exception))
        with self.assertRaises(SystemExit) as cm:
            self.check("personal", workspaces={"personal": "atira"})
        self.assertIn("workspace", str(cm.exception))

    def test_atira_needs_the_explicit_opt_in(self):
        self.assertEqual(self.check("atira", allow_atira=True), "atira")
        self.assertEqual(self.check("personal", workspaces={"personal": "atira"}, allow_atira=True), "personal")
        self.assertTrue(mt.is_atira("atira", "atira"))
        self.assertTrue(mt.is_atira("personal", "atira"))
        self.assertFalse(mt.is_atira("personal", "someone-personal"))
        # the opt-in does not relax anything else
        with self.assertRaises(SystemExit):
            self.check(None, allow_atira=True)
        with self.assertRaises(SystemExit):
            self.check("atira", env={"MODAL_TOKEN_SECRET": "x"}, allow_atira=True)
        with self.assertRaises(SystemExit):
            self.check("atira", workspaces={}, allow_atira=True)

    def test_atira_environment_must_be_named_and_not_prod(self):
        for bad in (None, "", "prod", "main", "staging"):
            with self.assertRaises(SystemExit):
                mt.check_atira_env(bad)
        self.assertEqual(mt.check_atira_env("testing"), "testing")

    def test_app_name_collision_is_refused(self):
        others = [{"name": "qwen", "state": "deployed", "id": "ap-1"}]
        self.assertTrue(mt.check_app_collision(others))
        self.assertTrue(mt.check_app_collision(others + [{"name": mt.APP_NAME, "state": "stopped", "id": "ap-2"}]))
        for state in ("deployed", "ephemeral", "ephemeral (detached)"):
            with self.assertRaises(SystemExit) as cm:
                mt.check_app_collision(others + [{"name": mt.APP_NAME, "state": state, "id": "ap-3"}])
            self.assertIn("already exists", str(cm.exception))
        with self.assertRaises(SystemExit):
            mt.check_app_collision(None)

    def test_hard_timeouts_and_never_deploy(self):
        self.assertEqual((mt.TRAIN_TIMEOUT_MIN, mt.SMOKE_TIMEOUT_MIN, mt.EVAL_TIMEOUT_MIN), (90, 25, 30))
        ap = mt.build_parser()
        self.assertEqual(mt._timeout_min(ap.parse_args(["train"])), 90)
        self.assertEqual(mt._timeout_min(ap.parse_args(["train", "--smoke"])), 25)
        self.assertEqual(mt._timeout_min(ap.parse_args(["eval"])), 30)
        with open(PATH) as f:
            src = f.read()
        self.assertNotIn(".deploy(", src)
        self.assertNotIn("8 * 3600", src)

    def test_smoke_caps_max_steps(self):
        with tempfile.TemporaryDirectory() as d:
            write_splits(d, [rec(i) for i in range(64)], [rec(100)], [rec(200)])
            with redirect_stdout(io.StringIO()):
                with self.assertRaises(SystemExit) as cm:
                    mt.main(["train", "--dry-run", "--smoke", "--max-steps", "500", "--data", d])
                self.assertIn("at most", str(cm.exception))
                buf = io.StringIO()
                with redirect_stdout(buf):
                    est = mt.main(["train", "--dry-run", "--smoke", "--data", d])
        self.assertEqual(est["total_steps"], 4)  # fewer than the default 20 smoke steps
        self.assertIn("hard cap: smoke function timeout 25 min", buf.getvalue())

    def test_unknown_profile_or_unverified_workspace_is_refused(self):
        with self.assertRaises(SystemExit):
            self.check("nope")
        with self.assertRaises(SystemExit):
            self.check("personal", workspaces={})
        self.assertEqual(self.check("personal", workspaces={}, skip_workspace_check=True), "personal")

    def test_token_env_vars_are_refused(self):
        with self.assertRaises(SystemExit):
            self.check("personal", env={"MODAL_TOKEN_ID": "x"})

    def test_gpu_jobs_need_confirm_spend(self):
        with tempfile.TemporaryDirectory() as d:
            write_splits(d, [rec(i) for i in range(8)], [rec(100)], [rec(200)])
            with redirect_stdout(io.StringIO()):
                with self.assertRaises(SystemExit) as cm:
                    mt.main(["train", "--data", d, "--profile", "personal"])
            self.assertIn("--confirm-spend", str(cm.exception))
            with self.assertRaises(SystemExit) as cm:
                mt.main(["eval", "--data", d, "--profile", "personal"])
            self.assertIn("--confirm-spend", str(cm.exception))

    def test_modal_cli_import_is_refused(self):
        saved = sys.argv[:]
        try:
            sys.argv = ["/opt/bin/modal", "run", PATH]
            with self.assertRaises(SystemExit) as cm:
                load("modal_train_cli")
            self.assertIn("disabled", str(cm.exception))
        finally:
            sys.argv = saved

    def test_profile_names_only(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "m.toml")
            with open(p, "w") as f:
                f.write('[alpha]\ntoken_id = "ak-zzzz"\ntoken_secret = "as-zzzz"\nactive = true\n\n[beta]\ntoken_id = "x"\n')
            self.assertEqual(mt.modal_profiles(p), ["alpha", "beta"])


class RemoteHelperTests(unittest.TestCase):
    def test_lora_targets_text_layers_only(self):
        r = re.compile(mt.lora_target_regex(mt.DEFAULT_LORA_MODULES))
        yes = ["model.layers.3.self_attn.q_proj", "model.layers.0.linear_attn.in_proj_qkv",
               "model.layers.0.linear_attn.out_proj", "model.layers.63.mlp.down_proj",
               "model.language_model.layers.1.mlp.up_proj"]
        no = ["model.layers.0.linear_attn.in_proj_a", "model.visual.blocks.0.attn.qkv",
              "model.visual.merger.linear_fc1", "mtp.layers.0.self_attn.q_proj", "lm_head",
              "model.embed_tokens"]
        for n in yes:
            self.assertTrue(r.match(n), n)
        for n in no:
            self.assertFalse(r.match(n), n)

    def test_blind_side_by_side_key(self):
        rows = [{"id": "b%d" % i, "input": "in %d" % i, "model": "model %d" % i, "target": "real %d" % i}
                for i in range(20)]
        md, key = mt.blind_side_by_side(rows)
        self.assertEqual(len(key), 20)
        self.assertNotIn("model:", md.lower())
        sides = {v["model"] for v in key.values()}
        self.assertEqual(sides, {"A", "B"})
        for n, v in key.items():
            block = md.split("## %s\n" % n, 1)[1].split("---", 1)[0]
            a = block.split("**A**", 1)[1].split("**B**", 1)[0]
            self.assertEqual("model" in a, v["model"] == "A")


if __name__ == "__main__":
    unittest.main()
