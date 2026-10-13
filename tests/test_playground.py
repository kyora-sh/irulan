"""Stdlib-only tests for serve/modal_playground.py: the token check, adapter
names and paths, the LoRA key rename, request validation and the CLI guard.
No Modal, no torch. Every value is invented."""

import importlib.util
import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATH = os.path.join(HERE, "serve", "modal_playground.py")


def load(name="modal_playground_under_test"):
    spec = importlib.util.spec_from_file_location(name, PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


pg = load()


class TokenTest(unittest.TestCase):
    def test_token(self):
        self.assertTrue(pg.token_ok("abc123", "abc123"))
        self.assertFalse(pg.token_ok("abc124", "abc123"))
        self.assertFalse(pg.token_ok("", "abc123"))
        self.assertFalse(pg.token_ok(None, "abc123"))

    def test_unset_secret_rejects_everything(self):
        self.assertFalse(pg.token_ok("", ""))
        self.assertFalse(pg.token_ok("anything", None))


class SystemPromptTest(unittest.TestCase):
    def test_matches_training(self):
        sys.path.insert(0, HERE)
        try:
            from irulan.prepare import SYSTEM_PROMPTS
        finally:
            sys.path.remove(HERE)
        self.assertEqual(pg.SYSTEM_V1, SYSTEM_PROMPTS["system_v1"])
        msgs = pg.chat_messages("hello")
        self.assertEqual([m["role"] for m in msgs], ["system", "user"])


class AdapterTest(unittest.TestCase):
    def test_paths(self):
        self.assertEqual(pg.adapter_path("/r", "run-a"), "/r/run-a/adapter")
        self.assertEqual(pg.adapter_path("/r", "run-a@checkpoint-20"), "/r/run-a/checkpoints/checkpoint-20")
        self.assertEqual(pg.adapter_path("/r", "run-a@epoch-1"), "/r/run-a/epoch-1")
        for bad in ("", "..", "../x", "a/b", "run@checkpoint-x", "@checkpoint-1", ".hidden", "run@step-1"):
            self.assertIsNone(pg.adapter_path("/r", bad), bad)

    def test_list_and_default(self):
        with tempfile.TemporaryDirectory() as d:
            def touch(*parts):
                p = os.path.join(d, *parts)
                os.makedirs(os.path.dirname(p), exist_ok=True)
                open(p, "w").close()
            touch("smoke-1", "adapter", "adapter_model.safetensors")
            touch("smoke-1", "checkpoints", "checkpoint-20", "adapter_model.safetensors")
            touch("smoke-1", "checkpoints", "checkpoint-10", "adapter_model.safetensors")
            touch("smoke-1", "checkpoints", "checkpoint-30", "optimizer.pt")  # no weights
            touch("smoke-1", "epoch-2", "adapter_model.safetensors")
            touch("smoke-1", "epoch-1", "adapter_model.safetensors")
            touch("empty-run", "config.json")
            touch("check-small", "adapter", "adapter_model.safetensors")
            with open(os.path.join(d, "check-small", "adapter", "adapter_config.json"), "w") as f:
                f.write('{"base_model_name_or_path": "Qwen/Qwen3.5-0.8B"}')
            names = [a["name"] for a in pg.list_adapters(d)]
            self.assertEqual(names, ["smoke-1", "smoke-1@epoch-1", "smoke-1@epoch-2",
                                     "smoke-1@checkpoint-10", "smoke-1@checkpoint-20"])
            self.assertEqual(pg.default_adapter(names), "smoke-1")
            touch("qlora-r32-v1", "adapter", "adapter_model.safetensors")
            names = [a["name"] for a in pg.list_adapters(d)]
            self.assertEqual(pg.default_adapter(names), "qlora-r32-v1")
            self.assertEqual(pg.default_adapter(names, "smoke-1@checkpoint-10"), "smoke-1@checkpoint-10")
            self.assertEqual(pg.default_adapter(names, "missing"), "qlora-r32-v1")
        self.assertEqual(pg.list_adapters("/nonexistent/dir"), [])
        self.assertIsNone(pg.default_adapter([]))

    def test_lora_key_rename(self):
        k = "base_model.model.model.layers.3.linear_attn.in_proj_qkv.lora_A.weight"
        self.assertEqual(pg.vllm_lora_key(k),
                         "base_model.model.model.language_model.layers.3.linear_attn.in_proj_qkv.lora_A.weight")
        done = "base_model.model.model.language_model.layers.0.mlp.up_proj.lora_B.weight"
        self.assertEqual(pg.vllm_lora_key(done), done)
        with self.assertRaises(ValueError):
            pg.vllm_lora_key("base_model.model.lm_head.lora_A.weight")
        keys = [pg.vllm_lora_key(k), done, "base_model.model.model.language_model.layers.3.self_attn.q_proj.lora_B.weight"]
        self.assertEqual(pg.lora_target_modules(keys), ["in_proj_qkv", "q_proj", "up_proj"])


class RequestTest(unittest.TestCase):
    def test_defaults_and_clamps(self):
        r = pg.clean_request({"text": "hi"})
        self.assertEqual((r["temperature"], r["max_tokens"], r["adapter"], r["compare"]), (0.3, 512, None, False))
        r = pg.clean_request({"text": "hi", "temperature": 9, "max_tokens": 0, "adapter": "smoke-1", "compare": 1})
        self.assertEqual((r["temperature"], r["max_tokens"], r["adapter"], r["compare"]), (1.5, 1, "smoke-1", True))
        self.assertIsNone(pg.clean_request({"text": "hi", "adapter": "base"})["adapter"])

    def test_rejects(self):
        for bad in (None, [], {"text": ""}, {"text": "   "}, {"text": "x" * (pg.MAX_INPUT_CHARS + 1)},
                    {"text": "hi", "adapter": "../etc"}, {"text": "hi", "temperature": "warm"}):
            with self.assertRaises(ValueError):
                pg.clean_request(bad)


class GuardTest(unittest.TestCase):
    def test_serve_only(self):
        env = {"MODAL_PROFILE": "personal"}
        pg.check_cli(["modal", "serve", "serve/modal_playground.py"], env)
        for cmd in ("deploy", "run", "shell"):
            with self.assertRaises(SystemExit):
                pg.check_cli(["modal", cmd, "serve/modal_playground.py"], env)

    def test_profile_and_env(self):
        with self.assertRaises(SystemExit):
            pg.check_cli(["modal", "serve", "x.py"], {})
        atira = {"MODAL_PROFILE": "atira"}
        with self.assertRaises(SystemExit):
            pg.check_cli(["modal", "serve", "x.py"], atira)
        with self.assertRaises(SystemExit):
            pg.check_cli(["modal", "serve", "x.py", "--env", "main"], atira)
        pg.check_cli(["modal", "serve", "x.py", "--env", "testing"], atira)
        pg.check_cli(["modal", "serve", "x.py", "-e", "dev"], atira)
        pg.check_cli(["modal", "serve", "x.py"], dict(atira, MODAL_ENVIRONMENT="testing"))
        with self.assertRaises(SystemExit):
            pg.check_cli(["modal", "serve", "x.py"], {"MODAL_PROFILE": "p", "MODAL_TOKEN_ID": "t"})


if __name__ == "__main__":
    unittest.main()
