"""Scrubber tests. Every value here is invented. Token-shaped strings are
assembled at runtime so no secret scanner mistakes this file for a leak."""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from irulan.scrub import Scrubber, load_denylist  # noqa: E402
from irulan import filters  # noqa: E402

FAKE = "Zq" * 20  # 40 letters of obvious nonsense


class SecretTests(unittest.TestCase):
    def setUp(self):
        self.s = Scrubber()

    def scrub(self, text):
        return self.s.scrub(text)

    def test_provider_tokens(self):
        cases = [
            "sk-" + "ant-" + "api03-" + FAKE,
            "sk-" + "proj-" + FAKE,
            "gh" + "p_" + FAKE,
            "github" + "_pat_" + "11" + FAKE,
            "xo" + "xb-" + "1234-" + FAKE,
            "AK" + "IA" + "FAKEFAKEFAKEFAKE",
            "AI" + "za" + "Sy" + FAKE[:33],
        ]
        for tok in cases:
            out, counts = self.scrub("use %s for the test pls" % tok)
            self.assertEqual(out, "use <SECRET> for the test pls", tok)
            self.assertEqual(counts["secret"], 1)

    def test_jwt_and_bearer(self):
        jwt = "ey" + "JhbGciOiJub25lIn0" + "." + "ey" + "JzdWIiOiJmYWtlIn0" + "." + "ZmFrZXNpZ25hdHVyZQ"
        out, _ = self.scrub("token is %s ok" % jwt)
        self.assertEqual(out, "token is <SECRET> ok")
        out, _ = self.scrub("curl -H 'Authorization: Bearer notarealtoken1234567'")
        self.assertIn("Bearer <SECRET>", out)

    def test_assignments_keep_the_key_name(self):
        out, _ = self.scrub("set password=hunter2fake and API_KEY=\"fake-value-123\"")
        self.assertEqual(out, "set password=<SECRET> and API_KEY=\"<SECRET>\"")
        # plain prose with "key:" is left alone
        out, _ = self.scrub("the key: speed matters more")
        self.assertEqual(out, "the key: speed matters more")

    def test_password_in_prose(self):
        out, _ = self.scrub("root pwd is x9@Fake" + "Pass42 for the box")
        self.assertEqual(out, "root pwd is <SECRET> for the box")
        out, _ = self.scrub("the password is wrong again")
        self.assertEqual(out, "the password is wrong again")

    def test_private_key_and_connection_string(self):
        pem = "-----BEGIN " + "PRIVATE KEY-----\nAAAAfakefake\n-----END " + "PRIVATE KEY-----"
        out, _ = self.scrub("here:\n" + pem + "\nthanks")
        self.assertEqual(out, "here:\n<SECRET>\nthanks")
        out, _ = self.scrub("db is postgres://bob:pw123fake@db.example.test:5432/app")
        self.assertEqual(out, "db is <SECRET>")

    def test_long_opaque_runs(self):
        out, _ = self.scrub("blob aGVsbG8gZmFrZSBibG9iIGZvciB0ZXN0cw9Kx2 end")
        self.assertEqual(out, "blob <SECRET> end")
        out, _ = self.scrub("hash " + "ab12" * 8)
        self.assertEqual(out, "hash <SECRET>")
        # long snake_case identifiers are words, not secrets
        ident = "test_an_edit_spanning_three_runs_keeps_links"
        self.assertEqual(self.scrub(ident)[0], ident)


class LocatorTests(unittest.TestCase):
    def setUp(self):
        self.s = Scrubber()

    def scrub(self, text):
        return self.s.scrub(text)[0]

    def test_urls(self):
        self.assertEqual(self.scrub("see https://github.com/fake-org/fake-repo/pull/12, then"), "see github.com, then")
        self.assertEqual(self.scrub("docs at https://docs.example.test/a/b"), "docs at docs.example.test")
        self.assertEqual(self.scrub("dash at https://dash.fakecorp.example/x?y=1"), "dash at <URL>")
        self.assertEqual(self.scrub("open app.fakecorp.io/projects/42 now"), "open <URL> now")

    def test_email_ip_host(self):
        self.assertEqual(self.scrub("mail jane.doe@fakecorp.example"), "mail <EMAIL>")
        self.assertEqual(self.scrub("ssh into 10.1.2.3:22 then"), "ssh into <IP> then")
        self.assertEqual(self.scrub("v6 fe80::1ff:fe23:4567:890a works"), "v6 <IP> works")
        self.assertEqual(self.scrub("at 12:30:45 today"), "at 12:30:45 today")
        self.assertEqual(self.scrub("hit build-box.internal and gpu.fake.modal.run"), "hit <HOST> and <HOST>")
        self.assertEqual(self.scrub("restart gpu-node-7 pls"), "restart <HOST> pls")
        self.assertEqual(self.scrub("try gpt-5 and utf-8"), "try gpt-5 and utf-8")
        self.assertEqual(self.scrub("fakecorp.ai is live"), "<HOST> is live")
        self.assertEqual(self.scrub("on Janes-MacBook-Pro now, my MacBook"), "on <HOST> now, my MacBook")
        self.assertEqual(self.scrub("check github.com"), "check github.com")

    def test_paths(self):
        self.assertEqual(self.scrub("look at /Users/someone/fake/notes.txt pls"), "look at <PATH> pls")
        self.assertEqual(self.scrub("edit ~/proj/fake/README.md"), "edit README.md")
        self.assertEqual(self.scrub("the /v1/chat/completions route"), "the /v1/chat/completions route")
        self.assertEqual(self.scrub("and/or w/o"), "and/or w/o")

    def test_ids(self):
        self.assertEqual(self.scrub("run 123e4567-e89b-12d3-a456-426614174000 failed"), "run <ID> failed")
        self.assertEqual(self.scrub("org org_" + "2FakeFakeFakeFake9" + " cfg"), "org <ID> cfg")
        self.assertEqual(self.scrub("rebased onto 4f3e2d1c"), "rebased onto <ID>")
        self.assertEqual(self.scrub("call_id=tooluse_" + "Fake9Fake9Fake9Fake9"), "call_id=<ID>")
        # all-letter hex-ish words stay
        self.assertEqual(self.scrub("defaced facade"), "defaced facade")


class DenylistTests(unittest.TestCase):
    def test_whole_word_case_insensitive(self):
        s = Scrubber(["Acme Widgets", "zorblax"])
        out, counts = s.scrub("ask ACME widgets and Zorblax, not zorblaxes")
        self.assertEqual(out, "ask <NAME> and <NAME>, not zorblaxes")
        self.assertEqual(counts["name"], 2)

    def test_denylisted_machine_prefix_becomes_host(self):
        s = Scrubber(["zorblax"])
        self.assertEqual(s.scrub("restart zorblax-3")[0], "restart <HOST>")

    def test_load_skips_comments(self):
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
            f.write("# comment\nfoo\n\n  Bar Baz  \n")
        try:
            self.assertEqual(load_denylist(f.name), ["foo", "Bar Baz"])
        finally:
            os.unlink(f.name)

    def test_voice_is_untouched(self):
        s = Scrubber(["zorblax"])
        text = "kk rq lmk ig, tbh this is p fkn sick yk?? ty"
        self.assertEqual(s.scrub(text), (text, {}))


class FilterTests(unittest.TestCase):
    def test_wrappers_and_drops(self):
        self.assertEqual(filters.clean_turn("<system-reminder>x</system-reminder>hey can u check")[0], "hey can u check")
        self.assertEqual(filters.clean_turn("[Request interrupted by user]")[1], "interrupted")
        self.assertEqual(filters.clean_turn("<command-name>/fake</command-name>")[1], "empty_after_strip")
        self.assertEqual(filters.clean_turn("/compact")[1], "slash_command")

    def test_fences_stripped_when_prose_remains(self):
        text, reason = filters.clean_turn("why does this fail\n\n```\nx = 1;\n```\n\nany idea?")
        self.assertIsNone(reason)
        self.assertEqual(text, "why does this fail\n\nany idea?")

    def test_pasted_output_paragraph_removed(self):
        text, _ = filters.clean_turn("Fake agent report line one and\n  its wrapped continuation\n\nok ship it")
        self.assertEqual(text, "ok ship it")

    def test_mostly_logs_dropped(self):
        logs = "\n".join("2026-01-01T00:00:0%d INFO fake.module started worker %d" % (i, i) for i in range(6))
        self.assertEqual(filters.clean_turn(logs)[1], "code_or_logs")


if __name__ == "__main__":
    unittest.main()
