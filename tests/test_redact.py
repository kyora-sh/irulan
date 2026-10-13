"""Local secret redaction. Every credential below is invented and fake."""

import unittest

from irulan.redact import redact


def r(text):
    return redact(text)[0]


class RedactTests(unittest.TestCase):
    def test_token_prefixes(self):
        for tok in ("sk-ant-api03-FAKEfakeFAKEfake1234", "ghp_FAKEfakeFAKEfakeFAKE1234",
                    "github_pat_FAKE1234fake5678FAKE90", "xoxb-1234-fake-token-here", "AKIAFAKEFAKEFAKE1234",
                    "hf_FAKEfakeFAKEfakeFAKEfa", "sk_live_FAKEfake1234"):
            self.assertEqual(r("use %s for it" % tok), "use <SECRET> for it", tok)

    def test_private_key_block(self):
        text = "here:\n-----BEGIN OPENSSH PRIVATE KEY-----\nAAAAfakeAAAA\n-----END OPENSSH PRIVATE KEY-----\nthx"
        self.assertEqual(r(text), "here:\n<SECRET>\nthx")

    def test_jwt_and_auth_header(self):
        jwt = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJmYWtlIn0.ZmFrZXNpZ25hdHVyZQ"
        self.assertEqual(r("token is %s ok" % jwt), "token is <SECRET> ok")
        self.assertEqual(r("curl -H 'Authorization: Bearer abcdefFAKE123456'"),
                         "curl -H 'Authorization: Bearer <SECRET>'")

    def test_assignments_and_prose(self):
        self.assertEqual(r("set password=hunter2fake and retry"), "set password=<SECRET> and retry")
        self.assertEqual(r('API_KEY: "abc123fakefake"'), 'API_KEY: "<SECRET>"')
        self.assertEqual(r("the root pwd is Fake!Pass99."), "the root pwd is <SECRET>.")
        self.assertEqual(r("postgres://app:s3cretFake@db.example.com/x"), "postgres://app:<SECRET>@db.example.com/x")
        self.assertEqual(r("conn AccountKey=FAKEfake123==;x"), "conn AccountKey=<SECRET>;x")

    def test_leaves_prose_names_and_hosts_alone(self):
        for text in ("the password is wrong", "auth=none for local", "the key: speed matters",
                     "ask Jane Fakename about gpu-node-3 on https://app.example.com/x",
                     "rename token_count to tokens", "api key rotation is next week"):
            self.assertEqual(r(text), text)

    def test_counts(self):
        _, counts = redact("ghp_FAKEfakeFAKEfakeFAKE1234 and password=Fake1234!")
        self.assertEqual(dict(counts), {"api_token": 1, "assignment": 1})


if __name__ == "__main__":
    unittest.main()
