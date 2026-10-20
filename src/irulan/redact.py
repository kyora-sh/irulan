"""A small local pass that replaces obvious credentials with <SECRET>.

    text, counts = redact(text)

It runs before upload when you pass `irulan upload --redact-secrets-locally`.
It only looks for things that are credentials by their shape: private key
blocks, credentials in connection strings, JWTs, well-known API token
prefixes, bearer/basic auth headers, `password=...` / `api_key: "..."`
assignments and "the password is ..." prose. It does NOT touch names, hosts,
URLs, paths or IDs; the platform scrubs those after upload (docs/API.md).
"""

import collections
import re

SECRET = "<SECRET>"

_PRIVATE_KEY = re.compile(
    r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY[A-Z ]*-----[\s\S]*?(?:-----END [A-Z0-9 ]*PRIVATE KEY[A-Z ]*-----|\Z)")
# scheme://user:password@host
_CONN_STRING = re.compile(r"\b([a-zA-Z][a-zA-Z0-9+.-]{1,20}://[^\s/:@'\"<>]+:)([^\s/@'\"<>]+)(@)")
# storage connection strings and signed URLs: AccountKey=..., SharedAccessKey=..., sig=...
_KEY_PARAM = re.compile(r"(?i)\b(AccountKey|SharedAccessKey|SharedAccessSignature|sig)=([^\s;&'\"]+)")
_JWT = re.compile(r"(?<![A-Za-z0-9])eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}(?:\.[A-Za-z0-9_-]{4,})?")
_TOKEN_PREFIXES = re.compile(
    r"(?<![\w-])(?:"
    r"sk-ant-[A-Za-z0-9_-]{8,}"
    r"|sk-(?:proj-|live-|test-)?[A-Za-z0-9_-]{16,}"
    r"|(?:sk|pk|rk)_(?:live|test)_[A-Za-z0-9]{8,}"
    r"|gh[pousr]_[A-Za-z0-9]{20,}"
    r"|github_pat_[A-Za-z0-9_]{20,}"
    r"|xox[bpasr]-[A-Za-z0-9-]{8,}"
    r"|(?:AKIA|ASIA)[0-9A-Z]{16}"
    r"|AIza[0-9A-Za-z_-]{30,}"
    r"|glpat-[A-Za-z0-9_-]{16,}"
    r"|hf_[A-Za-z0-9]{20,}"
    r"|npm_[A-Za-z0-9]{30,}"
    r"|whsec_[A-Za-z0-9+/=]{16,}"
    r"|r8_[A-Za-z0-9]{20,}"
    r"|dop_v1_[a-f0-9]{32,}|shpat_[a-fA-F0-9]{32}|SG\.[A-Za-z0-9_-]{16,}\.[A-Za-z0-9_-]{16,}"
    r"|lin_(?:api|oauth)_[A-Za-z0-9]{20,}|ntn_[A-Za-z0-9]{20,}"
    r"|ya29\.[A-Za-z0-9._-]{20,}"
    r")"
)
_AUTH_HEADER = re.compile(r"(?i)\b(Bearer|Basic|Token)\s+([A-Za-z0-9._~+/=-]{12,})")
_ASSIGN = re.compile(
    r"(?i)\b((?:[A-Za-z0-9]+[_-])*(?:password|passwd|pwd|pass|secret|token|api[_-]?key|apikey|key|"
    r"access[_-]?key|private[_-]?key|client[_-]?secret|auth|credentials?))"
    r"(\s*=\s*|\s*:\s*)(['\"]?)([^\s'\"`,;)&]{4,})"
)
_SAID_PASSWORD = re.compile(r"(?i)\b(password|passwd|pwd|pw)(\s+(?:is|was|=)\s+)(['\"]?)(\S{6,})")

KINDS = ("private_key", "connection_string", "key_param", "jwt", "api_token", "auth_header",
         "assignment", "password_prose")


def _assign_repl(m):
    name, sep, quote, value = m.groups()
    if value.startswith("<") or value.lower() in {"true", "false", "null", "none", "undefined"}:
        return None
    if ":" in sep:
        # "the key: speed" is prose; only a machine-looking value counts.
        if not quote and not (len(value) >= 12 and re.search(r"\d", value) and re.search(r"[A-Za-z]", value)):
            return None
    elif not quote and value.isalpha() and value.islower() and len(value) < 12:
        return None  # auth=none, key=name: config talk
    return name + sep + quote + SECRET


def _said_password_repl(m):
    value = m.group(4).rstrip(".,;:!?)")
    if value.startswith("<") or (value.isalpha() and value.islower()):
        return None  # "the password is wrong"
    return m.group(1) + m.group(2) + m.group(3) + SECRET + m.group(4)[len(value):]


_STEPS = (
    ("private_key", _PRIVATE_KEY, lambda m: SECRET),
    ("connection_string", _CONN_STRING, lambda m: m.group(1) + SECRET + m.group(3)),
    ("key_param", _KEY_PARAM, lambda m: m.group(1) + "=" + SECRET),
    ("jwt", _JWT, lambda m: SECRET),
    ("api_token", _TOKEN_PREFIXES, lambda m: SECRET),
    ("auth_header", _AUTH_HEADER, lambda m: m.group(1) + " " + SECRET),
    ("assignment", _ASSIGN, _assign_repl),
    ("password_prose", _SAID_PASSWORD, _said_password_repl),
)


def redact(text):
    """Return (text, Counter of replacements by kind)."""
    counts = collections.Counter()
    for kind, pattern, repl in _STEPS:
        def fn(m, kind=kind, repl=repl):
            r = repl(m)
            if r is None or r == m.group(0):
                return m.group(0)
            counts[kind] += 1
            return r
        text = pattern.sub(fn, text)
    return text, counts
