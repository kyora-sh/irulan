"""Replace secrets, URLs, emails, IPs, hosts, paths, IDs and denylisted names
with typed placeholders.

Only these spans are touched. Everything else in a message (shorthand, casing,
punctuation, typos) is left exactly as written: the cleaned text is training
target text, so the writer's voice must survive.

Usage as a library:

    s = Scrubber.from_files("scrub/denylist.txt")
    text, counts = s.scrub(raw_text)
"""

import collections
import ipaddress
import os
import re

PLACEHOLDERS = ("<SECRET>", "<URL>", "<EMAIL>", "<IP>", "<HOST>", "<PATH>", "<ID>", "<NAME>")

# Domains that are public, well known, and say something about the message
# ("check the github.com issue"). A URL on one of these keeps its bare domain;
# any other URL becomes <URL>.
PUBLIC_DOMAINS = {
    "github.com", "gist.github.com", "gitlab.com", "bitbucket.org",
    "stackoverflow.com", "superuser.com", "serverfault.com",
    "npmjs.com", "pypi.org", "crates.io", "docs.rs", "pkg.go.dev", "go.dev",
    "developer.mozilla.org", "wikipedia.org", "en.wikipedia.org",
    "google.com", "youtube.com", "x.com", "twitter.com", "news.ycombinator.com",
    "reddit.com", "medium.com", "dev.to",
    "vercel.com", "netlify.com", "cloudflare.com", "aws.amazon.com",
    "azure.microsoft.com", "portal.azure.com", "learn.microsoft.com",
    "cloud.google.com", "console.cloud.google.com",
    "openai.com", "platform.openai.com", "anthropic.com", "console.anthropic.com",
    "claude.ai", "claude.com", "huggingface.co", "modal.com", "linear.app",
    "notion.so", "figma.com", "slack.com", "discord.com",
    "python.org", "rust-lang.org", "nodejs.org", "react.dev", "nextjs.org",
    "tailwindcss.com", "developer.apple.com", "apple.com", "arxiv.org",
}

# Basenames that are generic enough to keep when an absolute path is scrubbed.
GENERIC_BASENAMES = {
    "readme.md", "claude.md", "agents.md", "changelog.md", "license", "makefile",
    "dockerfile", "package.json", "package-lock.json", "tsconfig.json",
    "pyproject.toml", "setup.py", "requirements.txt", "cargo.toml", "cargo.lock",
    "go.mod", "go.sum", ".env", ".env.local", ".gitignore", "docker-compose.yml",
    "docker-compose.yaml", "settings.json", "settings.local.json", "index.ts",
    "index.js", "index.tsx", "index.html", "main.py", "main.rs", "main.go",
    "lib.rs", "mod.rs", "app.py", "__init__.py", "config.json", "config.yaml",
    "config.yml", "bun.lock", "bun.lockb", "yarn.lock", "pnpm-lock.yaml",
    "next.config.js", "next.config.ts", "vite.config.ts", "manifest.json",
    "memory.md", "skill.md",
}

# Filesystem roots: an absolute path under one of these is a file path, not an
# API route like /v1/chat/completions.
FS_ROOTS = (
    "Users", "home", "private", "tmp", "var", "etc", "opt", "usr", "root", "mnt",
    "Volumes", "srv", "app", "workspace", "workspaces", "data", "proc", "System",
    "Library", "Applications", "nix", "bin", "sbin", "dev", "run", "cache",
)

# Segments that mark a dash-and-digit token as a machine name (gpu-node-3, build-box-12).
INFRA_SEGMENTS = {
    "vm", "host", "node", "worker", "runner", "srv", "server", "box", "db", "gpu",
    "prod", "staging", "stg", "tmp", "eu", "us", "ap", "asia", "west", "east",
    "north", "south", "central", "k8s", "pod", "svc", "edge", "lb", "proxy",
    "bastion", "ci", "build", "agent", "sandbox", "instance", "ec2", "droplet",
}

# First segments of dash-digit tokens that are never machine names.
NOT_HOST_PREFIXES = {
    "gpt", "claude", "opus", "sonnet", "haiku", "qwen", "llama", "gemini", "gemma",
    "mistral", "mixtral", "phi", "deepseek", "o1", "o3", "o4", "utf", "sha", "md",
    "iso", "rfc", "pep", "es", "ecma", "x86", "arm", "h", "step", "phase", "part",
    "option", "top", "v", "pr", "issue", "ticket", "item", "page", "level", "tier",
    "round", "batch", "chunk", "test", "case", "run", "day", "week", "wave",
    "track", "lane", "slot", "rank", "seed", "epoch", "layer", "fig", "figure",
    "table", "section", "chapter", "base", "int", "float", "uint", "bf", "fp",
    "covid", "web", "next", "vue", "react", "python", "node18", "py", "ipv",
    "rev", "version", "release", "sprint", "milestone", "q", "w", "s", "p", "t",
    "k", "n", "m", "a", "b", "c", "d", "e", "f", "g", "i", "j", "l", "r", "u",
    "x", "y", "z", "b200", "a100", "h100", "h200", "l4", "t4", "a10g",
}


def _w(pattern):
    return re.compile(pattern)


# Order matters: the most specific patterns run first so a JWT inside a URL is
# counted once, as a URL, and a key inside an assignment is counted as a secret.
_PRIVATE_KEY = _w(r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY[A-Z ]*-----[\s\S]*?(?:-----END [A-Z0-9 ]*PRIVATE KEY[A-Z ]*-----|\Z)")
_CONN_STRING = _w(r"\b[a-zA-Z][a-zA-Z0-9+.-]{1,20}://[^\s/:@'\"<>]+:[^\s/@'\"<>]+@[^\s'\"<>]+")
_AZURE_CONN = _w(r"(?i)\b(?:AccountKey|SharedAccessKey|SharedAccessSignature|sig)=[^\s;&'\"]+")
_JWT = _w(r"(?<![A-Za-z0-9])eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}(?:\.[A-Za-z0-9_-]{4,})?")
_TOKEN_PREFIXES = _w(
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
    r"|ak-[A-Za-z0-9]{16,}|as-[A-Za-z0-9]{16,}"
    r"|r8_[A-Za-z0-9]{20,}"
    r"|sntry[su]_[A-Za-z0-9+/=_-]{20,}"
    r"|dop_v1_[a-f0-9]{32,}|shpat_[a-fA-F0-9]{32}|SG\.[A-Za-z0-9_-]{16,}\.[A-Za-z0-9_-]{16,}"
    r"|lin_(?:api|oauth)_[A-Za-z0-9]{20,}|ntn_[A-Za-z0-9]{20,}|secret_[A-Za-z0-9]{30,}"
    r"|(?:ya29|1//)[A-Za-z0-9._-]{20,}"
    r")"
)
_BEARER = _w(r"(?i)\b(Bearer|Basic|Token)\s+([A-Za-z0-9._~+/=-]{12,})")
_ASSIGN_EQ = _w(
    r"(?i)\b((?:[A-Za-z0-9]+[_-])*(?:password|passwd|pwd|pass|secret|token|api[_-]?key|apikey|key|"
    r"access[_-]?key|private[_-]?key|client[_-]?secret|auth|credentials?))"
    r"(\s*=\s*|\s*:\s*)(['\"]?)([^\s'\"`,;)&]{4,})"
)
# "root pwd is <value>" / "the password was <value>": prose that hands over a credential.
_SAID_PASSWORD = _w(r"(?i)\b(password|passwd|passwort|pwd|pw)(\s+(?:is|was|=)\s+)(['\"]?)(\S{6,})")
_URL = _w(r"\b(?:https?|wss?|ftp|ssh|file)://[^\s<>\"'`)\]}]+")
_BARE_URL = _w(
    r"(?<![\w@./-])((?:[a-z0-9-]+\.)+(?:com|ai|io|dev|net|org|app|co|run|cloud|so|xyz|tech|gg|me|us|eu|de|ch|uk|sh))"
    r"(/[^\s<>\"'`)\]}]*)"
)
_EMAIL = _w(r"(?i)\b[a-z0-9._%+-]+@[a-z0-9-]+(?:\.[a-z0-9-]+)*\.[a-z]{2,}\b")
_IPV4 = _w(r"(?<![\w.])(\d{1,3}(?:\.\d{1,3}){3})(?:/\d{1,2})?(?::\d{2,5})?(?![\w.]*\d)")
_IPV6 = _w(r"(?<![\w:])([0-9A-Fa-f]{0,4}(?::[0-9A-Fa-f]{0,4}){2,7})(?:/\d{1,3})?(?![\w:])")
_INTERNAL_HOST = _w(
    r"(?i)(?<![\w.-])(?:[a-z0-9-]+\.)+(?:internal|local|lan|corp|intra|modal\.run|ts\.net|"
    r"svc\.cluster\.local|azurewebsites\.net|azurecr\.io|blob\.core\.windows\.net|"
    r"servicebus\.windows\.net|database\.windows\.net|cloudapp\.azure\.com|"
    r"amazonaws\.com|herokuapp\.com|fly\.dev|onrender\.com|ngrok(?:-free)?\.app|ngrok\.io|"
    r"vercel\.app|netlify\.app|pages\.dev|workers\.dev|up\.railway\.app)(?![\w-]|\.\w)"
)
_MAC_HOST = _w(r"(?i)\b(?:[\w]+-[\w-]*(?:MacBook|iMac|Mac-mini|Mac-Studio)[\w-]*|(?:MacBook|iMac)-[\w-]+)\b")
_BARE_DOMAIN = _w(
    r"(?i)(?<![\w@./-])((?:[a-z0-9-]+\.)+(?:com|ai|io|dev|net|org|app|co|run|cloud|so|xyz|tech|gg))(?![\w-]|\.\w)"
)
_DASH_DIGIT = _w(r"(?<![\w./-])([a-z][a-z0-9]*(?:-[a-z0-9]+)*-\d{1,3}[a-z]?)(?![\w-]|\.\w)")
_HOME_PATH = _w(r"(?<![\w])~/[^\s'\"`<>)\]}:,;]*")
_ABS_PATH = _w(r"(?<![\w.:/~<>-])(/(?:%s)(?:/[^\s'\"`<>)\]}:,;]*)?)" % "|".join(FS_ROOTS))
_ABS_FILE = _w(r"(?<![\w.:/~<>-])(/[\w.@+-]+(?:/[\w.@+-]+)+\.[A-Za-z0-9]{1,6})(?![\w/])")
_WIN_PATH = _w(r"\b[A-Za-z]:\\[^\s'\"`<>]+")
_UUID = _w(r"(?i)(?<![0-9a-f])[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}(?![0-9a-f])")
_PREFIXED_ID = _w(
    r"\b(?:org|user|sess|inv|ins|client|sms|email|idn|pak|sk|pk|toolu|msg|req|call|run|"
    r"chatcmpl|resp|evt|cus|sub|pi|ch|price|prod|acct|txn|ws|wf|job|ap|fc|file|asst|thread|batch)"
    r"_[A-Za-z0-9]{12,}\b"
)
# Any other lowercase_prefix + opaque suffix (tooluse_..., ses_...), when the
# suffix mixes letters and digits like a generated id does.
_GENERIC_ID = _w(r"\b[a-z]{2,12}_(?=[A-Za-z0-9]*\d)(?=[A-Za-z0-9]*[A-Za-z])[A-Za-z0-9]{14,}\b")
_LONG_HEX = _w(r"(?<![\w-])[0-9a-fA-F]{25,}(?![\w-])")
_SHA = _w(r"(?<![\w-])(?=[0-9a-f]*[0-9])(?=[0-9a-f]*[a-f])[0-9a-f]{7,40}(?![\w-])")
_LONG_B64 = _w(r"(?<![\w+/=-])[A-Za-z0-9+/_=-]{25,}(?![\w+/=-])")


class Scrubber:
    def __init__(self, denylist=()):
        terms = sorted({t.strip() for t in denylist if t.strip()}, key=len, reverse=True)
        self.denylist = terms
        if terms:
            alt = "|".join(re.escape(t) for t in terms)
            self._deny = re.compile(r"(?i)(?<![\w])(?:%s)(?![\w])" % alt)
        else:
            self._deny = None

    @classmethod
    def from_files(cls, *paths):
        terms = []
        for p in paths:
            if p and os.path.exists(p):
                terms.extend(load_denylist(p))
        return cls(terms)

    # Each step is (category, function(text, counter) -> text).
    def scrub(self, text):
        counts = collections.Counter()
        t = text

        def sub(category, pattern, repl, s):
            def fn(m):
                r = repl(m) if callable(repl) else repl
                if r is None:
                    return m.group(0)
                if r != m.group(0):
                    counts[category] += 1
                return r
            return pattern.sub(fn, s)

        t = sub("secret", _PRIVATE_KEY, "<SECRET>", t)
        t = sub("secret", _CONN_STRING, "<SECRET>", t)
        t = sub("secret", _AZURE_CONN, lambda m: m.group(0).split("=", 1)[0] + "=<SECRET>", t)
        t = sub("secret", _JWT, "<SECRET>", t)
        t = sub("secret", _TOKEN_PREFIXES, "<SECRET>", t)
        t = sub("secret", _BEARER, lambda m: m.group(1) + " <SECRET>", t)
        t = sub("url", _URL, _url_repl, t)
        t = sub("url", _BARE_URL, _bare_url_repl, t)
        t = sub("email", _EMAIL, "<EMAIL>", t)
        t = sub("secret", _ASSIGN_EQ, _assign_repl, t)
        t = sub("secret", _SAID_PASSWORD, _said_password_repl, t)
        t = sub("path", _HOME_PATH, _path_repl, t)
        t = sub("path", _ABS_PATH, _path_repl, t)
        t = sub("path", _ABS_FILE, _path_repl, t)
        t = sub("path", _WIN_PATH, _path_repl, t)
        t = sub("ip", _IPV4, _ipv4_repl, t)
        t = sub("ip", _IPV6, _ipv6_repl, t)
        t = sub("host", _INTERNAL_HOST, "<HOST>", t)
        t = sub("host", _MAC_HOST, "<HOST>", t)
        t = sub("host", _BARE_DOMAIN, _bare_domain_repl, t)
        t = sub("host", _DASH_DIGIT, self._dash_digit_repl, t)
        t = sub("id", _UUID, "<ID>", t)
        t = sub("id", _PREFIXED_ID, "<ID>", t)
        t = sub("id", _GENERIC_ID, _generic_id_repl, t)
        t = sub("id", _SHA, _sha_repl, t)
        t = sub("secret", _LONG_HEX, "<SECRET>", t)
        t = sub("secret", _LONG_B64, _b64_repl, t)
        if self._deny is not None:
            t = sub("name", self._deny, "<NAME>", t)
        return t, counts

    def _dash_digit_repl(self, m):
        tok = m.group(1)
        segs = tok.lower().split("-")
        if segs[0] in NOT_HOST_PREFIXES:
            return None
        if any(s in INFRA_SEGMENTS for s in segs[:-1]):
            return "<HOST>"
        if self._deny is not None and self._deny.search(" ".join(segs[:-1])):
            return "<HOST>"
        return None


def load_denylist(path):
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                out.append(line)
    return out


def _domain_of(url):
    rest = url.split("://", 1)[1] if "://" in url else url
    host = re.split(r"[/?#]", rest, 1)[0]
    host = host.rsplit("@", 1)[-1].split(":", 1)[0].lower().rstrip(".")
    if host.startswith("www."):
        host = host[4:]
    return host


def _is_public(host):
    if host in PUBLIC_DOMAINS:
        return True
    if host.startswith("docs.") or host.endswith(".readthedocs.io"):
        return True
    return False


def _strip_trailing_punct(s):
    trail = ""
    while s and s[-1] in ".,;:!?":
        trail = s[-1] + trail
        s = s[:-1]
    return s, trail


def _url_repl(m):
    url, trail = _strip_trailing_punct(m.group(0))
    host = _domain_of(url)
    return (host if _is_public(host) else "<URL>") + trail


def _bare_url_repl(m):
    host = m.group(1).lower()
    if host.startswith("www."):
        host = host[4:]
    _, trail = _strip_trailing_punct(m.group(2))
    return (host if _is_public(host) else "<URL>") + trail


def _bare_domain_repl(m):
    host = m.group(1).lower()
    if host.startswith("www."):
        host = host[4:]
    if _is_public(host):
        return None
    return "<HOST>"


def _assign_repl(m):
    name, sep, quote, value = m.group(1), m.group(2), m.group(3), m.group(4)
    if value.startswith("<") or value.lower() in {"true", "false", "null", "none", "undefined"}:
        return None
    if ":" in sep:
        # "key: value" is common in prose ("the key: speed"); only treat it as a
        # secret when the value looks machine-generated.
        if not quote and not (len(value) >= 12 and re.search(r"\d", value) and re.search(r"[A-Za-z]", value)):
            return None
    elif not quote and value.isalpha() and value.islower() and len(value) < 12:
        # key=value with a plain word ("auth=none", "key=name") is config talk.
        return None
    return name + sep + quote + "<SECRET>"


def _said_password_repl(m):
    value = m.group(4).rstrip(".,;:!?)")
    # "the password is wrong" / "pwd is <SECRET>" are not credentials
    if value.startswith("<") or (value.isalpha() and value.islower()):
        return None
    return m.group(1) + m.group(2) + m.group(3) + "<SECRET>" + m.group(4)[len(value):]


def _path_repl(m):
    path, trail = _strip_trailing_punct(m.group(0))
    base = re.split(r"[/\\]", path.rstrip("/\\"))[-1]
    if base.lower() in GENERIC_BASENAMES:
        return base + trail
    return "<PATH>" + trail


def _ipv4_repl(m):
    try:
        ipaddress.IPv4Address(m.group(1))
    except ValueError:
        return None
    return "<IP>"


def _ipv6_repl(m):
    cand = m.group(1)
    groups = [g for g in cand.split(":") if g]
    if "::" not in cand and len(cand.split(":")) < 8:
        return None
    if len(groups) < 2 and cand not in ("::1",):
        return None
    try:
        ipaddress.IPv6Address(cand)
    except ValueError:
        return None
    return "<IP>"


def _generic_id_repl(m):
    suffix = m.group(0).split("_", 1)[1]
    # a version-ish or numeric suffix (utf_8, item_20240101) is not an id
    if suffix.isdigit() or (suffix.islower() and sum(c.isdigit() for c in suffix) < 3):
        return None
    return "<ID>"


def _sha_repl(m):
    tok = m.group(0)
    if len(tok) not in (7, 8, 9, 10, 11, 12, 40):
        return None
    return "<ID>"


def _b64_repl(m):
    tok = m.group(0)
    if tok.startswith("<"):
        return None
    if not (re.search(r"[A-Z]", tok) and re.search(r"[a-z]", tok) and re.search(r"\d", tok)):
        return None
    # snake_case / kebab-case identifiers and long CamelCase names are words,
    # not encoded blobs: base64 has few separators and many case switches.
    parts = re.split(r"[_-]", tok)
    if len(parts) >= 3 and all(len(p) <= 14 for p in parts):
        return None
    switches = sum(1 for a, b in zip(tok, tok[1:]) if a.islower() != b.islower() and a.isalpha() and b.isalpha())
    if switches < len(tok) / 6:
        return None
    return "<SECRET>"


def placeholder_count(text):
    return sum(text.count(p) for p in PLACEHOLDERS)
