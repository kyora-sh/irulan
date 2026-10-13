"""Turn a Claude Code user turn into the text the human actually typed, or a
drop reason.

Nothing here rewrites the writer's words: it only removes harness wrappers,
pasted blocks and fenced code, and decides whether what is left is prose.
"""

import re
import unicodedata

# Harness wrapper tags. Their whole block is removed, content included.
WRAPPER_TAGS = (
    "system-reminder", "pasted_content", "command-name", "command-message",
    "command-args", "command-contents", "local-command-stdout", "local-command-stderr",
    "local-command-caveat", "task-notification", "teammate-message", "agent-message",
    "bash-input", "bash-stdout", "bash-stderr", "user-prompt-submit-hook",
    "user-memory-input", "channel-message", "tool-use-error", "summary",
    "new-diagnostics", "ide_selection", "ide_opened_file", "persisted-output",
)

_WRAPPER_BLOCK = re.compile(
    r"<(%s)\b[^>]*>[\s\S]*?</\1\b[^>]*>" % "|".join(re.escape(t) for t in WRAPPER_TAGS)
)
# Any other <kebab-or_snake-tag ...>...</kebab-or_snake-tag> block: harness
# style, never something a person types in prose.
_GENERIC_BLOCK = re.compile(r"<([a-z]+[-_][a-z0-9_-]+)\b[^>]*>[\s\S]*?</\1\b[^>]*>")
# An opening wrapper tag with no close (truncated paste): drop to the end.
_UNCLOSED = re.compile(r"<(%s)\b[^>]*>[\s\S]*\Z" % "|".join(re.escape(t) for t in WRAPPER_TAGS))
_IMAGE_REF = re.compile(r"\[Image(?: #\d+)?\]")
_FENCE = re.compile(r"(^|\n)[ \t]*(```|~~~)[^\n]*\n[\s\S]*?(?:\n[ \t]*\2[ \t]*(?=\n|$)|\Z)")

DROP_PREFIXES = (
    ("[Request interrupted", "interrupted"),
    ("This session is being continued", "continuation"),
    ("Caveat:", "caveat"),
    ("Base directory for this skill", "skill_body"),
    ("Stop hook feedback", "hook"),
    ("Goal check-in", "hook"),
    ("A session-scoped Stop hook", "hook"),
    ("Another Claude session sent a message", "peer_message"),
    ("[Image: ", "image_meta"),
    ("Continue from where you left off", "continuation"),
)

_SLASH_ONLY = re.compile(r"^/[\w:.-]+(?:\s+[\w:.-]+)?\s*$")

_CODE_LINE = re.compile(
    r"""^\s*(?:
        [{}\[\]()<>;,]+\s*$                       # brackets alone
      | (?:def|class|import|from|return|const|let|var|function|fn|pub|impl|use|
           export|async|await|if|elif|else|for|while|try|except|catch|finally|
           package|struct|enum|interface|type|SELECT|INSERT|UPDATE|DELETE|CREATE)\b.*[:{(=;]\s*$
      | .*[;{]\s*$                                 # statement / block ends
      | .*(?:=>|::|\+=|-=|==|!=|&&|\|\|).*[;{})]\s*$
      | [$#>]\s+\S+                                # shell prompt
      | (?:Traceback|\s+File\s+".*",\s+line\s+\d+|\s+at\s+\S+\s*\(|Caused\ by:)
      | .*\b(?:ERROR|WARN|WARNING|INFO|DEBUG|TRACE|FATAL)\b[\]:|]
      | \[?\d{4}-\d\d-\d\d[T\s]\d\d:\d\d
      | \d\d:\d\d:\d\d(?:\.\d+)?\s
      | "[^"]+"\s*:\s*.+,?\s*$                     # JSON field
      | [-+]{3}\s+[ab]/ | @@\s+-\d                  # diff headers
      | [+-](?![-+\s>])\S.*                        # diff lines
      | \s{4,}\S.*[^.!?]$                          # indented, not a sentence
    )""",
    re.VERBOSE,
)

_LONG_TOKEN = re.compile(r"\S{60,}")
_URLS = re.compile(r"\b[a-z]+://\S+")
_KV = re.compile(r"\b[A-Za-z_][\w.-]*=[^\s=]+")
_SRC_LOC = re.compile(r"\b[\w.-]+\.(?:ts|tsx|js|jsx|mjs|py|rs|go|java|kt|rb|c|cc|cpp|h|swift|css|html|vue):\d+(?::\d+)?\s*$")
_JSON_FIELD = re.compile(r'"[\w.-]+"\s*:')
_HTML_TAG = re.compile(r"</?[a-zA-Z][\w-]*(?:\s[^<>]*)?>")
# Pasted terminal or agent output: TUI glyphs at line start, or the two-space
# continuation indent that Claude Code puts on wrapped lines.
_TUI_LINE = re.compile(r"^\s*[\u23fa\u23bf\u25cf\u273b\u2502\u256d\u2570\u2514\u251c\u250c\u2500\u2714\u2717\u2610\u2612\u25b2\u276f\u2733\u00b7]")
_WRAP_INDENT = re.compile(r"^ {2,}\S")
# Chat and code-host UI pastes carry other people's words, so the whole message goes.
_CHAT_PASTE = re.compile(
    r"\[\d{1,2}:\d\d(?:\s?[AP]M)?\]"
    r"|^.{1,60}\s+\d{1,2}:\d\d\s?[AP]M\s*$"
    r"|opened \d+ (?:seconds?|minutes?|hours?|days?|weeks?|months?) ago by"
    r"|\bReview required\b|\u00b7 Last: |\bcommented (?:\d+ \w+ ago|on [A-Z][a-z]{2} \d)"
    r"|^\s*(?:Reply|Resolve conversation|Add a comment|View reply|Hide resolved|Load more|Pending)\s*$"
    r"|Copy file name to clipboard|Expand all lines|\bLines changed: \d|Load diff|Viewed\s*$"
    r"|\b(?:approved these changes|requested changes|commented now|left a comment|reviewed \d+ (?:minutes?|hours?|days?) ago)\b"
    r"|^\s*\d+[smhd] ago\s*$|^\s*\d+ (?:minutes?|hours?|days?) ago\s*$"
    r"|^\S+@\S+ \S+ [%$#](?: |$)"          # shell prompt user@host dir %
    r'|(?:\\"[\w-]+\\":.*){2}',            # escaped JSON

    re.M,
)
_URL_ONLY = re.compile(r"^(?:[a-z]+://\S+\s*)+$")


def normalise(text):
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [ln.rstrip() for ln in text.split("\n")]
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def dedupe_key(text):
    return re.sub(r"\s+", " ", text.lower()).strip()


def short_key(text):
    return re.sub(r"[^\w]+", "", text.lower())


def strip_wrappers(text):
    prev = None
    while prev != text:
        prev = text
        text = _WRAPPER_BLOCK.sub("", text)
        text = _GENERIC_BLOCK.sub("", text)
    text = _UNCLOSED.sub("", text)
    return text


def strip_fences(text):
    return _FENCE.sub(lambda m: m.group(1), text)


def is_code_line(line):
    s = line.strip()
    if not s:
        return False
    if _URL_ONLY.match(s):
        return False
    if _LONG_TOKEN.search(_URLS.sub("", s)):
        return True
    if _CODE_LINE.match(line):
        return True
    if len(_JSON_FIELD.findall(s)) >= 3 or len(_HTML_TAG.findall(s)) >= 2:
        return True
    if len(_KV.findall(s)) >= 2 or _SRC_LOC.search(s):
        return True
    letters = sum(c.isalpha() for c in s)
    if len(s) >= 8 and letters / len(s) < 0.45:
        return True
    return False


def prose_words(text):
    """Words that are actually language, not URLs, paths or symbols."""
    n = 0
    for tok in text.split():
        if "://" in tok or tok.startswith("/") or tok.startswith("~/"):
            continue
        if re.search(r"[A-Za-z]{2,}|^[A-Za-z]$", tok) and sum(c.isalpha() for c in tok) >= len(tok) / 2:
            n += 1
    return n


def remove_code_runs(text, min_run=3):
    """Drop runs of >= min_run consecutive code/log lines (unfenced pastes)."""
    lines = text.split("\n")
    flags = [is_code_line(ln) for ln in lines]
    out, i = [], 0
    while i < len(lines):
        if flags[i]:
            j = i
            while j < len(lines) and (flags[j] or not lines[j].strip()):
                j += 1
            # trim trailing blanks from the run
            k = j
            while k > i and not lines[k - 1].strip():
                k -= 1
            run = sum(flags[i:k])
            longest = max(len(lines[x]) for x in range(i, k) if flags[x])
            if run >= min_run or longest >= 200:
                out.append("")
                i = k
                continue
            out.extend(lines[i:k])
            i = k
        else:
            out.append(lines[i])
            i += 1
    return "\n".join(out)


def looks_repetitive(text):
    """Logs repeat themselves; people mostly don't."""
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    if len(lines) < 10:
        return False
    prefixes = {re.sub(r"\d+", "0", ln[:30]) for ln in lines}
    return len(prefixes) / len(lines) < 0.5


def code_share(text):
    """Share of non-empty lines, or of characters, that are code/log output,
    whichever is higher (one 5,000-char JSON line outweighs two prose lines)."""
    lines = [ln for ln in text.split("\n") if ln.strip()]
    if not lines:
        return 0.0
    flags = [is_code_line(ln) for ln in lines]
    by_lines = sum(flags) / len(lines)
    by_chars = sum(len(ln) for ln, f in zip(lines, flags) if f) / sum(len(ln) for ln in lines)
    return max(by_lines, by_chars)


def looks_pasted_ui(text):
    """Many short lines with no sentence punctuation: a copied web page or UI."""
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    if len(lines) < 15:
        return False
    short = sum(1 for ln in lines if len(ln.split()) <= 3)
    return short / len(lines) > 0.6


def is_pasted_paragraph(par):
    for ln in par.split("\n"):
        if _TUI_LINE.match(ln) or _WRAP_INDENT.match(ln):
            return True
    return False


def remove_pasted_paragraphs(text):
    """Drop blank-line-separated paragraphs that look like pasted terminal or
    agent output, plus a lead-in paragraph ending in ':' right before one (the
    pasted text's own intro, as often as not). Returns (text, n_removed)."""
    pars = re.split(r"\n\s*\n", text)
    flags = [is_pasted_paragraph(p) for p in pars]
    for i in range(len(pars) - 1):
        if flags[i + 1] and not flags[i] and pars[i].rstrip().endswith(":"):
            flags[i] = True
    kept = [p for p, f in zip(pars, flags) if not f]
    return "\n\n".join(kept), sum(flags)


def clean_turn(raw, notes=None):
    """Return (text, None) for a keeper or (None, reason) for a drop."""
    text = raw.replace("\u200e", "").replace("\u200b", "")
    stripped_lead = text.lstrip()
    for prefix, reason in DROP_PREFIXES:
        if stripped_lead.startswith(prefix):
            return None, reason
    had_wrapper = "<" in text
    text = strip_wrappers(text) if had_wrapper else text
    text = _IMAGE_REF.sub("", text)
    text = normalise(text)
    if not text:
        return None, "empty_after_strip"
    for prefix, reason in DROP_PREFIXES:
        if text.startswith(prefix):
            return None, reason
    if re.match(r"<[A-Za-z!/]", text):
        return None, "starts_with_tag"
    if _SLASH_ONLY.match(text):
        return None, "slash_command"

    if "```" in text or "~~~" in text:
        before = text
        text = normalise(strip_fences(text))
        if notes is not None:
            notes["fenced_blocks_stripped"] += 1
        if prose_words(text) < 3:
            return None, "code_or_logs"
        if prose_words(text) < 0.15 * prose_words(before) and prose_words(text) < 8:
            return None, "code_or_logs"

    if _CHAT_PASTE.search(text):
        return None, "pasted_chat_or_ui"
    mostly_code = code_share(text) > 0.5 or looks_repetitive(text)
    before = prose_words(text)
    text = normalise(remove_code_runs(text))
    text, n_pasted = remove_pasted_paragraphs(text)
    text = normalise(text)
    if mostly_code:
        # A message that is mostly a paste survives only if a real note of
        # the writer's own is left once the paste is gone.
        if not text or prose_words(text) < 5 or code_share(text) > 0.2 or looks_repetitive(text):
            return None, "code_or_logs"
        if notes is not None:
            notes["code_pastes_stripped"] += 1
    elif not text:
        return None, "code_or_logs"
    if n_pasted:
        if prose_words(text) < 3:
            return None, "pasted_output"
        if notes is not None:
            notes["pasted_paragraphs_removed"] += n_pasted
    if code_share(text) > 0.5:
        return None, "code_or_logs"
    if notes is not None and prose_words(text) < before:
        notes["messages_partly_stripped"] += 1
    if looks_pasted_ui(text):
        return None, "pasted_ui"
    if prose_words(text) == 0:
        return None, "no_prose"
    if _SLASH_ONLY.match(text):
        return None, "slash_command"
    return text, None
