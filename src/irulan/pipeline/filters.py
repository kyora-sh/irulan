"""Turn a Claude Code user turn into the text the human actually typed, or a
drop reason.

Nothing here rewrites the writer's words: it only removes harness wrappers,
pasted blocks and fenced code, and decides whether what is left is prose.
"""

import collections
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


# ---------------------------------------------------------------------------
# Authorship: is a message the writer's own typing, or something pasted in?
#
# The writer types short, mostly lowercase lines with little punctuation and a
# lot of shorthand. Pasted text (meeting transcripts, profiles and CVs, LLM
# answers and status reports, prompt templates, emails, docs) reads
# differently: capitalised sentences ending in periods, markdown structure,
# assistant phrasing, platform UI words, or lines that recur verbatim across
# many sessions. Each blank-line-separated paragraph gets a score; pasted
# paragraphs are stripped, and the message is kept only if enough of the
# writer's own prose is left. Strong whole-message markers (a transcript, a
# profile, an email) drop the message outright.

AUTHOR_DEFAULTS = {
    "enabled": True,
    "par_threshold": 3,         # paragraph score at or above which it counts as pasted
    "min_keep_words": 5,        # prose words that must remain after stripping a paste
    "max_words": 600,           # longer messages are length outliers and dropped
    "transcript_min_words": 30,
    "transcript_fillers_per_100": 1.5,
    "transcript_min_fillers": 3,
    "transcript_label_lines": 2,  # "Speaker 1:" or timestamp lines
    "profile_min_hits": 3,      # LinkedIn / CV markers
    "email_header_lines": 2,
    "residue_period_lines": 2,  # left-over lines ending in "." after stripping
    "template_min_words": 8,    # a line this long that recurs...
    "template_min_sessions": 4,  # ...in this many distinct sessions is a template
    "language": True,           # drop mainly non-English messages
    "lang_min_hits": 2,         # German markers needed before a message can be non-English
    "lang_max_share": 0.5,      # German share of (German + English) markers that drops it
}

SHORTHAND = frozenset("""
rq pls plz yk imo imho tho u ur kk k lfg preesh smh abt ig rn tbh bc obv wdym lmk ty tysm thx
sb yday tmrw btw idk nvm mb fkn fking fucking fuck fucked shit gonna wanna gotta ye yeah yep yo alr
ok okay hm hmm mh sick chill dope g2g g2 sth smth ofc rly v lol haha sesh prio iirc afaik w/ w/o
nah ngl fr ya yup lemme dunno cuz p bit ish aight bro vibe vibes rip oof yikes legit
""".split())

_ABBREV = re.compile(r"\b(?:e\.g|i\.e|etc|vs|approx|incl|cf|resp|esp|min|max|no|nr|ca|mr|ms|dr|st)\.", re.I)
_PERIOD_END = re.compile(r"[A-Za-z0-9)\]\"'`%*]\.(?=\s|$)")
# CI, build and package-manager output, one line at a time.
_LOG_LINE = re.compile(
    r"^\s*(?:#\d+\s+(?:\[|\d+\.\d+\s|DONE|CACHED|ERROR|transferring|load|importing|exporting|resolve|sha256|naming|writing)"
    r"|Run\s+[a-z][\w./@-]*(?:\s|$)|\[(?:gw\d+|\d+/\d+|internal|builder[\w /]*|stage[\w /-]*)\]\s"
    r"|(?:Downloading|Downloaded|Installed|Installing|Collecting|Compiling|Resolved|Prepared|Uninstalled|"
    r"Fetching|Pulling|Pushing|Step \d+/\d+)\b)"
    r"|\b(?:DONE|CACHED|FAILED|PASSED)\b(?:\s+\d+(?:\.\d+)?s)?\s*$"
    r"|[\w\]-]\.\w{1,5}:\d+:\d+\b"                 # file.ext:line:col
    r"|[\u2501\u2500\u2550=_]{8,}"                    # rule lines
    r"|^E\s{2,}\S|^\s*>\s+[\w@/.-]+@\d+\.\d+|^\[\s?\d+%\]|\b0x[0-9a-f]{8,}\b"
    r"|^\s*(?:self|[a-z_]+)\s=\s<|(?:\b\w+ = [^,\s]+, ){2,}|\.py:\d+:\s"  # pytest output
    r"|^\s*(?:[\w.]+\.)?[A-Z]\w*(?:Error|Exception|Warning)\b:\s|^\s*\[SQL:\s"  # exceptions
    r"|^\s*(?:SELECT|FROM|WHERE|JOIN|LEFT JOIN|ORDER BY|GROUP BY|LIMIT|VALUES|INSERT INTO|UPDATE)\s"
    r"|^\s*[^\x00-\x7f\w\u2022\u00b7]\s{1,2}\S"            # tool output with an emoji or glyph prefix
    r"|^\s*(?:The above exception was|During handling of the above|\(Background on this error at:)"
    r"|^\s*(?:INFO|DEBUG|WARNING|ERROR|CRITICAL):[\w.]+:"
)
_QUOTE_LINE = re.compile(r"^\s*(?:[\u258e\u258f\u2503\u2502]|>)\s")
_WHOLE_QUOTE = re.compile(r'^(?:"[^"]+"|\u201c[^\u201d]+\u201d|\'[^\']+\')[.,:;!?]?$')
_BULLET_LEAD = re.compile(r"^\s*(?:[-*\u2022\u25aa\u25b8>]|\d+[.)])\s+")
_CAP_START = re.compile(r"(?:^|\n|[.!?]\s+)(?:\s*(?:[-*\u2022>]|\d+[.)])\s+)?[\"'(*]*[A-Z]")
# Log lines share a "source:" prefix ("@pkg/web build: ...", "worker-1 | ...").
_LOG_PREFIX = re.compile(r"^\s*([^\s:|]{2,40}(?: [^\s:|]{1,20})?)\s?[:|]\s")
# A capitalised sentence with its own end punctuation (questions included).
_FORMAL_SENT = re.compile(r"(?:^|(?<=[.!?]\s)|(?<=\n))[\"'(*]*[A-Z][^.!?\n]{10,}[.!?](?=\s|$)")
# "Goal: ...", "Issue: ...": three or more of these in a message is a pasted report.
_REPORT_LINE = re.compile(r"(?m)^[A-Z][A-Za-z]+(?: [A-Za-z]+){0,2}: \S")
# Report-item lead: "Weak spot: ...", "Fix: ...", "Worth fixing: ..."
_LABEL_LEAD = re.compile(r"^[A-Z][a-z]+(?: [a-z]+){0,2}: [a-z\"`(]")
_EM_DASH = re.compile(r"\s[\u2014\u2013]\s|\w\u2014\w")
_CURLY = re.compile("[\u2018\u2019\u201c\u201d]")
_ASSISTANT = re.compile(
    r"\b(?:I've|I'll|I'm going to|Here's|Here is|Here are|Let me|I can also|I'd (?:suggest|recommend|keep|go with)"
    r"|Want me to|Should I|Shall I|Happy to|Your call|Next steps?|In summary|To summari[sz]e|Summary:"
    r"|Root cause|Overall,|Note:|Worth (?:a|noting|flagging|fixing|knowing)|You can now|This (?:means|ensures|keeps|makes|is because)"
    r"|Great question|You're (?:right|absolutely)|I (?:have|had) (?:added|updated|fixed|changed|implemented|created|verified)"
    r"|Key (?:changes|findings|points)|What (?:changed|I did)|Verdict|TL;DR|Bottom line)\b"
)
_MD_HEADER = re.compile(r"(?m)^\s{0,3}#{1,6}\s+\S")
_MD_TABLE = re.compile(r"(?m)^\s*\|.*\|\s*$")
_MD_BOLD = re.compile(r"\*\*[^*\n]+\*\*|__[^_\n]+__")
_MD_LINK = re.compile(r"\[[^\]\n]{2,80}\]\([^)\s]+\)")
_MD_CHECKBOX = re.compile(r"(?m)^\s*[-*]\s+\[[ xX]\]\s")
_CHECKS = re.compile("[\u2705\u274c\u26a0\u2713\u2717\u2714\u2716\U0001F534\U0001F7E2\U0001F7E1]")
_CAP_BULLET = re.compile(r"(?m)^\s*(?:[-*\u2022]|\d+\.)\s+[`*\"(]*[A-Z]\S*(?:\s+\S+){4,}")
_FILLER = re.compile(
    r"\b(?:Um+|Uh+|Uhm+|Erm+|Mm+-?hm+|Mhm+|Uh-huh|Hmm\.)(?=[\s.,!?]|$)"
    r"|(?:^|(?<=[.!?]\s))(?:Okay|Yeah|Right|All right|Exactly|Sure|Yes|No|Cool|Perfect|Thank you|Thanks)[.!](?=\s|$)",
    re.M,
)
_TRANSCRIPT_LABEL = re.compile(
    r"(?m)^\s*(?:\[?\(?\d{1,2}:\d\d(?::\d\d)?\)?\]?\s|(?:Speaker|Sprecher|Interviewer|Interviewee|Candidate|"
    r"Participant|Host|Guest)\s*\d*\s*[:\-(]|[A-Z][a-z]+(?: [A-Z][a-z]+)?\s+\d{1,2}:\d\d(?::\d\d)?\s*$)"
)
_PROFILE = (
    re.compile(r"View .{1,60}?['\u2019]s? (?:full )?profile"),
    re.compile(r"\b\d[\d,.]*\s?[KkM]? followers\b"),
    re.compile(r"(?m)^\s*Promoted\s*$"),
    re.compile(r"\u2022\s*(?:1st|2nd|3rd\+?)\b"),
    re.compile(r"\b\d+\+? connections\b"),
    re.compile(r"(?m)^\s*Endorse\s*$"),
    re.compile(r"(?m)^\s*Show all\b"),
    re.compile(r"\u00b7\s*(?:Full-time|Part-time|Internship|Contract|Freelance|Self-employed|Hybrid|On-site|Remote|Apprenticeship)\b"),
    re.compile(r"\b\d+\s(?:mos?|yrs?)\b"),
    re.compile(r"(?m)\S logo\s*$"),
    re.compile(r"(?m)^\s*(?:Message|Follow|Connect|Like|Comment|Repost|Send|\+ Follow)\s*$"),
    re.compile(r"(?m)^\s*(?:Experience|Education|Skills|About|Activity|Licenses & certifications|Featured|"
               r"Recommendations|Interests|Languages|Honors & awards|Volunteering|Projects)\s*$"),
    re.compile(r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{4}\s*[-\u2013]\s*"
               r"(?:Present|Heute|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{4})"),
    re.compile(r"\b\d+ notifications?\b|\bReactivate Premium\b|\bOpen to work\b|\bpeople you may know\b", re.I),
)
_EMAIL_HEADER = re.compile(r"(?m)^\s*(?:From|To|Cc|Bcc|Subject|Sent|Date|Von|An|Betreff|Gesendet):\s+\S")
_EMAIL_QUOTE = re.compile(r"(?m)^On .{5,120} wrote:\s*$|^-{2,}\s*(?:Original Message|Forwarded message)\s*-{2,}")
_TEMPLATE_OPEN = re.compile(r"^(?:You are (?:a|an|the|Claude)\b|Your task is\b|# [A-Z]|<[a-z_]+>\s*$|##\s)")
_TITLE_LINE = re.compile(r"^[A-Z0-9][\w&'./+-]*(?:\s+[\w&'./+()-]+){0,3}$")


def _norm_line(line):
    return re.sub(r"\s+", " ", line.strip().lower())


def _tokens(text):
    return [t.strip(".,!?;:()[]{}\"'`*").lower() for t in text.split()]


_EMOTICON = re.compile(r"(?:^|(?<=\s))(?:\^\^|:\)|:D|xD|xd)(?=\s|$|[.,!?])")


def shorthand_hits(text):
    toks = _tokens(text)
    n = sum(1 for t in toks if t in SHORTHAND)
    return n + text.count("->") + len(_EMOTICON.findall(text))


def period_sentences(text):
    """Sentences that end in a full stop (abbreviations and ellipses aside)."""
    t = _ABBREV.sub("", text).replace("...", "").replace("\u2026", "")
    return len(_PERIOD_END.findall(t))


def template_lines(records, min_words=8, min_sessions=4):
    """Lines that recur verbatim in at least min_sessions distinct sessions.
    Lines with the writer's shorthand are their own stock phrases, not pastes."""
    seen = {}
    for r in records:
        sid = r.get("session_id") or ""
        for ln in set(r["text"].split("\n")):
            k = _norm_line(ln)
            if len(k.split()) < min_words:
                continue
            seen.setdefault(k, set()).add(sid)
    return {k for k, s in seen.items() if len(s) >= min_sessions and not shorthand_hits(k)}


_SOFT = frozenset(("register", "llm_output", "template"))


def paragraph_score(par, hot_lines=frozenset()):
    """Score how likely a paragraph is pasted rather than typed. Returns
    (score, {category: points})."""
    words = par.split()
    n = len(words)
    cats = collections.Counter()
    if not n:
        return 0, cats
    periods = period_sentences(par)
    caps = len(_CAP_START.findall(par))
    short = shorthand_hits(par)
    lead = _BULLET_LEAD.sub("", par.lstrip())
    # register
    if n >= 15 and periods >= 2 and caps >= 2:
        cats["register"] += 2
        if periods >= 4:
            cats["register"] += 1
    if n >= 12 and par.rstrip().endswith(".") and not par.rstrip().endswith(".."):
        cats["register"] += 1
        if lead[:1].isupper() and short == 0:
            cats["register"] += 1
            if re.search(r"[,:;]\s", par):
                cats["register"] += 1
    if n >= 25 and periods >= 1 and short == 0:
        cats["register"] += 1
    if n >= 12 and periods < 2 and short == 0 and len(_FORMAL_SENT.findall(par)) >= 2:
        cats["register"] += 2
    if n >= 12 and re.match(r"\s*\d+\.\s", par) and par.rstrip().endswith("."):
        cats["register"] += 1   # a numbered report item, quoted
    if n >= 12 and _LABEL_LEAD.match(lead):
        cats["register"] += 1
    if _EM_DASH.search(par):
        cats["register"] += 1
    if _CURLY.search(par):
        cats["register"] += 1
    # assistant output and markdown structure
    a = len(_ASSISTANT.findall(par))
    if a:
        cats["llm_output"] += min(a, 2)
    if _MD_HEADER.search(par):
        cats["llm_output"] += 3
    if len(_MD_TABLE.findall(par)) >= 2:
        cats["llm_output"] += 3
    if _CHECKS.search(par):
        cats["llm_output"] += 2
    if _MD_BOLD.search(par) or _MD_LINK.search(par):
        cats["llm_output"] += 1
    if _MD_CHECKBOX.search(par):
        cats["llm_output"] += 2
    b = len(_CAP_BULLET.findall(par))
    if b >= 2:
        cats["llm_output"] += 2 if b == 2 else 3
    prefixes = collections.Counter()
    for ln in par.split("\n"):
        m = _LOG_PREFIX.match(ln)
        if m and m.group(1).lower() not in ("re", "q", "e.g", "i.e", "note", "edit"):
            prefixes[m.group(1)] += 1
    if prefixes and prefixes.most_common(1)[0][1] >= 3:
        cats["log"] += 3
    lines = [ln for ln in par.split("\n") if ln.strip()]
    if lines and sum(1 for ln in lines if _LOG_LINE.search(ln)) * 2 >= len(lines):
        cats["log"] += 3
    # quoted text: Claude Code's quote bar, markdown quotes, or a paragraph
    # that is one long quotation
    if any(_QUOTE_LINE.match(ln) and len(ln.split()) >= 6 for ln in par.split("\n")):
        cats["quote"] += 3
    if n >= 12 and _WHOLE_QUOTE.match(par.strip()):
        cats["quote"] += 3
    if any(ln.rstrip().endswith(";") and len(ln.split()) >= 5 for ln in par.split("\n")):
        cats["quote"] += 3   # list items ending in ";" are always someone else's
    # transcripts, profiles, templates
    fillers = len(_FILLER.findall(par))
    if fillers >= 2:
        cats["transcript"] += 2 + (fillers >= 4)
    if _TRANSCRIPT_LABEL.search(par):
        cats["transcript"] += 2
    if sum(1 for p in _PROFILE if p.search(par)) >= 2:
        cats["profile"] += 3
    if _EMAIL_HEADER.search(par) or _EMAIL_QUOTE.search(par):
        cats["email"] += 3
    if _TEMPLATE_OPEN.match(lead):
        cats["template"] += 2
    if hot_lines and any(_norm_line(ln) in hot_lines for ln in par.split("\n")):
        cats["template"] += 3
    # The writer's own register pulls the soft signals down; structural ones
    # (logs, quotes, transcripts, profiles, emails) stand on their own.
    soft = sum(v for k, v in cats.items() if k in _SOFT)
    soft -= min(short, 3)
    if lead[:1].islower() and periods < 2:
        soft -= 1
    hard = sum(v for k, v in cats.items() if k not in _SOFT)
    return (max(soft, 0) + hard if hard else soft), cats


def _profile_hits(text):
    return sum(1 for p in _PROFILE if p.search(text))


def whole_message_paste(text, cfg):
    """Markers strong enough to drop the whole message. Returns a category or None."""
    words = len(text.split())
    if words > cfg["max_words"]:
        return "length_outlier"
    if _profile_hits(text) >= cfg["profile_min_hits"]:
        return "profile"
    if len(_EMAIL_HEADER.findall(text)) >= cfg["email_header_lines"] or _EMAIL_QUOTE.search(text):
        return "email"
    if words >= cfg["transcript_min_words"]:
        fillers = len(_FILLER.findall(text))
        if fillers >= cfg["transcript_min_fillers"] and fillers * 100.0 / words >= cfg["transcript_fillers_per_100"]:
            return "transcript"
    if len(_TRANSCRIPT_LABEL.findall(text)) >= cfg["transcript_label_lines"]:
        return "transcript"
    return None


def _residue(text, cfg):
    """What is left after stripping still reads as a paste: several lines
    ending in a full stop, or a stack of short Title Case UI lines."""
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    if not lines:
        return False
    dotted = sum(1 for ln in lines if ln.endswith(".") and not ln.endswith("..") and len(ln.split()) >= 1)
    if dotted >= cfg["residue_period_lines"] and dotted / len(lines) >= 0.3:
        return True
    titles = sum(1 for ln in lines if _TITLE_LINE.match(ln) and not shorthand_hits(ln))
    return titles >= 4 and titles / len(lines) >= 0.4


def _formal_sentence(par):
    s = par.strip()
    return (len(s.split()) >= 4 and s[:1].isupper() and s.endswith(".") and not s.endswith("..")
            and not shorthand_hits(s))


def author_check(text, cfg=None, hot_lines=frozenset()):
    """Decide whether a cleaned message is the writer's own typing.

    Returns (text, None, info) for a keeper (pasted paragraphs may have been
    stripped; info["stripped"] lists their categories) or (None, reason, info)
    for a drop. Reasons are "not_typed_<category>".
    """
    cfg = dict(AUTHOR_DEFAULTS, **(cfg or {}))
    info = {"stripped": [], "scores": []}
    if not cfg["enabled"]:
        return text, None, info
    cat = whole_message_paste(text, cfg)
    if cat:
        return None, "not_typed_" + cat, info
    pars = re.split(r"\n\s*\n", text)
    report = len(_REPORT_LINE.findall(text)) >= 3
    kept, removed = [], collections.Counter()
    for p in pars:
        score, cats = paragraph_score(p, hot_lines)
        if report and _REPORT_LINE.match(p.lstrip()):
            cats["report"] += 3
            score += 3
        info["scores"].append(score)
        if score >= cfg["par_threshold"] and cats:
            top = cats.most_common(1)[0][0]
            removed[top] += sum(cats.values())
            info["stripped"].append(top)
        else:
            kept.append(p)
    if not removed:
        return text, None, info
    top = removed.most_common(1)[0][0]
    if _residue("\n\n".join(kept), cfg):
        return None, "not_typed_" + top, info
    # A paste's own lead-in or tail (a capitalised sentence with a full stop
    # and none of the writer's shorthand) goes with it.
    rest_pars = []
    for p in kept:
        if _formal_sentence(p):
            info["stripped"].append("register")
            removed["register"] += 1
        else:
            rest_pars.append(p)
    top = removed.most_common(1)[0][0]
    rest = normalise("\n\n".join(rest_pars))
    if prose_words(rest) < cfg["min_keep_words"] or _residue(rest, cfg) or code_share(rest) > 0.5:
        return None, "not_typed_" + top, info
    return rest, None, info


# ---------------------------------------------------------------------------
# Language: English only for v1. Stopword counts for English and German, with
# umlauts and sharp s as extra German evidence.

_EN_STOP = frozenset("""
the and is are was were be been to of in on at for with this that these those it its it's you your we
our they them i i'm i'd me my not but or if then so just can could would should will what why how when
where which who do does did don't doesn't didn't have has had there here from by as an a about into
also still only all any some no yes than more most very really think need want like get got make
""".split())
_DE_STOP = frozenset("""
und der die das den dem des ist sind war nicht ich du er sie wir ihr es ein eine einen einem einer
zu mit auf f\u00fcr aber auch noch wie was dass wenn oder nur mal schon jetzt kann k\u00f6nnen wird werden
hast habe hab bin bist sich man mir mich dir dich uns euch kein keine keinen nach bei aus vom zum zur
doch halt eben gerade bitte danke genau also ja nein vielleicht weil damit dann denn hier dort sehr
gibt geht macht machen muss m\u00fcssen soll sollte w\u00fcrde w\u00e4re glaube einfach ganz immer wieder etwas
""".split())
# Words that are both English and German enough to say nothing either way.
_AMBIGUOUS = frozenset("also so man was die hast war bin in an am um".split())
_GERMAN_CHARS = re.compile("[\u00e4\u00f6\u00fc\u00c4\u00d6\u00dc\u00df]")


def language_counts(text):
    en = de = 0
    for tok in re.findall(r"[^\W\d_]+(?:'[a-z]+)?", text.lower()):
        if tok in _AMBIGUOUS:
            continue
        if tok in _DE_STOP or _GERMAN_CHARS.search(tok):
            de += 1
        elif tok in _EN_STOP:
            en += 1
    return en, de


def is_mainly_non_english(text, cfg=None):
    cfg = dict(AUTHOR_DEFAULTS, **(cfg or {}))
    en, de = language_counts(text)
    if de < cfg["lang_min_hits"]:
        return False
    return de / float(en + de) > cfg["lang_max_share"]


def language_check(text, cfg=None):
    """English only. A mainly non-English message loses its non-English
    paragraphs and is kept if at least min_keep_words of mainly English prose
    remain. Returns (text, None, n_stripped) or (None, "non_english", 0)."""
    cfg = dict(AUTHOR_DEFAULTS, **(cfg or {}))
    if not is_mainly_non_english(text, cfg):
        return text, None, 0
    kept, n = [], 0
    for p in re.split(r"\n\s*\n", text):
        en, de = language_counts(p)
        if de and de >= en:
            n += 1
        else:
            kept.append(p)
    rest = normalise("\n\n".join(kept))
    if n and prose_words(rest) >= cfg["min_keep_words"] and not is_mainly_non_english(rest, cfg):
        en, de = language_counts(rest)
        if en > de:
            return rest, None, n
    return None, "non_english", 0

