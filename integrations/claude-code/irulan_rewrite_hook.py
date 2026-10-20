"""Claude Code PreToolUse hook: rewrite the --body of `gh pr create`,
`gh pr comment` and `gh issue comment` through irulan before the command runs.

Claude Code passes the tool call as JSON on stdin. When the Bash command is
one of those gh calls with a literal --body/-b, this prints a PreToolUse
decision of "ask" with `updatedInput` holding the same command and the
rewritten body, so you see the new text in the permission prompt and approve
it or not. Anything else (other commands, --body-file, a body built from
$variables) passes through untouched.

If irulan is not reachable the command runs unchanged (fail open); set
IRULAN_HOOK_STRICT=1 to block it instead.

Environment: IRULAN_BASE_URL, IRULAN_API_KEY (as for the client),
IRULAN_HOOK_MODEL or IRULAN_MODEL (default irulan-swe).
Needs Python 3.10+ and the irulan package importable, or the `irulan` CLI on PATH.
"""

import json
import os
import re
import shlex
import shutil
import subprocess
import sys

GH_CALL = re.compile(r"(?:^|[\s;&|(])gh\s+(?:pr\s+(?:create|comment)|issue\s+comment)\b")
BODY_FLAG = re.compile(r"(?:^|(?<=\s))(--body|-b)(=|[ \t]+)")
# "$(cat <<'EOF'\n...\nEOF\n)": how Claude Code usually passes a multi-line body.
HEREDOC = re.compile(r'"\$\(cat\s+<<-?\s*([\'"]?)(\w+)\1[ \t]*\n(.*?)\n\2[ \t]*\n?\s*\)"', re.S)
WORD_END = " \t\n;&|<>()"


def parse_word(s, i):
    """Decode the shell word starting at s[i] when it is a plain literal.
    Returns (end, text) or None when it holds an expansion or is unterminated."""
    out = []
    while i < len(s) and s[i] not in WORD_END:
        c = s[i]
        if c == "'":
            j = s.find("'", i + 1)
            if j < 0:
                return None
            out.append(s[i + 1:j])
            i = j + 1
        elif c == '"':
            m = HEREDOC.match(s, i)
            if m:
                if not m.group(1) and re.search(r"[$`\\]", m.group(3)):
                    return None  # unquoted delimiter: the body would be expanded
                out.append(m.group(3))
                i = m.end()
                continue
            j, buf = i + 1, []
            while j < len(s) and s[j] != '"':
                if s[j] == "\\" and j + 1 < len(s) and s[j + 1] in '"\\$`\n':
                    if s[j + 1] != "\n":
                        buf.append(s[j + 1])
                    j += 2
                    continue
                if s[j] in "$`":
                    return None
                buf.append(s[j])
                j += 1
            if j >= len(s):
                return None
            out.append("".join(buf))
            i = j + 1
        elif c == "\\":
            if i + 1 < len(s) and s[i + 1] != "\n":
                out.append(s[i + 1])
            i += 2
        elif c in "$`":
            return None
        else:
            out.append(c)
            i += 1
    return i, "".join(out)


def find_body(command):
    """(start, end, text) of the first literal --body value of a matching gh
    call, or None."""
    m = GH_CALL.search(command)
    if not m:
        return None
    for f in BODY_FLAG.finditer(command, m.end()):
        start = f.end()
        parsed = parse_word(command, start)
        if parsed is None:
            return None
        end, text = parsed
        return start, end, text
    return None


def replace_body(command, rewrite):
    """Return the command with its --body text passed through rewrite(), or None."""
    found = find_body(command)
    if not found:
        return None
    start, end, text = found
    if not text.strip():
        return None
    new = rewrite(text)
    if not new or not new.strip() or new == text:
        return None
    return command[:start] + shlex.quote(new) + command[end:]


def make_rewriter(model):
    try:
        import irulan
    except ImportError:
        irulan = None
    if irulan is not None:
        client = irulan.Client(timeout=45, max_retries=1)
        return lambda text: client.rewrite(text, model=model)
    exe = shutil.which("irulan")
    if not exe:
        raise RuntimeError("irulan is neither importable nor on PATH")

    def via_cli(text):
        p = subprocess.run([exe, "rewrite", "--model", model], input=text, capture_output=True, text=True, timeout=50)
        if p.returncode != 0:
            raise RuntimeError(p.stderr.strip() or "irulan rewrite failed")
        return p.stdout.rstrip("\n")
    return via_cli


def main():
    try:
        event = json.load(sys.stdin)
    except ValueError:
        return 0
    if event.get("hook_event_name") != "PreToolUse" or event.get("tool_name") != "Bash":
        return 0
    tool_input = event.get("tool_input") or {}
    command = tool_input.get("command") or ""
    if not find_body(command):
        return 0
    model = os.environ.get("IRULAN_HOOK_MODEL") or os.environ.get("IRULAN_MODEL") or "irulan-swe"
    try:
        new_command = replace_body(command, make_rewriter(model))
    except Exception as e:  # noqa: BLE001 - any failure: fail open (or closed, if strict)
        if os.environ.get("IRULAN_HOOK_STRICT") == "1":
            print("irulan hook: rewrite failed, blocking (IRULAN_HOOK_STRICT=1): %s" % e, file=sys.stderr)
            return 2
        print("irulan hook: rewrite skipped: %s" % e, file=sys.stderr)
        return 0
    if not new_command:
        return 0
    print(json.dumps({"hookSpecificOutput": {
        "hookEventName": "PreToolUse",
        "permissionDecision": "ask",
        "permissionDecisionReason": "irulan rewrote the --body with %s. Check the new text before it is posted." % model,
        "updatedInput": dict(tool_input, command=new_command),
    }}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
