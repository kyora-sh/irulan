# Claude Code hook: rewrite PR and issue text

`irulan_rewrite_hook.py` is a `PreToolUse` hook. When Claude Code is about to
run `gh pr create`, `gh pr comment` or `gh issue comment` with a literal
`--body` / `-b`, it sends the body through `irulan rewrite`. It then asks you to
approve the command with the rewritten body in place. Every other command
runs untouched.

## Setup

1. Install irulan where the hook can find it. Either the package is
   importable by the Python that runs the hook, or the `irulan` command is on
   `PATH` (for example `pipx install .` from a checkout). Set
   `IRULAN_BASE_URL` and `IRULAN_API_KEY` in the environment Claude Code runs
   in. Don't put the key in a settings file you commit.
2. Copy the script into your project or user hooks dir:

   ```sh
   mkdir -p .claude/hooks
   cp integrations/claude-code/irulan_rewrite_hook.py .claude/hooks/
   ```

3. Merge `settings.example.json` into `.claude/settings.json` (project),
   `.claude/settings.local.json` (project, not committed) or
   `~/.claude/settings.json` (all projects):

   ```json
   {"hooks": {"PreToolUse": [{"matcher": "Bash", "hooks": [{
     "type": "command", "if": "Bash(gh *)",
     "command": "python3 -I \"$CLAUDE_PROJECT_DIR\"/.claude/hooks/irulan_rewrite_hook.py",
     "timeout": 60}]}]}}
   ```

   - `matcher: "Bash"` runs the hook for Bash tool calls only.
   - `if: "Bash(gh *)"` narrows it to `gh` commands, so other commands don't
     start Python. The script checks the exact subcommand itself, so the hook
     works the same where `if` is not supported.
   - `timeout` is in seconds.
   - If irulan lives in a virtualenv, use that interpreter in `command`
     (`/path/to/venv/bin/python -I ...`). The script also falls back to the
     `irulan` CLI on `PATH`.

## What it does

The hook reads the tool call from stdin:
`{"hook_event_name": "PreToolUse", "tool_name": "Bash", "tool_input": {"command": ...}, ...}`.
When the body can be rewritten, it prints:

```json
{"hookSpecificOutput": {"hookEventName": "PreToolUse", "permissionDecision": "ask",
  "permissionDecisionReason": "irulan rewrote the --body with irulan-swe. Check the new text before it is posted.",
  "updatedInput": {"command": "gh pr comment 7 --body '...rewritten...'", "description": "..."}}}
```

- `updatedInput` replaces the whole tool input, so the hook copies every
  other field unchanged.
- `ask` means you see the new command in the permission prompt. The hook
  never approves a command for you.

It handles a body in single quotes, double quotes (no `$` or backticks),
`--body=...`, or `"$(cat <<'EOF' ... EOF)"` heredocs. It leaves these alone:
`--body-file`, a body built from variables or command substitution, and
unquoted-delimiter heredocs that would expand.

Failures, such as no server or a timeout, let the command run unchanged.
`IRULAN_HOOK_STRICT=1` blocks it instead (exit 2). Use `IRULAN_HOOK_MODEL`
(or `IRULAN_MODEL`) to pick your own model, for example `irulan-jdoe-voice`.

The hook schema follows the Claude Code hooks reference:
https://code.claude.com/docs/en/hooks
