"""The Claude Code hook example in integrations/claude-code."""

import importlib.util
import json
import os
import shlex
import subprocess
import sys
import unittest

from .fakeserver import FakeServer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HOOK = os.path.join(ROOT, "integrations", "claude-code", "irulan_rewrite_hook.py")
spec = importlib.util.spec_from_file_location("irulan_rewrite_hook", HOOK)
hook = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hook)

UPPER = lambda t: t.upper()  # noqa: E731


def body_of(command):
    """What bash would pass as the --body argument."""
    toks = shlex.split(command)
    for i, t in enumerate(toks):
        if t in ("--body", "-b"):
            return toks[i + 1]
        if t.startswith("--body="):
            return t[len("--body="):]
    return None


class ParseTests(unittest.TestCase):
    def test_quoting_forms(self):
        cases = [
            ('gh pr comment 12 --body "looks good, merging"', "looks good, merging"),
            ("gh pr comment 12 -b 'it'\"'\"'s fine'", "it's fine"),
            ('gh issue comment 3 --body=plain', "plain"),
            ('gh pr create --title "x" --body "say \\"hi\\" now" --base main', 'say "hi" now'),
        ]
        for cmd, text in cases:
            self.assertEqual(hook.find_body(cmd)[2], text, cmd)
            new = hook.replace_body(cmd, UPPER)
            self.assertEqual(body_of(new), text.upper(), cmd)

    def test_heredoc_body(self):
        cmd = ("gh pr create --title \"Add retries\" --body \"$(cat <<'EOF'\n## Summary\n- add retries\n"
               "- costs $5 in `ci`\nEOF\n)\" --draft")
        self.assertEqual(hook.find_body(cmd)[2], "## Summary\n- add retries\n- costs $5 in `ci`")
        new = hook.replace_body(cmd, UPPER)
        self.assertEqual(body_of(new), "## SUMMARY\n- ADD RETRIES\n- COSTS $5 IN `CI`")
        self.assertTrue(new.startswith('gh pr create --title "Add retries" --body '))
        self.assertTrue(new.endswith(" --draft"))

    def test_left_alone(self):
        for cmd in ('gh pr view 12', 'git commit -m "x"', 'gh pr comment 1 --body-file notes.md',
                    'gh pr comment 1 --body "$MSG"', "gh pr comment 1 --body \"$(cat <<EOF\nhi $USER\nEOF\n)\"",
                    'gh pr comment 1 --body "unterminated', 'echo gh pr --body x'):
            self.assertIsNone(hook.replace_body(cmd, UPPER), cmd)


class RunTests(unittest.TestCase):
    def run_hook(self, event, env):
        env = dict(env, PYTHONPATH=os.path.join(ROOT, "src"), PATH=os.environ.get("PATH", ""))
        # no -I here so PYTHONPATH can stand in for an installed package
        p = subprocess.run([sys.executable, HOOK], input=json.dumps(event), capture_output=True, text=True, env=env)
        return p.returncode, p.stdout, p.stderr

    def event(self, command):
        return {"session_id": "x", "hook_event_name": "PreToolUse", "tool_name": "Bash", "tool_use_id": "t1",
                "tool_input": {"command": command, "description": "Comment on the PR", "timeout": 120000}}

    def test_rewrites_and_asks(self):
        with FakeServer() as srv:
            env = {"IRULAN_BASE_URL": srv.base_url, "IRULAN_API_KEY": srv.api_key}
            code, out, err = self.run_hook(self.event('gh pr comment 7 --body "I Have Reviewed The Change."'), env)
        self.assertEqual(code, 0, err)
        res = json.loads(out)["hookSpecificOutput"]
        self.assertEqual((res["hookEventName"], res["permissionDecision"]), ("PreToolUse", "ask"))
        self.assertEqual(res["updatedInput"]["description"], "Comment on the PR")
        self.assertEqual(res["updatedInput"]["timeout"], 120000)
        self.assertEqual(body_of(res["updatedInput"]["command"]), "terse: i have reviewed the change.")

    def test_fail_open_and_strict(self):
        ev = self.event('gh pr comment 7 --body "hello there"')
        code, out, err = self.run_hook(ev, {})
        self.assertEqual((code, out), (0, ""))
        self.assertIn("rewrite skipped", err)
        code, out, err = self.run_hook(ev, {"IRULAN_HOOK_STRICT": "1"})
        self.assertEqual(code, 2)

    def test_other_tools_pass(self):
        code, out, _ = self.run_hook({"hook_event_name": "PreToolUse", "tool_name": "Edit", "tool_input": {}}, {})
        self.assertEqual((code, out), (0, ""))


if __name__ == "__main__":
    unittest.main()
