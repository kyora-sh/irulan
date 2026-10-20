"""Generators for the studio's assistant-style text.

CodexGenerator runs the Codex CLI once per call:

    codex exec -m MODEL -c model_reasoning_effort=EFFORT -s read-only
               --skip-git-repo-check --ephemeral -C <empty temp dir> -o <file> -

from a fresh empty temp dir, with the prompt on stdin, and reads only the
last message from the `-o` file. It needs `codex` on PATH and a `codex login`.
FakeGenerator is deterministic and needs no network.
"""

import os
import re
import shutil
import subprocess
import tempfile

from .prompts import PROMPT_VERSION, StudioError

DEFAULT_MODEL = "gpt-6.1-sol"
DEFAULT_EFFORT = "low"
DEFAULT_TIMEOUT = 240


class FakeGenerator:
    """Deterministic, no network; for tests and for trying the page."""

    def __init__(self):
        self.info = {"backend": "fake", "model": "fake-1", "prompt_version": PROMPT_VERSION}
        self.calls = 0

    def status(self):
        return True, ""

    def generate(self, prompt):
        self.calls += 1
        text = ("**Summary:** This change moves the `ledger-sync` retry loop into "
                "services/ledger/retry.py and cuts p95 latency from 840 ms to 310 ms. Additionally, "
                "it bumps the client to v2.3.1.\n\nThat said, it might be worth a second look.")
        if "TOPIC:" in prompt:
            return "TOPIC: retry loop for the ledger sync service\n---\n" + text
        return text


class CodexGenerator:
    def __init__(self, model=DEFAULT_MODEL, effort=DEFAULT_EFFORT, timeout=DEFAULT_TIMEOUT, tmp_root=None,
                 codex_bin="codex"):
        self.model = model
        self.effort = effort or None
        self.timeout = timeout
        self.tmp_root = tmp_root or None
        self.codex_bin = codex_bin
        self.info = {"backend": "codex", "model": model, "prompt_version": PROMPT_VERSION}
        self._status = None

    def status(self):
        """(ok, reason), checked once: codex on PATH and logged in."""
        if self._status is None:
            if not shutil.which(self.codex_bin):
                self._status = (False, "codex CLI not found on PATH")
                return self._status
            try:
                p = subprocess.run([self.codex_bin, "login", "status"], capture_output=True, text=True, timeout=30)
                out = (p.stdout + p.stderr).strip()
                if p.returncode != 0 or re.search(r"(?i)not logged in", out):
                    self._status = (False, "codex is not logged in (run `codex login`): %s" % out[-200:])
                else:
                    self._status = (True, "")
            except (OSError, subprocess.TimeoutExpired) as e:
                self._status = (False, "could not check codex login: %s" % e)
        return self._status

    def command(self, work, last):
        cmd = [self.codex_bin, "exec", "-m", self.model]
        if self.effort:
            cmd += ["-c", "model_reasoning_effort=%s" % self.effort]
        return cmd + ["-s", "read-only", "--skip-git-repo-check", "--ephemeral", "-C", work, "-o", last, "-"]

    def generate(self, prompt):
        if self.tmp_root:
            os.makedirs(self.tmp_root, exist_ok=True)
        work = tempfile.mkdtemp(prefix="irulan-studio-", dir=self.tmp_root)
        try:
            last = os.path.join(work, ".last_message")
            try:
                p = subprocess.run(self.command(work, last), input=prompt, capture_output=True, text=True,
                                   cwd=work, timeout=self.timeout)
            except subprocess.TimeoutExpired:
                raise StudioError("codex timed out after %ss" % self.timeout, 504)
            except OSError as e:
                raise StudioError("could not run codex: %s" % e, 502)
            text = ""
            if os.path.exists(last):
                with open(last, encoding="utf-8") as f:
                    text = f.read()
            if not text.strip():
                why = p.stderr.strip()[-300:] or "empty output (exit %d)" % p.returncode
                raise StudioError("codex failed: %s" % why, 502)
            return text
        finally:
            shutil.rmtree(work, ignore_errors=True)
