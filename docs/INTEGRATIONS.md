# Integrations

Every integration is a call to `Client.rewrite()` or `irulan rewrite`. Both
need `IRULAN_BASE_URL` and `IRULAN_API_KEY` (the hosted endpoint is planned,
see [API.md](API.md)).

## Python

```python
import irulan

client = irulan.Client()                        # IRULAN_API_KEY / IRULAN_BASE_URL
text = client.rewrite("I wanted to follow up on whether you had a chance to review the PR.")
mine = client.rewrite(text, model="irulan-jdoe-voice")   # your own fine-tune
```

`rewrite` sends the versioned system instruction the models are trained with
(`irulan.SYSTEM_VERSION`, currently `system_v1`). If you call
`/chat/completions` yourself, send `irulan.SYSTEM_PROMPT` as the system
message.

## Shell and scripts

```sh
irulan rewrite "Could you please take a look at the failing test when you have a moment?"
git log -1 --format=%B | irulan rewrite              # stdin
irulan rewrite --file draft.md --model irulan-jdoe-voice > final.md
```

The exit code is 0 on success, 1 on an API error, 2 on a usage error and 3
when the API is not available yet.

## Claude Code hook

[`integrations/claude-code/`](../integrations/claude-code/) has a
`PreToolUse` hook that rewrites the `--body` of `gh pr create`,
`gh pr comment` and `gh issue comment` and asks you to approve the result.
Its README covers the setup.

## GitHub Action

Rewrite a PR description when the PR is opened. Store the key as a repository
secret. The text goes to the irulan API, so only use this in repositories
where that is acceptable.

```yaml
name: irulan
on:
  pull_request:
    types: [opened]
permissions:
  pull-requests: write
jobs:
  rewrite-body:
    if: github.event.pull_request.body != ''
    runs-on: ubuntu-latest
    steps:
      - uses: actions/setup-python@v5
        with: {python-version: "3.12"}
      - run: pip install "git+https://github.com/kyora-sh/irulan"
      - env:
          IRULAN_BASE_URL: ${{ vars.IRULAN_BASE_URL }}
          IRULAN_API_KEY: ${{ secrets.IRULAN_API_KEY }}
          GH_TOKEN: ${{ github.token }}
          BODY: ${{ github.event.pull_request.body }}
        run: |
          printf '%s' "$BODY" | irulan rewrite > body.md
          gh pr edit ${{ github.event.pull_request.number }} --repo ${{ github.repository }} --body-file body.md
```

The body is passed through an environment variable rather than inlined into
the script, so its contents are never run as shell.
