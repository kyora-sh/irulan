# Irulan

Princess Irulan wrote the chronicles of Paul Atreides, telling his story in her own words. Irulan does the same for your text.

<!-- vision: written by the repo owner -->

## Install

Python 3.10+, no dependencies.

```sh
pip install "git+https://github.com/kyora-sh/irulan"
# or, from a checkout
pip install -e .
```

This installs the `irulan` command and the `irulan` package.

## Quickstart: rewrite

```python
import irulan

client = irulan.Client()          # reads IRULAN_API_KEY and IRULAN_BASE_URL
print(client.rewrite("I wanted to follow up on whether you had a chance to review the PR."))
```

```sh
irulan rewrite "I wanted to follow up on whether you had a chance to review the PR."
```

The models are `irulan-swe`, or your own `irulan-<user>-<name>`. The hosted
endpoint is planned. Until it is live, set `IRULAN_BASE_URL` to a server that
implements [docs/API.md](docs/API.md); without it, calls raise
`irulan.NotAvailableYet`.

## Prepare your own dataset

These steps run locally and call no model:

```sh
irulan extract                  # ~/.claude/projects -> data/raw/cc_messages.jsonl
irulan stats                    # counts only
less data/raw/cc_messages.jsonl # read what you would upload
irulan upload --dry-run --redact-secrets-locally --payload-out data/upload_preview.jsonl
irulan upload --redact-secrets-locally
```

`extract` keeps only the text you typed yourself. It drops harness wrappers,
tool output, pasted code, logs, transcripts and LLM answers, and
non-English messages, then deduplicates. Filters and flags are described in
[docs/PIPELINE.md](docs/PIPELINE.md).

## Fine-tune on kyora (planned)

```sh
irulan datasets status DS --wait scrubbed
irulan datasets review DS --out data/review.tsv
irulan datasets approve DS --denylist data/denylist.local.txt
irulan finetune create --dataset DS --name my-voice --wait
irulan rewrite --model irulan-<user>-my-voice "..."
```

The platform scrubs the upload, builds the name review list, generates
back-translation pairs, prepares the training set and trains a LoRA adapter
on `irulan-swe`. The spec is in [docs/API.md](docs/API.md). The client already
implements it, and the dataset and fine-tune endpoints are not live yet.
`examples/prep_and_finetune.sh` runs the whole flow.

## Integrations

- Python and shell: `Client.rewrite()` and `irulan rewrite` (stdin, `--file`).
- Claude Code: a `PreToolUse` hook that rewrites the `--body` of
  `gh pr create`, `gh pr comment` and `gh issue comment`, in
  [integrations/claude-code](integrations/claude-code/).
- GitHub Actions and scripts: [docs/INTEGRATIONS.md](docs/INTEGRATIONS.md).

## Privacy

- `irulan extract` reads `~/.claude/projects` and writes to `data/` on your
  machine. Nothing leaves the machine until you run `irulan upload`.
- The upload contains your message text unscrubbed. It may hold names,
  hostnames, URLs, paths and IDs. `upload` shows a count summary and asks
  first (`--yes` skips the question). `--dry-run --payload-out FILE` writes
  the exact payload so you can read it.
- `--redact-secrets-locally` replaces obvious credentials with `<SECRET>`
  before upload: private keys, API tokens with known prefixes, JWTs, bearer
  headers, `password=` / `api_key: "..."` and "the password is ..." forms.
  It is pattern-based and misses credentials that have none of those shapes.
  It does not touch names or hosts.
- On upload, kyora first scrubs the data. Only the scrubbed version is
  stored, encrypted at rest. The original (unscrubbed) upload is not
  retained. `irulan datasets delete DS` deletes a dataset and everything
  derived from it.
- Scrubbing replaces secrets, URLs, emails, IPs, internal hosts, paths, IDs
  and detected names with placeholders. It catches only what matches its
  patterns and what is on the denylist. Names it does not recognise (a
  colleague's first name, a customer, an internal project) stay in unless
  you add them. Read the review list (`irulan datasets review`) and add
  every such name to your denylist before approving.
- The upload format leaves out the local project directory name, which
  contains your home path. Session ids are sent only as a hash.
