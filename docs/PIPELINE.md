# Data pipeline

`irulan-swe` learns to rewrite LLM-written text into the voice of a working
software engineer. Training pairs come from back-translation: real messages a
person wrote are the targets, and a model later writes a "Claude-style"
version of each as the input.

This document covers step 1: extracting and scrubbing the human side. It
calls no model and spends nothing.

Everything runs on Python 3 with the standard library only. Commands assume
the repo root as the working directory.

## Never commit data

This repo is public. `data/`, `*.local.*` and `scrub/denylist.txt` are
git-ignored. Before committing, run `git status --porcelain` and
`git diff --cached --stat` and check that only code, templates and docs are
staged. Test fixtures are invented strings.

## Step 1a: Claude Code messages

```sh
make extract-cc          # or: python3 -I pipeline.py extract-cc
```

Walks every `*.jsonl` under `~/.claude/projects` (override with `--src`),
line by line, and keeps only what the person typed:

- `type == "user"` turns that are not `isMeta`, not `isSidechain`, not a
  compaction summary, and whose `origin` / `promptSource` / `turnOrigin` mark
  them as human (task notifications, peer and SDK messages are dropped).
- Text parts only; tool results and images are ignored.
- Harness wrapper blocks are removed with their content
  (`<system-reminder>`, `<pasted_content …>`, `<command-*>`,
  `<local-command-*>`, `<task-notification>`, `<bash-*>`, and any other
  `<kebab-or_snake-tag>…</…>` block). A message that is empty after that, or
  still starts with a tag, is dropped.
- Dropped outright: interrupt markers, session continuations, `Caveat:`
  messages, skill bodies, hook feedback, slash-command-only messages, and
  pasted chat threads or code-host UI (they carry other people's words).
- Fenced code blocks, runs of 3+ code/log lines, single code lines of 200+
  characters, and paragraphs that look like pasted terminal or agent output
  (TUI glyphs, wrapped-line indentation) are stripped. A message that was
  mostly code or logs is kept only if at least 5 words of the writer's own
  prose remain.
- Exact dedupe after whitespace and case normalisation. Short messages
  (3 words or fewer) are not deduped but capped at 3 copies each.

The writer's own words are never rewritten: no spelling, casing or
shorthand normalisation.

Outputs:

| file | contents |
|---|---|
| `data/raw/cc_messages.jsonl` | filtered and deduped, not scrubbed |
| `data/clean/cc_messages.jsonl` | scrubbed training targets |
| `data/entity_candidates.tsv` | possible names to review |
| `data/stats/cc_stats.json` | counts only |

Each record has `id`, `project` (the transcript's project directory name),
`session_id`, `timestamp`, `words`, `lines` and `text`.

## Scrubbing

`irulan/scrub.py` replaces sensitive spans with typed placeholders and
leaves everything else alone:

| placeholder | what |
|---|---|
| `<SECRET>` | provider API keys and tokens, JWTs, `Bearer …`, `password=`-style assignments (the key name stays), private key blocks, connection strings with credentials, long hex or base64 runs |
| `<URL>` | URLs, except on well-known public dev domains, which keep the bare domain (`github.com`, `docs.*`) |
| `<EMAIL>` | email addresses |
| `<IP>` | IPv4 and IPv6 addresses |
| `<HOST>` | internal hostnames (`*.internal`, `*.local`, `*.modal.run`, `*.ts.net`, cloud app hosts), machine names like `gpu-node-3`, bare non-public domains |
| `<PATH>` | home-dir and absolute filesystem paths; generic basenames such as `README.md` are kept |
| `<ID>` | UUIDs, prefixed ids (`org_…`, `user_…`, `toolu_…`), commit SHAs |
| `<NAME>` | terms from `scrub/denylist.txt` |

Per-category replacement counts go to `data/stats/cc_stats.json`.

## The denylist

```sh
cp scrub/denylist.example.txt scrub/denylist.txt   # once
```

One term per line, case-insensitive, whole word; `#` starts a comment. Each
match becomes `<NAME>`, and `term-<digits>` becomes `<HOST>`.

To fill it, open `data/entity_candidates.tsv` (columns: count, term, a short
scrubbed example). It lists capitalised, camelCase and multi-word proper-noun
spans and frequent lowercase tokens that are not common English or dev
vocabulary, most frequent first. Copy real customer, company, person and
infrastructure names into the denylist, then rescrub:

```sh
make rescrub             # or: python3 -I pipeline.py rescrub
```

`rescrub` re-reads `data/raw/cc_messages.jsonl` and rewrites the clean
output, the candidates and the stats without walking the transcripts again.

## Stats and tests

```sh
make stats               # print data/stats/cc_stats.json
make test                # stdlib unittest, invented fixtures only
```
