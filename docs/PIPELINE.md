# Dataset prep

`irulan-swe` rewrites LLM-written text in the voice of a working software
engineer. A personal model learns your voice from messages you typed
yourself. Training pairs come from back-translation: your real messages are
the targets, and a model writes a "Claude-style" version of each one as the
input.

The SDK does the first step on your machine. It extracts what you typed into
Claude Code and nothing else, and it calls no model. Everything after that
runs on the kyora platform once you upload: scrubbing, the name review, pair
generation, building the training set and training. See [API.md](API.md).

The local step runs on Python 3.10+ with the standard library only. Paths
below are relative to the working directory. Set `--data-dir` or
`IRULAN_DATA_DIR` to put `data/` somewhere else.

## Never commit data

`data/` and `*.local.*` are git-ignored in this repo. If you work in a
checkout, run `git status --porcelain` before committing and check that only
code and docs are staged. Test fixtures are invented strings.

## Extract

```sh
irulan extract            # from a checkout without installing: python3 -I pipeline.py extract
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
- Text the person did not type is stripped or dropped (see "Authorship"
  below), and mainly non-English messages are dropped (see "Language").
- Exact dedupe after whitespace and case normalisation. Short messages
  (3 words or fewer) are not deduped but capped at 3 copies each.

The writer's own words are never rewritten: no spelling, casing or
shorthand normalisation.

Outputs:

| file | contents |
|---|---|
| `data/raw/cc_messages.jsonl` | filtered and deduped, **not scrubbed** |
| `data/stats/cc_stats.json` | counts only |
| `data/stats/cc_drops.jsonl` | every authorship or language drop and strip, with its reason and text |

Each record has `id`, `project` (the transcript's project directory name),
`session_id`, `timestamp`, `words`, `lines` and `text`.

### Authorship

The targets are text the person typed themselves, so `filters.author_check`
(in `irulan.pipeline.filters`) scores every blank-line-separated paragraph for signs of a paste:

- register: two or more capitalised sentences ending in full stops, a
  capitalised sentence with a closing full stop and no shorthand, em dashes,
  curly quotes, `Label: text` report leads (shorthand such as `rq`, `pls`,
  `yk`, `->` pulls the score down);
- LLM output: markdown headers, tables, bold, links, checkboxes, capitalised
  bullet lists, check marks, assistant phrases ("I've", "Here's", "Let me",
  "Next steps", "Your call");
- logs: lines sharing a `source:` prefix, build and package-manager output,
  exceptions and SQL, pytest dumps, `file:line:col`;
- quotes: Claude Code's quote bar, `> ` lines, a paragraph that is one long
  quotation, list items ending in `;`;
- transcripts, LinkedIn profiles and feeds, emails, `You are ...` templates,
  and lines that recur verbatim in 4+ sessions (computed over the whole
  extract before dedupe; lines with the writer's shorthand are their own
  stock phrases and are exempt).

A paragraph scoring at or above `par_threshold` (3) is stripped, along with
any capitalised full-stop sentence left next to it. The message is kept if
at least `min_keep_words` (5) prose words remain and what is left does not
itself read as a paste; otherwise it is dropped as `not_typed_<category>`.
Transcripts (filler words, speaker or timestamp lines), profiles (3+
LinkedIn/CV markers), emails (header lines) and messages over `max_words`
(600) are dropped whole.

### Language

English only for v1. `filters.language_check` counts English and German
stopwords (umlauts and `ß` count as German). A message with mostly German
markers loses its German paragraphs and is kept only if 5+ words of mainly
English prose remain; a few German words in an English message are fine.
`--keep-non-english` turns this off.

### Tuning

All thresholds live in `filters.AUTHOR_DEFAULTS`. Override them with a JSON
file (`--filter-config my.json`) or the common flags `--par-threshold`,
`--max-words` and `--min-keep-words`; `--no-author-filter` turns the
authorship pass off. The effective config is recorded in the stats file.

The defaults are tuned for a terse, lowercase, shorthand-heavy writer. If
you write in full sentences, raise `--par-threshold` or run with
`--no-author-filter`, then read `data/stats/cc_drops.jsonl` to see what was
dropped and why.

## Stats

```sh
irulan stats              # prints data/stats/cc_stats.json
```

Stats files written by earlier versions, which still had `clean` and `scrub`
sections, render the same way.

## Upload

```sh
irulan upload --dry-run --redact-secrets-locally --payload-out data/upload_preview.jsonl
irulan upload --redact-secrets-locally
```

`upload` converts each raw record to the `irulan.raw.v1` upload format
(`{id, text, meta}`, see [API.md](API.md#datasets-planned)):

- `project` is dropped, because the transcript directory name contains your
  local path.
- `session` is a hash of project and session id, used to split by session.

It then prints a count summary and asks before sending. Without a terminal
it needs `--yes`. `--dry-run` stops after the summary, and `--payload-out`
writes the exact bytes that would be sent.

The text is not scrubbed locally. `--redact-secrets-locally` does replace
obvious credentials with `<SECRET>` first (`irulan.redact`). It covers
private key blocks, connection-string passwords, JWTs, well-known API token
prefixes, bearer headers, `password=` and `api_key: "..."` assignments, and
"the password is ..." prose. Names, hosts, URLs, paths and IDs are left to
the platform.

On upload, kyora first scrubs the data. Only the scrubbed version is stored,
encrypted at rest. The original (unscrubbed) upload is not retained.

## After upload

```sh
irulan datasets status DS --wait scrubbed
irulan datasets review DS --out data/review.tsv       # names the platform found
irulan datasets approve DS --denylist data/denylist.local.txt
irulan datasets status DS --wait ready
irulan finetune create --dataset DS --name my-voice --wait
```

Add every name in `review.tsv` that should not reach training (people,
customers, companies, internal products and machines) to the denylist file,
one per line. The platform rescrubs with those terms before it generates
pairs.

## Tests

```sh
make test                 # or: python3 -I pipeline.py test
```

The tests use the stdlib unittest and invented fixtures only. The client
tests run against a local fake server (`tests/fakeserver.py`).
