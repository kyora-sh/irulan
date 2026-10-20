# Data pipeline

`irulan-swe` learns to rewrite LLM-written text into the voice of a working
software engineer. Training pairs come from back-translation: real messages a
person wrote are the targets, and a model later writes a "Claude-style"
version of each as the input.

Step 1 extracts and scrubs the human side; it calls no model and spends
nothing. Step 2 generates the inputs, step 3 builds the training set and
step 4 trains and evaluates on Modal.

Everything runs on Python 3 with the standard library only (the training job
itself runs in a Modal image with its own pinned packages). Commands assume
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
- Text the person did not type is stripped or dropped (see "Authorship"
  below), and mainly non-English messages are dropped (see "Language").
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
| `data/stats/cc_drops.jsonl` | every authorship or language drop and strip, with its reason |

Each record has `id`, `project` (the transcript's project directory name),
`session_id`, `timestamp`, `words`, `lines` and `text`.

### Authorship

The targets are text the person typed themselves, so `filters.author_check`
scores every blank-line-separated paragraph for signs of a paste:

- register: two or more capitalised sentences ending in full stops, a
  capitalised sentence with a closing full stop and no shorthand, em dashes,
  curly quotes, `Label: text` report leads (the writer types short lowercase
  lines with shorthand such as `rq`, `pls`, `yk`, `->`, which pull the score
  down);
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

## Step 2: generating the input side

```sh
make sample                                   # fixed 50-message sample
make generate ARGS="--backend codex --model luna --run luna_s50 \
  --ids data/samples/sample50.jsonl --variants assigned"
make check-pairs ARGS="data/pairs/luna_s50.jsonl"
make review ARGS="--runs data/pairs/luna_s50.jsonl data/pairs/sol_s50.jsonl"
```

`irulan/generate.py` asks a model how an LLM assistant would have written
each clean message (same meaning, asks, facts and placeholders; assistant
register; no preamble; roughly 1.5-2x the length through phrasing only) and
writes `{id, backend, model, variant, input, target, meta}` lines to
`data/pairs/<run>.jsonl`. The prompt and its examples are invented and
versioned (`--prompt-version`, recorded in `meta.prompt_version`).

Backends:

| backend | what | notes |
|---|---|---|
| `codex` | `codex exec -m <model>` | runs in a fresh empty temp dir (`--tmp-root` / `IRULAN_TMP`) with `-s read-only --skip-git-repo-check --ephemeral --json`; the prompt goes on stdin and the last agent message is the output. Aliases: `sol` = `gpt-6.1-sol`, `luna` = `gpt-6-luna`. `--effort` sets `model_reasoning_effort` (default `low`). |
| `fake` | no network | deterministic output, for tests and dry runs. |

Variants: `plain` (prose, no markdown) and `formatted` (light markdown, only
for messages of 30+ words; shorter ones get the plain rules). `--variants
random` picks one per message with a seeded RNG (same choice for every
model), `both` writes both, `assigned` takes the `variant` field from
`--ids`.

Runs use a small worker pool (`--workers`, default 4) with retries and
exponential backoff, and are resumable: pairs already in the output file are
skipped and new ones are appended as they finish (a half-written last line
from a hard kill is dropped on resume). Every attempt is logged to
`data/pairs/<run>.log.jsonl` with latency, token usage and error kind.

Rate-limit errors (429, "rate limit", overloaded) get up to `--rate-retries`
(6) attempts with a backoff starting at `--rate-backoff` (30 s), doubling up
to 10 minutes. The run stops cleanly, starting no new calls and letting
running ones finish, on a usage-limit error, after `--max-rate-failures` (3)
jobs in a row gave up on rate limits, after `--max-consecutive-failures` (10)
failed jobs in a row, or on SIGTERM/SIGINT. It then exits with code 3; rerun
the same command to resume.

`irulan/check_pairs.py` flags inputs that add or drop placeholders, are
shorter than the target, exceed 4x its length, open with a preamble, look like
they answer the target's question, use markdown in a plain variant, or came
from a call that ran tools. `sample` picks 15 messages of 15-29 words, 20 of
30-59 and 15 of 60+ (skipping mostly-placeholder ones) and splits them evenly
between the two variants; `review` writes a side-by-side markdown file and a
per-model summary to `data/review/`.

## Studio: handwritten pairs

```sh
make studio              # or: python3 -I pipeline.py studio [--port 8765]
make studio-stats        # or: python3 -I pipeline.py studio --stats
```

A local page (stdlib `http.server` plus one static HTML file, bound to
127.0.0.1 only, no external assets) for writing pairs by hand. Pick a shape
(PR description, review comment, issue, Slack status or thread reply,
internal or customer email, design doc section, incident summary, release
notes, LinkedIn post, tweet or thread, commit message, standup, or a custom
one you describe), a length and an optional topic. Generate (Cmd/Ctrl+G)
makes one `codex exec` call with `gpt-6.1-sol` at low effort through
`generate.CodexBackend`, so it runs from an empty temp dir with the same
flags as step 2. The prompt (`studio.PROMPTS`, version `studio_v1`) asks for
assistant-register text with invented specifics (numbers, service names,
relative paths, versions); with no topic, the model invents one in a random
area and reports it on a `TOPIC:` line that is stored, not shown as text.
Only one generation runs at a time. If codex is missing or not logged in,
generation is disabled and "Paste my own" still works.

Write your version on the right and save (Cmd/Ctrl+Enter); Esc skips. Each
save appends one line to `data/handwritten/pairs.jsonl` (`--data` to
change):

```json
{"id": "hw_<16 hex of sha256(input, target)>", "source": "handwritten",
 "shape": "pr_description", "length": "medium", "topic": "...",
 "generator": {"backend": "codex", "model": "gpt-6.1-sol", "prompt_version": "studio_v1"},
 "input": "<assistant-style text>", "target": "<your version>", "created_at": "...Z"}
```

`generator` is `"pasted"` unless the saved input is exactly a text this
server generated. "Saved pairs" lists them newest first with edit and delete
(the file is rewritten through a temp file and `os.replace`); an edit
recomputes the id. Save and edit also run the pair through `prepare` and
warn if it would be dropped, for example when a URL or a denylisted name is
on one side only. "Stats" (or `--stats`) prints counts per shape, length and
generator, word counts and what `prepare` would keep.

Into training: `prepare --extra-pairs data/handwritten/pairs.jsonl`
(alongside or instead of `--pairs`). Handwritten records need no
`data/clean` session: each day's pairs are one session,
`handwritten/<date>`. The `shorter`, `too_long` and `markdown_in_plain`
checks do not drop them; everything else (rescrub, placeholder surrogates,
duplicates, the split) applies as for generated pairs.

## Step 3: the training set

```sh
make prepare ARGS="--pairs data/pairs/full_sol_v2.jsonl"
make prepare ARGS="--pairs data/pairs/full_sol_v2.jsonl --extra-pairs data/handwritten/pairs.jsonl"
```

`irulan/prepare.py` turns a pairs file into `data/train/` (git-ignored). It
reads the pairs line by line and skips lines that do not parse, so it can run
on a file a generation run is still appending to. Per pair, in order:

1. ids in `data/exclude_ids.local.txt` (git-ignored, one id per line, `#`
   comments) are dropped, as are ids no longer in `data/clean` (the extract
   dropped them since; `--keep-missing-session` keeps them as their own
   session).
2. Input and target are rescrubbed with the current `scrub/denylist.txt` and
   `irulan/scrub.py` rules, so a name added to the denylist after generation is
   removed from both sides without touching `data/clean`.
3. Targets under `--min-target-words` (2) words are dropped; short messages of
   2-4 words are kept.
4. `check_pairs` runs on the rescrubbed pair. Every flag drops the pair except
   those in `--allow-flags` (default `preamble,answered`: `preamble` fires on
   assistant lead-ins that mirror the target, such as "Okay, here is what is
   left:", and `answered` is a loose heuristic). An input that talks about
   the rewrite itself ("Here's the rewritten message:") is dropped regardless.
5. Exact-duplicate inputs (after whitespace normalisation) are dropped.
6. Placeholders are replaced with invented surrogates so the model never
   learns to emit `<NAME>` and friends: fake first names, `example.com` /
   `.org` / `.net` URLs and emails, RFC 5737 IPs (`192.0.2.x`, ...),
   `host-07.internal`-style hosts, made-up repo paths, hex ids and obviously
   fake secrets (`sk-test-...`). The n-th `<NAME>` of the input and the n-th
   `<NAME>` of the target get the same value, per type; a pair whose input and
   target hold a different number of some placeholder is dropped. Choices are
   seeded per pair id, and denylisted terms are never drawn.

The kept pairs are split 90/5/5 into train/val/test by session (project and
session id from `data/clean`), so a session never spans two splits: sessions
are shuffled with a fixed seed (`--seed`), val and test are filled to their
share, and the rest is train. Large sessions therefore tend to land in train.

Each split is chat-format jsonl:

```json
{"messages": [{"role": "system", "content": "<system_v1>"},
              {"role": "user", "content": "<assistant-style input>"},
              {"role": "assistant", "content": "<the writer's message>"}],
 "id": "...", "system_version": "system_v1", "variant": "plain"}
```

The system prompt is versioned (`--system`, `prepare.SYSTEM_PROMPTS`).
`data/train/stats.json` has counts only (per split, drop reasons, flags kept,
rescrub replacements, surrogates, a chars/3.5 token estimate and the share
over `--max-seq-len`); `data/train/drops.jsonl` has ids and reasons only.

## Step 4: training and eval on Modal

`train/modal_train.py` fine-tunes `Qwen/Qwen3.8-27B` (Apache 2.0, revision
pinned) with QLoRA on Modal. It never spends money by accident:

- `--dry-run` is local and needs neither Modal nor torch: it checks the
  schema of the three splits (roles, empty content, leftover placeholders, ids
  in two splits), counts examples over `--max-seq-len`, simulates batch
  padding and prints steps, hours and dollars under a stated throughput
  assumption. `--scale-to N` adds the estimate for N pairs.
- Every GPU job (train and eval) needs `--confirm-spend`.
- The Modal profile must be named explicitly (`--profile` or `MODAL_PROFILE`;
  the active profile is never used implicitly) and must exist in
  `~/.modal.toml`. `MODAL_TOKEN_ID` / `MODAL_TOKEN_SECRET` in the environment
  are refused, since they override the profile. `modal run` / `modal deploy`
  on the file are refused at import; jobs run as ephemeral apps
  (`app.run()`), and nothing here deploys.
- A profile whose name or workspace (looked up with
  `modal profile list --json`) matches `atira` is refused unless
  `--allow-atira-workspace` is passed. With it, the workspace check cannot be
  skipped, `--modal-env` must name `testing` or `dev` (never prod), and the
  launcher lists the apps there (`modal app list --json`, read-only) and
  refuses if an app named `irulan-train` is deployed or running.
- Hard Modal timeouts: train 90 min, `--smoke` 25 min, eval 30 min. Before
  starting, the launcher prints the estimate and the timeout's dollar
  ceiling. `--max-steps N` stops training after N optimizer steps; `--smoke`
  defaults it to 20 and refuses more than 50.

```sh
make train-dry-run ARGS="--scale-to 10100"           # local, free

M="uv run --no-project --with modal==1.4.3 python -I train/modal_train.py"
P="--profile <profile>"   # Atira: --profile atira --allow-atira-workspace --modal-env testing
$M upload $P                                          # data/train -> volume irulan-data:/train
$M train  $P --confirm-spend [--gpu H100|A100-80GB|B200] [--run NAME] [--max-steps N] [--detach]
$M eval   $P --confirm-spend --run NAME [--limit 200] [--no-adapter]

# smoke test: a separate training set and volume prefix, 20 steps, small val
python3 -I pipeline.py prepare --pairs <snapshot>.jsonl --out data/train_smoke
$M upload $P --data data/train_smoke --data-subdir smoke
$M train  $P --confirm-spend --smoke --data data/train_smoke --data-subdir smoke \
   --run smoke-1 --eval-steps 10 --val-limit 32 --logging-steps 1
$M eval   $P --confirm-spend --data-subdir smoke --run smoke-1 --limit 10
```

`--base-model` / `--base-revision` swap in a small model of the same family
(for example `Qwen/Qwen3.5-0.8B` on `--gpu L4`) to check the whole pipeline
for cents before the 27B run.

`train` uploads the data, fills the HF cache volume (`irulan-hf-cache`) on a
CPU container, then trains on one GPU (H100 80GB by default). Outputs go to
the `irulan-runs` volume under the run name: `checkpoints/` (latest and
best), `adapter/` (the best checkpoint's adapter), `epoch-N/` (the adapter at
the end of each epoch; an epoch end is also evaluated and checkpointed, so it
competes for best), `log.jsonl`, `config.json`,
`steps.jsonl` (seconds, tokens and peak memory per step) and `summary.json`
(load time, mean step time, tokens/s, peak memory, losses). Fetch them with
`modal volume get irulan-runs NAME/adapter`.

Training setup (all flags):

| setting | default |
|---|---|
| quantisation | 4-bit NF4, double quant, bf16 compute; `lm_head`, `in_proj_a`, `in_proj_b` stay bf16 |
| LoRA | r 32, alpha 32, dropout 0.05 (`--lora-r/--lora-alpha/--lora-dropout`) |
| LoRA targets | text layers only: `self_attn.{q,k,v,o}_proj`, `linear_attn.{in_proj_qkv,in_proj_z,out_proj}`, `mlp.{gate,up,down}_proj` (`--lora-modules`) |
| loss | assistant tokens only (the target and its end-of-turn token) |
| prompt | the model's chat template with `enable_thinking=False`, so the assistant turn starts after an empty think block, as at inference |
| length | `--max-seq-len` 1024; longer examples are dropped, never truncated |
| schedule | 2 epochs, lr 1e-4 cosine, 3% warmup, effective batch 32 (micro-batch 8 x accumulation 4), paged 8-bit AdamW |
| memory | gradient checkpointing, length-grouped batches (`--no-length-grouping` for plain random batches) |
| eval | val loss every `--eval-steps` (default: about ten per run); best by val loss |

Model notes: the checkpoint is `Qwen3_5ForConditionalGeneration` (a
vision-language model; 64 text layers, 3 Gated DeltaNet linear-attention
layers per full-attention layer). `AutoModelForCausalLM` maps it to
`Qwen3_5ForCausalLM`, which loads the text stack only and skips the vision
tower and the MTP head, so nothing needs freezing. It needs transformers
>= 5.8; the image pins transformers 5.18.0, peft 0.21.2, bitsandbytes 0.50.2,
accelerate 1.15.0, torch 2.13.0 and flash-linear-attention 0.5.2 (without it
the DeltaNet layers fall back to a much slower PyTorch path). Unsloth also
supports Qwen3.8 and is an option if speed matters.

`eval` loads the base model in 4-bit plus the run's adapter (or no adapter,
`--no-adapter`, as a baseline), generates rewrites for the test inputs
(non-thinking sampling: temperature 0.7, top-p 0.8, top-k 20) and writes
`data/eval/NAME/test_adapter.jsonl` (`id`, `input`, `model`, `target`), a
blind side-by-side `test_adapter_blind.md` (A/B in seeded random order) and
its key, `test_adapter_blind_key.json`.

`--inputs FILE.jsonl` (`{id, input}` per line) rewrites texts from outside the
splits instead, with the training system prompt (`--system`, default
`system_v1`); there is no target, so it writes only
`data/eval/NAME/inputs-FILE_adapter.jsonl`.
