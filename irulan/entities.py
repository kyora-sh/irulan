"""Find tokens in the scrubbed text that may be names of people, customers,
companies, products or machines, for a human to review into the denylist.

Output is a TSV sorted by frequency: count, term, example (scrubbed, <= 80 chars).
"""

import collections
import os
import re

WORDLIST_PATHS = ("/usr/share/dict/words", "/usr/share/dict/web2")

# Function words and very common words; these are never candidates, even
# when capitalised at the start of a sentence.
STOPWORDS = set("""
a about above after again against all also am an and any are aren't as at be because been before being
below between both but by can can't cannot could couldn't did didn't do does doesn't doing don't down
during each few for from further had hadn't has hasn't have haven't having he he'd he'll he's her here
here's hers herself him himself his how how's i i'd i'll i'm i've if in into is isn't it it's its itself
let's me more most mustn't my myself no nor not of off on once only or other ought our ours ourselves
out over own same shan't she she'd she'll she's should shouldn't so some such than that that's the their
theirs them themselves then there there's these they they'd they'll they're they've this those through
to too under until up very was wasn't we we'd we'll we're we've were weren't what what's when when's
where where's which while who who's whom why why's with won't would wouldn't you you'd you'll you're
you've your yours yourself yourselves yes yeah yep nope ok okay hey hi hello thanks thank please pls
just like also now then still even really actually maybe
""".split())

# Dev, chat and shorthand vocabulary that is not in a dictionary but is not a
# name either.
DEV_VOCAB = set("""
api apis app apps repo repos pr prs ci cd cli sdk ui ux url urls json yaml yml toml csv tsv md html css js
ts tsx jsx py rs go rb sql db dbs env envs config configs dev prod staging infra devops backend frontend
fullstack async await auth oauth jwt sso saml http https ssh tcp udp dns ip ips vpn vm vms gpu gpus cpu
cpus ram ssd hdd os macos linux ubuntu debian docker k8s kubectl helm npm pnpm bun yarn pip uv venv
node deno git github gitlab gh diff diffs repo regex regexes stdout stderr stdin tty tmux zsh bash
llm llms ai ml rag gpt eval evals benchmark benchmarks prompt prompts token tokens embedding embeddings
lgtm lfg ty tysm thx np nvm imo imho tbh afaik iirc btw fyi asap eta wip tldr tl dr lol lmao omg wtf
rn ig yk lmk kk idk idc ofc pls plz smth sth bc cuz tho ur u r ya yea yup nah mb ngl fr irl
gonna wanna gotta kinda sorta lemme dunno ain't y'all
todo todos fixme readme changelog codebase monorepo worktree worktrees subagent subagents
frontend backend middleware webhook webhooks endpoint endpoints localhost runtime runtimes
refactor refactors refactoring lint linter linting formatter typecheck typecheck typescript javascript
python rust golang java kotlin swift jsonl ndjson uuid uuids sha shas hash hashes cron crons
stdlib params param args arg kwargs bool bools int ints str enum enums struct structs impl dict dicts
tuple tuples iter init deinit ctor dtor src dist lib libs pkg pkgs util utils misc tmp temp
gh repo rebase rebased rebasing cherry pick merge merged merging upstream downstream
database databases dataset datasets metadata baseline baselines pdf pdfs docx xlsx pptx csv png jpg svg
perf max min avg demo demos e2e etc smh fkn fking fk abt wdyt rly bs alr obv prolly preesh uh hmm
hm ah oh ooh aha mhm sec secs ms ns min mins hr hrs tmrw tmr l8 l8r rq ap atm ppl sb smb pov ofc
w wo w/o b4 dm dms msg msgs pic pics img imgs vid vids ui fe be qa pm eng mvp poc roadmap
xhigh xlow v1 v2 v3 v4 v5 mcp sse ws wss grpc rpc rest graphql orm sdk sdks cdn dns tls ssl
email emails login logout signup onboarding dashboard dashboards analytics workflow workflows
subtask subtasks codex fullscreen screenshot screenshots changeset changesets lockfile
typo typos ack nit nits oof yikes damn shit fuck fucking crap bruh dope sick lit chill vibe vibes
lgtm ptal wip rfc rfcs okr okrs kpi kpis slo slos sla slas p50 p90 p95 p99 webapp
""".split())

# Public tools, vendors and models. Naming them is not sensitive, so they are
# not offered as denylist candidates.
PUBLIC_NAMES = set("""
python javascript typescript rust go golang java kotlin swift ruby node nodejs deno bun react nextjs next
vue svelte tailwind vite webpack docker kubernetes k8s helm terraform github gitlab bitbucket git
slack discord notion linear figma jira confluence vercel netlify cloudflare aws azure gcp google
microsoft apple amazon meta openai anthropic claude opus sonnet haiku fable gpt gemini gemma qwen llama
mistral deepseek codex copilot cursor chatgpt grok temporal postgres postgresql mysql sqlite redis
mongodb supabase firebase clerk stripe sentry grafana prometheus datadog modal bedrock vertex
huggingface pypi npm pnpm yarn pip uv macos linux ubuntu windows ios android chrome firefox safari
excel word powerpoint office libreoffice pdf playwright puppeteer pytest jest vitest fastapi django
flask express nestjs prisma drizzle zod pydantic celery kafka rabbitmq nginx caddy ngrok tailscale
keda azurite cosmos cosmosdb blob terraform ansible pulumi elastic elasticsearch opensearch
youtube twitter reddit hackernews hn wikipedia stackoverflow eu us uk ai llm api pr
""".split())


class Vocab:
    def __init__(self, words):
        self.words = words

    @classmethod
    def load(cls):
        words = set(STOPWORDS) | set(DEV_VOCAB)
        for p in WORDLIST_PATHS:
            if os.path.exists(p):
                with open(p, encoding="utf-8", errors="replace") as f:
                    for line in f:
                        w = line.strip()
                        # Lowercase entries are common words; capitalised ones
                        # in the system list are proper nouns, which we want to see.
                        if w and w.islower():
                            words.add(w)
        return cls(words)

    def known(self, w):
        w = w.lower()
        if w in self.words:
            return True
        if "'" in w:
            return self.known(w.split("'", 1)[0])
        if "-" in w:
            parts = [p for p in w.split("-") if p and not p.isdigit()]
            return bool(parts) and all(p in PUBLIC_NAMES or self.known(p) for p in parts)
        # crude inflection handling
        for suf, rep in (("s", ""), ("es", ""), ("ed", ""), ("ed", "e"), ("ing", ""), ("ing", "e"),
                         ("ies", "y"), ("'s", ""), ("ly", ""), ("er", ""), ("ers", "")):
            stem = w[: -len(suf)] + rep
            if w.endswith(suf) and len(stem) >= 3 and stem in self.words:
                return True
        return False


_DOTTED = re.compile(r"\S*[\w-]\.[A-Za-z][\w.-]*")  # file names, domains, e.g.
_TOKEN = re.compile(r"@?[A-Za-z][A-Za-z0-9]*(?:[-'][A-Za-z0-9]+)*")
_PLACEHOLDER = re.compile(r"<(?:SECRET|URL|EMAIL|IP|HOST|PATH|ID|NAME)>")
_CAMEL = re.compile(r"^[a-z]+[A-Z]|^[A-Z][a-z]+[A-Z]")
_SENT_END = re.compile(r"[.!?:\n]\s*$")


def _context(text, start, end, width=80):
    pad = max(0, (width - (end - start)) // 2)
    a = max(0, start - pad)
    b = min(len(text), a + width)
    a = max(0, b - width)
    s = text[a:b].replace("\t", " ").replace("\n", " ")
    return s.strip()


def find_candidates(texts, vocab=None, min_count=2):
    vocab = vocab or Vocab.load()
    total = collections.Counter()       # by lowercase form
    surface = collections.defaultdict(collections.Counter)
    cap_mid = collections.Counter()     # capitalised and not sentence-initial
    lower = collections.Counter()
    example = {}
    spans = collections.Counter()
    span_example = {}

    for text in texts:
        # Blank out placeholders, escaped newlines and dotted names without
        # moving offsets, so examples can be cut from the scrubbed text itself.
        blank = lambda m: " " * len(m.group(0))  # noqa: E731
        clean = _DOTTED.sub(blank, _PLACEHOLDER.sub(blank, text.replace("\\n", "  ")))
        run = []  # consecutive capitalised tokens for multi-word spans

        def flush():
            if len(run) >= 2:
                words = [w for w, _, _ in run]
                if not all(vocab.known(w) for w in words):
                    phrase = " ".join(words)
                    spans[phrase] += 1
                    if phrase not in span_example:
                        span_example[phrase] = _context(text, run[0][1], run[-1][2])
            run.clear()

        prev_end = 0
        for m in _TOKEN.finditer(clean):
            tok = m.group(0)
            between = clean[prev_end:m.start()]
            prev_end = m.end()
            if run and between.strip():
                flush()
            handle = tok.startswith("@")
            word = tok.lstrip("@")
            if len(word) < 2 or word.isdigit():
                flush()
                continue
            key = word.lower()
            initial = m.start() == 0 or bool(_SENT_END.search(clean[: m.start()]))
            is_cap = word[0].isupper()
            is_camel = bool(_CAMEL.match(word))
            if is_cap and not initial and key not in STOPWORDS:
                run.append((word, m.start(), m.end()))
            else:
                flush()
            if key in STOPWORDS or key in PUBLIC_NAMES:
                continue
            total[key] += 1
            surface[key][("@" if handle else "") + word] += 1
            if handle:
                cap_mid[key] += 3  # @handles are names by definition
            elif (is_cap and not initial and word[1:].islower()) or is_camel:
                cap_mid[key] += 1
            elif word.islower():
                lower[key] += 1
            if key not in example:
                example[key] = _context(text, m.start(), m.end())
        flush()

    rows = []
    for key, n in total.items():
        if n < min_count:
            continue
        known = vocab.known(key)
        if key in DEV_VOCAB:
            continue
        if known:
            # A dictionary word only shows up when it is mostly written as a
            # proper noun mid-sentence ("ask Linear", "on Modal").
            if not (cap_mid[key] >= 3 and cap_mid[key] > lower[key]):
                continue
        else:
            if len(key) < 3 and cap_mid[key] == 0:
                continue
        term = surface[key].most_common(1)[0][0]
        rows.append((n, term, example[key]))
    for phrase, n in spans.items():
        if n >= min_count:
            rows.append((n, phrase, span_example[phrase]))
    rows.sort(key=lambda r: (-r[0], r[1].lower()))
    return rows


def write_tsv(rows, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("count\tterm\texample\n")
        for n, term, ex in rows:
            f.write("%d\t%s\t%s\n" % (n, term, ex[:80]))
