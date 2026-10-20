"""The studio's shapes and its generation prompt (studio_v1).

A prompt asks for one piece of LLM-assistant-style text of a given shape and
length. With no topic, the model picks one in a random area and reports it on
a TOPIC line, so topics do not converge.
"""

import collections
import random
import re

PROMPT_VERSION = "studio_v1"
LENGTHS = ("short", "medium", "long")


class StudioError(Exception):
    def __init__(self, msg, status=400):
        super().__init__(msg)
        self.status = status


Shape = collections.namedtuple("Shape", "label words structure tone")

# words: (short, medium, long) word ranges.
SHAPES = collections.OrderedDict([
    ("pr_description", Shape(
        "PR description", ((80, 150), (150, 300), (300, 550)),
        "Markdown. A one-line summary, then `## Summary`, `## Changes` (a bulleted list), `## Testing` and "
        "optionally `## Notes` or `## Risks`. File paths, flags and config keys in inline code.",
        "confident, thorough, reviewer-friendly")),
    ("pr_review_comment", Shape(
        "PR review comment", ((30, 60), (60, 120), (120, 220)),
        "One inline review comment on a specific file and line: a brief acknowledgement, the concern, a "
        "suggested fix (optionally a short code block), and a closing question.",
        "collegial and hedged (\"nit:\", \"might be worth\", \"wdyt?\" written out in full)")),
    ("issue", Shape(
        "Issue", ((60, 120), (120, 250), (250, 450)),
        "Markdown with a title line, then `## Summary`, `## Steps to reproduce` (numbered), `## Expected "
        "behavior`, `## Actual behavior`, `## Environment` and optionally `## Possible cause`.",
        "precise, neutral, a little formal")),
    ("slack_status", Shape(
        "Slack status update", ((30, 60), (60, 120), (120, 200)),
        "A channel update with bolded labels (**Done:**, **In progress:**, **Blockers:**, **Next:**), short "
        "bullets under each, maybe one emoji code such as :white_check_mark:.",
        "upbeat, organised, reassuring")),
    ("slack_thread_reply", Shape(
        "Slack reply in a thread", ((20, 50), (50, 100), (100, 180)),
        "A reply to a teammate's question in a thread: an acknowledging opener (\"Great question!\"), the "
        "answer, a few bullets if there are several points, an offer to help further.",
        "friendly, helpful, slightly over-explained")),
    ("email_internal", Shape(
        "Email (internal)", ((80, 150), (150, 280), (280, 450)),
        "A `Subject:` line, a greeting, a context paragraph, a bulleted list of points or asks, next steps, "
        "and a sign-off with an invented first name.",
        "professional, warm, structured")),
    ("email_external", Shape(
        "Email (external / customer)", ((100, 180), (180, 300), (300, 450)),
        "A `Subject:` line, a greeting by an invented name, appreciation or an apology, a clear explanation "
        "without internal jargon, numbered next steps, an offer to set up a call, and a signature with an "
        "invented name and role.",
        "polite, empathetic, customer-facing")),
    ("design_doc_section", Shape(
        "Design doc section", ((150, 250), (250, 450), (450, 700)),
        "One section of a design doc (such as `## Proposed approach` or `## Alternatives considered`) with "
        "subheadings, trade-off bullets and optionally a small markdown table.",
        "analytical, balanced, thorough")),
    ("incident_summary", Shape(
        "Incident summary / postmortem", ((120, 220), (220, 400), (400, 650)),
        "Markdown: `## Summary`, `## Impact` (duration, share of requests, users affected), `## Timeline` "
        "(UTC times as bullets), `## Root cause`, `## Resolution`, `## Action items` (checkboxes with owners).",
        "blameless, calm, precise")),
    ("release_notes", Shape(
        "Release notes", ((60, 120), (120, 250), (250, 450)),
        "A version heading, then sections such as `### New`, `### Improvements`, `### Fixes` and "
        "`### Breaking changes`, bullets that start with a **bold feature name**.",
        "polished, enthusiastic, user-facing")),
    ("linkedin_post", Shape(
        "LinkedIn post", ((80, 150), (150, 250), (250, 400)),
        "A one-line hook, short paragraphs separated by blank lines, a short list of lessons, a reflective "
        "close, a question to readers and three to five hashtags.",
        "inspirational, humble-brag, reflective")),
    ("tweet_thread", Shape(
        "Tweet / thread", ((25, 45), (90, 160), (180, 300)),
        "Short: a single tweet under 280 characters. Medium and long: a thread of 3-5 or 6-9 tweets "
        "numbered `1/`, `2/`, ..., each under 280 characters, the first one a hook.",
        "punchy, engagement-seeking, a little breathless")),
    ("commit_message", Shape(
        "Commit message", ((15, 40), (40, 90), (90, 160)),
        "A conventional-commit subject line under 72 characters, a blank line, then a body wrapped at 72 "
        "columns with a short bulleted list of changes and a closing line on why.",
        "formal, complete, explanatory")),
    ("standup", Shape(
        "Standup update", ((25, 50), (50, 90), (90, 150)),
        "Bolded labels **Yesterday:**, **Today:** and **Blockers:**, each with one to three bullets.",
        "tidy, positive, complete sentences")),
    ("custom", Shape(
        "Custom", ((50, 100), (100, 250), (250, 450)),
        "", "")),
])

# Invented first names offered per call, so the model does not settle on one.
PEOPLE = (
    "Anouk", "Bilal", "Carmen", "Desmond", "Esra", "Farid", "Gwen", "Hiroshi", "Ilse", "Jarek",
    "Kemal", "Lucia", "Magnus", "Nadia", "Oren", "Paloma", "Quentin", "Rosa", "Stellan", "Tamsin",
    "Ugo", "Vera", "Wendell", "Ximena", "Yara", "Zoltan", "Amara", "Bastian", "Celeste", "Dov",
)

# Areas for an invented topic when none is given, so topics do not converge.
AREAS = (
    "payments and billing", "search indexing", "authentication and sessions", "CI pipelines",
    "mobile app releases", "data pipelines and ETL", "observability and alerting", "caching",
    "database migrations", "feature flags", "rate limiting", "background job queues",
    "frontend performance", "API versioning", "on-call tooling", "cloud cost reduction",
    "ML model serving", "access control and permissions", "file uploads and storage",
    "notifications and email delivery", "developer tooling", "dependency upgrades",
    "multi-region failover", "data retention and deletion", "load testing", "schema validation",
)

PROMPT_V1 = """You are helping build a style-transfer dataset. Write one realistic example of the text an LLM assistant such as Claude or ChatGPT produces when a software engineer asks it to draft the piece below. A person will later rewrite it in their own voice, so it must read unmistakably like assistant-drafted text.

Shape: {label}
Structure: {structure}
Tone: {tone}
Length: about {lo}-{hi} words.
{topic_line}

Write in the assistant register: polished, complete sentences; mild hedging ("It might be worth...", "I think we should consider..."); smooth transitions ("Additionally,", "That said,", "Overall,"); em dashes; bolded labels and tidy bullet lists wherever the shape allows them; a short summary or an offer to help at the end where it fits the shape.

Include concrete specifics that a rewrite has to keep: numbers (latencies, percentages, counts, dates, durations), names of invented services, components and people, relative file paths in a repository, config keys, version numbers and ticket numbers. Invent fresh, varied names and values for this piece; reuse nothing from these instructions. If people are mentioned, name them something like {people}.

Keep everything invented: no real companies, products or people, apart from widely used open-source tools and languages (Postgres, Kafka, Python and the like). Use no URLs, email addresses, IP addresses, absolute or home-directory paths, commit hashes, or anything that looks like a key or secret. Write as if everything were real: never call anything invented, fictional, hypothetical or an example in the text itself.

Do not use tools or run commands. {output_rule}"""

PROMPTS = {"studio_v1": PROMPT_V1}

OUTPUT_ONLY = "Output only the text itself: no preamble, no notes, no quotation marks around it."
OUTPUT_WITH_TOPIC = ("First write one line `TOPIC: <the topic you chose, under 15 words>`, then a line `---`, "
                     "then the text itself. Nothing else: no preamble, no notes.")
TOPIC_RE = re.compile(r"(?is)^\s*\**\s*topic\s*:\s*\**\s*(?P<topic>[^\n]*)\n(?:\s*-{3,}\s*\n)?(?P<text>.*)$")



def build_prompt(shape, length, topic="", custom="", area=None, version=PROMPT_VERSION, rng=None):
    """Prompt for one piece of assistant-style text. With no topic, the model
    invents one in `area` and reports it on a TOPIC line."""
    rng = rng or random
    if shape not in SHAPES:
        raise StudioError("unknown shape: %s" % shape)
    if length not in LENGTHS:
        raise StudioError("unknown length: %s" % length)
    s = SHAPES[shape]
    lo, hi = s.words[LENGTHS.index(length)]
    label, structure, tone = s.label, s.structure, s.tone
    if shape == "custom":
        custom = (custom or "").strip()
        if not custom:
            raise StudioError("the custom shape needs a description")
        label = custom
        structure = "whatever is typical for this kind of text"
        tone = "the usual assistant register for it"
    topic = (topic or "").strip()
    if topic:
        topic_line = "Topic: %s" % topic
        output_rule = OUTPUT_ONLY
    else:
        area = area or rng.choice(AREAS)
        topic_line = ("Topic: invent a plausible, specific software-engineering situation in the area of %s. "
                      "It must be generic and invented, not about any real company." % area)
        output_rule = OUTPUT_WITH_TOPIC
    people = " or ".join(rng.sample(PEOPLE, 3))
    return PROMPTS[version].format(label=label, structure=structure, tone=tone, lo=lo, hi=hi,
                                   topic_line=topic_line, output_rule=output_rule, people=people)


def split_topic(text):
    """(topic, text) from a reply that may open with a TOPIC line."""
    t = (text or "").strip()
    m = TOPIC_RE.match(t)
    if not m:
        return "", t
    return m.group("topic").strip().strip("*").strip(), m.group("text").strip()
