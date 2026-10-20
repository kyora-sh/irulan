"""Rewrite text through an irulan model over OpenAI-compatible /chat/completions."""

# The system instruction the training data is built with. Inference must send
# the exact same text, so a model sees at inference what it saw in training.
# The platform's data prep must use the same version; bump the key (system_v2,
# ...) rather than editing a published text.
SYSTEM_PROMPTS = {
    "system_v1": ("Rewrite the text in the writer's own voice: terse, direct, short lines. "
                  "Keep the meaning, asks, facts and numbers. Return only the rewrite."),
}
SYSTEM_VERSION = "system_v1"
SYSTEM_PROMPT = SYSTEM_PROMPTS[SYSTEM_VERSION]
DEFAULT_MODEL = "irulan-swe"


def build_messages(text, system_version=SYSTEM_VERSION):
    if system_version not in SYSTEM_PROMPTS:
        raise ValueError("unknown system_version %r (known: %s)" % (system_version, ", ".join(sorted(SYSTEM_PROMPTS))))
    return [
        {"role": "system", "content": SYSTEM_PROMPTS[system_version]},
        {"role": "user", "content": text},
    ]


def chat_body(text, model=DEFAULT_MODEL, system_version=SYSTEM_VERSION, temperature=None, max_tokens=None, **extra):
    body = {"model": model, "messages": build_messages(text, system_version)}
    if temperature is not None:
        body["temperature"] = temperature
    if max_tokens is not None:
        body["max_tokens"] = max_tokens
    body.update(extra)
    return body


def completion_text(resp):
    try:
        content = resp["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        raise ValueError(("unexpected /chat/completions response: %r" % (resp,))[:300])
    if not isinstance(content, str):
        raise ValueError("completion has no text content")
    return content
