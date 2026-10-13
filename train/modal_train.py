"""QLoRA fine-tune of Qwen3.8-27B on Modal, plus an eval entrypoint.

Nothing here spends money unless it is told to twice: the dry run is local
and needs no Modal or torch, and every GPU job needs --confirm-spend on top
of an explicitly selected Modal profile. A profile on the Atira workspace is
refused unless --allow-atira-workspace is passed, and then only in a
non-production environment (--modal-env) where no app named irulan-train
is running or deployed. Every GPU function has a hard timeout (train 90 min,
smoke 25 min, eval 30 min) and the launcher prints the estimate and the
timeout's dollar ceiling before starting. Jobs run as ephemeral apps
(`app.run()`, like `modal run`); nothing here ever deploys.

    # local, free: validate data/train and print the cost estimate
    python3 -I train/modal_train.py train --dry-run [--scale-to 10100]

    # real runs need the modal client (the CLI's own venv, or uv):
    uv run --no-project --with modal==1.4.3 python -I train/modal_train.py \\
        upload --profile <profile>
    uv run --no-project --with modal==1.4.3 python -I train/modal_train.py \\
        train --profile <profile> --confirm-spend [--gpu H100] [--run NAME] [--max-steps N]
    uv run --no-project --with modal==1.4.3 python -I train/modal_train.py \\
        eval --profile <profile> --confirm-spend --run NAME [--limit 200]
    # on the Atira workspace, add: --allow-atira-workspace --modal-env testing

`modal run` / `modal deploy` on this file are refused on purpose: the
launcher checks the profile before the Modal client is even imported.

Model notes (from the Hub config, revision pinned below):
- `Qwen3_5ForConditionalGeneration`, model_type `qwen3_5`, a vision-language
  model whose text stack is 64 layers in a 3:1 pattern of Gated DeltaNet
  linear attention (`linear_attn.in_proj_qkv/in_proj_z/in_proj_a/in_proj_b/
  out_proj`) and gated full attention (`self_attn.q/k/v/o_proj`), each with a
  SwiGLU MLP (`mlp.gate/up/down_proj`); hidden 5120, vocab 248320, untied
  lm_head, plus one MTP layer for speculative decoding.
- `AutoModelForCausalLM` maps `qwen3_5` to `Qwen3_5ForCausalLM`, which
  builds the text model only, strips the `model.language_model.` prefix and
  skips `model.visual.*` and `mtp.*` on load: the vision tower is never
  materialised, so there is nothing to freeze.
- Needs transformers >= 5.8 (the config was written by 5.8.0.dev0); the
  DeltaNet layers use flash-linear-attention's Triton kernels when it is
  installed and fall back to a much slower PyTorch reference otherwise.
- bitsandbytes NF4 quantises every nn.Linear except the ones skipped below;
  peft LoRA wraps bnb Linear4bit layers as usual.
- Unsloth supports Qwen3.8 (unsloth/Qwen3.8-27B-unsloth-bnb-4bit) and would
  be faster, but plain transformers + peft + bitsandbytes is the default.
"""

import argparse
import hashlib
import json
import math
import os
import random
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

BASE_MODEL = "Qwen/Qwen3.8-27B"
BASE_REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
APP_NAME = "irulan-train"
DATA_VOLUME = "irulan-data"
HF_VOLUME = "irulan-hf-cache"
RUNS_VOLUME = "irulan-runs"
DATA_MOUNT, HF_MOUNT, RUNS_MOUNT = "/vol/data", "/vol/hf", "/vol/runs"

# Exact versions for the Modal image (nothing here is installed locally).
PINNED = (
    "torch==2.13.0",
    "transformers==5.18.0",
    "tokenizers==0.23.3",
    "peft==0.21.2",
    "bitsandbytes==0.50.2",
    "accelerate==1.15.0",
    "flash-linear-attention==0.5.2",
    "fla-core==0.5.2",
    "huggingface-hub==2.1.1",
    "hf-xet==1.6.0",
)
MODAL_CLIENT = "modal==1.4.3"

ROLES = ("system", "user", "assistant")
SPLITS = ("train", "val", "test")
PLACEHOLDER_RE = re.compile(r"<(?:SECRET|URL|EMAIL|IP|HOST|PATH|ID|NAME)>")
CHARS_PER_TOKEN = 3.5
TEMPLATE_OVERHEAD_TOKENS = 16

# LoRA targets: text-layer attention (full and linear) and MLP projections.
# in_proj_a / in_proj_b are 5120 -> 48 gate projections; they stay in bf16
# (skipped by bnb) and untouched by LoRA.
DEFAULT_LORA_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj", "in_proj_qkv", "in_proj_z",
                        "out_proj", "gate_proj", "up_proj", "down_proj")
BNB_SKIP_MODULES = ("lm_head", "in_proj_a", "in_proj_b")

# Throughput assumptions for the estimate, in trained tokens per second
# (real plus padding) for QLoRA on a 27B dense model: NF4 weights, bf16
# compute, gradient checkpointing, SDPA, fla kernels, micro-batch 8 of short
# sequences. Rough guesses until a first run measures it.
GPUS = {
    "H100": {"usd_per_hour": 3.95, "tokens_per_s": 1000},
    "A100-80GB": {"usd_per_hour": 2.50, "tokens_per_s": 450},
    "B200": {"usd_per_hour": 6.25, "tokens_per_s": 1800},
    # only for a pipeline check with a small --base-model; far too small for the 27B
    "L4": {"usd_per_hour": 0.80, "tokens_per_s": 3000},
}
# Hard Modal timeouts for the GPU functions, in minutes.
TRAIN_TIMEOUT_MIN = 90
SMOKE_TIMEOUT_MIN = 25
EVAL_TIMEOUT_MIN = 30
SMOKE_MAX_STEPS = 50  # --smoke refuses a longer run
SETUP_MINUTES = 12   # GPU minutes for loading + quantising the weights and kernel warm-up
EVAL_COST_FACTOR = 1 / 3  # an eval token costs about a third of a training token (forward only)

# The profile guard: a profile whose name or workspace matches this is refused
# unless --allow-atira-workspace is passed; then only these environments may
# be used (never prod), and APP_NAME must not collide with an existing app.
BLOCKED_WORKSPACE_RE = re.compile(os.environ.get("IRULAN_BLOCKED_MODAL_RE", r"atira"), re.I)
ATIRA_ALLOWED_ENVS = ("testing", "dev")


class Refused(SystemExit):
    def __init__(self, msg):
        super().__init__("refused: " + msg)


# ---------------------------------------------------------------------------
# Profile guard (stdlib only; runs before the modal client is imported)

def modal_profiles(path=None):
    """Profile names in ~/.modal.toml (never the tokens)."""
    path = path or os.path.expanduser("~/.modal.toml")
    if not os.path.exists(path):
        return []
    try:
        import tomllib
    except ImportError:  # Python < 3.11
        names = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                m = re.match(r"^\s*\[([^\]]+)\]\s*$", line)
                if m:
                    names.append(m.group(1).strip().strip('"'))
        return names
    with open(path, "rb") as f:
        return [k for k, v in tomllib.load(f).items() if isinstance(v, dict)]


def lookup_workspaces():
    """{profile: workspace} via `modal profile list --json`, or None if the
    CLI is missing or the lookup fails. Read-only."""
    try:
        out = subprocess.run(["modal", "profile", "list", "--json"], capture_output=True,
                             text=True, timeout=60, check=True).stdout
        return {p["name"]: p.get("workspace") for p in json.loads(out)}
    except (OSError, subprocess.SubprocessError, ValueError, KeyError, TypeError):
        return None


def check_profile(profile, env=None, profiles=None, workspaces=None, skip_workspace_check=False,
                  allow_atira=False):
    """Return the profile to use, or raise Refused. The profile must be named
    explicitly (--profile or MODAL_PROFILE) and exist; unless allow_atira is
    set, neither its name nor its workspace may look like Atira's. Token env
    vars are refused because they override the profile and hide which
    workspace is used."""
    env = os.environ if env is None else env
    if env.get("MODAL_TOKEN_ID") or env.get("MODAL_TOKEN_SECRET"):
        raise Refused("MODAL_TOKEN_ID/MODAL_TOKEN_SECRET are set; they override the profile. Unset them.")
    profile = profile or env.get("MODAL_PROFILE")
    if not profile:
        raise Refused("no Modal profile selected. Pass --profile NAME or set MODAL_PROFILE "
                      "to a personal (non-Atira) profile; the active profile is never used implicitly.")
    if BLOCKED_WORKSPACE_RE.search(profile) and not allow_atira:
        raise Refused("profile %r looks like an Atira profile; pass --allow-atira-workspace only "
                      "if running there was approved." % profile)
    profiles = modal_profiles() if profiles is None else profiles
    if profile not in profiles:
        raise Refused("profile %r is not in ~/.modal.toml (found: %s)." % (profile, ", ".join(profiles) or "none"))
    if not skip_workspace_check:
        workspaces = lookup_workspaces() if workspaces is None else workspaces
        if workspaces is None or profile not in workspaces:
            raise Refused("could not confirm the workspace of profile %r (pass "
                          "--skip-workspace-check only if you know it)." % profile)
        ws = workspaces[profile] or ""
        if BLOCKED_WORKSPACE_RE.search(ws) and not allow_atira:
            raise Refused("profile %r points at workspace %r, which looks like Atira's; pass "
                          "--allow-atira-workspace only if running there was approved." % (profile, ws))
    return profile


def is_atira(profile, workspace=None):
    return bool(BLOCKED_WORKSPACE_RE.search(profile or "") or BLOCKED_WORKSPACE_RE.search(workspace or ""))


def check_atira_env(modal_env):
    """On the Atira workspace the environment must be named and allowlisted."""
    if not modal_env:
        raise Refused("on the Atira workspace pass --modal-env (one of %s); the profile's "
                      "default environment is never used there." % ", ".join(ATIRA_ALLOWED_ENVS))
    if modal_env not in ATIRA_ALLOWED_ENVS:
        raise Refused("environment %r is not allowed on the Atira workspace (allowed: %s)." % (
            modal_env, ", ".join(ATIRA_ALLOWED_ENVS)))
    return modal_env


def list_apps(modal_env):
    """[{name, state, id}] via `modal app list --json` (read-only), or None."""
    cmd = ["modal", "app", "list", "--json"] + (["--env", modal_env] if modal_env else [])
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=120, check=True,
                             env=dict(os.environ, NO_COLOR="1")).stdout
        return [{"name": a.get("Description"), "state": a.get("State"), "id": a.get("App ID")}
                for a in json.loads(out)]
    except (OSError, subprocess.SubprocessError, ValueError, AttributeError, TypeError):
        return None


def check_app_collision(apps, app_name=None):
    """Refuse if an app with our name is deployed or running (stopped is fine)."""
    app_name = app_name or APP_NAME
    if apps is None:
        raise Refused("could not list the apps in the environment; not starting %s." % app_name)
    live = [a for a in apps if a.get("name") == app_name and "stopped" not in (a.get("state") or "")]
    if live:
        raise Refused("an app named %r already exists (%s); not starting another one." % (
            app_name, ", ".join("%s %s" % (a.get("id"), a.get("state")) for a in live)))
    return True


def _early_profile_from_argv(argv):
    """Export --profile as MODAL_PROFILE (and --modal-env as
    MODAL_ENVIRONMENT) before modal is imported (the client reads the
    profile once, at import time)."""
    for flag, var in (("--profile", "MODAL_PROFILE"), ("--modal-env", "MODAL_ENVIRONMENT")):
        for i, a in enumerate(argv):
            if a == flag and i + 1 < len(argv):
                os.environ[var] = argv[i + 1]
            elif a.startswith(flag + "="):
                os.environ[var] = a.split("=", 1)[1]


if __name__ == "__main__":
    _early_profile_from_argv(sys.argv[1:])
    try:
        import modal  # noqa: E402
    except ImportError:
        modal = None
elif os.path.basename(sys.argv[0] if sys.argv else "") == "modal" and os.environ.get("MODAL_IS_REMOTE") != "1":
    # Imported by the local `modal` CLI (modal run / deploy / serve): refuse
    # before any app is created on whatever profile happens to be active.
    raise Refused("use `python -I train/modal_train.py ...` (it checks the profile and "
                  "--confirm-spend first); `modal run`/`modal deploy` on this file are disabled.")
else:
    try:
        import modal  # noqa: E402  (inside the container, or tests with modal installed)
    except ImportError:
        modal = None


# ---------------------------------------------------------------------------
# Data checks and estimates (stdlib only)

def est_tokens(text_or_texts):
    texts = [text_or_texts] if isinstance(text_or_texts, str) else text_or_texts
    return int(round(sum(len(t) for t in texts) / CHARS_PER_TOKEN))


def validate_split(path):
    """Return (records, problems). Checks the chat schema the trainer expects."""
    recs, problems = [], []
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except ValueError:
                problems.append("%s:%d: not JSON" % (os.path.basename(path), n))
                continue
            msgs = r.get("messages") if isinstance(r, dict) else None
            if not isinstance(msgs, list) or [m.get("role") for m in msgs if isinstance(m, dict)] != list(ROLES):
                problems.append("%s:%d: messages must be system, user, assistant" % (os.path.basename(path), n))
                continue
            if not all(isinstance(m.get("content"), str) and m["content"].strip() for m in msgs):
                problems.append("%s:%d: empty or non-string content" % (os.path.basename(path), n))
                continue
            if any(PLACEHOLDER_RE.search(m["content"]) for m in msgs[1:]):
                problems.append("%s:%d: placeholder token left in the text" % (os.path.basename(path), n))
            recs.append(r)
    return recs, problems


def record_tokens(r):
    return est_tokens([m["content"] for m in r["messages"]]) + TEMPLATE_OVERHEAD_TOKENS


def padding_factor(lengths, micro_batch, grouped=True, seed=0):
    """Padded tokens / real tokens for one epoch of batches, simulated."""
    if not lengths:
        return 1.0
    rng = random.Random(seed)
    if grouped:
        batches = length_grouped_batches(lengths, micro_batch, rng)
    else:
        idx = list(range(len(lengths)))
        rng.shuffle(idx)
        batches = [idx[i:i + micro_batch] for i in range(0, len(idx), micro_batch)]
    padded = sum(max(lengths[i] for i in b) * len(b) for b in batches)
    return padded / sum(lengths)


def length_grouped_batches(lengths, micro_batch, rng, mega=64):
    """Shuffle, sort within mega-batches of mega*micro_batch by length, cut
    into micro-batches and shuffle those: little padding, still random."""
    idx = list(range(len(lengths)))
    rng.shuffle(idx)
    size = mega * micro_batch
    batches = []
    for i in range(0, len(idx), size):
        chunk = sorted(idx[i:i + size], key=lambda j: lengths[j], reverse=True)
        batches.extend(chunk[k:k + micro_batch] for k in range(0, len(chunk), micro_batch))
    rng.shuffle(batches)
    return batches


def estimate(stats, gpu="H100", epochs=2, effective_batch=32, micro_batch=8, eval_every=0,
             scale=1.0, tokens_per_s=None, max_steps=0, val_limit=0):
    """Steps, time and dollars from the per-split token counts. max_steps > 0
    stops the run early (the estimate scales the trained tokens down);
    val_limit > 0 evaluates on that many val examples only."""
    g = GPUS[gpu]
    tps = tokens_per_s or g["tokens_per_s"]
    n_train = stats["train"]["n"] * scale
    train_tokens = stats["train"]["tokens"] * scale * stats["train"]["pad"]
    val_tokens = stats["val"]["tokens"] * scale * stats["val"]["pad"]
    grad_accum = max(1, round(effective_batch / micro_batch))
    steps_per_epoch = math.ceil(n_train / (micro_batch * grad_accum))
    total_steps = steps_per_epoch * epochs
    full_steps = total_steps
    if max_steps and max_steps < total_steps:
        total_steps = max_steps
    eval_steps = eval_every or default_eval_steps(total_steps)
    n_evals = total_steps // eval_steps + 1
    if val_limit and stats["val"]["n"]:
        val_tokens *= min(1.0, val_limit / (stats["val"]["n"] * scale))
    work_tokens = train_tokens * epochs * total_steps / full_steps + val_tokens * n_evals * EVAL_COST_FACTOR
    hours = work_tokens / tps / 3600 + SETUP_MINUTES / 60
    return {
        "gpu": gpu, "assumed_tokens_per_s": tps, "usd_per_hour": g["usd_per_hour"],
        "train_examples": int(n_train), "epochs": epochs, "micro_batch": micro_batch,
        "grad_accum": grad_accum, "effective_batch": micro_batch * grad_accum,
        "steps_per_epoch": steps_per_epoch, "total_steps": total_steps, "max_steps": max_steps,
        "eval_steps": eval_steps, "evals": n_evals,
        "train_tokens_per_epoch_padded": int(train_tokens), "work_tokens": int(work_tokens),
        "hours": round(hours, 2), "usd": round(hours * g["usd_per_hour"], 2),
        "usd_range": [round((work_tokens / (tps * k) / 3600 + SETUP_MINUTES / 60) * g["usd_per_hour"], 2)
                      for k in (1.5, 0.6)],
    }


def default_eval_steps(total_steps):
    """About ten evals per run, at least every 10 steps apart."""
    return max(10, total_steps // 10)


def summarise_data(data_dir, max_seq_len, micro_batch):
    """Validate the three splits and return (stats, problems)."""
    stats, problems, ids = {}, [], {}
    for split in SPLITS:
        path = os.path.join(data_dir, "%s.jsonl" % split)
        if not os.path.exists(path):
            problems.append("missing %s" % path)
            stats[split] = {"n": 0, "tokens": 0, "pad": 1.0, "over": 0, "max": 0}
            continue
        recs, probs = validate_split(path)
        problems.extend(probs)
        for r in recs:
            rid = r.get("id")
            if rid is not None:
                if rid in ids and ids[rid] != split:
                    problems.append("id %s is in both %s and %s" % (rid, ids[rid], split))
                ids[rid] = split
        systems = {r["messages"][0]["content"] for r in recs}
        if len(systems) > 1:
            problems.append("%s: %d different system prompts" % (split, len(systems)))
        lengths = [record_tokens(r) for r in recs]
        fitting = [n for n in lengths if n <= max_seq_len]
        stats[split] = {
            "n": len(fitting), "n_all": len(recs), "tokens": sum(fitting),
            "target_tokens": sum(est_tokens(r["messages"][2]["content"]) + 1 for r in recs),
            "over": len(lengths) - len(fitting), "max": max(lengths) if lengths else 0,
            "pad": padding_factor(fitting, micro_batch) if fitting else 1.0,
            "mean": round(sum(lengths) / len(lengths), 1) if lengths else 0,
        }
    return stats, problems


def print_estimate(est, scale_note=""):
    print("cost estimate%s:" % scale_note)
    print("  %d train examples x %d epochs, micro-batch %d x grad-accum %d = effective batch %d" % (
        est["train_examples"], est["epochs"], est["micro_batch"], est["grad_accum"], est["effective_batch"]))
    print("  %d steps (%d per epoch%s), eval every %d steps (%d evals)" % (
        est["total_steps"], est["steps_per_epoch"],
        ", stopped by --max-steps" if est.get("max_steps") and est["max_steps"] == est["total_steps"] else "",
        est["eval_steps"], est["evals"]))
    print("  ~%.2fM padded train tokens per epoch, ~%.2fM token-equivalents of work" % (
        est["train_tokens_per_epoch_padded"] / 1e6, est["work_tokens"] / 1e6))
    print("  assumption: %d tokens/s on one %s (QLoRA 27B, NF4, grad checkpointing), plus %d min setup" % (
        est["assumed_tokens_per_s"], est["gpu"], SETUP_MINUTES))
    print("  => ~%.2f GPU-hours, ~$%.2f at $%.2f/h (range $%.2f-$%.2f for 1.5x faster to 0.6x slower)" % (
        est["hours"], est["usd"], est["usd_per_hour"], est["usd_range"][0], est["usd_range"][1]))


# ---------------------------------------------------------------------------
# Remote side (only defined when the modal client is importable)

def lora_target_regex(modules):
    names = "|".join(re.escape(m) for m in modules)
    # Text decoder layers only (model.layers.N or model.language_model.layers.N);
    # the vision tower (model.visual.*) and the MTP head (mtp.*) never match.
    return r"^model\.(?:language_model\.)?layers\.\d+\.(?:self_attn|linear_attn|mlp)\.(?:%s)$" % names


def _encode(tok, rec, max_len):
    """Prompt tokens are masked out of the loss; the assistant turn up to and
    including <|im_end|> is trained."""
    msgs = rec["messages"]
    prompt = tok.apply_chat_template(msgs[:2], tokenize=False, add_generation_prompt=True,
                                     enable_thinking=False)
    p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    c_ids = tok(msgs[2]["content"] + "<|im_end|>", add_special_tokens=False)["input_ids"]
    ids = p_ids + c_ids
    if len(ids) > max_len:
        return None
    return {"input_ids": ids, "labels": [-100] * len(p_ids) + c_ids}


def _read_jsonl(path, limit=0):
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
                if limit and len(out) >= limit:
                    break
    return out


def _load_model(cfg, for_training):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    import time
    from importlib.metadata import version
    print("env: %s, %s" % (torch.cuda.get_device_name(0), ", ".join(
        "%s %s" % (p, version(p)) for p in ("torch", "transformers", "peft", "bitsandbytes", "fla-core"))))
    t0 = time.time()
    base, rev = cfg.get("base_model") or BASE_MODEL, cfg.get("base_revision") or BASE_REVISION
    tok = AutoTokenizer.from_pretrained(base, revision=rev)
    if tok.pad_token_id is None:
        tok.pad_token = "<|endoftext|>"
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True, bnb_4bit_quant_storage=torch.bfloat16,
        llm_int8_skip_modules=list(BNB_SKIP_MODULES))
    model = AutoModelForCausalLM.from_pretrained(
        base, revision=rev, quantization_config=bnb, dtype=torch.bfloat16,
        attn_implementation="sdpa", device_map={"": 0})
    cfg["_load_s"] = round(time.time() - t0, 1)
    print("loaded %s %s (%s) in %.0f s, %.1f GB on GPU" % (base, type(model).__name__, rev[:8],
                                                       cfg["_load_s"], torch.cuda.memory_allocated() / 1e9))
    try:
        _report_fast_path(model)
    except Exception as e:  # diagnostics only
        print("kernels: could not inspect (%s)" % e)
    if any(".visual." in n or n.startswith("visual.") for n, _ in model.named_modules()):
        print("note: vision tower present; it is frozen and not targeted by LoRA")
    if for_training:
        model.config.use_cache = False
        for p in model.parameters():
            p.requires_grad_(False)
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    return model, tok


def _report_fast_path(model):
    """Print whether the DeltaNet layers got the fla / causal-conv1d kernels."""
    import importlib
    mod = importlib.import_module(type(model).__module__)
    flags = {n: getattr(mod, n) for n in dir(mod)
             if any(k in n.lower() for k in ("fast_path", "causal_conv1d", "gated_delta", "fla_"))}
    shown = {n: (v if isinstance(v, (bool, type(None))) else ("available" if v else v))
             for n, v in flags.items() if not isinstance(v, type(mod))}
    print("kernels: %s" % json.dumps({k: str(v) if not isinstance(v, (bool, type(None))) else v
                                      for k, v in shown.items()}))
    for name, m in model.named_modules():
        if name.endswith("linear_attn"):
            print("linear_attn impl: %s" % json.dumps({
                k: getattr(getattr(m, k, None), "__module__", None)
                for k in ("causal_conv1d_fn", "chunk_gated_delta_rule", "recurrent_gated_delta_rule")}))
            break


def _check_template(tok, rec):
    msgs = rec["messages"]
    full = tok.apply_chat_template(msgs, tokenize=False, enable_thinking=False)
    prompt = tok.apply_chat_template(msgs[:2], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    expected = prompt + msgs[2]["content"] + "<|im_end|>"
    if not full.startswith(expected):
        print("warning: the chat template renders the assistant turn differently from the "
              "generation prompt; training uses prompt + target + <|im_end|>, as at inference")


def run_training(cfg):
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import Trainer, TrainerCallback, TrainingArguments

    if cfg.get("confirm_spend") is not True:
        raise RuntimeError("refused: training needs confirm_spend=True")
    run_dir = os.path.join(RUNS_MOUNT, cfg["run"])
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    model, tok = _load_model(cfg, for_training=True)
    data = {}
    for split in ("train", "val"):
        rows = _read_jsonl(os.path.join(DATA_MOUNT, cfg["data_subdir"], "%s.jsonl" % split),
                           cfg.get("val_limit", 0) if split == "val" else 0)
        if split == "train" and rows:
            _check_template(tok, rows[0])
        enc = [_encode(tok, r, cfg["max_seq_len"]) for r in rows]
        data[split] = [e for e in enc if e is not None]
        n_tok = sum(len(e["input_ids"]) for e in data[split])
        print("%s: %d examples, %d dropped over %d tokens, %d tokens (%.0f%% trained)" % (
            split, len(data[split]), len(enc) - len(data[split]), cfg["max_seq_len"], n_tok,
            100 * sum(sum(1 for x in e["labels"] if x != -100) for e in data[split]) / max(1, n_tok)))

    lora = LoraConfig(r=cfg["lora_r"], lora_alpha=cfg["lora_alpha"], lora_dropout=cfg["lora_dropout"],
                      bias="none", task_type="CAUSAL_LM", target_modules=lora_target_regex(cfg["lora_modules"]))
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    pad = tok.pad_token_id

    def collate(batch):
        m = max(len(b["input_ids"]) for b in batch)
        ids = [b["input_ids"] + [pad] * (m - len(b["input_ids"])) for b in batch]
        att = [[1] * len(b["input_ids"]) + [0] * (m - len(b["input_ids"])) for b in batch]
        lab = [b["labels"] + [-100] * (m - len(b["labels"])) for b in batch]
        return {"input_ids": torch.tensor(ids), "attention_mask": torch.tensor(att), "labels": torch.tensor(lab)}

    micro, accum = cfg["micro_batch"], max(1, round(cfg["effective_batch"] / cfg["micro_batch"]))
    steps_per_epoch = math.ceil(len(data["train"]) / (micro * accum))
    total_steps = steps_per_epoch * cfg["epochs"]
    if cfg.get("max_steps") and cfg["max_steps"] < total_steps:
        total_steps = cfg["max_steps"]
    eval_steps = cfg["eval_steps"] or default_eval_steps(total_steps)
    print("steps: %d (%d per epoch%s), eval/save every %d" % (
        total_steps, steps_per_epoch, ", --max-steps" if cfg.get("max_steps") else "", eval_steps))

    runs_vol = volumes[RUNS_MOUNT]
    import time
    meter = {"real": 0, "padded": 0, "steps": []}

    class CommitVolume(TrainerCallback):
        def on_save(self, args, state, control, **kw):
            runs_vol.commit()

        def on_log(self, args, state, control, logs=None, **kw):
            with open(os.path.join(run_dir, "log.jsonl"), "a") as f:
                f.write(json.dumps(dict(logs or {}, step=state.global_step)) + "\n")

        def on_step_begin(self, args, state, control, **kw):
            meter["t"], meter["real"], meter["padded"] = time.time(), 0, 0

        def on_step_end(self, args, state, control, **kw):
            dt = time.time() - meter["t"]
            row = {"step": state.global_step, "sec": round(dt, 2), "real_tokens": meter["real"],
                   "padded_tokens": meter["padded"], "tok_s": round(meter["padded"] / max(dt, 1e-6)),
                   "real_tok_s": round(meter["real"] / max(dt, 1e-6)),
                   "peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 1)}
            meter["steps"].append(row)
            print("step " + json.dumps(row))
            with open(os.path.join(run_dir, "steps.jsonl"), "a") as f:
                f.write(json.dumps(row) + "\n")

    lengths = [len(e["input_ids"]) for e in data["train"]]

    class LengthGroupedTrainer(Trainer):
        """Batches of similar length (see length_grouped_batches), so short
        messages are not padded to the longest one in a random batch."""
        epoch = 0

        def training_step(self, model, inputs, *a, **kw):
            m = inputs.get("attention_mask")
            if m is not None:
                meter["real"] += int(m.sum())
                meter["padded"] += int(m.numel())
            return super().training_step(model, inputs, *a, **kw)

        def get_train_dataloader(self):
            if not cfg["length_grouping"]:
                return super().get_train_dataloader()
            from torch.utils.data import DataLoader
            outer = self

            class Sampler:
                def __iter__(self):
                    rng = random.Random("%s:%d" % (cfg["seed"], outer.epoch))
                    outer.epoch += 1
                    return iter(length_grouped_batches(lengths, micro, rng))

                def __len__(self):
                    return math.ceil(len(lengths) / micro)

            dl = DataLoader(self.train_dataset, batch_sampler=Sampler(), collate_fn=collate,
                            num_workers=2, pin_memory=True)
            return self.accelerator.prepare(dl)

    args = TrainingArguments(
        output_dir=os.path.join(run_dir, "checkpoints"),
        num_train_epochs=cfg["epochs"], max_steps=cfg.get("max_steps") or -1,
        learning_rate=cfg["lr"], lr_scheduler_type="cosine",
        warmup_steps=max(1, math.ceil(cfg["warmup_ratio"] * total_steps)),
        per_device_train_batch_size=micro, per_device_eval_batch_size=micro,
        gradient_accumulation_steps=accum, gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=True, optim="paged_adamw_8bit", weight_decay=0.0, max_grad_norm=1.0,
        logging_steps=cfg.get("logging_steps", 5), eval_strategy="steps", eval_steps=eval_steps,
        save_strategy="steps", save_steps=eval_steps, save_total_limit=2,
        load_best_model_at_end=True, metric_for_best_model="eval_loss", greater_is_better=False,
        report_to="none", remove_unused_columns=False, label_names=["labels"], seed=cfg["seed_int"],
        dataloader_num_workers=2)
    trainer = LengthGroupedTrainer(model=model, args=args, train_dataset=data["train"],
                                   eval_dataset=data["val"], data_collator=collate,
                                   callbacks=[CommitVolume()])
    t_train = time.time()
    result = trainer.train()
    train_s = time.time() - t_train
    # load_best_model_at_end has put the best checkpoint back in place.
    best_dir = os.path.join(run_dir, "adapter")
    trainer.model.save_pretrained(best_dir)
    tok.save_pretrained(best_dir)
    steps = meter["steps"][1:] or meter["steps"]  # the first step includes kernel warm-up
    summary = {"train": result.metrics, "best_checkpoint": trainer.state.best_model_checkpoint,
               "best_eval_loss": trainer.state.best_metric, "total_steps": trainer.state.global_step,
               "load_s": cfg.get("_load_s"), "train_wall_s": round(train_s, 1),
               "first_step_s": meter["steps"][0]["sec"] if meter["steps"] else None,
               "mean_step_s": round(sum(r["sec"] for r in steps) / max(1, len(steps)), 2),
               "padded_tok_s": round(sum(r["padded_tokens"] for r in steps) / max(1e-6, sum(r["sec"] for r in steps))),
               "real_tok_s": round(sum(r["real_tokens"] for r in steps) / max(1e-6, sum(r["sec"] for r in steps))),
               "peak_allocated_gb": round(torch.cuda.max_memory_allocated() / 1e9, 1),
               "peak_reserved_gb": round(torch.cuda.max_memory_reserved() / 1e9, 1),
               "losses": [{k: r.get(k) for k in ("step", "loss", "eval_loss", "grad_norm") if k in r}
                          for r in trainer.state.log_history]}
    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    runs_vol.commit()
    print(json.dumps(summary, indent=2))
    return summary


def run_eval(cfg):
    import torch

    if cfg.get("confirm_spend") is not True:
        raise RuntimeError("refused: eval needs confirm_spend=True")
    run_dir = os.path.join(RUNS_MOUNT, cfg["run"])
    model, tok = _load_model(cfg, for_training=False)
    if not cfg["no_adapter"]:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, os.path.join(run_dir, "adapter"))
    model.eval()
    tok.padding_side = "left"
    rows = _read_jsonl(os.path.join(DATA_MOUNT, cfg["data_subdir"], "%s.jsonl" % cfg["split"]), cfg["limit"])
    import time
    t_gen, new_tokens = time.time(), 0
    stop = [tok.convert_tokens_to_ids("<|im_end|>"), tok.convert_tokens_to_ids("<|endoftext|>")]
    torch.manual_seed(cfg["seed_int"])
    out = []
    bs = cfg["eval_batch"]
    for i in range(0, len(rows), bs):
        batch = rows[i:i + bs]
        prompts = [tok.apply_chat_template(r["messages"][:2], tokenize=False, add_generation_prompt=True,
                                           enable_thinking=False) for r in batch]
        enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=cfg["max_new_tokens"], do_sample=cfg["temperature"] > 0,
                                 temperature=cfg["temperature"] or None, top_p=0.8, top_k=20,
                                 eos_token_id=stop, pad_token_id=tok.pad_token_id)
        new = gen[:, enc["input_ids"].shape[1]:]
        new_tokens += int(sum(int((g != tok.pad_token_id).sum()) for g in new))
        for r, g in zip(batch, new):
            out.append({"id": r.get("id"), "input": r["messages"][1]["content"],
                        "model": tok.decode(g, skip_special_tokens=True).strip(),
                        "target": r["messages"][2]["content"]})
        print("generated %d/%d (%.0f s, %d new tokens so far, peak %.1f GB)" % (
            len(out), len(rows), time.time() - t_gen, new_tokens, torch.cuda.max_memory_allocated() / 1e9))
    tag = "base" if cfg["no_adapter"] else "adapter"
    eval_dir = os.path.join(run_dir, "eval")
    os.makedirs(eval_dir, exist_ok=True)
    with open(os.path.join(eval_dir, "%s_%s.jsonl" % (cfg["split"], tag)), "w") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    volumes[RUNS_MOUNT].commit()
    return out


def blind_side_by_side(records, seed="blind-v1"):
    """(markdown, key): each item shows the input and two unlabelled rewrites
    in a seeded random order; the key says which one is the model's."""
    lines, key = ["# Blind side-by-side", "", "Pick the rewrite that sounds like the writer.", ""], {}
    for n, r in enumerate(records, 1):
        h = hashlib.sha256(("%s:%s" % (seed, r.get("id"))).encode()).digest()
        model_first = h[0] % 2 == 0
        a, b = (r["model"], r["target"]) if model_first else (r["target"], r["model"])
        key[str(n)] = {"id": r.get("id"), "model": "A" if model_first else "B"}
        lines += ["## %d" % n, "", "**Input**", "", r["input"], "", "**A**", "", a, "", "**B**", "", b, "", "---", ""]
    return "\n".join(lines), key


if modal is not None:
    image = (
        modal.Image.debian_slim(python_version="3.12")
        .pip_install(*PINNED)
        .env({"HF_HOME": HF_MOUNT, "HF_XET_HIGH_PERFORMANCE": "1", "TOKENIZERS_PARALLELISM": "false",
              "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    )
    app = modal.App(APP_NAME, image=image)
    volumes = {
        DATA_MOUNT: modal.Volume.from_name(DATA_VOLUME, create_if_missing=True),
        HF_MOUNT: modal.Volume.from_name(HF_VOLUME, create_if_missing=True),
        RUNS_MOUNT: modal.Volume.from_name(RUNS_VOLUME, create_if_missing=True),
    }

    @app.function(volumes={HF_MOUNT: volumes[HF_MOUNT]}, cpu=4, memory=8192, timeout=3600)
    def download_model(base_model=BASE_MODEL, base_revision=BASE_REVISION):
        """CPU only: fill the HF cache volume so the GPU never waits on a download."""
        import time
        from huggingface_hub import snapshot_download
        t0 = time.time()
        path = snapshot_download(base_model, revision=base_revision)
        t1 = time.time()
        volumes[HF_MOUNT].commit()
        size = sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(path) for f in fs)
        print("hf cache: %.1f GB, snapshot %.0f s, commit %.0f s" % (size / 1e9, t1 - t0, time.time() - t1))
        return {"path": path, "gb": round(size / 1e9, 1), "snapshot_s": round(t1 - t0), "commit_s": round(time.time() - t1)}

    # Hard caps: with_options may only lower these (see _timeout).
    @app.function(volumes=volumes, gpu="H100", cpu=8, memory=65536, timeout=TRAIN_TIMEOUT_MIN * 60)
    def train_remote(cfg):
        return run_training(cfg)

    @app.function(volumes=volumes, gpu="H100", cpu=8, memory=65536, timeout=EVAL_TIMEOUT_MIN * 60)
    def eval_remote(cfg):
        return run_eval(cfg)


# ---------------------------------------------------------------------------
# Launcher

def _need_modal():
    if modal is None:
        raise SystemExit("the modal client is not importable here; run with\n"
                         "  uv run --no-project --with %s python -I train/modal_train.py ..." % MODAL_CLIENT)


def _guard(args, starts_app=True):
    """Profile, workspace and environment checks; on the Atira workspace also
    the environment allowlist and the app-name collision check."""
    allow = getattr(args, "allow_atira_workspace", False)
    workspaces = None if args.skip_workspace_check else lookup_workspaces()
    profile = check_profile(args.profile, workspaces=workspaces, skip_workspace_check=args.skip_workspace_check,
                            allow_atira=allow)
    ws = (workspaces or {}).get(profile)
    atira = is_atira(profile, ws)
    if atira and args.skip_workspace_check:
        raise Refused("--skip-workspace-check is not allowed on the Atira workspace.")
    modal_env = args.modal_env or os.environ.get("MODAL_ENVIRONMENT")
    if atira:
        check_atira_env(modal_env)
    os.environ["MODAL_PROFILE"] = profile
    if modal_env:
        os.environ["MODAL_ENVIRONMENT"] = modal_env
    _need_modal()
    active = getattr(getattr(modal, "config", None), "_profile", None)
    if active != profile:
        raise Refused("the modal client loaded profile %r, not %r" % (active, profile))
    if modal_env and modal.config.config.get("environment") != modal_env:
        raise Refused("the modal client resolves environment %r, not %r" % (
            modal.config.config.get("environment"), modal_env))
    print("modal profile: %s, workspace %s, environment %s%s" % (
        profile, ws or "NOT checked", modal_env or "(profile default)",
        " (Atira workspace, allowed by --allow-atira-workspace)" if atira else ""))
    if atira and starts_app:
        apps = list_apps(modal_env)
        check_app_collision(apps)
        print("apps in %s: %d, none named %s (ephemeral run, nothing is deployed)" % (
            modal_env, len(apps), APP_NAME))
    return profile


def upload(args):
    _guard(args, starts_app=False)
    _upload_data(args)


def _upload_data(args):
    vol = modal.Volume.from_name(DATA_VOLUME, create_if_missing=True)
    with vol.batch_upload(force=True) as b:
        for split in SPLITS:
            b.put_file(os.path.join(args.data, "%s.jsonl" % split), "/%s/%s.jsonl" % (args.data_subdir, split))
    print("uploaded %s/{train,val,test}.jsonl to volume %s:/%s" % (args.data, DATA_VOLUME, args.data_subdir))


def _cfg(args, **extra):
    cfg = {k: getattr(args, k) for k in (
        "run", "data_subdir", "max_seq_len", "lora_r", "lora_alpha", "lora_dropout", "epochs", "lr",
        "warmup_ratio", "effective_batch", "micro_batch", "eval_steps", "seed", "length_grouping")}
    for k in ("max_steps", "val_limit", "logging_steps", "smoke"):
        if hasattr(args, k):
            cfg[k] = getattr(args, k)
    cfg["lora_modules"] = [m for m in args.lora_modules.split(",") if m]
    cfg["seed_int"] = int(hashlib.sha256(args.seed.encode()).hexdigest()[:8], 16)
    cfg["gpu"] = args.gpu
    if args.base_model != BASE_MODEL and args.base_revision == BASE_REVISION:
        raise Refused("--base-model %s needs its own --base-revision (a commit sha)." % args.base_model)
    cfg["base_model"], cfg["base_revision"], cfg["pinned"] = args.base_model, args.base_revision, list(PINNED)
    cfg.update(extra)
    return cfg


def _timeout_min(args):
    """The hard GPU timeout for this command, in minutes."""
    if args.cmd == "eval":
        return EVAL_TIMEOUT_MIN
    return SMOKE_TIMEOUT_MIN if getattr(args, "smoke", False) else TRAIN_TIMEOUT_MIN


def print_ceiling(args):
    rate = GPUS[args.gpu]["usd_per_hour"]
    minutes = _timeout_min(args)
    print("  hard cap: %s function timeout %d min => at most $%.2f of %s time for this job" % (
        args.cmd if not getattr(args, "smoke", False) else "smoke", minutes, minutes / 60 * rate, args.gpu))


def train(args):
    if args.smoke:
        args.max_steps = args.max_steps or 20
        if args.max_steps > SMOKE_MAX_STEPS:
            raise Refused("--smoke runs at most %d steps (got --max-steps %d)." % (SMOKE_MAX_STEPS, args.max_steps))
    if args.max_steps < 0:
        raise Refused("--max-steps must be >= 0")
    stats, problems = summarise_data(args.data, args.max_seq_len, args.micro_batch)
    print("data: %s" % args.data)
    for split in SPLITS:
        s = stats[split]
        print("  %-5s %5d examples, ~%d tokens (mean %s, max %d), %d over %d, padding x%.2f" % (
            split, s.get("n_all", 0), s["tokens"], s.get("mean", 0), s["max"], s["over"],
            args.max_seq_len, s["pad"]))
    total = sum(stats[k].get("n_all", 0) for k in SPLITS)
    over = sum(stats[k]["over"] for k in SPLITS)
    print("  share over max_seq_len %d: %.2f%% (%d of %d, estimated with chars/%.1f)" % (
        args.max_seq_len, 100 * over / max(1, total), over, total, CHARS_PER_TOKEN))
    if problems:
        for p in problems[:20]:
            print("  problem: " + p)
        raise SystemExit("data check failed (%d problems)" % len(problems))
    if not stats["train"]["n"]:
        raise SystemExit("no training examples")
    est = estimate(stats, args.gpu, args.epochs, args.effective_batch, args.micro_batch, args.eval_steps,
                   tokens_per_s=args.assume_tokens_per_s, max_steps=args.max_steps, val_limit=args.val_limit)
    print_estimate(est)
    print_ceiling(args)
    if est["hours"] * 60 > _timeout_min(args):
        print("  WARNING: the estimate (%.0f min) is over the %d min timeout; the job would be cut off "
              "(checkpoints are kept every eval)." % (est["hours"] * 60, _timeout_min(args)))
    if args.scale_to:
        scale = args.scale_to / max(1, total)
        print_estimate(estimate(stats, args.gpu, args.epochs, args.effective_batch, args.micro_batch,
                                args.eval_steps, scale=scale, tokens_per_s=args.assume_tokens_per_s),
                       " scaled to %d pairs (x%.2f)" % (args.scale_to, scale))
    print("  config: LoRA r=%d alpha=%d dropout=%.2f on %s; lr %.0e cosine, warmup %.0f%%, max_seq_len %d" % (
        args.lora_r, args.lora_alpha, args.lora_dropout, args.lora_modules, args.lr,
        100 * args.warmup_ratio, args.max_seq_len))
    if args.dry_run:
        print("dry run: nothing was sent to Modal.")
        return est
    if not args.confirm_spend:
        raise Refused("this starts a paid %s job; rerun with --confirm-spend once spending is approved." % args.gpu)
    _guard(args)
    _upload_data(args)
    cfg = _cfg(args, confirm_spend=True)
    timeout = _timeout_min(args) * 60
    import time
    with modal.enable_output():
        with app.run(detach=args.detach, environment_name=args.modal_env or None):
            print("filling the HF cache (CPU)...")
            t0 = time.time()
            print("hf cache: %s, %.0f s wall" % (json.dumps(download_model.remote(args.base_model, args.base_revision)),
                                                 time.time() - t0))
            call = train_remote.with_options(gpu=args.gpu, timeout=timeout).spawn(cfg)
            print("training started: function call %s, run %s, gpu %s" % (call.object_id, args.run, args.gpu))
            if args.detach:
                print("detached; outputs land in volume %s:/%s" % (RUNS_VOLUME, args.run))
                return est
            print(json.dumps(call.get(), indent=2))
    return est


def evaluate(args):
    if not args.confirm_spend:
        raise Refused("eval starts a paid %s job; rerun with --confirm-spend once spending is approved." % args.gpu)
    print("eval: run %s, %s split of %s, limit %s, max_new_tokens %d, batch %d" % (
        args.run, args.split, args.data_subdir, args.limit or "none", args.max_new_tokens, args.eval_batch))
    print_ceiling(args)
    _guard(args)
    cfg = _cfg(args, confirm_spend=True, split=args.split, limit=args.limit, no_adapter=args.no_adapter,
               max_new_tokens=args.max_new_tokens, temperature=args.temperature, eval_batch=args.eval_batch)
    with modal.enable_output():
        with app.run(environment_name=args.modal_env or None):
            download_model.remote(args.base_model, args.base_revision)
            records = eval_remote.with_options(gpu=args.gpu, timeout=_timeout_min(args) * 60).remote(cfg)
    out_dir = os.path.join(ROOT, "data/eval", args.run)
    os.makedirs(out_dir, exist_ok=True)
    tag = "base" if args.no_adapter else "adapter"
    with open(os.path.join(out_dir, "%s_%s.jsonl" % (args.split, tag)), "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    md, key = blind_side_by_side(records)
    with open(os.path.join(out_dir, "%s_%s_blind.md" % (args.split, tag)), "w", encoding="utf-8") as f:
        f.write(md)
    with open(os.path.join(out_dir, "%s_%s_blind_key.json" % (args.split, tag)), "w", encoding="utf-8") as f:
        json.dump(key, f, indent=2)
    print("wrote %d outputs to %s (blind side-by-side + key)" % (len(records), os.path.relpath(out_dir, ROOT)))


def build_parser():
    ap = argparse.ArgumentParser(prog="modal_train", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--profile", help="Modal profile (or MODAL_PROFILE); required, never implicit")
        p.add_argument("--modal-env", default=None,
                       help="Modal environment (or MODAL_ENVIRONMENT); required on the Atira workspace (%s)"
                       % "/".join(ATIRA_ALLOWED_ENVS))
        p.add_argument("--allow-atira-workspace", action="store_true",
                       help="opt in to the Atira workspace (approved spend only); refused otherwise")
        p.add_argument("--skip-workspace-check", action="store_true")
        p.add_argument("--data", default=os.path.join(ROOT, "data/train"))
        p.add_argument("--data-subdir", default="train", help="directory inside the irulan-data volume")
        p.add_argument("--run", default="qlora-r32-v1", help="output directory in the irulan-runs volume")
        p.add_argument("--gpu", default="H100", choices=sorted(GPUS))
        p.add_argument("--confirm-spend", action="store_true", help="required for any GPU job")
        p.add_argument("--max-seq-len", type=int, default=1024)
        p.add_argument("--lora-r", type=int, default=32)
        p.add_argument("--lora-alpha", type=int, default=32)
        p.add_argument("--lora-dropout", type=float, default=0.05)
        p.add_argument("--lora-modules", default=",".join(DEFAULT_LORA_MODULES))
        p.add_argument("--epochs", type=int, default=2)
        p.add_argument("--lr", type=float, default=1e-4)
        p.add_argument("--warmup-ratio", type=float, default=0.03)
        p.add_argument("--effective-batch", type=int, default=32)
        p.add_argument("--micro-batch", type=int, default=8)
        p.add_argument("--eval-steps", type=int, default=0, help="0 = about ten evals per run")
        p.add_argument("--seed", default="irulan-train-v1")
        p.add_argument("--base-model", default=BASE_MODEL,
                       help="override only for a pipeline check with a small model of the same family")
        p.add_argument("--base-revision", default=BASE_REVISION)
        p.add_argument("--no-length-grouping", dest="length_grouping", action="store_false")
        p.add_argument("--max-steps", type=int, default=0, help="stop after this many optimizer steps (0 = full)")
        p.add_argument("--val-limit", type=int, default=0, help="evaluate on the first N val examples (0 = all)")
        p.add_argument("--logging-steps", type=int, default=5)

    t = sub.add_parser("train", help="validate, estimate, and (with --confirm-spend) train")
    common(t)
    t.add_argument("--dry-run", action="store_true", help="local checks and estimate only; never contacts Modal")
    t.add_argument("--scale-to", type=int, default=0, help="also estimate for this many pairs")
    t.add_argument("--assume-tokens-per-s", type=float, default=None)
    t.add_argument("--detach", action="store_true", help="return once the job is started")
    t.add_argument("--smoke", action="store_true",
                   help="short check run: %d min timeout, --max-steps default 20 (at most %d)" % (
                       SMOKE_TIMEOUT_MIN, SMOKE_MAX_STEPS))

    e = sub.add_parser("eval", help="generate rewrites for held-out inputs (needs --confirm-spend)")
    common(e)
    e.add_argument("--split", default="test", choices=SPLITS)
    e.add_argument("--limit", type=int, default=0)
    e.add_argument("--no-adapter", action="store_true", help="base model only, as a baseline")
    e.add_argument("--max-new-tokens", type=int, default=512)
    e.add_argument("--temperature", type=float, default=0.7)
    e.add_argument("--eval-batch", type=int, default=16)

    u = sub.add_parser("upload", help="put data/train on the irulan-data volume")
    common(u)
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.cmd == "train":
        return train(args)
    if args.cmd == "eval":
        return evaluate(args)
    if args.cmd == "upload":
        return upload(args)


if __name__ == "__main__":
    main()
