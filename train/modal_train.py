"""QLoRA fine-tune of Qwen3.8-27B on Modal, plus an eval entrypoint.

Nothing here spends money unless it is told to twice: the dry run is local
and needs no Modal or torch, and every GPU job needs --confirm-spend on top
of an explicitly selected, non-Atira Modal profile.

    # local, free: validate data/train and print the cost estimate
    python3 -I train/modal_train.py train --dry-run [--scale-to 10100]

    # real runs need the modal client (the CLI's own venv, or uv):
    uv run --no-project --with modal==1.4.3 python -I train/modal_train.py \\
        upload --profile <personal-profile>
    uv run --no-project --with modal==1.4.3 python -I train/modal_train.py \\
        train --profile <personal-profile> --confirm-spend [--gpu H100] [--run NAME]
    uv run --no-project --with modal==1.4.3 python -I train/modal_train.py \\
        eval --profile <personal-profile> --confirm-spend --run NAME [--limit 200]

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
APP_NAME = "irulan-swe-train"
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
}
SETUP_MINUTES = 12   # GPU minutes for loading + quantising the weights and kernel warm-up
EVAL_COST_FACTOR = 1 / 3  # an eval token costs about a third of a training token (forward only)

# The profile guard: a profile whose name or workspace matches this is refused.
BLOCKED_WORKSPACE_RE = re.compile(os.environ.get("IRULAN_BLOCKED_MODAL_RE", r"atira"), re.I)


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


def check_profile(profile, env=None, profiles=None, workspaces=None, skip_workspace_check=False):
    """Return the profile to use, or raise Refused. The profile must be named
    explicitly (--profile or MODAL_PROFILE), exist, and neither its name nor
    its workspace may look like Atira's. Token env vars are refused because
    they override the profile and hide which workspace is used."""
    env = os.environ if env is None else env
    if env.get("MODAL_TOKEN_ID") or env.get("MODAL_TOKEN_SECRET"):
        raise Refused("MODAL_TOKEN_ID/MODAL_TOKEN_SECRET are set; they override the profile. Unset them.")
    profile = profile or env.get("MODAL_PROFILE")
    if not profile:
        raise Refused("no Modal profile selected. Pass --profile NAME or set MODAL_PROFILE "
                      "to a personal (non-Atira) profile; the active profile is never used implicitly.")
    if BLOCKED_WORKSPACE_RE.search(profile):
        raise Refused("profile %r looks like an Atira profile; this project must not run there." % profile)
    profiles = modal_profiles() if profiles is None else profiles
    if profile not in profiles:
        raise Refused("profile %r is not in ~/.modal.toml (found: %s)." % (profile, ", ".join(profiles) or "none"))
    if not skip_workspace_check:
        workspaces = lookup_workspaces() if workspaces is None else workspaces
        if workspaces is None or profile not in workspaces:
            raise Refused("could not confirm the workspace of profile %r (pass "
                          "--skip-workspace-check only if you know it)." % profile)
        ws = workspaces[profile] or ""
        if BLOCKED_WORKSPACE_RE.search(ws):
            raise Refused("profile %r points at workspace %r, which looks like Atira's." % (profile, ws))
    return profile


def _early_profile_from_argv(argv):
    """Export --profile as MODAL_PROFILE before modal is imported (the client
    reads it once, at import time)."""
    for i, a in enumerate(argv):
        if a == "--profile" and i + 1 < len(argv):
            os.environ["MODAL_PROFILE"] = argv[i + 1]
        elif a.startswith("--profile="):
            os.environ["MODAL_PROFILE"] = a.split("=", 1)[1]


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
             scale=1.0, tokens_per_s=None):
    """Steps, time and dollars from the per-split token counts."""
    g = GPUS[gpu]
    tps = tokens_per_s or g["tokens_per_s"]
    n_train = stats["train"]["n"] * scale
    train_tokens = stats["train"]["tokens"] * scale * stats["train"]["pad"]
    val_tokens = stats["val"]["tokens"] * scale * stats["val"]["pad"]
    grad_accum = max(1, round(effective_batch / micro_batch))
    steps_per_epoch = math.ceil(n_train / (micro_batch * grad_accum))
    total_steps = steps_per_epoch * epochs
    eval_steps = eval_every or default_eval_steps(total_steps)
    n_evals = total_steps // eval_steps + 1
    work_tokens = train_tokens * epochs + val_tokens * n_evals * EVAL_COST_FACTOR
    hours = work_tokens / tps / 3600 + SETUP_MINUTES / 60
    return {
        "gpu": gpu, "assumed_tokens_per_s": tps, "usd_per_hour": g["usd_per_hour"],
        "train_examples": int(n_train), "epochs": epochs, "micro_batch": micro_batch,
        "grad_accum": grad_accum, "effective_batch": micro_batch * grad_accum,
        "steps_per_epoch": steps_per_epoch, "total_steps": total_steps,
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
    print("  %d steps (%d per epoch), eval every %d steps (%d evals)" % (
        est["total_steps"], est["steps_per_epoch"], est["eval_steps"], est["evals"]))
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

    tok = AutoTokenizer.from_pretrained(BASE_MODEL, revision=BASE_REVISION)
    if tok.pad_token_id is None:
        tok.pad_token = "<|endoftext|>"
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True, bnb_4bit_quant_storage=torch.bfloat16,
        llm_int8_skip_modules=list(BNB_SKIP_MODULES))
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, revision=BASE_REVISION, quantization_config=bnb, dtype=torch.bfloat16,
        attn_implementation="sdpa", device_map={"": 0})
    print("loaded %s (%s), %.1f GB on GPU" % (type(model).__name__, BASE_REVISION[:8],
                                              torch.cuda.memory_allocated() / 1e9))
    if any(".visual." in n or n.startswith("visual.") for n, _ in model.named_modules()):
        print("note: vision tower present; it is frozen and not targeted by LoRA")
    if for_training:
        model.config.use_cache = False
        for p in model.parameters():
            p.requires_grad_(False)
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    return model, tok


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
        rows = _read_jsonl(os.path.join(DATA_MOUNT, cfg["data_subdir"], "%s.jsonl" % split))
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
    eval_steps = cfg["eval_steps"] or default_eval_steps(total_steps)
    print("steps: %d (%d per epoch), eval/save every %d" % (total_steps, steps_per_epoch, eval_steps))

    runs_vol = volumes[RUNS_MOUNT]

    class CommitVolume(TrainerCallback):
        def on_save(self, args, state, control, **kw):
            runs_vol.commit()

        def on_log(self, args, state, control, logs=None, **kw):
            with open(os.path.join(run_dir, "log.jsonl"), "a") as f:
                f.write(json.dumps(dict(logs or {}, step=state.global_step)) + "\n")

    lengths = [len(e["input_ids"]) for e in data["train"]]

    class LengthGroupedTrainer(Trainer):
        """Batches of similar length (see length_grouped_batches), so short
        messages are not padded to the longest one in a random batch."""
        epoch = 0

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
        num_train_epochs=cfg["epochs"], learning_rate=cfg["lr"], lr_scheduler_type="cosine",
        warmup_steps=max(1, math.ceil(cfg["warmup_ratio"] * total_steps)),
        per_device_train_batch_size=micro, per_device_eval_batch_size=micro,
        gradient_accumulation_steps=accum, gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=True, optim="paged_adamw_8bit", weight_decay=0.0, max_grad_norm=1.0,
        logging_steps=5, eval_strategy="steps", eval_steps=eval_steps,
        save_strategy="steps", save_steps=eval_steps, save_total_limit=2,
        load_best_model_at_end=True, metric_for_best_model="eval_loss", greater_is_better=False,
        report_to="none", remove_unused_columns=False, seed=cfg["seed_int"],
        dataloader_num_workers=2)
    trainer = LengthGroupedTrainer(model=model, args=args, train_dataset=data["train"],
                                   eval_dataset=data["val"], data_collator=collate,
                                   callbacks=[CommitVolume()])
    result = trainer.train()
    # load_best_model_at_end has put the best checkpoint back in place.
    best_dir = os.path.join(run_dir, "adapter")
    trainer.model.save_pretrained(best_dir)
    tok.save_pretrained(best_dir)
    summary = {"train": result.metrics, "best_checkpoint": trainer.state.best_model_checkpoint,
               "best_eval_loss": trainer.state.best_metric, "total_steps": trainer.state.global_step}
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
        for r, g in zip(batch, gen[:, enc["input_ids"].shape[1]:]):
            out.append({"id": r.get("id"), "input": r["messages"][1]["content"],
                        "model": tok.decode(g, skip_special_tokens=True).strip(),
                        "target": r["messages"][2]["content"]})
        print("generated %d/%d" % (len(out), len(rows)))
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
    def download_model():
        """CPU only: fill the HF cache volume so the GPU never waits on a download."""
        from huggingface_hub import snapshot_download
        path = snapshot_download(BASE_MODEL, revision=BASE_REVISION)
        volumes[HF_MOUNT].commit()
        return path

    @app.function(volumes=volumes, gpu="H100", cpu=8, memory=65536, timeout=8 * 3600)
    def train_remote(cfg):
        return run_training(cfg)

    @app.function(volumes=volumes, gpu="H100", cpu=8, memory=65536, timeout=2 * 3600)
    def eval_remote(cfg):
        return run_eval(cfg)


# ---------------------------------------------------------------------------
# Launcher

def _need_modal():
    if modal is None:
        raise SystemExit("the modal client is not importable here; run with\n"
                         "  uv run --no-project --with %s python -I train/modal_train.py ..." % MODAL_CLIENT)


def _guard(args):
    profile = check_profile(args.profile, skip_workspace_check=args.skip_workspace_check)
    os.environ["MODAL_PROFILE"] = profile
    _need_modal()
    active = getattr(getattr(modal, "config", None), "_profile", None)
    if active != profile:
        raise Refused("the modal client loaded profile %r, not %r" % (active, profile))
    print("modal profile: %s (workspace checked)" % profile if not args.skip_workspace_check
          else "modal profile: %s (workspace NOT checked)" % profile)
    return profile


def upload(args):
    _guard(args)
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
    cfg["lora_modules"] = [m for m in args.lora_modules.split(",") if m]
    cfg["seed_int"] = int(hashlib.sha256(args.seed.encode()).hexdigest()[:8], 16)
    cfg["gpu"] = args.gpu
    cfg["base_model"], cfg["base_revision"], cfg["pinned"] = BASE_MODEL, BASE_REVISION, list(PINNED)
    cfg.update(extra)
    return cfg


def train(args):
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
                   tokens_per_s=args.assume_tokens_per_s)
    print_estimate(est)
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
    timeout = int(min(24, max(2, est["hours"] * 2.5)) * 3600)
    with modal.enable_output():
        with app.run(detach=args.detach):
            print("filling the HF cache (CPU)...")
            download_model.remote()
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
    _guard(args)
    cfg = _cfg(args, confirm_spend=True, split=args.split, limit=args.limit, no_adapter=args.no_adapter,
               max_new_tokens=args.max_new_tokens, temperature=args.temperature, eval_batch=args.eval_batch)
    with modal.enable_output():
        with app.run():
            download_model.remote()
            records = eval_remote.with_options(gpu=args.gpu).remote(cfg)
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
        p.add_argument("--profile", help="Modal profile (or MODAL_PROFILE); must be personal, never Atira")
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
        p.add_argument("--no-length-grouping", dest="length_grouping", action="store_false")

    t = sub.add_parser("train", help="validate, estimate, and (with --confirm-spend) train")
    common(t)
    t.add_argument("--dry-run", action="store_true", help="local checks and estimate only; never contacts Modal")
    t.add_argument("--scale-to", type=int, default=0, help="also estimate for this many pairs")
    t.add_argument("--assume-tokens-per-s", type=float, default=None)
    t.add_argument("--detach", action="store_true", help="return once the job is started")

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
