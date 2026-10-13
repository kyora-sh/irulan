"""irulan-swe playground: paste text, get the rewrite back. Ephemeral, on Modal.

    MODAL_PROFILE=<profile> modal serve serve/modal_playground.py --env testing

Two functions in one app (`irulan-playground`):

- `web` (CPU, FastAPI): serves the page and the JSON API. Every /api route
  needs the access token (header X-Playground-Token, compared in constant
  time against the `irulan-playground-token` secret). Loading the page never
  starts a GPU.
- `Engine` (one H100): vLLM on the BF16 base model with the run adapters as
  runtime LoRA. It is only reachable through `web` (or Modal's own auth), runs
  at most one container, and scales to zero after SCALEDOWN_S idle seconds.

Inference matches training: the `system_v1` prompt, the model's chat
template with enable_thinking=False, and stop on <|im_end|>. Request texts
are never logged.

Adapters are read from the `irulan-runs` volume: `<run>` is
`<run>/adapter`, `<run>@epoch-N` is `<run>/epoch-N` and `<run>@checkpoint-N`
is `<run>/checkpoints/checkpoint-N`.
The default is the first of DEFAULT_ADAPTERS that exists (override with
PLAYGROUND_DEFAULT_ADAPTER at serve time).

`modal deploy` on this file is refused at import: the playground only lives
while `modal serve` runs. Modal disables memory and GPU snapshots for
ephemeral apps, so every cold start loads the weights and rebuilds the
engine; the torch.compile, Triton and FlashInfer caches live on the
`irulan-playground-cache` volume so only the first boot compiles. On a profile whose name looks like Atira's the
environment must be named and be `testing` or `dev`.
"""

import hashlib
import hmac
import json
import os
import re
import sys
import time

_T_IMPORT = time.time()

APP_NAME = "irulan-playground"
BASE_MODEL = "Qwen/Qwen3.8-27B"
BASE_REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
VLLM = "vllm==0.30.0"  # Qwen3_5 + LoRA on the fused GDN in_proj_qkvz (qkv + z)
TRANSFORMERS = "transformers==5.18.0"  # same as training: same chat template rendering
FASTAPI = "fastapi[standard]==0.142.2"
CUDA_IMAGE = "nvidia/cuda:13.0.2-devel-ubuntu24.04"  # torch 2.13 wheels are cu13

# Must equal irulan.prepare.SYSTEM_PROMPTS["system_v1"] (a test checks it).
SYSTEM_V1 = ("Rewrite the text in the writer's own voice: terse, direct, short lines. "
             "Keep the meaning, asks, facts and numbers. Return only the rewrite.")

HF_VOLUME, RUNS_VOLUME, CACHE_VOLUME = "irulan-hf-cache", "irulan-runs", "irulan-playground-cache"
HF_MOUNT, RUNS_MOUNT, CACHE_MOUNT = "/vol/hf", "/vol/runs", "/vol/cache"
TOKEN_SECRET, TOKEN_ENV, TOKEN_HEADER = "irulan-playground-token", "PLAYGROUND_TOKEN", "x-playground-token"

DEFAULT_ADAPTERS = ("qlora-r32-v1", "smoke-1")
GPU = "H100"
SCALEDOWN_S = 300         # idle seconds before the GPU container stops
REQUEST_TIMEOUT_S = 300   # per rewrite request
STARTUP_TIMEOUT_S = 1200  # cold start: weights from the volume, compile, CUDA graphs
MAX_MODEL_LEN = 8192
MAX_LORA_RANK = int(os.environ.get("PLAYGROUND_MAX_LORA_RANK", "32"))
DEFAULT_OVERRIDE = os.environ.get("PLAYGROUND_DEFAULT_ADAPTER", "")

MAX_INPUT_CHARS = 20000
DEFAULTS = {"temperature": 0.3, "max_tokens": 512, "top_p": 0.8, "top_k": 20}
LIMITS = {"temperature": (0.0, 1.5), "max_tokens": (1, 2048)}
ATIRA_RE = re.compile(r"atira", re.I)
ALLOWED_ENVS = ("testing", "dev")


# ---------------------------------------------------------------------------
# Pure helpers (stdlib only, unit-tested)

def token_ok(given, expected):
    """Constant-time check; an unset expected token rejects everything."""
    if not expected:
        return False
    a = hashlib.sha256((given or "").encode()).digest()
    b = hashlib.sha256(expected.encode()).digest()
    return hmac.compare_digest(a, b)


_RUN_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,99}$")
_ADAPTER_RE = re.compile(r"^(?P<run>[A-Za-z0-9][A-Za-z0-9._-]{0,99})(?:@(?P<kind>checkpoint|epoch)-(?P<step>\d{1,9}))?$")


def adapter_path(root, name):
    """Directory of adapter `name` under root, or None for a malformed name."""
    m = _ADAPTER_RE.match(name or "")
    if not m or m.group("run") in (".", ".."):
        return None
    if m.group("kind") == "epoch":
        return os.path.join(root, m.group("run"), "epoch-%d" % int(m.group("step")))
    if m.group("kind") == "checkpoint":
        return os.path.join(root, m.group("run"), "checkpoints", "checkpoint-%d" % int(m.group("step")))
    return os.path.join(root, m.group("run"), "adapter")


def _same_base(adapter_dir):
    """False if the adapter config names another base model (e.g. a 0.8B pipeline check)."""
    try:
        with open(os.path.join(adapter_dir, "adapter_config.json"), encoding="utf-8") as f:
            base = json.load(f).get("base_model_name_or_path")
    except (OSError, ValueError, AttributeError):
        return True
    return base in (None, "", BASE_MODEL)


def list_adapters(root):
    """[{name, run, step}] for every run adapter, per-epoch adapter and checkpoint
    of BASE_MODEL that has weights."""
    out = []
    try:
        runs = sorted(os.listdir(root))
    except OSError:
        return out
    for run in runs:
        if not _RUN_RE.match(run):
            continue
        if not _same_base(os.path.join(root, run, "adapter")):
            continue
        if os.path.exists(os.path.join(root, run, "adapter", "adapter_model.safetensors")):
            out.append({"name": run, "run": run, "step": None})
        epochs = []
        for d in os.listdir(os.path.join(root, run)) if os.path.isdir(os.path.join(root, run)) else ():
            m = re.match(r"^epoch-(\d+)$", d)
            if m and os.path.exists(os.path.join(root, run, d, "adapter_model.safetensors")):
                epochs.append(int(m.group(1)))
        for e in sorted(epochs):
            out.append({"name": "%s@epoch-%d" % (run, e), "run": run, "step": None, "epoch": e})
        ck = os.path.join(root, run, "checkpoints")
        steps = []
        if os.path.isdir(ck):
            for d in os.listdir(ck):
                m = re.match(r"^checkpoint-(\d+)$", d)
                if m and os.path.exists(os.path.join(ck, d, "adapter_model.safetensors")):
                    steps.append(int(m.group(1)))
        for s in sorted(steps):
            out.append({"name": "%s@checkpoint-%d" % (run, s), "run": run, "step": s})
    return out


def default_adapter(names, override=None):
    if override and override in names:
        return override
    for n in DEFAULT_ADAPTERS:
        if n in names:
            return n
    return names[0] if names else None


_LORA_PREFIXES = ("base_model.model.model.layers.", "base_model.model.model.language_model.layers.")


def vllm_lora_key(key):
    """Adapter tensor name for vLLM's Qwen3_5ForConditionalGeneration.

    Training loads the text stack only (Qwen3_5ForCausalLM), so peft saves
    `base_model.model.model.layers.N...`. vLLM serves the checkpoint's own
    architecture, whose decoder is `language_model.model.layers.N`, and maps
    adapter names with the checkpoint mapper (`model.language_model.` ->
    `language_model.model.`). Without the `language_model.` segment the LoRA
    would load but match no module and silently do nothing."""
    if key.startswith(_LORA_PREFIXES[1]):
        return key
    if key.startswith(_LORA_PREFIXES[0]):
        return _LORA_PREFIXES[1] + key[len(_LORA_PREFIXES[0]):]
    raise ValueError("unexpected adapter tensor name: %s" % key)


def lora_target_modules(keys):
    """Leaf module names the adapter actually targets (e.g. in_proj_qkv)."""
    names = set()
    for k in keys:
        parts = k.split(".")
        if len(parts) >= 3 and parts[-2] in ("lora_A", "lora_B"):
            names.add(parts[-3])
    return sorted(names)


def chat_messages(text):
    return [{"role": "system", "content": SYSTEM_V1}, {"role": "user", "content": text}]


def clean_request(req):
    """Validated request dict; raises ValueError with a user-facing message."""
    if not isinstance(req, dict):
        raise ValueError("request must be a JSON object")
    text = req.get("text")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("text is empty")
    if len(text) > MAX_INPUT_CHARS:
        raise ValueError("text is over %d characters" % MAX_INPUT_CHARS)
    out = {"text": text, "compare": bool(req.get("compare"))}
    for k, (lo, hi) in LIMITS.items():
        v = req.get(k, DEFAULTS[k])
        try:
            v = float(v) if k == "temperature" else int(v)
        except (TypeError, ValueError):
            raise ValueError("%s must be a number" % k)
        out[k] = min(hi, max(lo, v))
    adapter = req.get("adapter")
    if adapter in (None, "", "base"):
        out["adapter"] = None
    elif isinstance(adapter, str) and _ADAPTER_RE.match(adapter):
        out["adapter"] = adapter
    else:
        raise ValueError("bad adapter name")
    return out


# ---------------------------------------------------------------------------
# Guard: only `modal serve`, only on an explicit profile and allowed env

def _cli_env(argv, environ):
    for i, a in enumerate(argv):
        if a in ("--env", "-e") and i + 1 < len(argv):
            return argv[i + 1]
        if a.startswith("--env="):
            return a.split("=", 1)[1]
    return environ.get("MODAL_ENVIRONMENT")


def check_cli(argv, environ):
    """Raise SystemExit unless this is `modal serve` on a named profile; on an
    Atira-looking profile the environment must be testing or dev."""
    cmd = argv[1] if len(argv) > 1 else ""
    if cmd != "serve":
        raise SystemExit("refused: the playground runs only under `modal serve` (got %r); "
                         "nothing here is ever deployed." % cmd)
    profile = environ.get("MODAL_PROFILE")
    if not profile:
        raise SystemExit("refused: set MODAL_PROFILE explicitly; the active profile is never used implicitly.")
    if ATIRA_RE.search(profile):
        env = _cli_env(argv[2:], environ)
        if env not in ALLOWED_ENVS:
            raise SystemExit("refused: on profile %r pass --env %s (got %r)." % (
                profile, " or --env ".join(ALLOWED_ENVS), env))
    if environ.get("MODAL_TOKEN_ID") or environ.get("MODAL_TOKEN_SECRET"):
        raise SystemExit("refused: MODAL_TOKEN_ID/MODAL_TOKEN_SECRET override the profile; unset them.")


_LOCAL_CLI = (os.path.basename(sys.argv[0] if sys.argv else "") == "modal"
              and os.environ.get("MODAL_IS_REMOTE") != "1")
if _LOCAL_CLI:
    check_cli(sys.argv, os.environ)

try:
    import modal
except ImportError:  # unit tests without the modal client
    modal = None


# ---------------------------------------------------------------------------
# Modal app

if modal is not None:
    app = modal.App(APP_NAME)
    hf_vol = modal.Volume.from_name(HF_VOLUME)
    runs_vol = modal.Volume.from_name(RUNS_VOLUME)
    cache_vol = modal.Volume.from_name(CACHE_VOLUME, create_if_missing=True)
    token_secret = modal.Secret.from_name(TOKEN_SECRET)

    engine_image = (
        modal.Image.from_registry(CUDA_IMAGE, add_python="3.12")
        .entrypoint([])
        .uv_pip_install(VLLM, TRANSFORMERS)
        .env({
            "HF_HOME": HF_MOUNT, "HF_HUB_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false",
            "VLLM_CACHE_ROOT": CACHE_MOUNT + "/vllm", "TRITON_CACHE_DIR": CACHE_MOUNT + "/triton",
            "FLASHINFER_WORKSPACE_BASE": CACHE_MOUNT + "/flashinfer",
            "VLLM_NO_USAGE_STATS": "1", "DO_NOT_TRACK": "1",
            "PLAYGROUND_MAX_LORA_RANK": str(MAX_LORA_RANK),
            "PLAYGROUND_DEFAULT_ADAPTER": DEFAULT_OVERRIDE,
        })
    )
    web_image = (
        modal.Image.debian_slim(python_version="3.12")
        .uv_pip_install(FASTAPI)
        .env({"PLAYGROUND_DEFAULT_ADAPTER": DEFAULT_OVERRIDE})
        .add_local_file(os.path.join(os.path.dirname(os.path.abspath(__file__)), "playground.html"),
                        "/root/playground.html")
    )

    def _snapshot_dir():
        d = os.path.join(HF_MOUNT, "hub", "models--" + BASE_MODEL.replace("/", "--"), "snapshots", BASE_REVISION)
        if not os.path.exists(os.path.join(d, "config.json")):
            raise RuntimeError("base model revision %s is not in the %s volume" % (BASE_REVISION[:8], HF_VOLUME))
        return d

    engine_kwargs = dict(
        image=engine_image, gpu=GPU, cpu=4, memory=32768,
        volumes={HF_MOUNT: hf_vol.read_only(), RUNS_MOUNT: runs_vol.read_only(), CACHE_MOUNT: cache_vol},
        timeout=REQUEST_TIMEOUT_S, startup_timeout=STARTUP_TIMEOUT_S,
        scaledown_window=SCALEDOWN_S, max_containers=1,
    )

    @app.cls(**engine_kwargs)
    @modal.concurrent(max_inputs=8)
    class Engine:
        async def _boot(self):
            t0 = time.time()
            import torch  # noqa: F401
            from transformers import AutoTokenizer
            from vllm import AsyncEngineArgs
            from vllm.v1.engine.async_llm import AsyncLLM
            t1 = time.time()
            path = _snapshot_dir()
            self.tok = AutoTokenizer.from_pretrained(path)
            self.stop_ids = [self.tok.convert_tokens_to_ids(t) for t in ("<|im_end|>", "<|endoftext|>")]
            args = AsyncEngineArgs(
                model=path, tokenizer=path, served_model_name=BASE_MODEL, dtype="bfloat16",
                max_model_len=MAX_MODEL_LEN, gpu_memory_utilization=0.90, max_num_seqs=8,
                language_model_only=True, enable_lora=True, max_lora_rank=MAX_LORA_RANK,
                max_loras=2, max_cpu_loras=4, seed=0)
            self.engine = AsyncLLM.from_engine_args(args)
            t2 = time.time()
            self.loras = {}
            await self._warmup()
            t3 = time.time()
            try:
                await cache_vol.commit.aio()  # keep the compile caches for the next cold start
            except Exception as e:
                print("cache commit failed: %s" % e)
            self.timings = {"import_s": round(t1 - t0, 1), "engine_init_s": round(t2 - t1, 1),
                            "warmup_s": round(t3 - t2, 1), "container_to_ready_s": round(t3 - _T_IMPORT, 1)}
            print("engine ready: %s" % json.dumps(self.timings))

        async def _warmup(self):
            names = [a["name"] for a in list_adapters(RUNS_MOUNT)]
            default = default_adapter(names, DEFAULT_OVERRIDE)
            lora = await self._lora(default) if default else None
            # A short and a longer prompt, so the LoRA kernels for both shapes are compiled here.
            for text in ("warm up", "warm up the longer prompt shapes too. " * 60):
                async for _ in self._stream(text, lora, {"temperature": 0.0, "max_tokens": 8}):
                    pass

        @modal.enter()
        async def start(self):
            await self._boot()

        async def _lora(self, name):
            """LoRARequest for adapter `name`, converted to vLLM's names once per container."""
            from vllm.lora.request import LoRARequest
            if name in self.loras:
                return self.loras[name]
            src = adapter_path(RUNS_MOUNT, name)
            if src is None or not os.path.exists(os.path.join(src, "adapter_model.safetensors")):
                runs_vol.reload()
                if src is None or not os.path.exists(os.path.join(src, "adapter_model.safetensors")):
                    raise ValueError("no adapter %r on the runs volume" % name)
            from safetensors.torch import load_file, save_file
            sd = load_file(os.path.join(src, "adapter_model.safetensors"))
            renamed = {vllm_lora_key(k): v for k, v in sd.items()}
            with open(os.path.join(src, "adapter_config.json")) as f:
                cfg = json.load(f)
            cfg["target_modules"] = lora_target_modules(renamed)
            if cfg.get("r", 0) > MAX_LORA_RANK:
                raise ValueError("adapter rank %s is over max_lora_rank %d" % (cfg.get("r"), MAX_LORA_RANK))
            dst = os.path.join("/tmp/lora", name.replace("@", "__"))
            os.makedirs(dst, exist_ok=True)
            save_file(renamed, os.path.join(dst, "adapter_model.safetensors"))
            with open(os.path.join(dst, "adapter_config.json"), "w") as f:
                json.dump(cfg, f)
            req = LoRARequest(name, len(self.loras) + 1, dst)
            await self.engine.add_lora(req)
            self.loras[name] = req
            print("adapter %s: %d tensors, r=%s, targets %s" % (name, len(renamed), cfg.get("r"),
                                                              ",".join(cfg["target_modules"])))
            return req

        async def _stream(self, text, lora, params):
            """Yield (delta_text, n_tokens_so_far, finish_reason, prompt_tokens) for one request."""
            import uuid
            from vllm import SamplingParams
            from vllm.sampling_params import RequestOutputKind
            prompt = self.tok.apply_chat_template(chat_messages(text), tokenize=False,
                                                  add_generation_prompt=True, enable_thinking=False)
            ids = self.tok(prompt, add_special_tokens=False)["input_ids"]
            sp = SamplingParams(temperature=params["temperature"], top_p=DEFAULTS["top_p"],
                                top_k=DEFAULTS["top_k"], max_tokens=params["max_tokens"],
                                stop_token_ids=self.stop_ids, output_kind=RequestOutputKind.DELTA)
            n = 0
            async for out in self.engine.generate({"prompt_token_ids": ids}, sp, uuid.uuid4().hex,
                                                  lora_request=lora):
                o = out.outputs[0]
                n += len(o.token_ids)
                yield o.text, n, o.finish_reason, len(ids)

        @modal.method()
        async def generate(self, req):
            """Stream events for one rewrite: start, delta*, done per side, end."""
            import asyncio
            t_req = time.time()
            req = clean_request(req)
            lora = await self._lora(req["adapter"]) if req["adapter"] else None
            sides = [("main", lora)]
            if req["compare"] and lora is not None:
                sides.append(("base", None))
            yield {"type": "start", "adapter": req["adapter"], "sides": [s for s, _ in sides],
                   "engine": self.timings}
            q = asyncio.Queue()

            async def run(side, lr):
                t0, t_first, n, fin, p = time.time(), None, 0, None, 0
                try:
                    async for delta, n, fin, p in self._stream(req["text"], lr, req):
                        if t_first is None and n:
                            t_first = time.time()
                        if delta:
                            await q.put({"type": "delta", "side": side, "text": delta})
                    t_end = time.time()
                    dec = t_end - (t_first or t_end)
                    await q.put({"type": "done", "side": side, "tokens": n, "prompt_tokens": p,
                                 "finish_reason": fin, "ttft_s": round((t_first or t_end) - t0, 3),
                                 "total_s": round(t_end - t0, 3),
                                 "tok_s": round((n - 1) / dec, 1) if n > 1 and dec > 0 else None})
                except Exception as e:  # report, never the input text
                    await q.put({"type": "error", "side": side, "error": "%s: %s" % (type(e).__name__, e)})

            tasks = [asyncio.create_task(run(s, lr)) for s, lr in sides]
            try:
                pending = len(tasks)
                while pending:
                    ev = await q.get()
                    if ev["type"] in ("done", "error"):
                        pending -= 1
                    yield ev
                yield {"type": "end", "server_s": round(time.time() - t_req, 3)}
            finally:
                for t in tasks:
                    t.cancel()

        @modal.method()
        async def status(self):
            return {"timings": self.timings, "loras": sorted(self.loras),
                    "gpu": __import__("torch").cuda.get_device_name(0)}

        @modal.exit()
        def stop(self):
            try:
                self.engine.shutdown()
            except Exception:
                pass

    @app.function(image=web_image, secrets=[token_secret], volumes={RUNS_MOUNT: runs_vol.read_only()},
                  cpu=1, memory=1024, timeout=REQUEST_TIMEOUT_S + STARTUP_TIMEOUT_S,
                  scaledown_window=SCALEDOWN_S, max_containers=1)
    @modal.concurrent(max_inputs=32)
    @modal.asgi_app()
    def web():
        from fastapi import FastAPI, Request
        from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

        api = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
        expected = os.environ.get(TOKEN_ENV, "")
        with open("/root/playground.html", encoding="utf-8") as f:
            page = f.read()
        state = {"reloaded": 0.0}

        def denied(request):
            if token_ok(request.headers.get(TOKEN_HEADER), expected):
                return None
            return JSONResponse({"error": "unauthorized"}, status_code=401)

        @api.get("/", response_class=HTMLResponse)
        def index():
            return HTMLResponse(page, headers={"Cache-Control": "no-store", "X-Robots-Tag": "noindex",
                                               "Referrer-Policy": "no-referrer"})

        @api.get("/api/adapters")
        def adapters(request: Request):
            if (r := denied(request)) is not None:
                return r
            if time.time() - state["reloaded"] > 10:
                try:
                    runs_vol.reload()
                except Exception:
                    pass
                state["reloaded"] = time.time()
            items = list_adapters(RUNS_MOUNT)
            names = [a["name"] for a in items]
            return {"adapters": items, "default": default_adapter(names, DEFAULT_OVERRIDE),
                    "defaults": DEFAULTS, "limits": LIMITS, "base_model": BASE_MODEL,
                    "base_revision": BASE_REVISION[:8], "system_prompt": "system_v1"}

        @api.post("/api/warm")
        async def warm(request: Request):
            if (r := denied(request)) is not None:
                return r
            call = await Engine().status.spawn.aio()
            return {"started": call.object_id}

        @api.get("/api/status")
        async def status(request: Request):
            if (r := denied(request)) is not None:
                return r
            return await Engine().status.remote.aio()

        @api.post("/api/rewrite")
        async def rewrite(request: Request):
            if (r := denied(request)) is not None:
                return r
            try:
                req = clean_request(await request.json())
            except ValueError as e:
                return JSONResponse({"error": str(e)}, status_code=400)

            async def events():
                t0 = time.time()
                try:
                    async for ev in Engine().generate.remote_gen.aio(req):
                        if ev.get("type") == "start":
                            ev["queue_s"] = round(time.time() - t0, 3)
                        yield json.dumps(ev) + "\n"
                except Exception as e:
                    yield json.dumps({"type": "error", "side": "main", "error": "%s: %s" % (type(e).__name__, e)}) + "\n"
                yield json.dumps({"type": "wall", "wall_s": round(time.time() - t0, 3)}) + "\n"

            return StreamingResponse(events(), media_type="application/x-ndjson",
                                     headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"})

        return api
