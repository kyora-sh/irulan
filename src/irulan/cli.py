"""irulan: prepare your dataset locally, fine-tune on kyora, rewrite text.

Local (no network):
  extract      walk ~/.claude/projects and write the raw extract of what you typed
  stats        print the counts of the last extract
  studio       a local page for writing handwritten pairs (generated with codex, or pasted)

kyora (needs IRULAN_BASE_URL and IRULAN_API_KEY; the dataset and fine-tune
endpoints are planned, see docs/API.md):
  upload       upload the raw extract as a dataset (asks first)
  datasets     list | status | review | approve | delete
  finetune     create | status | list | cancel
  models       list the models you can use
  rewrite      rewrite text with irulan-swe or your own model
"""

import argparse
import json
import os
import sys

from . import __version__, errors, upload as _upload
from .client import Client
from .pipeline import extract_cc, stats as _stats
from .rewrite import DEFAULT_MODEL, SYSTEM_VERSION
from .studio import server as _studio

EXIT_API = 1
EXIT_USAGE = 2
EXIT_NOT_AVAILABLE = 3
DEFAULT_DATA_DIR = extract_cc.DEFAULT_DATA_DIR


def _data_dir():
    return os.environ.get("IRULAN_DATA_DIR") or DEFAULT_DATA_DIR


def _print_json(obj):
    print(json.dumps(obj, indent=2, ensure_ascii=False))


def _client(args):
    return Client(api_key=getattr(args, "api_key", None), base_url=getattr(args, "base_url", None))


def _status_line(obj):
    bits = [str(obj.get("id")), str(obj.get("status"))]
    if obj.get("fine_tuned_model"):
        bits.append(obj["fine_tuned_model"])
    print("  ".join(bits), file=sys.stderr)


# --- local -----------------------------------------------------------------

def cmd_extract(args):
    extract_cc.run(args)
    return 0


def cmd_stats(args):
    path = args.path or os.path.join(_data_dir(), "stats", "cc_stats.json")
    if not os.path.exists(path):
        print("no stats at %s; run `irulan extract` first" % path, file=sys.stderr)
        return EXIT_USAGE
    with open(path, encoding="utf-8") as f:
        print(_stats.render(json.load(f)))
    return 0


def cmd_studio(args):
    return _studio.run(args)


# --- upload ----------------------------------------------------------------

def _confirm(prompt):
    if not sys.stdin.isatty():
        return False
    try:
        return input(prompt).strip().lower() in ("y", "yes")
    except EOFError:
        return False


def cmd_upload(args):
    path = args.path or os.path.join(_data_dir(), "raw", "cc_messages.jsonl")
    if not os.path.exists(path):
        print("no raw extract at %s; run `irulan extract` first" % path, file=sys.stderr)
        return EXIT_USAGE
    # The summary goes to stderr and the dataset object to stdout, so scripts
    # can read the result as JSON.
    data, summary = _upload.build(path, redact_secrets=args.redact_secrets_locally)
    print(_upload.render_summary(summary), file=sys.stderr)
    if args.payload_out:
        with open(args.payload_out, "wb") as f:
            f.write(data)
        print("wrote the exact upload payload to %s" % args.payload_out, file=sys.stderr)
    if args.dry_run:
        print("dry run: nothing was uploaded.", file=sys.stderr)
        return 0
    client = _client(args)
    client._t.url("datasets")  # raises NotAvailableYet before asking
    print("destination: %s" % client.base_url, file=sys.stderr)
    if not args.yes and not _confirm("Upload %d messages to %s? [y/N] " % (summary["records"], client.base_url)):
        print("not uploaded: confirm interactively or pass --yes.", file=sys.stderr)
        return EXIT_USAGE
    ds = client.datasets.upload_bytes(data, name=args.name or os.path.basename(path),
                                      review="skip" if args.skip_review else "required")
    _print_json(ds)
    return 0


# --- datasets --------------------------------------------------------------

def cmd_datasets(args):
    c = _client(args)
    if args.action == "list":
        _print_json(c.datasets.list())
    elif args.action == "status":
        if args.wait:
            d = c.datasets.wait(args.id, until=args.wait, poll=args.poll, on_update=_status_line)
        else:
            d = c.datasets.get(args.id)
        _print_json(d)
    elif args.action == "review":
        r = c.datasets.review(args.id)
        if args.out:
            with open(args.out, "w", encoding="utf-8") as f:
                f.write("count\tterm\texample\n")
                for cand in r.get("candidates", []):
                    f.write("%s\t%s\t%s\n" % (cand.get("count", ""), cand.get("term", ""),
                                              (cand.get("example") or "").replace("\t", " ").replace("\n", " ")))
            print("%d candidates -> %s" % (len(r.get("candidates", [])), args.out))
        else:
            _print_json(r)
    elif args.action == "approve":
        terms = []
        if args.denylist:
            with open(args.denylist, encoding="utf-8") as f:
                terms = [ln.strip() for ln in f if ln.strip() and not ln.lstrip().startswith("#")]
        _print_json(c.datasets.approve(args.id, denylist=terms))
    elif args.action == "delete":
        if not args.yes and not _confirm("Delete dataset %s and everything derived from it? [y/N] " % args.id):
            print("not deleted: confirm interactively or pass --yes.", file=sys.stderr)
            return EXIT_USAGE
        _print_json(c.datasets.delete(args.id))
    return 0


# --- fine-tunes ------------------------------------------------------------

def _hyperparameters(args):
    hp = json.loads(args.hyperparameters) if args.hyperparameters else {}
    if not isinstance(hp, dict):
        raise SystemExit("--hyperparameters must be a JSON object")
    for key in ("epochs", "learning_rate_multiplier", "lora_rank"):
        v = getattr(args, key)
        if v is not None:
            hp[key] = v
    return hp or None


def cmd_finetune(args):
    c = _client(args)
    if args.action == "create":
        job = c.finetunes.create(args.dataset, name=args.name, base=args.base, hyperparameters=_hyperparameters(args))
        if args.wait:
            _status_line(job)
            job = c.finetunes.wait(job["id"], poll=args.poll, on_update=_status_line)
        _print_json(job)
    elif args.action == "status":
        if args.wait:
            job = c.finetunes.wait(args.id, poll=args.poll, on_update=_status_line)
        else:
            job = c.finetunes.get(args.id)
        _print_json(job)
    elif args.action == "list":
        _print_json(c.finetunes.list())
    elif args.action == "cancel":
        _print_json(c.finetunes.cancel(args.id))
    return 0


# --- inference -------------------------------------------------------------

def cmd_models(args):
    _print_json(_client(args).models.list())
    return 0


def cmd_rewrite(args):
    if args.file:
        with open(args.file, encoding="utf-8") as f:
            text = f.read()
    elif args.text:
        text = " ".join(args.text)
    elif not sys.stdin.isatty():
        text = sys.stdin.read()
    else:
        print("give the text as arguments, with --file, or on stdin", file=sys.stderr)
        return EXIT_USAGE
    if not text.strip():
        print("nothing to rewrite", file=sys.stderr)
        return EXIT_USAGE
    out = _client(args).rewrite(text, model=args.model, temperature=args.temperature, max_tokens=args.max_tokens)
    sys.stdout.write(out if out.endswith("\n") else out + "\n")
    return 0


# --- parser ----------------------------------------------------------------

def _api_flags(p):
    p.add_argument("--base-url", help="API base URL incl. /v1 (default: $IRULAN_BASE_URL)")
    p.add_argument("--api-key", help="API key (default: $IRULAN_API_KEY; prefer the env var)")


def build_parser():
    ap = argparse.ArgumentParser(prog="irulan", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", action="version", version="irulan %s" % __version__)
    sub = ap.add_subparsers(dest="command", metavar="COMMAND")

    p = sub.add_parser("extract", aliases=["extract-cc"], help="extract what you typed from Claude Code transcripts",
                       description=extract_cc.__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    extract_cc.add_arguments(p)
    p.set_defaults(func=cmd_extract)

    p = sub.add_parser("stats", help="print the counts of the last extract")
    p.add_argument("path", nargs="?", help="stats file (default: DATA_DIR/stats/cc_stats.json)")
    p.set_defaults(func=cmd_stats)

    p = sub.add_parser("studio", help="a local page for writing handwritten pairs",
                       description=_studio.__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    _studio.add_arguments(p)
    p.set_defaults(func=cmd_studio)

    p = sub.add_parser("upload", help="upload the raw extract as a dataset (planned API)",
                       description="Upload the raw extract to kyora as a dataset. The text is NOT scrubbed "
                                   "locally; the platform scrubs it after upload (docs/API.md). Shows a "
                                   "summary first and needs --yes or an interactive yes.")
    p.add_argument("path", nargs="?", help="raw extract (default: DATA_DIR/raw/cc_messages.jsonl)")
    p.add_argument("--name", help="dataset name (default: the file name)")
    p.add_argument("--redact-secrets-locally", action="store_true",
                   help="replace obvious credentials (API keys, tokens, private keys, JWTs, passwords) "
                        "with <SECRET> before upload; names and hosts are left to the platform")
    p.add_argument("--skip-review", action="store_true",
                   help="let the platform generate pairs without waiting for your review of its name list")
    p.add_argument("--yes", "-y", action="store_true", help="upload without asking")
    p.add_argument("--dry-run", action="store_true", help="print the summary and stop")
    p.add_argument("--payload-out", metavar="FILE", help="also write the exact payload to FILE for inspection")
    _api_flags(p)
    p.set_defaults(func=cmd_upload)

    p = sub.add_parser("datasets", help="list | status | review | approve | delete (planned API)")
    dsub = p.add_subparsers(dest="action", metavar="ACTION")
    dsub.required = True
    q = dsub.add_parser("list", help="your datasets")
    _api_flags(q)
    q = dsub.add_parser("status", help="a dataset's processing stage")
    q.add_argument("id")
    q.add_argument("--wait", metavar="STAGE", choices=("uploaded", "scrubbed", "pairs", "ready"),
                   help="poll until the dataset reaches STAGE")
    q.add_argument("--poll", type=float, default=10.0, help="seconds between polls")
    _api_flags(q)
    q = dsub.add_parser("review", help="download the platform's name/entity review list")
    q.add_argument("id")
    q.add_argument("--out", metavar="FILE", help="write it as TSV (count, term, example)")
    _api_flags(q)
    q = dsub.add_parser("approve", help="approve the review, optionally adding terms to scrub")
    q.add_argument("id")
    q.add_argument("--denylist", metavar="FILE", help="one term per line, scrubbed as <NAME>")
    _api_flags(q)
    q = dsub.add_parser("delete", help="delete a dataset and everything derived from it")
    q.add_argument("id")
    q.add_argument("--yes", "-y", action="store_true")
    _api_flags(q)
    p.set_defaults(func=cmd_datasets)

    p = sub.add_parser("finetune", help="create | status | list | cancel (planned API)")
    fsub = p.add_subparsers(dest="action", metavar="ACTION")
    fsub.required = True
    q = fsub.add_parser("create", help="fine-tune your own irulan-<user>-<name> adapter")
    q.add_argument("--dataset", required=True, help="dataset id (status ready)")
    q.add_argument("--name", required=True, help="model name; the model id becomes irulan-<user>-<name>")
    q.add_argument("--base", default=DEFAULT_MODEL, help="base model (default: %(default)s)")
    q.add_argument("--epochs", type=int)
    q.add_argument("--learning-rate-multiplier", type=float)
    q.add_argument("--lora-rank", type=int)
    q.add_argument("--hyperparameters", metavar="JSON", help="extra hyperparameters as a JSON object")
    q.add_argument("--wait", action="store_true", help="poll until the job ends")
    q.add_argument("--poll", type=float, default=15.0)
    _api_flags(q)
    q = fsub.add_parser("status", help="a fine-tune's status, events and metrics")
    q.add_argument("id")
    q.add_argument("--wait", action="store_true", help="poll until the job ends")
    q.add_argument("--poll", type=float, default=15.0)
    _api_flags(q)
    q = fsub.add_parser("list", help="your fine-tunes")
    _api_flags(q)
    q = fsub.add_parser("cancel", help="cancel a queued or running fine-tune")
    q.add_argument("id")
    _api_flags(q)
    p.set_defaults(func=cmd_finetune)

    p = sub.add_parser("models", help="list the models you can use")
    _api_flags(p)
    p.set_defaults(func=cmd_models)

    p = sub.add_parser("rewrite", help="rewrite text with an irulan model",
                       description="Rewrite text with irulan-swe or your own model. Sends the %s system "
                                   "instruction the training data uses." % SYSTEM_VERSION)
    p.add_argument("text", nargs="*", help="text to rewrite (or --file, or stdin)")
    p.add_argument("--file", "-f")
    p.add_argument("--model", "-m", default=os.environ.get("IRULAN_MODEL") or DEFAULT_MODEL,
                   help="model id (default: $IRULAN_MODEL or %s)" % DEFAULT_MODEL)
    p.add_argument("--temperature", type=float)
    p.add_argument("--max-tokens", type=int)
    _api_flags(p)
    p.set_defaults(func=cmd_rewrite)
    return ap


def main(argv=None):
    ap = build_parser()
    args = ap.parse_args(argv)
    if not getattr(args, "func", None):
        ap.print_help()
        return EXIT_USAGE
    try:
        return args.func(args)
    except errors.NotAvailableYet as e:
        print("irulan: %s" % e, file=sys.stderr)
        return EXIT_NOT_AVAILABLE
    except errors.FineTuneFailed as e:
        print("irulan: %s" % e, file=sys.stderr)
        _print_json(e.job)
        return EXIT_API
    except errors.IrulanError as e:
        print("irulan: %s" % e, file=sys.stderr)
        return EXIT_API
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
