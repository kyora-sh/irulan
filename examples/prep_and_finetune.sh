#!/usr/bin/env bash
# From your own Claude Code transcripts to your own irulan model.
#
#   IRULAN_BASE_URL=... IRULAN_API_KEY=... examples/prep_and_finetune.sh my-voice
#
# Steps 1-2 run locally. Step 3 uploads UNSCRUBBED text (obvious credentials
# redacted locally); kyora scrubs it on receipt and stores only the scrubbed
# version. The dataset and fine-tune endpoints are planned (docs/API.md).
set -euo pipefail
name="${1:?usage: $0 MODEL_NAME}"

# 1. Extract what you typed. Writes data/raw/cc_messages.jsonl and data/stats/.
irulan extract
irulan stats

# 2. Preview exactly what would be uploaded. Read it before going on.
irulan upload --dry-run --redact-secrets-locally --payload-out data/upload_preview.jsonl
read -r -p "Reviewed data/upload_preview.jsonl? Upload it? [y/N] " ok
[[ "$ok" == [yY]* ]] || exit 1

# 3. Upload. The platform scrubs it and builds the review list.
ds=$(irulan upload --yes --redact-secrets-locally | python3 -c 'import json,sys; print(json.load(sys.stdin)["id"])')
irulan datasets status "$ds" --wait scrubbed

# 4. Review the names the platform found. Add any it should also scrub to a
#    file, one per line, then approve.
irulan datasets review "$ds" --out data/review.tsv
${EDITOR:-vi} data/review.tsv
touch data/denylist.local.txt
${EDITOR:-vi} data/denylist.local.txt
irulan datasets approve "$ds" --denylist data/denylist.local.txt
irulan datasets status "$ds" --wait ready

# 5. Fine-tune and try it.
model=$(irulan finetune create --dataset "$ds" --name "$name" --wait | python3 -c 'import json,sys; print(json.load(sys.stdin)["fine_tuned_model"])')
irulan rewrite --model "$model" "I wanted to follow up on whether you had a chance to look at the PR."
