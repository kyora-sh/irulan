PY := python3 -I

.PHONY: extract-cc rescrub stats test generate check-pairs sample review prepare train-dry-run

# Walk ~/.claude/projects, write data/raw + data/clean + entity candidates.
# Pass filter flags via ARGS, e.g. ARGS="--keep-non-english".
extract-cc:
	$(PY) pipeline.py extract-cc $(ARGS)

# Reapply scrubbing to data/raw after editing scrub/denylist.txt (no re-walk).
rescrub:
	$(PY) pipeline.py rescrub

stats:
	$(PY) pipeline.py stats

test:
	$(PY) pipeline.py test

# Back-translate clean messages into assistant-style inputs, e.g.
#   make generate ARGS="--backend codex --model luna --run luna_try --limit 20"
generate:
	$(PY) pipeline.py generate $(ARGS)

# Flag rates for generated pairs: make check-pairs ARGS="data/pairs/luna_try.jsonl"
check-pairs:
	$(PY) pipeline.py check-pairs $(ARGS)

# Fixed stratified 50-message sample for model comparisons.
sample:
	$(PY) pipeline.py sample $(ARGS)

# Side-by-side review: make review ARGS="--runs data/pairs/a.jsonl data/pairs/b.jsonl"
review:
	$(PY) pipeline.py review $(ARGS)

# Training set from a pairs file: make prepare ARGS="--pairs data/pairs/full_sol_v2.jsonl"
prepare:
	$(PY) pipeline.py prepare $(ARGS)

# Check data/train and print the cost estimate. Local, no Modal, no GPU.
train-dry-run:
	$(PY) train/modal_train.py train --dry-run $(ARGS)
