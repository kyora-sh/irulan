PY := python3 -I

.PHONY: extract stats test

# Walk ~/.claude/projects and write data/raw + data/stats. Nothing leaves the machine.
# Pass filter flags via ARGS, e.g. ARGS="--keep-non-english".
extract:
	$(PY) pipeline.py extract $(ARGS)

stats:
	$(PY) pipeline.py stats

test:
	$(PY) pipeline.py test
