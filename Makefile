PY := python3 -I

.PHONY: extract stats studio test

# Walk ~/.claude/projects and write data/raw + data/stats. Nothing leaves the machine.
# Pass filter flags via ARGS, e.g. ARGS="--keep-non-english".
extract:
	$(PY) pipeline.py extract $(ARGS)

stats:
	$(PY) pipeline.py stats

# Local page for handwritten pairs, e.g. ARGS="--port 8765" or ARGS="--fake".
studio:
	$(PY) pipeline.py studio $(ARGS)

test:
	$(PY) pipeline.py test
