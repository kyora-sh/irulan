PY := python3 -I

.PHONY: extract-cc rescrub stats test

# Walk ~/.claude/projects, write data/raw + data/clean + entity candidates.
extract-cc:
	$(PY) pipeline.py extract-cc

# Reapply scrubbing to data/raw after editing scrub/denylist.txt (no re-walk).
rescrub:
	$(PY) pipeline.py rescrub

stats:
	$(PY) pipeline.py stats

test:
	$(PY) pipeline.py test
