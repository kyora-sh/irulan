"""Entry point for the irulan data pipeline, runnable with `python3 -I`.

`-I` keeps the current directory off sys.path, so this file adds its own
directory explicitly and dispatches to the package.

    python3 -I pipeline.py extract-cc [--src DIR] [--keep-non-english] [--filter-config F] ...
    python3 -I pipeline.py rescrub
    python3 -I pipeline.py stats
    python3 -I pipeline.py test
    python3 -I pipeline.py generate --backend codex|fake --run NAME ...
    python3 -I pipeline.py check-pairs data/pairs/NAME.jsonl ...
    python3 -I pipeline.py sample
    python3 -I pipeline.py review --runs data/pairs/A.jsonl data/pairs/B.jsonl
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(__doc__)
        return
    cmd, rest = sys.argv[1], sys.argv[2:]
    if cmd == "extract-cc":
        from irulan import extract_cc
        extract_cc.main(rest)
    elif cmd == "rescrub":
        from irulan import extract_cc
        extract_cc.main(["--rescrub"] + rest)
    elif cmd == "stats":
        import json
        from irulan import stats
        path = rest[0] if rest else "data/stats/cc_stats.json"
        with open(path, encoding="utf-8") as f:
            print(stats.render(json.load(f)))
    elif cmd == "generate":
        from irulan import generate
        sys.exit(generate.main(rest))
    elif cmd == "check-pairs":
        from irulan import check_pairs
        check_pairs.main(rest)
    elif cmd == "sample":
        from irulan import sample
        sample.main_sample(rest)
    elif cmd == "review":
        from irulan import sample
        sample.main_review(rest)
    elif cmd == "test":
        import unittest
        suite = unittest.defaultTestLoader.discover(os.path.join(HERE, "tests"), top_level_dir=HERE)
        result = unittest.TextTestRunner(verbosity=1).run(suite)
        sys.exit(0 if result.wasSuccessful() else 1)
    else:
        sys.exit("unknown command: %s" % cmd)


if __name__ == "__main__":
    main()
