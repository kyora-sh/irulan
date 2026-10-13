"""Entry point for the irulan data pipeline, runnable with `python3 -I`.

`-I` keeps the current directory off sys.path, so this file adds its own
directory explicitly and dispatches to the package.

    python3 -I pipeline.py extract-cc [--src DIR] ...
    python3 -I pipeline.py rescrub
    python3 -I pipeline.py stats
    python3 -I pipeline.py test
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
    elif cmd == "test":
        import unittest
        suite = unittest.defaultTestLoader.discover(os.path.join(HERE, "tests"), top_level_dir=HERE)
        result = unittest.TextTestRunner(verbosity=1).run(suite)
        sys.exit(0 if result.wasSuccessful() else 1)
    else:
        sys.exit("unknown command: %s" % cmd)


if __name__ == "__main__":
    main()
