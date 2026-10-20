"""Run the irulan CLI from a checkout without installing it.

    python3 -I pipeline.py extract [--src DIR] [--keep-non-english] ...
    python3 -I pipeline.py stats
    python3 -I pipeline.py studio [--port 8765] [--fake] [--stats]
    python3 -I pipeline.py upload --dry-run
    python3 -I pipeline.py test

Same commands as the `irulan` console script (`pip install -e .`); `test`
runs the unit tests. `-I` keeps the current directory off sys.path, so this
file adds ./src explicitly.
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "src"))


def main():
    if sys.argv[1:2] == ["test"]:
        import unittest
        suite = unittest.defaultTestLoader.discover(os.path.join(HERE, "tests"), top_level_dir=HERE)
        result = unittest.TextTestRunner(verbosity=1).run(suite)
        return 0 if result.wasSuccessful() else 1
    from irulan.cli import main as cli_main
    return cli_main(sys.argv[1:])


if __name__ == "__main__":
    sys.exit(main())
