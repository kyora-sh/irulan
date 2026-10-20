"""irulan studio: write handwritten pairs on a local page (`irulan studio`)."""

from .generators import CodexGenerator, FakeGenerator
from .pairs import FIELDS, PairStore, check_pair, make_record, pair_id
from .prompts import LENGTHS, PROMPT_VERSION, SHAPES, StudioError, build_prompt
from .server import StudioServer, add_arguments, run

__all__ = [
    "CodexGenerator", "FakeGenerator", "FIELDS", "PairStore", "check_pair", "make_record", "pair_id",
    "LENGTHS", "PROMPT_VERSION", "SHAPES", "StudioError", "build_prompt", "StudioServer", "add_arguments", "run",
]
