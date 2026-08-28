"""Tiny test utilities shared by the L1 tests (no pytest dependency).

``SkipTest`` lets a test say "I can't run here, but that's not a failure"
(e.g. the ~500 MB L1 data shard isn't downloaded on this checkout).  verify.py
catches it and reports status=SKIP, which does NOT count as pass or fail.
"""


class SkipTest(Exception):
    """Raised to skip a test without counting it as pass or fail."""
    pass


def skip_if_no_l1(shard: str) -> None:
    from pathlib import Path
    if not Path(shard).exists():
        raise SkipTest(
            f"L1 data shard missing: {shard}\n"
            f"  (L1 uses a ~500 MB HuggingFace parquet shard that isn't in "
            f"every checkout. Fetch it via scripts/inspect_vibravox_local.py "
            f"to enable the L1 tests.)")
