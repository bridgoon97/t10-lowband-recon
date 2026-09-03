"""Static check (T13-A, post-rework R3): prove the ALGORITHM path (now INCLUDING
config.py — the switch location) never references the clean ref X, degrade
internals, OR an oracle-F0 backdoor.  R3 mutation sanity: re-introduce the
oracle backdoor in a temp file and show the static check now FAILS.
"""
import os
import re
from pathlib import Path
import subprocess
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FUSION = os.path.join(ROOT, "fusion")
# config.py is NOW in scope — it is where a switch (e.g. f0_use_oracle) would
# live; excluding it was the loophole the reviewer caught.
ALGO_FILES = [os.path.join(FUSION, fn) for fn in
              ("fusion.py", "align.py", "decision.py", "synthesis.py",
               "f0.py", "stft.py", "utils.py", "config.py",
               "trust.py", "voicing.py", "shape.py")]  # +N1 modules


def _grep_files(pattern, files):
    rx = re.compile(pattern)
    hits = []
    for fp in files:
        if not os.path.exists(fp):
            continue
        with open(fp, errors="ignore") as f:
            for i, line in enumerate(f, 1):
                if rx.search(line):
                    hits.append(f"{os.path.relpath(fp, ROOT)}:{i}:{line.rstrip()}")
    return len(hits), hits[:8]


# Forbidden tokens: clean-ref X, degrade internals, AND oracle-F0 backdoor.
FORBIDDEN = [
    r"\bX\b\s*=.*clean|clean_ref|x_clean|oracle_kill|kill_mask|true_snr",
    r"from\s+\.?degrade\s+import|import\s+degrade",
    r"degrade\.|apply_d1|DegradationConfig",
    r"\boracle\b|f0_use_oracle|_oracle_f0",
    # T13-N1: D5 internals (OFFLINE degrade) must not leak into the algorithm
    r"apply_d5|E_peak|d5_valley|d5_peak|d5_level|valley_mask|peak_mask",
]
# trust.py is the module that REJECTS oracle trust — it must be able to name
# what it rejects, so the generic oracle-token pattern is scoped out for it.
# (Its guard is separately unit-tested: test_N1_oracle_rejected + mutation.)
EXEMPT = {r"\boracle\b|f0_use_oracle|_oracle_f0": {"trust.py"}}


def test_static_no_X_in_algorithm():
    """Algorithm path (incl. config.py) must have 0 forbidden refs."""
    total = 0
    for pat in FORBIDDEN:
        files = [f for f in ALGO_FILES
                 if Path(f).name not in EXEMPT.get(pat, set())]
        n, sample = _grep_files(pat, files)
        total += n
        print(f"  grep -InE '{pat}' <algo+config>  -> {n} hit(s)")
        for s in sample:
            print(f"      {s}")
    print(f"  static check: {total} forbidden references in algorithm path "
          f"({'PASS (algorithm path clean)' if total == 0 else 'FAIL -- leakage!'})")
    assert total == 0, f"static check FAILED: {total} forbidden refs in algorithm path"


def test_static_fusion_imports():
    """Record what the algorithm modules import (allowed surface)."""
    n, sample = _grep_files(r"^\s*(import|from)\s", ALGO_FILES)
    print(f"  algorithm modules import statements: {n} (allowed surface):")
    for s in sample:
        print(f"      {s}")


def test_R3_oracle_mutation():
    """R3 mutation sanity: re-introduce the oracle-F0 backdoor in a temp config
    file (``f0_use_oracle: bool = True`` + the prod-path override line) and show
    the static check now FAILS (finds it).  Proves the check has teeth for the
    exact loophole the reviewer flagged (B-stage: switch opened ⇒ static FAIL)."""
    mutant = (
        "from dataclasses import dataclass\n"
        "@dataclass\n"
        "class FusionConfig:\n"
        "    f0_use_oracle: bool = True   # BACKDOOR re-introduced (mutation)\n"
        "    # in prod path: if cfg.f0_use_oracle: f0 = self._oracle_f0\n"
    )
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(mutant)
        tmp = f.name
    try:
        n, sample = _grep_files(r"\boracle\b|f0_use_oracle|_oracle_f0", [tmp])
    finally:
        os.unlink(tmp)
    caught = n > 0
    print(f"  R3 mutation (re-introduced f0_use_oracle backdoor): "
          f"static grep finds {n} hit(s) -> "
          f"{'FAIL-of-mutant (static catches it) PASS' if caught else 'NOT caught PROBLEM'}")
    for s in sample:
        print(f"      {s.split(os.sep)[-1]} :: {s.split(':', 2)[-1] if ':' in s else s}")
    assert caught, "R3 mutation not caught: static check would NOT fail if the oracle switch were opened"


if __name__ == "__main__":
    test_static_no_X_in_algorithm()
    test_static_fusion_imports()
    test_R3_oracle_mutation()
    print("static check tests: PASS")
