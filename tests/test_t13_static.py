"""Static check (T13-A): prove the FUSION ALGORITHM path never references
  * X — the clean FF reference (must not enter the algorithm)
  * degrade-model internals — the real kill mask / kill positions / real SNR
    (degrade is OFFLINE data-prep; the algorithm estimates its own F0).

Grep-level proof across the ALGORITHM modules only
(fusion.py / align.py / decision.py / synthesis.py / f0.py / stft.py / utils.py)
— NOT __init__.py (it exports degrade for the test harness), NOT degrade.py
itself, NOT signals.py (synthetic builders for tests), NOT config.py.
"""
import os
import re
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FUSION = os.path.join(ROOT, "fusion")
ALGO_FILES = [os.path.join(FUSION, fn) for fn in
              ("fusion.py", "align.py", "decision.py", "synthesis.py",
               "f0.py", "stft.py", "utils.py")]


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


def test_static_no_X_in_algorithm():
    """Algorithm path must not reference clean-ref X or degrade internals."""
    forbidden = [
        r"\bX\b\s*=.*clean|clean_ref|x_clean|oracle_kill|kill_mask|true_snr",
        r"from\s+\.?degrade\s+import|import\s+degrade",
        r"degrade\.|apply_d1|DegradationConfig",
    ]
    total = 0
    for pat in forbidden:
        n, sample = _grep_files(pat, ALGO_FILES)
        total += n
        print(f"  grep -InE '{pat}' <algo files>  -> {n} hit(s)")
        for s in sample:
            print(f"      {s}")
    print(f"  static check: {total} forbidden references in algorithm path "
          f"({'PASS (algorithm path clean)' if total == 0 else 'FAIL -- leakage!'})")
    assert total == 0, f"static check FAILED: {total} forbidden refs in algorithm path"


def test_static_fusion_imports():
    """Record what the algorithm modules import (the allowed surface)."""
    n, sample = _grep_files(r"^\s*(import|from)\s", ALGO_FILES)
    print(f"  algorithm modules import statements: {n} (allowed surface):")
    for s in sample:
        print(f"      {s}")


if __name__ == "__main__":
    test_static_no_X_in_algorithm()
    test_static_fusion_imports()
    print("static check tests: PASS")
