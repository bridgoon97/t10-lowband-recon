#!/usr/bin/env python3
"""T13-A test runner — runs the streaming (G5), mechanism (M1–M7), ablation,
and static test modules, prints a count + raw PASS/FAIL summary.

Usage:  python3 fusion/run_t13_tests.py
Exit code: 0 if all PASS, 1 if any FAIL.
"""
import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests._testutil import SkipTest

MODULES = [
    "tests.test_t13_streaming",
    "tests.test_t13_mechanisms",
    "tests.test_t13_real",
    "tests.test_t13_ablation",
    "tests.test_t13_static",
]


def main():
    total = 0
    passed = 0
    failed = 0
    skipped = 0
    for mod in MODULES:
        m = __import__(mod, fromlist=["x"])
        tests = sorted(f for f in dir(m) if f.startswith("test_"))
        for t in tests:
            total += 1
            print(f"\n{'=' * 70}\n  {mod}::{t}\n{'=' * 70}")
            try:
                getattr(m, t)()
                passed += 1
                print(f"  >>> {t}: PASS")
            except SkipTest as e:
                skipped += 1
                print(f"  >>> {t}: SKIP — {str(e).splitlines()[0]}")
            except Exception as e:
                failed += 1
                print(f"  >>> {t}: FAIL — {e}")
                traceback.print_exc()
    print(f"\n{'=' * 70}\n  T13-A SUMMARY: {passed}/{total} passed, {failed} failed, "
          f"{skipped} skipped\n{'=' * 70}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
