#!/usr/bin/env python3
"""Unified verification script — runs all §5 tests and writes a report.

    python verify.py                    # run all, write reports/verification_report.md
    python verify.py --only streaming   # run just one category
    python verify.py --arm arm_a_ddsp   # limit to one arm where applicable
"""
from __future__ import annotations

import argparse
import io
import sys
import time
import traceback
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent))

from tests._testutil import SkipTest


def _run(name, fn, results):
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    t0 = time.time()
    try:
        fn()
        results[name] = {"status": "PASS", "time_s": time.time() - t0,
                          "error": None}
        print(f"  >>> {name}: PASS ({time.time()-t0:.1f}s)")
    except SkipTest as e:
        # Not a failure: e.g. L1 data shard not downloaded on this checkout.
        results[name] = {"status": "SKIP", "time_s": time.time() - t0,
                          "error": str(e)}
        print(f"  >>> {name}: SKIP ({time.time()-t0:.1f}s) — {str(e).splitlines()[0]}")
    except Exception as e:
        results[name] = {"status": "FAIL", "time_s": time.time() - t0,
                          "error": str(e)}
        print(f"  >>> {name}: FAIL ({time.time()-t0:.1f}s)")
        traceback.print_exc()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None,
                    help="run only tests whose category name contains this "
                         "substring (matches both L0 and L1)")
    ap.add_argument("--report", default="reports/verification_report.md")
    args = ap.parse_args()

    # L0 = ideal-lowpass synthetic data (lowpass_sim).  These were the original
    # 21 tests; they all run on synthetic clean-speech → ideal-lowpass pairs.
    L0_TESTS = {
        "5.1_complexity": "tests.test_complexity",
        "5.2_causality": "tests.test_causality",
        "5.3_streaming": "tests.test_streaming",
        "5.4_overfit": "tests.test_overfit",
        "5.5_gradient": "tests.test_gradient",
        "5.6_stft_roundtrip": "tests.test_stft_roundtrip",
        "5.7_loss_stability": "tests.test_loss_stability",
        "5.8_degradation": "tests.test_degradation",
        "5.9_export": "tests.test_export",
        "5.10_smoke": "tests.test_smoke",
        "5.12_streaming_memory": "tests.test_streaming_memory",
        "6.1_ddsp_antialias": "tests.test_ddsp_antialias",
        "6.5_f0": "tests.test_f0",
    }
    # L1 = REAL body-conduction data (Vibravox forehead accelerometer → headset
    # air ref).  These SKIP (not fail) if the ~500 MB parquet shard is absent.
    L1_TESTS = {
        "4_l1_adapter": "tests.test_l1_adapter",
        "5.11_smoke_l1": "tests.test_smoke_l1",
    }

    def _filter(d):
        if not args.only:
            return d
        return {k: v for k, v in d.items() if args.only in k}

    l0_tests, l1_tests = _filter(L0_TESTS), _filter(L1_TESTS)
    results = {"L0": {}, "L1": {}}

    for level, tests in (("L0", l0_tests), ("L1", l1_tests)):
        for name, mod_path in tests.items():
            try:
                mod = __import__(mod_path, fromlist=["x"])
                for fname in dir(mod):
                    if fname.startswith("test_"):
                        _run(f"{level}::{name}::{fname}",
                              getattr(mod, fname), results[level])
            except SkipTest as e:
                results[level][name] = {"status": "SKIP",
                                        "time_s": 0.0, "error": str(e)}
                print(f"  >>> {name}: SKIP (import) — {str(e).splitlines()[0]}")
            except Exception as e:
                results[level][name] = {"status": "FAIL",
                                        "time_s": 0.0, "error": str(e)}
                print(f"  >>> {name}: FAIL (import)")
                traceback.print_exc()

    Path("reports").mkdir(exist_ok=True)
    with open(args.report, "w") as f:
        f.write("# Verification Report (layered: L0 ideal-lowpass / L1 real body-conduction)\n\n")
        f.write("Tests are split by DATA DOMAIN, not by feature.  L0 = synthetic\n")
        f.write("clean speech → ideal lowpass (``lowpass_sim``).  L1 = real Vibravox\n")
        f.write("body-conduction (forehead accelerometer) ↔ headset air reference.\n")
        f.write("L1 tests SKIP (not fail) if the local parquet shard is absent —\n")
        f.write("see ``reports/vibravox_schema_diff.md`` for how to fetch it.\n\n")
        for level, label in (("L0", "L0 — ideal lowpass (lowpass_sim)"),
                              ("L1", "L1 — real body-conduction (Vibravox)")):
            rs = results[level]
            f.write(f"## {label}\n\n")
            f.write("| Test | Status | Time (s) | Error |\n")
            f.write("|------|--------|----------|-------|\n")
            for name, r in rs.items():
                err = (r["error"] or "")
                if len(err) > 80:
                    err = err[:80] + "..."
                f.write(f"| {name} | {r['status']} | {r['time_s']:.1f} | {err} |\n")
            n_pass = sum(1 for r in rs.values() if r["status"] == "PASS")
            n_fail = sum(1 for r in rs.values() if r["status"] == "FAIL")
            n_skip = sum(1 for r in rs.values() if r["status"] == "SKIP")
            total = len(rs)
            f.write(f"\n**{level} summary: {n_pass}/{total} passed, "
                    f"{n_fail} failed, {n_skip} skipped**\n\n")
        # NO single aggregate number — a flat "N passed" hides the L0/L1 split,
        # which is exactly what the rework demanded.

    print(f"\n{'='*60}")
    print(f"  Report written to {args.report}")
    for level in ("L0", "L1"):
        rs = results[level]
        n_pass = sum(1 for r in rs.values() if r["status"] == "PASS")
        n_fail = sum(1 for r in rs.values() if r["status"] == "FAIL")
        n_skip = sum(1 for r in rs.values() if r["status"] == "SKIP")
        print(f"  {level}: {n_pass} passed, {n_fail} failed, {n_skip} skipped")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
