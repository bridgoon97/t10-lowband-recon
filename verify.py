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
    except Exception as e:
        results[name] = {"status": "FAIL", "time_s": time.time() - t0,
                          "error": str(e)}
        print(f"  >>> {name}: FAIL ({time.time()-t0:.1f}s)")
        traceback.print_exc()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None,
                    help="run only this category (streaming/overfit/etc)")
    ap.add_argument("--report", default="reports/verification_report.md")
    args = ap.parse_args()

    tests = {
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
        "6.1_ddsp_antialias": "tests.test_ddsp_antialias",
        "6.5_f0": "tests.test_f0",
    }

    if args.only:
        tests = {k: v for k, v in tests.items() if args.only in k}

    results = {}
    for name, mod_path in tests.items():
        try:
            mod = __import__(mod_path, fromlist=["x"])
            # Find the main test function
            for fname in dir(mod):
                if fname.startswith("test_"):
                    _run(f"{name}::{fname}", getattr(mod, fname), results)
        except Exception as e:
            _run(name, lambda: (_ for _ in ()).throw(e), results)

    # Write report
    Path("reports").mkdir(exist_ok=True)
    with open(args.report, "w") as f:
        f.write("# Verification Report (§5 + §6 tests)\n\n")
        f.write("| Test | Status | Time (s) | Error |\n")
        f.write("|------|--------|----------|-------|\n")
        for name, r in results.items():
            err = r["error"][:60] + "..." if r["error"] and len(r["error"]) > 60 else (r["error"] or "")
            f.write(f"| {name} | {r['status']} | {r['time_s']:.1f} | {err} |\n")
        n_pass = sum(1 for r in results.values() if r["status"] == "PASS")
        n_fail = sum(1 for r in results.values() if r["status"] == "FAIL")
        f.write(f"\n**Summary: {n_pass} passed, {n_fail} failed**\n")

    print(f"\n{'='*60}")
    print(f"  Report written to {args.report}")
    n_pass = sum(1 for r in results.values() if r["status"] == "PASS")
    n_fail = sum(1 for r in results.values() if r["status"] == "FAIL")
    print(f"  {n_pass} passed, {n_fail} failed")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
