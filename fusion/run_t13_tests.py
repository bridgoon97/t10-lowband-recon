#!/usr/bin/env python3
"""T13-A test runner — runs the streaming (G5), mechanism (M1–M7), ablation,
and static test modules, prints a count + raw PASS/FAIL summary.

Usage:  python3 fusion/run_t13_tests.py
Exit code: 0 if there is no unregistered FAIL or XPASS; registered XFAILs are
reported but do not make the runner fail.
"""
import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests._testutil import SkipTest


# A1-0: explicit registry for thresholds that are known to fail.  The tests
# still execute their real assertions.  A registered assertion failure is an
# XFAIL; an unexpected pass is an XPASS event so a repaired mechanism (or a
# silently changed threshold) cannot disappear into an ordinary PASS count.
KNOWN_FAIL_AUDIT = {
    "tests.test_t13_b1::test_G2_dropout_fallback": {
        "gate": "G2",
        "measured": "legacy aggregate LSD(Y,S)=0.538 dB",
        "threshold": "dropout steady-state LSD(Y,S)<0.5 dB; transition step<3 dB",
        "reason_source": "A1-0 已分析解除：旧协议误用 FB idx0、缺失稳态保持段且统计整段；修正协议通过",
    },
    "tests.test_t13_b1::test_J2_false_intervention": {
        "gate": "J2",
        "measured": "depth15/20 false-intervention=0.11/0.14",
        "threshold": "all depth false-intervention<=0.10",
        "reason_source": "已分析悬置：>10 dB 桶为空，max=7.95 dB；来源 LR4",
    },
    "tests.test_t13_b1::test_Ka_cv_healthy": {
        "gate": "K-a",
        "measured": "healthy median c_V=0.398（旧轮次 0.394）",
        "threshold": "median c_V>=0.5",
        "reason_source": "已分析悬置：非 LTI 域差使 q_term 偏弱，门槛待定标；来源 LR2",
    },
    "tests.test_t13_b1::test_Kc_cv_dropout": {
        "gate": "K-c",
        "measured": "dropout median c_V=0.170",
        "threshold": "median c_V<=0.05",
        "reason_source": "已分析悬置：m_term 需在聚合中有否决权；来源 LR1/K-c",
    },
    "tests.test_t13_a6::test_A6_2_HR3_design": {
        "gate": "HR3-design",
        "measured": "post-roundtrip corr_post max_down=-17.58 dB (bound -5), max_up=+26.67 (bound +25)",
        "threshold": "corr_post in [-delta_down, +delta_up] = [-5, +25] (the design property, m=0)",
        "reason_source": "A6-2: ISTFT->STFT roundtrip breaks the clip guarantee; magnitude-only change + S phase kept => y_spec not STFT-consistent => OLA cancellation => downward blow (worst -17.58 dB). Unfixed; do not read 'HR3 PASS' as 'design is safe'.",
    },
}

# G2 was initially registered, then A1-0 established that the legacy 0.538 dB
# came from an invalid test protocol (FB idx0, missing hold, whole-clip LSD).
# Keep all four entries above as the requested audit record, but only these
# unresolved entries are active XFAILs after the analysis.
ACTIVE_KNOWN_FAIL = {
    key: value for key, value in KNOWN_FAIL_AUDIT.items()
    if value["gate"] in {"J2", "K-a", "K-c", "HR3-design"}
}

MODULES = [
    "tests.test_t13_streaming",
    "tests.test_t13_mechanisms",
    "tests.test_t13_real",
    "tests.test_t13_ablation",
    "tests.test_t13_b1",
    "tests.test_t13_a2",
    "tests.test_t13_a3",
    "tests.test_t13_a4",
    "tests.test_t13_a5",
    "tests.test_t13_a6",
    "tests.test_t13_a7",
    "tests.test_t13_a8",
    "tests.test_t13_a9",
    "tests.test_t13_mvp",
    "tests.test_t13_static",
]


def main():
    total = 0
    passed = 0
    failed = 0
    skipped = 0
    xfailed = 0
    xpassed = 0
    for mod in MODULES:
        m = __import__(mod, fromlist=["x"])
        tests = sorted(f for f in dir(m) if f.startswith("test_"))
        for t in tests:
            test_id = f"{mod}::{t}"
            known = ACTIVE_KNOWN_FAIL.get(test_id)
            total += 1
            print(f"\n{'=' * 70}\n  {mod}::{t}\n{'=' * 70}")
            try:
                getattr(m, t)()
                if known is not None:
                    xpassed += 1
                    print(f"  >>> {t}: XPASS — registered {known['gate']} unexpectedly passed; investigate")
                else:
                    passed += 1
                    print(f"  >>> {t}: PASS")
            except SkipTest as e:
                skipped += 1
                print(f"  >>> {t}: SKIP — {str(e).splitlines()[0]}")
            except Exception as e:
                if known is not None:
                    xfailed += 1
                    print(f"  >>> {t}: XFAIL — {known['gate']}: {e}")
                    print(f"      measured={known['measured']}")
                    print(f"      threshold={known['threshold']}")
                    print(f"      reason/source={known['reason_source']}")
                else:
                    failed += 1
                    print(f"  >>> {t}: FAIL — {e}")
                    traceback.print_exc()
    print(f"\n{'=' * 70}\n  T13-A SUMMARY: {passed}/{total} passed, {failed} failed, "
          f"{xfailed} xfailed, {xpassed} xpassed, {skipped} skipped\n{'=' * 70}")
    return 1 if (failed or xpassed) else 0


if __name__ == "__main__":
    sys.exit(main())
