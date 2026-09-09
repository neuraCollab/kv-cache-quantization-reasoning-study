"""
Master test runner. Executes all four test suites in sequence and reports
overall pass/fail status for the thesis's theoretical claims.

Usage:
    python3 run_all_tests.py
"""
import subprocess
import sys
import time

SUITES = [
    ("Suite 1: Basic Invariants (INV-1..INV-4)", "test_01_invariants.py"),
    ("Suite 2: Main Theorem Predictions (PRED-A..C)", "test_02_main_theorem.py"),
    ("Suite 3: Trichotomy Takedown (FAIL-A..C)", "test_03_trichotomy.py"),
    ("Suite 4: Falsifiability Guards", "test_04_falsifiability.py"),
]


def main():
    print()
    print("#" * 78)
    print("#" + " " * 18 + "KV-QUANT LONG-CoT THESIS TEST HARNESS" + " " * 21 + "#")
    print("#" * 78)
    print()

    t_total_start = time.time()
    total_pass = 0
    total_fail = 0
    suite_results = []

    for suite_name, script in SUITES:
        print()
        print("=" * 78)
        print(f"  {suite_name}")
        print("=" * 78)
        t0 = time.time()
        proc = subprocess.run(
            [sys.executable, script],
            capture_output=True, text=True,
        )
        elapsed = time.time() - t0
        print(proc.stdout, end="")
        if proc.stderr:
            print("STDERR:", proc.stderr)

        # Parse "N passed, M failed" from output
        import re
        m = re.search(r"(\d+) passed, (\d+) failed", proc.stdout)
        if m:
            p, f = int(m.group(1)), int(m.group(2))
            total_pass += p
            total_fail += f
            suite_results.append((suite_name, p, f, elapsed))
        else:
            suite_results.append((suite_name, 0, 1, elapsed))
            total_fail += 1

    total_elapsed = time.time() - t_total_start

    # Final summary
    print()
    print("#" * 78)
    print("#" + " " * 30 + "FINAL SUMMARY" + " " * 33 + "#")
    print("#" * 78)
    for name, p, f, el in suite_results:
        status = "\033[92mPASS\033[0m" if f == 0 else "\033[91mFAIL\033[0m"
        print(f"  [{status}]  {name:<55} {p:3d}/{p+f:3d} tests  [{el:.2f}s]")
    print("-" * 78)
    overall = "\033[92m ALL TESTS PASSED \033[0m" if total_fail == 0 else "\033[91m SOME TESTS FAILED \033[0m"
    print(f"  Total: {total_pass} passed, {total_fail} failed  in {total_elapsed:.2f}s")
    print(f"  Overall: {overall}")
    print("#" * 78)
    print()

    return total_fail


if __name__ == "__main__":
    sys.exit(main())
