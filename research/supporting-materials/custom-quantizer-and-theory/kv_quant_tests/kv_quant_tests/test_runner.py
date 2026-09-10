"""
Lightweight test runner. Replicates `pytest -v` UX without pytest.

Usage:
    from test_runner import test, run_all
    @test
    def test_something():
        assert 1 == 1

    if __name__ == "__main__":
        run_all()
"""
import sys
import time
import traceback


_TESTS = []


def test(fn):
    """Decorator: register a test function."""
    _TESTS.append(fn)
    return fn


def _fmt_duration(s: float) -> str:
    if s < 1e-3:
        return f"{s*1e6:.0f}us"
    if s < 1:
        return f"{s*1e3:.0f}ms"
    return f"{s:.2f}s"


def run_all(verbose: bool = True) -> int:
    """Run every registered test. Returns number of failures."""
    n_pass = 0
    n_fail = 0
    failures = []
    total_t0 = time.time()
    for fn in _TESTS:
        t0 = time.time()
        name = fn.__name__
        try:
            fn()
            dt = time.time() - t0
            if verbose:
                print(f"  \033[92mPASS\033[0m  {name}  [{_fmt_duration(dt)}]")
            n_pass += 1
        except Exception as e:
            dt = time.time() - t0
            tb = traceback.format_exc()
            failures.append((name, tb))
            if verbose:
                print(f"  \033[91mFAIL\033[0m  {name}  [{_fmt_duration(dt)}]")
            n_fail += 1

    total_dt = time.time() - total_t0
    print()
    if failures:
        print("=" * 70)
        print("FAILURE DETAILS")
        print("=" * 70)
        for name, tb in failures:
            print(f"\n--- {name} ---")
            print(tb)
    print("=" * 70)
    status = "\033[92mOK\033[0m" if n_fail == 0 else "\033[91mFAILED\033[0m"
    print(f"{status}  {n_pass} passed, {n_fail} failed  in {_fmt_duration(total_dt)}")
    print("=" * 70)
    return n_fail


def approx(actual, expected, rtol=1e-6, atol=1e-9, msg=""):
    """Assert approximately equal (relative or absolute)."""
    diff = abs(actual - expected)
    tol = max(atol, rtol * abs(expected))
    if diff > tol:
        raise AssertionError(
            f"{msg} expected ~{expected:.6g}, got {actual:.6g} "
            f"(|diff|={diff:.3g}, tol={tol:.3g})"
        )
