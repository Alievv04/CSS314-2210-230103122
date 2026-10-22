
import os
import platform
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from numba import njit

STUDENT_ID = 230103122      # <-- your numeric Student ID
SIZES = [100, 1_000, 10_000, 100_000, 1_000_000, 10_000_000]
REFINE_POINTS = 14          # extra sizes inside the crossover bracket
A, B = 2.5, 1.0
WORKERS = os.cpu_count() or 1


@njit(nogil=True)
def kernel(x, y, a, b, lo, hi):
    for i in range(lo, hi):
        y[i] = a * x[i] + b


def time_call(fn, target=0.02, repeats=5):
    """Best (minimum) per-call time in seconds, using batches of ~`target` s."""
    fn()                                            # warm-up
    t0 = time.perf_counter()
    fn()
    est = max(time.perf_counter() - t0, 1e-8)
    reps = int(min(max(target / est, 1), 200_000))
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        for _ in range(reps):
            fn()
        best = min(best, (time.perf_counter() - t0) / reps)
    return best


def measure(n, pool):
    x = np.random.default_rng(STUDENT_ID).random(n)
    y = np.empty(n)
    bounds = np.linspace(0, n, WORKERS + 1, dtype=np.int64)
    ranges = [(int(bounds[i]), int(bounds[i + 1])) for i in range(WORKERS)]

    def serial():
        kernel(x, y, A, B, 0, n)

    def parallel():
        futures = [pool.submit(kernel, x, y, A, B, lo, hi) for lo, hi in ranges]
        for f in futures:
            f.result()

    t_ser = time_call(serial)
    ref = y.copy()
    t_par = time_call(parallel)
    assert np.allclose(y, ref) and np.allclose(y, A * x + B), "wrong result!"
    return t_ser, t_par


def find_crossover(sizes, speedups):
    """Return (last_N_slower, first_N_faster) where parallel becomes faster
    and stays faster for every larger size. Either may be None."""
    first = None
    for i in range(len(sizes)):
        if all(s > 1.0 for s in speedups[i:]):
            first = i
            break
    if first is None:
        return sizes[-1], None                      # never (stably) faster
    if first == 0:
        return None, sizes[0]                       # faster everywhere tested
    return sizes[first - 1], sizes[first]


def print_rows(rows):
    header = (f"{'N':>12} | {'Serial (us)':>12} | {'Parallel (us)':>14} | "
              f"{'Speedup':>8} | Winner")
    print(header)
    print("-" * len(header))
    for n, ts, tp in rows:
        sp = ts / tp
        print(f"{n:>12,} | {ts * 1e6:>12.2f} | {tp * 1e6:>14.2f} | {sp:>7.2f}x | "
              f"{'PARALLEL' if sp > 1 else 'serial'}")


def main():
    print(f"Student ID    : {STUDENT_ID}")
    print(f"OS            : {platform.system()} {platform.release()}")
    print(f"CPU / Platform: {platform.processor() or platform.machine()}")
    print(f"Logical cores : {os.cpu_count()} (workers used: {WORKERS})")
    print(f"Operation     : y[i] = {A} * x[i] + {B} (float64), best of 5 batches per size")
    print()

    kernel(np.zeros(4), np.zeros(4), A, B, 0, 4)    # compile before timing

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        print("Coarse sweep")
        rows = [(n, *measure(n, pool)) for n in SIZES]
        print_rows(rows)
        print()

        speedups = [ts / tp for _, ts, tp in rows]
        slower_n, faster_n = find_crossover(SIZES, speedups)

        if slower_n is not None and faster_n is not None:
            sizes_fine = sorted({int(round(v)) for v in
                                 np.geomspace(slower_n, faster_n, REFINE_POINTS + 2)})
            print(f"Refinement sweep between N = {slower_n:,} and N = {faster_n:,}")
            fine_rows = [(n, *measure(n, pool)) for n in sizes_fine]
            print_rows(fine_rows)
            print()
            fine_speed = [ts / tp for _, ts, tp in fine_rows]
            s2, f2 = find_crossover(sizes_fine, fine_speed)
            if s2 is not None and f2 is not None:
                slower_n, faster_n = s2, f2         # clean, refined crossover
            else:
                print("Note: the refinement sweep was noisy (no clean crossover); "
                      "reporting the coarse bracket instead.")
                print()

    print("=== CROSSOVER RESULT ===")
    if faster_n is None:
        print(f"Parallel was never stably faster than serial up to N = {SIZES[-1]:,} "
              f"on this machine.")
    elif slower_n is None:
        print(f"Parallel was already faster than serial at the smallest tested size "
              f"(N = {faster_n:,}); crossover is below that.")
    else:
        print(f"Parallel is SLOWER than serial up to N = {slower_n:,} and "
              f"FASTER from N = {faster_n:,} onwards.")
        print(f"Crossover range: {slower_n:,} < N <= {faster_n:,}")


if __name__ == "__main__":
    main()