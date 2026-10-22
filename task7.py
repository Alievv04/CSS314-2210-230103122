
import math
import os
import platform
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from numba import njit

STUDENT_ID = 230103122
N = 30_000_000
RUNS = 5
STALL_EFFICIENCY = 0.5


@njit(nogil=True)
def kernel_copy(a, b, lo, hi):
    for i in range(lo, hi):
        b[i] = a[i]


@njit(nogil=True)
def kernel_trig(a, b, lo, hi):
    for i in range(lo, hi):
        v = a[i]
        b[i] = math.sin(v) * math.cos(v) + math.tan(v)


def run_once(kernel, a, b, pool, p):
    bounds = np.linspace(0, a.size, p + 1, dtype=np.int64)
    t0 = time.perf_counter()
    futures = [pool.submit(kernel, a, b, int(bounds[i]), int(bounds[i + 1]))
               for i in range(p)]
    for f in futures:
        f.result()
    return time.perf_counter() - t0


def bench(kernel, a, b, p):
    with ThreadPoolExecutor(max_workers=p) as pool:
        run_once(kernel, a, b, pool, p)                 # warm-up (discarded)
        return min(run_once(kernel, a, b, pool, p) for _ in range(RUNS))


def study(name, kernel, a, b, core_counts):
    rows = []
    t1 = None
    for p in core_counts:
        t = bench(kernel, a, b, p)
        if p == 1:
            t1 = t
        speedup = t1 / t
        eff = speedup / p
        gbs = (2 * 8 * a.size) / t / 1e9
        rows.append((p, t, speedup, eff, gbs))
    header = (f"{'Cores':>6} | {'Time (ms)':>10} | {'Speedup':>8} | "
              f"{'Efficiency':>10} | {'Mem traffic (GB/s)':>19}")
    print(name)
    print(header)
    print("-" * len(header))
    for p, t, s, e, g in rows:
        print(f"{p:>6} | {t * 1000:>10.2f} | {s:>7.2f}x | {e * 100:>9.1f}% | {g:>19.2f}")
    print()
    return rows


def main():
    max_cores = os.cpu_count() or 1
    core_counts = sorted({1, 2, max_cores})

    print(f"Student ID    : {STUDENT_ID}")
    print(f"OS            : {platform.system()} {platform.release()}")
    print(f"CPU / Platform: {platform.processor() or platform.machine()}")
    print(f"Logical cores : {max_cores} (tested: {core_counts})")
    print(f"Array size    : N = {N:,} float64 ({N * 8 / 1e6:.0f} MB per array), "
          f"best of {RUNS} runs")
    print()

    a = np.random.default_rng(STUDENT_ID).random(N)
    b = np.empty(N)
    b[:] = 0.0

    kernel_copy(a, b, 0, 8)
    kernel_trig(a, b, 0, 8)

    rows_a = study("Kernel A (memory-bound): B[i] = A[i]", kernel_copy, a, b, core_counts)
    assert np.array_equal(a, b), "copy kernel produced a wrong result"

    rows_b = study("Kernel B (compute-bound): B[i] = sin(A[i])*cos(A[i]) + tan(A[i])",
                   kernel_trig, a, b, core_counts)
    expected = np.sin(a[:1000]) * np.cos(a[:1000]) + np.tan(a[:1000])
    assert np.allclose(b[:1000], expected), "trig kernel produced a wrong result"

    eff_a, eff_b = rows_a[-1][3], rows_b[-1][3]
    sp_a, sp_b = rows_a[-1][2], rows_b[-1][2]
    peak_a = max(r[4] for r in rows_a)

    print("=== VERDICT ===")
    print(f"At {core_counts[-1]} cores: Kernel A efficiency = {eff_a * 100:.1f}% "
          f"(speedup {sp_a:.2f}x), Kernel B efficiency = {eff_b * 100:.1f}% "
          f"(speedup {sp_b:.2f}x)")
    print(f"Kernel A memory traffic peaked at {peak_a:.2f} GB/s "
          f"(Kernel B only {max(r[4] for r in rows_b):.2f} GB/s).")
    if len(core_counts) < 3:
        print("Too few cores to judge scaling.")
    elif eff_a < eff_b and eff_a < STALL_EFFICIENCY:
        print("RESULT: Kernel A (vector copy) STALLED in scaling due to memory channel "
              "saturation: extra cores only wait on the RAM bus.")
        if eff_b >= STALL_EFFICIENCY:
            print("        Kernel B (sin/cos/tan) kept scaling because it is compute-bound.")
        else:
            print("        Kernel B scaled better, but its efficiency is also reduced "
                  "because logical cores beyond the physical count share execution units (SMT).")
    elif eff_b < eff_a and eff_b < STALL_EFFICIENCY:
        print("RESULT: Kernel B stalled more than Kernel A on this run (unexpected). "
              "Likely cause: SMT/thermal limits or background load; re-run on an idle machine.")
    else:
        print("RESULT: neither kernel dropped clearly below "
              f"{STALL_EFFICIENCY * 100:.0f}% efficiency; memory saturation was not "
              "conclusively reached on this machine.")


if __name__ == "__main__":
    main()