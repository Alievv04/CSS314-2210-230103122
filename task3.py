
import os
import platform
import sys
import threading
import time

import numpy as np
from numba import njit

STUDENT_ID = 230103122      # <-- your numeric Student ID
ITERS = 50_000_000
RUNS = 3                    # timed runs per variant (after warm-up)
PAD = 8                     # 8 x int64 = 64 bytes = one cache line


@njit(nogil=True)
def kernel_shared(counts, ones, offs, slot, iters):
    """Counter lives in shared memory and is updated on every iteration."""
    for k in range(iters):
        counts[slot + offs[k & 7]] += ones[k & 7]


@njit(nogil=True)
def kernel_local(counts, ones, offs, slot, iters):
    """Counter lives in a local variable; one write to shared memory at the end."""
    local = 0
    for k in range(iters):
        local += ones[k & 7]
    counts[slot] = local


def physical_cores():
    try:
        import psutil
        n = psutil.cpu_count(logical=False)
        if n:
            return n
    except ImportError:
        pass
    return max(1, (os.cpu_count() or 2) // 2)


def run_threads(kernel, counts, ones, offs, workers, stride, iters):
    threads = [threading.Thread(target=kernel, args=(counts, ones, offs, i * stride, iters))
               for i in range(workers)]
    t0 = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return time.perf_counter() - t0


def bench(kernel, workers, stride, ones, offs, iters):
    size = workers * stride
    # Warm-up (compiles the function; discarded)
    run_threads(kernel, np.zeros(size, dtype=np.int64), ones, offs, workers, stride, 1000)
    times = []
    for _ in range(RUNS):
        counts = np.zeros(size, dtype=np.int64)
        times.append(run_threads(kernel, counts, ones, offs, workers, stride, iters))
        assert all(counts[i * stride] == iters for i in range(workers)), "wrong count!"
    return sum(times) / len(times)


def main():
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else physical_cores()
    ones = np.ones(8, dtype=np.int64)
    offs = np.zeros(8, dtype=np.int64)

    print(f"Student ID    : {STUDENT_ID}")
    print(f"OS            : {platform.system()} {platform.release()}")
    print(f"CPU / Platform: {platform.processor() or platform.machine()}")
    print(f"Logical cores : {os.cpu_count()}")
    print(f"Worker threads: {workers} (physical core count)")
    print(f"Increments    : {ITERS:,} per thread | mean of {RUNS} runs after warm-up")
    print(f"Cache line    : 64 bytes (int64 counters, padding = {PAD} elements)")
    print()

    t_unpadded = bench(kernel_shared, workers, 1, ones, offs, ITERS)
    t_padded = bench(kernel_shared, workers, PAD, ones, offs, ITERS)
    t_local = bench(kernel_local, workers, 1, ones, offs, ITERS)

    print(f"{'Variant':<32} | {'Wall Time (s)':>13} | {'Ratio vs Unpadded':>18}")
    print("-" * 70)
    for name, t in [("Unpadded shared (false sharing)", t_unpadded),
                    ("Padded shared (64-byte pad)", t_padded),
                    ("Local variable", t_local)]:
        print(f"{name:<32} | {t:>13.4f} | {t_unpadded / t:>17.2f}x")
    print()
    print(f"Execution ratio Unpadded / Padded = {t_unpadded / t_padded:.2f}x")
    print(f"Execution ratio Unpadded / Local  = {t_unpadded / t_local:.2f}x")


if __name__ == "__main__":
    main()