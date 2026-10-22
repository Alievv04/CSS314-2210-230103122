
import os
import platform
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

STUDENT_ID = 230103122
N = 20_000_000
P = 0.95
RUNS = 5


def chunk_sum(view):
    return view.sum()


def parallel_sum(data, workers, executor):
    # Split into `workers` contiguous slices (views: no copying).
    bounds = np.linspace(0, data.size, workers + 1, dtype=np.int64)
    slices = [data[bounds[i]:bounds[i + 1]] for i in range(workers)]
    return sum(executor.map(chunk_sum, slices))


def benchmark(data, workers):
    """Mean wall time (seconds) of the parallel sum with `workers` threads."""
    with ThreadPoolExecutor(max_workers=workers) as ex:
        parallel_sum(data, workers, ex)                 # warm-up (discarded)
        times = []
        for _ in range(RUNS):
            t0 = time.perf_counter()
            result = parallel_sum(data, workers, ex)
            times.append(time.perf_counter() - t0)
    return sum(times) / len(times), result


def amdahl(p_workers, parallel_fraction=P):
    return 1.0 / ((1.0 - parallel_fraction) + parallel_fraction / p_workers)


def main():
    max_cores = os.cpu_count() or 1
    print(f"Student ID    : {STUDENT_ID}")
    print(f"OS            : {platform.system()} {platform.release()}")
    print(f"CPU / Platform: {platform.processor() or platform.machine()}")
    print(f"Logical cores : {max_cores}")
    print(f"Array size    : {N} (float64), seed = {STUDENT_ID}")
    print(f"Runs / count  : {RUNS} (mean, after 1 warm-up) | Amdahl P = {P}")
    print()

    rng = np.random.default_rng(STUDENT_ID)
    data = rng.random(N, dtype=np.float64)
    reference = data.sum()

    counts = sorted({1, 2, 4, max_cores, max_cores * 2})

    results = []
    for w in counts:
        t, value = benchmark(data, w)
        assert np.isclose(value, reference, rtol=1e-9), f"sum mismatch at {w} workers"
        results.append((w, t))

    t1 = dict(results)[1]

    header = f"{'Cores':>6} | {'Wall Time (ms)':>15} | {'Speedup':>9} | {'Amdahl Upper Bound':>19}"
    print(header)
    print("-" * len(header))
    for w, t in results:
        print(f"{w:>6} | {t * 1000:>15.3f} | {t1 / t:>8.2f}x | {amdahl(w):>18.2f}x")


if __name__ == "__main__":
    main()