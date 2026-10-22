
import os
import platform
import threading
import time

import numpy as np
from numba import njit

STUDENT_ID = 230103122      # <-- your numeric Student ID
N_ITEMS = 1000
WORKERS = 4
RUNS = 3                    # timed runs per strategy (after warm-up)
UNIT = 0.0001               # seconds


@njit(nogil=True)
def spin(n):
    """Busy loop of n dependent integer operations (cannot be folded away)."""
    x = 1
    for _ in range(n):
        x = (x * 1664525 + 1013904223) & 0xFFFFFFFF
    return x


def calibrate():
    """Return seconds per spin iteration on this machine."""
    spin(1000)                                  # compile
    n = 20_000_000
    t0 = time.perf_counter()
    spin(n)
    return (time.perf_counter() - t0) / n


def cost_spec(i):
    return UNIT * (i % 20)


def cost_ramp(i):
    return UNIT * (i // 50)


def run_strategy(kind, chunk, iters, workers=WORKERS):
    """Run one strategy; return (total_s, first_finish_s, last_finish_s)."""
    n = len(iters)
    finish = [0.0] * workers
    done = [0] * workers
    lock = threading.Lock()
    nxt = [0]
    t_start = [0.0]

    def worker_static(w):
        size = (n + workers - 1) // workers
        lo, hi = w * size, min((w + 1) * size, n)
        for i in range(lo, hi):
            spin(iters[i])
        done[w] = hi - lo
        finish[w] = time.perf_counter() - t_start[0]

    def worker_dynamic(w):
        count = 0
        while True:
            with lock:                          # shared work queue
                lo = nxt[0]
                nxt[0] += chunk
            if lo >= n:
                break
            hi = min(lo + chunk, n)
            for i in range(lo, hi):
                spin(iters[i])
            count += hi - lo
        done[w] = count
        finish[w] = time.perf_counter() - t_start[0]

    target = worker_static if kind == "static" else worker_dynamic
    threads = [threading.Thread(target=target, args=(w,)) for w in range(workers)]
    t_start[0] = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    total = time.perf_counter() - t_start[0]
    assert sum(done) == n, "not all items were processed!"
    return total, min(finish), max(finish)


def scenario(title, cost_fn, sec_per_iter):
    iters = np.array([int(round(cost_fn(i) / sec_per_iter)) for i in range(N_ITEMS)],
                     dtype=np.int64)
    total_work = sum(cost_fn(i) for i in range(N_ITEMS))
    print(f"{title}")
    print(f"Total work = {total_work * 1000:.1f} ms | ideal time on {WORKERS} workers = "
          f"{total_work / WORKERS * 1000:.1f} ms")

    header = (f"{'Strategy':<26} | {'Total (ms)':>10} | {'First done (ms)':>15} | "
              f"{'Last done (ms)':>14} | {'Delta (ms)':>10}")
    print(header)
    print("-" * len(header))

    configs = [("A: Static (blocks of 250)", "static", 0),
               ("B1: Dynamic, chunk = 1", "dynamic", 1),
               ("B2: Dynamic, chunk = 5", "dynamic", 5)]
    for name, kind, chunk in configs:
        run_strategy(kind, chunk, iters)                       # warm-up (discarded)
        res = [run_strategy(kind, chunk, iters) for _ in range(RUNS)]
        total = sum(r[0] for r in res) / RUNS
        first = sum(r[1] for r in res) / RUNS
        last = sum(r[2] for r in res) / RUNS
        print(f"{name:<26} | {total * 1000:>10.2f} | {first * 1000:>15.2f} | "
              f"{last * 1000:>14.2f} | {(last - first) * 1000:>10.2f}")
    print()


def main():
    print(f"Student ID    : {STUDENT_ID}")
    print(f"OS            : {platform.system()} {platform.release()}")
    print(f"CPU / Platform: {platform.processor() or platform.machine()}")
    print(f"Logical cores : {os.cpu_count()}")
    print(f"Items         : {N_ITEMS} | Workers: {WORKERS} | mean of {RUNS} runs after warm-up")
    sec_per_iter = calibrate()
    print(f"Calibration   : {sec_per_iter * 1e9:.2f} ns per spin iteration")
    print()

    scenario("Scenario 1 (task spec): cost(i) = 0.0001 * (i % 20) s", cost_spec, sec_per_iter)
    scenario("Scenario 2 (extra, ramp skew): cost(i) = 0.0001 * (i // 50) s", cost_ramp, sec_per_iter)


if __name__ == "__main__":
    main()