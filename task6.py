
import os
import platform
import threading
import time

import numpy as np
from llvmlite import ir
from numba import njit, types
from numba.extending import intrinsic

STUDENT_ID = 230103122      # <-- your numeric Student ID
THREADS = 4
ITERS = 2_000_000
EXPECTED = THREADS * ITERS  # 8,000,000
RUNS = 3


# ---------------------------------------------------------------- Part 1 ----
@intrinsic
def atomic_add(typingctx, arr, idx, val):
    """Hardware atomic fetch-and-add on arr[idx] (int64). Returns the old value."""
    sig = types.int64(arr, idx, val)

    def codegen(context, builder, signature, args):
        arr_t, idx_t, val_t = signature.args
        ary = context.make_array(arr_t)(context, builder, args[0])
        i = context.cast(builder, args[1], idx_t, types.intp)
        v = context.cast(builder, args[2], val_t, types.int64)
        ptr = builder.gep(ary.data, [i])
        return builder.atomic_rmw("add", ptr, v, "seq_cst")

    return sig, codegen


@njit(nogil=True)
def kernel_unsafe(counter, offs, iters):
    for k in range(iters):
        counter[offs[k & 7]] += 1           # load, add, store: NOT atomic


@njit(nogil=True)
def kernel_atomic(counter, offs, iters):
    for k in range(iters):
        atomic_add(counter, offs[k & 7], 1)  # single atomic instruction


def run_numba(kernel, offs):
    counter = np.zeros(1, dtype=np.int64)
    barrier = threading.Barrier(THREADS + 1)    # all threads start together

    def worker():
        barrier.wait()
        kernel(counter, offs, ITERS)

    threads = [threading.Thread(target=worker) for _ in range(THREADS)]
    for t in threads:
        t.start()
    barrier.wait()
    t0 = time.perf_counter()
    for t in threads:
        t.join()
    return int(counter[0]), time.perf_counter() - t0


def report_runs(name, kernel, offs):
    print(f"{name}")
    print(f"{'Run':>4} | {'Expected':>10} | {'Actual':>10} | {'Lost':>10} | "
          f"{'Lost %':>8} | {'Time (ms)':>10}")
    print("-" * 66)
    times = []
    for r in range(1, RUNS + 1):
        actual, t = run_numba(kernel, offs)
        lost = EXPECTED - actual
        print(f"{r:>4} | {EXPECTED:>10,} | {actual:>10,} | {lost:>10,} | "
              f"{100.0 * lost / EXPECTED:>7.2f}% | {t * 1000:>10.2f}")
        times.append(t)
    print()
    return sum(times) / len(times)


# ---------------------------------------------------------------- Part 2 ----
def py_run(use_lock):
    c = [0]
    lock = threading.Lock()

    def worker_plain():
        for _ in range(ITERS):
            c[0] += 1

    def worker_locked():
        for _ in range(ITERS):
            with lock:
                c[0] += 1

    target = worker_locked if use_lock else worker_plain
    threads = [threading.Thread(target=target) for _ in range(THREADS)]
    t0 = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return c[0], time.perf_counter() - t0


def main():
    print(f"Student ID    : {STUDENT_ID}")
    print(f"OS            : {platform.system()} {platform.release()}")
    print(f"CPU / Platform: {platform.processor() or platform.machine()}")
    print(f"Logical cores : {os.cpu_count()}")
    print(f"Threads       : {THREADS} x {ITERS:,} increments | expected = {EXPECTED:,}")
    print()

    offs = np.zeros(8, dtype=np.int64)
    kernel_unsafe(np.zeros(1, dtype=np.int64), offs, 10)        # compile
    atomic_ok = True
    try:
        kernel_atomic(np.zeros(1, dtype=np.int64), offs, 10)
    except Exception as exc:                                    # pragma: no cover
        atomic_ok = False
        print(f"[atomic intrinsic unavailable in this Numba build: {exc}]\n")

    print("=== PART 1: hardware race (Numba threads, real parallelism) ===")
    t_unsafe = report_runs("Unsynchronized counter[0] += 1", kernel_unsafe, offs)
    if atomic_ok:
        t_atomic = report_runs("Atomic add (hardware LOCK XADD)", kernel_atomic, offs)
        print(f"Mean time unsynchronized : {t_unsafe * 1000:.2f} ms")
        print(f"Mean time atomic         : {t_atomic * 1000:.2f} ms")
        print(f"Synchronization penalty  : +{(t_atomic - t_unsafe) * 1000:.2f} ms "
              f"({t_atomic / t_unsafe:.2f}x slower)")
        print()

    print("=== PART 2: Python Mutex (threading.Lock) ===")
    py_run(False)                                               # warm-up
    v_plain, t_plain = py_run(False)
    v_lock, t_lock = py_run(True)
    for name, v, t in [("No lock", v_plain, t_plain), ("With Lock", v_lock, t_lock)]:
        lost = EXPECTED - v
        print(f"{name:<10}: expected {EXPECTED:,} | actual {v:,} | "
              f"lost {lost:,} ({100.0 * lost / EXPECTED:.2f}%) | {t * 1000:.2f} ms")
    print(f"Lock penalty            : +{(t_lock - t_plain) * 1000:.2f} ms "
          f"({t_lock / t_plain:.2f}x slower)")


if __name__ == "__main__":
    main()