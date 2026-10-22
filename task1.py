
import os

for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[var] = "1"

import platform
import time
import numpy as np

STUDENT_ID = 230103122
N = 20_000_000


def main():
    # 1. Hardware probe
    print(f"Student ID    : {STUDENT_ID}")
    print(f"OS            : {platform.system()} {platform.release()} ({platform.version()})")
    print(f"CPU / Platform: {platform.processor() or platform.machine()}")
    print(f"Logical cores : {os.cpu_count()}")

    # 2. Generate 20,000,000 float64 elements, seeded by the Student ID
    rng = np.random.default_rng(STUDENT_ID)
    data = rng.random(N, dtype=np.float64)
    print(f"Array size    : {data.size} ({data.dtype})")

    # 3 + 4. Serial sum and standard deviation, timed (wall clock)
    t0 = time.perf_counter()
    total = data.sum()
    std = data.std()
    t1 = time.perf_counter()

    print(f"Sum           : {total:.6f}")
    print(f"Std deviation : {std:.10f}")
    print(f"Execution time: {(t1 - t0) * 1000:.3f} ms")


if __name__ == "__main__":
    main()