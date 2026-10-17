import time
import numpy as np

rng = np.random.default_rng(7)
sizes = [1, 2, 4, 8, 16, 32, 64]
warmup = 50
iterations = 2000

print(
    f"{'n':>4} {'FLOPs':>12} "
    f"{'preallocated us':>17} {'new output us':>15}"
)

for n in sizes:
    a = rng.random((n, n), dtype=np.float32)
    b = rng.random((n, n), dtype=np.float32)
    c = np.empty((n, n), dtype=np.float32)

    # Warm up both versions.
    for _ in range(warmup):
        np.matmul(a, b, out=c)
        result = np.matmul(a, b)

    preallocated_times = []
    allocated_times = []

    for _ in range(iterations):
        start = time.perf_counter_ns()
        np.matmul(a, b, out=c)
        preallocated_times.append(time.perf_counter_ns() - start)

        start = time.perf_counter_ns()
        result = np.matmul(a, b)
        allocated_times.append(time.perf_counter_ns() - start)

    preallocated_us = np.median(preallocated_times) / 1000
    allocated_us = np.median(allocated_times) / 1000

    print(
        f"{n:4d} {2 * n**3:12,d} "
        f"{preallocated_us:17.3f} {allocated_us:15.3f}"
    )