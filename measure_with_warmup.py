import time
import poly_match_v1 as poly_match
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"

print(f"Measuring `{poly_match.__name__}`")

t0 = time.perf_counter()
polygons, points = poly_match.generate_example()
t1 = time.perf_counter()

NUM_ITER = 50

# Warmup.
for _ in range(NUM_ITER):
    poly_match.main(polygons, points)

t0 = time.perf_counter()
for _ in range(NUM_ITER):
    poly_match.main(polygons, points)
t1 = time.perf_counter()

took = (t1 - t0) / NUM_ITER

print(f"Took an avg of {took * 1000:.2f}ms per iteration")
