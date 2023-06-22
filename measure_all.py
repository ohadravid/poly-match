import importlib
import os
import time
from pathlib import Path

os.environ["OPENBLAS_NUM_THREADS"] = "1"

benches = [bench.stem for bench in sorted(Path(__file__).parent.glob("poly_match_*.py"), key=lambda path: path.name)]
baseline = benches.pop(benches.index("poly_match_v1"))

def benchmark(poly_match_path):
    print(f"Measuring `{poly_match_path}`")
    poly_match = importlib.import_module(poly_match_path)
    polygons, points = poly_match.generate_example()

    t0 = time.perf_counter()
    for i in range(500):
        poly_match.main(polygons, points)

        if i >= 5 and time.perf_counter() - t0 > 3.:
            break
    t1 = time.perf_counter()

    num_iter = i + 1

    took = (t1 - t0) / num_iter
    return took, num_iter

baseline_time, num_iter = benchmark(baseline)
print(f"Took an avg of {baseline_time * 1000:6.2f}ms per iteration ({num_iter:>3d} iterations) {1:6.2f}x speedup")
for poly_match_path in benches:
    benchmark_time, num_iter = benchmark(poly_match_path)
    speedup = baseline_time / benchmark_time
    print(f"Took an avg of {benchmark_time * 1000:6.2f}ms per iteration ({num_iter:>3d} iterations) {speedup:6.2f}x speedup")
