import time
import importlib
import pathlib
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"

all_version = list(pathlib.Path(__file__).parent.glob("poly_match*.py"))

# Always test "poly_match.py" last.
all_version.sort(key=lambda path: (path.name == "poly_match.py", path.name))

for poly_match_path in all_version:
    poly_match = importlib.import_module(poly_match_path.stem)
    print(f"Measuring `{poly_match.__name__}`")

    t0 = time.perf_counter()
    polygons, points = poly_match.generate_example()
    t1 = time.perf_counter()

    t0 = time.perf_counter()
    for i in range(500):
        poly_match.main(polygons, points)

        if i >= 5 and time.perf_counter() - t0 > 3.:
            break
    t1 = time.perf_counter()
    
    num_iter = i + 1

    took = (t1 - t0) / num_iter
    print(f"Took an avg of {took * 1000:.2f}ms per iteration ({num_iter} iterations)")
