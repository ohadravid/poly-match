import numpy as np
from poly_match_v1_6_excessively_vectorized import generate_example
import poly_match_v1_6_excessively_vectorized
from numba import njit

@njit("Tuple((f8[:, :], i8[:]))(f8[:, :], i8[:], f8[:, :], f8)", cache=True)
def smallest_polygon_in_range(polygons, num_vertices, points, max_dist):
    num_polys = len(num_vertices)
    areas = np.zeros(num_polys)
    centers = np.zeros((num_polys, 2))
    radius = np.zeros(num_polys)

    i = 0
    for i_poly in range(num_polys):
        n = num_vertices[i_poly]
        cx = 0.0
        cy = 0.0
        area = 0.0
        ax = polygons[i + n - 1, 0]
        ay = polygons[i + n - 1, 1]
        for k in range(n):
            bx = polygons[i + k, 0]
            by = polygons[i + k, 1]
            area += ax * by - ay * bx
            ax = bx
            ay = by
            cx += bx
            cy += by
        cx /= n
        cy /= n
        d = 0.0
        for k in range(n):
            dx = polygons[i, 0] - cx
            dy = polygons[i, 1] - cy
            d = max(d, np.hypot(dx, dy))
        i += n
        centers[i_poly, 0] = cx
        centers[i_poly, 1] = cy
        radius[i_poly] = np.sqrt(d)
        areas[i_poly] = 0.5 * area

    polygon_indices = np.zeros(len(points), dtype=np.int64)

    for i_point in range(len(points)):
        x = points[i_point, 0]
        y = points[i_point, 1]

        smallest_poly = -1
        min_area = np.inf

        i = 0
        for i_poly in range(num_polys):
            n = num_vertices[i_poly]
            dx = x - centers[i_poly, 0]
            dy = y - centers[i_poly, 1]

            if np.sqrt(dx * dx + dy * dy) > max_dist + radius[i_poly]:
                i += n
                continue

            in_range = False
            for k in range(n):
                dx = polygons[i + k, 0] - x
                dy = polygons[i + k, 1] - y
                if dx * dx + dy * dy <= max_dist * max_dist:
                    in_range = True
                    break

            i += n

            if in_range and areas[i_poly] < min_area:
                min_area = areas[i_poly]
                smallest_poly = i_poly

        polygon_indices[i_point] = smallest_poly

    valid = polygon_indices != -1
    return points[valid], polygon_indices[valid]

def test():
    (polygons, num_vertices), points =  generate_example()

    max_dist = 10.0

    for _ in range(5):
        from time import perf_counter
        t = perf_counter()
        valid_points, indices = smallest_polygon_in_range(polygons, num_vertices, points, max_dist)
        print(perf_counter() - t)

        valid_points2, indices2 = poly_match_v1_6_excessively_vectorized.smallest_polygon_in_range(polygons, num_vertices, points, max_dist)

        assert np.allclose(indices, indices2)
        assert np.allclose(valid_points, valid_points2)

def main(polygons_num_vertices, points):
    polygons, num_vertices = polygons_num_vertices

    smallest_polygon_in_range(polygons, num_vertices, points, max_dist=10.0)

if __name__ == "__main__":
    test()
