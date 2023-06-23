from typing import List, Tuple
import numpy as np
import poly_match_rs
from poly_match_v1_6_excessively_vectorized import make_polygons, split_polygons


poly_match_rs = poly_match_rs.v4

Point = np.array


class Polygon(poly_match_rs.Polygon):
    _area: float = None

    def area(self) -> float:
        if self._area is None:
            self._area = 0.5 * np.abs(
                np.dot(self.x, np.roll(self.y, 1)) - np.dot(self.y, np.roll(self.x, 1))
            )
        return self._area

def generate_example() -> Tuple[List[Polygon], List[np.array]]:
    points = np.random.default_rng(0).uniform(0.0, 100.0, size=(100, 2))

    polygons = [Polygon(*polygon.T)
        for polygon in split_polygons(*make_polygons())]

    return polygons, points


def select_best_polygon(
    polygon_sets: List[Tuple[Point, List[Polygon]]]
) -> List[Tuple[Point, Polygon]]:
    best_polygons = []
    for point, polygons in polygon_sets:
        best_polygon = polygons[0]

        for poly in polygons:
            if poly.area() < best_polygon.area():
                best_polygon = poly

        best_polygons.append((point, best_polygon))

    return best_polygons


def main(polygons: List[Polygon], points: np.ndarray) -> List[Tuple[Point, Polygon]]:
    max_dist = 10.0
    polygon_sets = poly_match_rs.find_all_close_polygons(polygons, points, max_dist)

    best_polygons = select_best_polygon(polygon_sets)

    return best_polygons


if __name__ == "__main__":
    polygons, points = generate_example()
    main(polygons, points)
