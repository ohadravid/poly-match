from functools import cached_property
from typing import List, Tuple
import numpy as np
from dataclasses import dataclass

Point = np.array


@dataclass
class Polygon:
    x: np.array
    y: np.array
    _area: float = None

    @cached_property
    def center(self) -> np.array:
        centroid = np.array([self.x, self.y]).mean(axis=1)
        return centroid

    def area(self) -> float:
        if self._area is None:
            self._area = 0.5 * np.abs(
                np.dot(self.x, np.roll(self.y, 1)) - np.dot(self.y, np.roll(self.x, 1))
            )
        return self._area


def generate_one_polygon() -> Polygon:
    x = np.arange(0.0, 1.0, 0.1)
    y = np.sqrt(1.0 - x**2)
    return Polygon(x=x, y=y)


def generate_example() -> Tuple[List[Polygon], List[np.array]]:
    rng = np.random.RandomState(6)
    xs = np.arange(0.0, 100.0, 1.0)
    rng.shuffle(xs)

    ys = np.arange(0.0, 100.0, 1.0)
    rng.shuffle(ys)

    points = [np.array([x, y]) for x, y in zip(xs, ys)]

    ex_poly = generate_one_polygon()
    polygons = [
        Polygon(
            x=ex_poly.x + rng.randint(0.0, 100.0),
            y=ex_poly.y + rng.randint(0.0, 100.0),
        )
        for _ in range(1000)
    ]

    return polygons, points


def find_close_polygons(
    polygon_subset: List[Polygon], point: np.array, max_dist: float
) -> List[Polygon]:
    close_polygons = []
    for poly in polygon_subset:
        if np.linalg.norm(poly.center - point) < max_dist:
            close_polygons.append(poly)

    return close_polygons


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
    polygon_sets = []
    for point in points:
        close_polygons = find_close_polygons(polygons, point, max_dist)

        if len(close_polygons) == 0:
            continue

        polygon_sets.append((point, close_polygons))

    best_polygons = select_best_polygon(polygon_sets)

    return best_polygons


if __name__ == "__main__":
    polygons, points = generate_example()
    main(polygons, points)
