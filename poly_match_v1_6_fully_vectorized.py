from dataclasses import dataclass
from functools import cache, cached_property
from typing import Dict, List, Tuple

import numpy as np

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
            self._area = 0.5 * np.abs(np.dot(self.x, np.roll(self.y, 1)) - np.dot(self.y, np.roll(self.x, 1)))
        return self._area


def generate_one_polygon() -> Polygon:
    x = np.arange(0.0, 1.0, 0.1)
    y = np.sqrt(1.0 - x**2)
    return Polygon(x=x, y=y)


def generate_example() -> Tuple[List[Polygon], np.array]:
    """returns Tuple of M random polygons and N points"""
    rng = np.random.RandomState(6)
    xs = np.arange(0.0, 100.0, 1.0)
    rng.shuffle(xs)

    ys = np.arange(0.0, 100.0, 1.0)
    rng.shuffle(ys)

    points = np.array([[x, y] for x, y in zip(xs, ys)])

    ex_poly = generate_one_polygon()
    polygons = [
        Polygon(
            x=ex_poly.x + rng.randint(0.0, 100.0),
            y=ex_poly.y + rng.randint(0.0, 100.0),
        )
        for _ in range(1000)
    ]

    return polygons, points


def find_close_polygons(polygon_to_points_dist: np.ndarray, max_dist: float) -> np.ndarray:
    # return a matrix of MxN booleans specifying whether the distance from
    # the centroid of polygon m to the point n is smaller than max_dist
    return polygon_to_points_dist < max_dist


def select_best_polygon_index(polygon_areas: np.ndarray, close_polygon_indices: np.ndarray) -> np.ndarray:
    # for every point, returns the index of the smallest polygon (by area)
    # that is closer than max_dist to that point
    area_matrix = polygon_areas[:, np.newaxis].repeat(close_polygon_indices.shape[1], axis=1)
    area_matrix[np.where(np.logical_not(close_polygon_indices))] = np.inf
    return np.argmin(area_matrix, axis=0)


def find_dist_to_points(centroids: np.ndarray, points: np.ndarray) -> np.ndarray:
    return np.linalg.norm(centroids - points[:, np.newaxis], axis=2).T


def main(polygons: List[Polygon], points: np.ndarray) -> List[Tuple[Point, Polygon]]:
    max_dist = 10.0

    polygon_centers = np.asarray([polygon.center for polygon in polygons])
    polygon_areas = np.asarray([polygon.area() for polygon in polygons])
    polygon_to_points_dist = find_dist_to_points(polygon_centers, points)

    close_polygon_indices = find_close_polygons(polygon_to_points_dist, max_dist)
    best_polygon_indices = select_best_polygon_index(polygon_areas, close_polygon_indices)

    return [(point, polygons[index]) for point, index in enumerate(best_polygon_indices)]


if __name__ == "__main__":
    polygons, points = generate_example()
    main(polygons, points)
