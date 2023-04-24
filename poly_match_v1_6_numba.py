import numpy as np
from numba import njit, prange


def generate_example():
    rng = np.random.RandomState(6)
    # generate points
    xs = np.arange(0.0, 100.0, 1.0)
    rng.shuffle(xs)
    ys = np.arange(0.0, 100.0, 1.0)
    rng.shuffle(ys)
    points = np.column_stack([xs, ys])

    # generate polygons
    x = np.arange(0.0, 1.0, 0.1)
    y = np.sqrt(1.0 - x**2)
    # polygons shape (#points per polygon, #coordinates, #polygons)
    polygons = np.column_stack([x, y]).reshape(-1, 2, 1) + \
               rng.randint(0.0, 100.0, size=(x.size, 2, 1000))
    return polygons, points

def main(polygons: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Pure numpy implementation"""
    max_dist = 10.0
    # polygon_to_points_dist shape = (# points, # polygons)
    polygon_to_points_dist = np.linalg.norm(
                                np.mean(polygons, axis=0, keepdims=True) - points[..., np.newaxis],
                                axis=1)
    
    # area of polygons
    x, y = polygons[:, 0, :].T, polygons[:, 1, :].T
    areas = 0.5*np.abs(np.sum(x * (np.roll(y, 1, axis=1) - np.roll(y, -1, axis=1)), axis=1))  # shoelace
    areas = np.ones_like(polygon_to_points_dist) * areas.reshape(1, -1)
    # mask those polygons whose distances are beyond max_dist
    areas[polygon_to_points_dist >= max_dist] = np.inf
    poly_indices_per_point = np.argmin(areas, axis=1)
    # returns vector of indices with best point to polygon matches
    return poly_indices_per_point  # size of #points

@njit
def main_jitted(polygons: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Numba implementation"""
    max_dist = 10.0
    
    centers = []
    for i in range(polygons.shape[1]):
        for j in range(polygons.shape[2]):
            centers.append(np.mean(polygons[:, i, j]))
    centers = np.array(centers).reshape(1, polygons.shape[1], polygons.shape[2])
    
    diffs = centers - points.reshape(points.shape[0], points.shape[1], 1)
    dists = []
    for i in prange(points.shape[0]):
        for j in prange(centers.shape[2]):
            dists.append(np.linalg.norm(diffs[i, :, j]))
    # polygon_to_points_dist shape = (# points, # polygons)
    polygon_to_points_dist = np.array(dists).reshape(points.shape[0], centers.shape[2])
    
    # area of polygons
    areas = np.zeros(polygons.shape[-1])
    for pol_i in prange(polygons.shape[-1]):
        x, y = polygons[:, 0, pol_i], polygons[:, 1, pol_i]
        areas[pol_i] = 0.5*np.abs(np.sum(x * (np.roll(y, 1) - np.roll(y, -1))))  # shoelace

    poly_indices_per_point = []
    for point_i in prange(polygon_to_points_dist.shape[0]):
        filtered_areas = np.where(polygon_to_points_dist[point_i, :] >= max_dist, np.inf, areas)
        poly_indices_per_point.append(np.argmin(filtered_areas))

    # returns vector of indices with best point to polygon matches
    return np.array(poly_indices_per_point)  # size of #points


if __name__ == "__main__":
    polygons, points = generate_example()
    main(polygons, points)
