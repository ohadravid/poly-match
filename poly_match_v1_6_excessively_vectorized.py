import numpy as np
import scipy.spatial
import matplotlib.pyplot as plt
from matplotlib import patches

def split_polygons(polygons, num_vertices):
    # Split densely packed polygons into individual polygons.
    # This function should not be used if performance is important.
    i = 0
    for n in num_vertices:
        yield polygons[i:i+n]
        i += n

def make_polygons(n=1000, min_vertices=5, max_vertices=15, min_radius=0.5, max_radius=1.5):
    rng = np.random.default_rng(0)

    # Number of vertices of each polygon
    num_vertices = rng.integers(min_vertices, max_vertices + 1, n)

    # Compute angle for each vertex of polygon
    poly_index = np.repeat(np.arange(n), num_vertices)
    c = cumsum0(num_vertices)
    ranges = np.arange(num_vertices.sum()) - c[poly_index]
    angles = 2 * np.pi * ranges / np.repeat(num_vertices, num_vertices)

    # Distance from center to vertex of polygon
    distance = rng.uniform(min_radius, max_radius, len(angles))

    # Centers of polygons (not "real" center)
    centers = rng.uniform(0.0, 100.0, (n, 2))

    # Vertex coordinates of polygons
    x = distance * np.cos(angles) + np.repeat(centers[:, 0], num_vertices)
    y = distance * np.sin(angles) + np.repeat(centers[:, 1], num_vertices)

    polygons = np.column_stack([x, y])

    return polygons, num_vertices

def cumsum0(values, axis=0):
    # np.cumsum for array prepended with zero
    shape = list(values.shape)
    shape[axis] = 1
    zeros = np.zeros(shape, dtype=values.dtype)
    return np.cumsum(np.concatenate([zeros, values], axis=axis), axis)

def gather_sum(values, lengths, axis=0):
    # Sum over consecutive value ranges of given length
    return np.diff(cumsum0(values, axis)[cumsum0(lengths, axis)], axis=axis)

def roll1(values, lengths):
    # Roll one value back to front for given value ranges
    indices = cumsum0(lengths)
    extended = np.insert(values, indices[:-1], values[indices[1:] - 1])
    del_indices = indices[1:] + np.arange(len(lengths))
    return np.delete(extended, del_indices)

def polygon_areas(polygons, num_vertices):
    # https://en.wikipedia.org/wiki/Shoelace_formula
    bx, by = polygons.T
    ax = roll1(bx, num_vertices)
    ay = roll1(by, num_vertices)
    return 0.5 * gather_sum(ax * by - ay * bx, num_vertices)

def smallest_polygon_in_range(polygons, num_vertices, points, max_dist):
    # For each point, find the polygon with smallest area within max_dist.
    # If no polygon is found, no point is returned.
    centers = gather_sum(polygons, num_vertices, axis=0) / num_vertices[:, None]
    areas = polygon_areas(polygons, num_vertices)

    areas2d = np.tile(areas, (len(points), 1))

    out_of_range = scipy.spatial.distance.cdist(points, centers) > max_dist
    areas2d[out_of_range] = np.inf

    polygon_indices = np.argmin(areas2d, axis=1)

    # Discard points where no polygon is within range
    valid = ~np.all(out_of_range, axis=1)
    return points[valid], polygon_indices[valid]

def draw():
    polygons, num_vertices = make_polygons()

    points = np.random.default_rng(0).uniform(0.0, 100.0, size=(100, 2))

    max_dist = 10.0

    valid_points, indices = smallest_polygon_in_range(polygons, num_vertices, points, max_dist)

    centers = gather_sum(polygons, num_vertices, axis=0) / num_vertices[:, None]

    angles = np.linspace(0, 2 * np.pi, 30)
    cx = max_dist * np.cos(angles)
    cy = max_dist * np.sin(angles)

    plt.figure(figsize=(10, 10))

    plt.scatter(*centers.T, color="cornflowerblue", s=5)
    plt.scatter(*points.T, color="red", zorder=3)

    for p, q in zip(valid_points, centers[indices]):
        plt.plot([p[0], q[0]], [p[1], q[1]], color="green")
        plt.plot(cx + p[0], cy + p[1], color="orange", alpha=0.2)

    for polygon in split_polygons(polygons, num_vertices):
        plt.gca().add_patch(patches.Polygon(polygon, facecolor="cornflowerblue", edgecolor="k", alpha=0.5))

    plt.tight_layout()
    plt.axis("off")
    plt.tight_layout()
    plt.savefig("polygons.png", dpi=150, bbox_inches="tight", pad_inches=0)
    plt.show()

def generate_example():
    points = np.random.default_rng(0).uniform(0.0, 100.0, size=(100, 2))

    return make_polygons(), points

def main(polygons_num_vertices, points):
    polygons, num_vertices = polygons_num_vertices

    smallest_polygon_in_range(polygons, num_vertices, points, max_dist=10.0)

if __name__ == "__main__":
    draw()
