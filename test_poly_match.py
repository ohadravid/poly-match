import poly_match_v4_avoid_allocs as poly_match
import numpy as np


def test_poly():
    polygon = poly_match.generate_one_polygon()
    assert np.isclose(polygon.area(), 0.108, 0.1)

    assert np.all(np.isclose(polygon.center, [0.45, 0.826], 0.1))


def test_box():
    box = poly_match.Polygon(x=np.array([0., 1., 1., 0.]), y=np.array([0., 0., 1., 1.]))
    assert np.isclose(box.area(), 1.0, 0.1)

    assert np.all(np.isclose(box.center, [0.5, 0.5], 0.1))


def test_main():
    polygons, points = poly_match.generate_example()
    
    results = sorted(poly_match.main(polygons, points), key=lambda point_polygons: tuple(point_polygons[0]))

    point, polygon = results[0]

    assert np.all(np.isclose(point, [0.0, 25.0], 0.1))
    assert np.isclose(polygon.x[0], 9., 0.1)
    assert np.isclose(polygon.y[0], 27., 0.1)