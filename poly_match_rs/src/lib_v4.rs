use pyo3::prelude::*;

use ndarray::Array1;
use ndarray_linalg::Scalar;
use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2, ToPyArray};

#[pyclass(subclass)]
struct Polygon {
    x: Array1<f64>,
    y: Array1<f64>,
    center: (f64, f64),
}

#[pymethods]
impl Polygon {
    #[new]
    fn new(x: PyReadonlyArray1<f64>, y: PyReadonlyArray1<f64>) -> Polygon {
        let x = x.as_array();
        let y = y.as_array();
        let center = Array1::from_vec(vec![x.mean().unwrap(), y.mean().unwrap()]);

        Polygon {
            x: x.to_owned(),
            y: y.to_owned(),
            center: (center[0], center[1]),
        }
    }

    #[getter]
    fn x<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        Ok(self.x.to_pyarray_bound(py))
    }

    #[getter]
    fn y<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        Ok(self.y.to_pyarray_bound(py))
    }

    #[getter]
    fn center<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let center = Array1::from_vec(vec![self.center.0, self.center.1]);
        Ok(center.to_pyarray_bound(py))
    }
}

#[pyfunction]
fn find_close_polygons(
    py: Python<'_>,
    polygons: Vec<PyRef<Polygon>>,
    point: PyReadonlyArray1<f64>,
    max_dist: f64,
) -> PyResult<Vec<PyObject>> {
    find_close_polygons_impl(py, &polygons, &point, max_dist)
}

fn find_close_polygons_impl(
    py: Python<'_>,
    polygons: &[PyRef<Polygon>],
    point: &PyReadonlyArray1<f64>,
    max_dist: f64,
) -> PyResult<Vec<PyObject>> {
    let mut close_polygons = vec![];
    let point = point.as_array();
    let point = (point[0], point[1]);
    let max_dist_2 = max_dist.square();

    for poly in polygons {
        let norm_2 = {
            let center = &poly.center;

            (center.0 - point.0).square() + (center.1 - point.1).square()
        };

        if norm_2 < max_dist_2 {
            close_polygons.push(poly.into_py(py));
        }
    }

    Ok(close_polygons)
}

#[pyfunction]
fn find_all_close_polygons(
    py: Python<'_>,
    polygons: Vec<PyRef<Polygon>>,
    points: PyReadonlyArray2<f64>,
    max_dist: f64,
) -> PyResult<Vec<(PyObject, Vec<PyObject>)>> {
    let mut polygon_sets = vec![];

    for point in points.iter()? {
        let point = point?.extract()?;

        let close_polygons = find_close_polygons_impl(py, &polygons, &point, max_dist)?;

        if close_polygons.len() == 0 {
            continue;
        }

        polygon_sets.push((point.to_object(py), close_polygons));
    }

    Ok(polygon_sets)
}

pub fn poly_match_rs(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<Polygon>()?;
    m.add_function(wrap_pyfunction!(find_close_polygons, m)?)?;
    m.add_function(wrap_pyfunction!(find_all_close_polygons, m)?)?;
    Ok(())
}
