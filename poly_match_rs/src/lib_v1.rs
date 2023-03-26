use pyo3::prelude::*;

use ndarray_linalg::Norm;
use numpy::PyReadonlyArray1;

#[pyfunction]
fn find_close_polygons(
    py: Python<'_>,
    polygons: Vec<PyObject>,
    point: PyReadonlyArray1<f64>,
    max_dist: f64,
) -> PyResult<Vec<PyObject>> {
    let mut close_polygons = vec![];
    let point = point.as_array();
    for poly in polygons {
        let center = poly
            .getattr(py, "center")?
            .extract::<PyReadonlyArray1<f64>>(py)?
            .as_array()
            .to_owned();

        if (center - point).norm() < max_dist {
            close_polygons.push(poly)
        }
    }

    Ok(close_polygons)
}

pub fn poly_match_rs(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(find_close_polygons, m)?)?;
    Ok(())
}
