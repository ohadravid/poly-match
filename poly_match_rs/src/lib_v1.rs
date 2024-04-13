use pyo3::prelude::*;

use ndarray_linalg::Norm;
use numpy::PyReadonlyArray1;

#[pyfunction]
fn find_close_polygons<'py>(
    polygons: Vec<Bound<'py, PyAny>>,
    point: PyReadonlyArray1<f64>,
    max_dist: f64,
) -> PyResult<Vec<Bound<'py, PyAny>>> {
    let mut close_polygons = vec![];
    let point = point.as_array();
    for poly in polygons {
        let center = poly
            .getattr("center")?
            .extract::<PyReadonlyArray1<f64>>()?
            .as_array()
            .to_owned();

        if (center - point).norm() < max_dist {
            close_polygons.push(poly)
        }
    }

    Ok(close_polygons)
}

pub fn poly_match_rs(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(find_close_polygons, m)?)?;
    Ok(())
}
