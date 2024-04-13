#![allow(unused_variables)]
use pyo3::prelude::*;

#[pyfunction]
fn find_close_polygons<'py>(
    polygons: Vec<Bound<'py, PyAny>>,
    point: &Bound<'py, PyAny>,
    max_dist: f64,
) -> PyResult<Vec<Bound<'py, PyAny>>> {
    Ok(vec![])
}

pub fn poly_match_rs(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(find_close_polygons, m)?)?;
    Ok(())
}
