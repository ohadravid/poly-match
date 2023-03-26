#![allow(unused_variables)]
use pyo3::prelude::*;

#[pyfunction]
fn find_close_polygons(polygons: Vec<PyObject>, point: PyObject, max_dist: f64) -> PyResult<Vec<PyObject>> {
    Ok(vec![])
}

pub fn poly_match_rs(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(find_close_polygons, m)?)?;
    Ok(())
}