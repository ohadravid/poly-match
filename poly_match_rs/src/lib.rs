mod lib_v0;
mod lib_v1;
mod lib_v2;
mod lib_v3;
mod lib_v4;

use pyo3::prelude::*;

use ndarray::Array1;
use ndarray_linalg::Scalar;
use numpy::{PyArray1, PyReadonlyArray1, ToPyArray};

#[pyclass(subclass)]
struct Polygon {
    x: Array1<f64>,
    y: Array1<f64>,
    center: Array1<f64>,
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
            center,
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
        Ok(self.center.to_pyarray_bound(py))
    }
}

#[pyfunction]
fn find_close_polygons<'py>(
    polygons: Vec<Bound<'py, Polygon>>,
    point: PyReadonlyArray1<'py, f64>,
    max_dist: f64,
) -> PyResult<Vec<Bound<'py, Polygon>>> {
    let mut close_polygons = vec![];
    let point = point.as_array();
    for poly in polygons {
        let norm = {
            let center = &poly.borrow().center;

            ((center[0] - point[0]).square() + (center[1] - point[1]).square()).sqrt()
        };

        if norm < max_dist {
            close_polygons.push(poly)
        }
    }

    Ok(close_polygons)
}

#[pymodule]
pub fn poly_match_rs(py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    m.add_class::<Polygon>()?;
    m.add_function(wrap_pyfunction!(find_close_polygons, m)?)?;

    // Just for easier testing of different versions.
    let v0 = PyModule::new_bound(py, "v0")?;
    lib_v0::poly_match_rs(py, &v0)?;
    m.add_submodule(&v0)?;

    let v1 = PyModule::new_bound(py, "v1")?;
    lib_v1::poly_match_rs(py, &v1)?;
    m.add_submodule(&v1)?;

    let v2 = PyModule::new_bound(py, "v2")?;
    lib_v2::poly_match_rs(py, &v2)?;
    m.add_submodule(&v2)?;

    let v3 = PyModule::new_bound(py, "v3")?;
    lib_v3::poly_match_rs(py, &v3)?;
    m.add_submodule(&v3)?;

    let v4 = PyModule::new_bound(py, "v4")?;
    lib_v4::poly_match_rs(py, &v4)?;
    m.add_submodule(&v4)?;

    Ok(())
}
