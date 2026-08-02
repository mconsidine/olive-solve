// Copyright (c) 2026 Omair Kamil
// See LICENSE file in root directory for license terms.

use crate::FusedSolver;
use numpy::PyReadonlyArray2;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

// --- We will append the helpers here ---
// --- Helper Functions to Map Python kwargs to Rust Structs ---

#[pyclass(name = "FusedSolver")]
/// A Python wrapper for the FusedSolver, providing plate solving and star extraction.
/// The `FusedSolver` unifies standard and fast extraction pipelines, along with IMU support.
pub struct PyFusedSolver {
    inner: FusedSolver,
}

#[pymethods]
impl PyFusedSolver {
    #[new]
    /// Initializes a new FusedSolver instance.
    ///
    /// Args:
    ///     database_path (str): The file path to the npz star database.
    pub fn new(database_path: &str) -> PyResult<Self> {
        let inner = FusedSolver::new(std::path::Path::new(database_path), None, None)
            .map_err(pyo3::exceptions::PyRuntimeError::new_err)?;
        Ok(Self { inner })
    }

    #[pyo3(signature = (image, **kwargs))]
    /// Extracts star centroids using the standard pipeline.
    ///
    /// Args:
    ///     image (numpy.ndarray): 2D float32 image array.
    ///     **kwargs: Extraction options (sigma, max_returned, downsample, etc).
    ///
    /// Returns:
    ///     numpy.ndarray: An array of shape (N, 2) containing (y, x) centroid coordinates.
    pub fn extract<'py>(
        &self,
        py: Python<'py>,
        image: PyReadonlyArray2<'py, f32>,
        kwargs: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let options = tetra3::extractor::ExtractOptions::from_kwargs(kwargs)?;
        let img_view = image.as_array();

        let result = self
            .inner
            .extract(&img_view, options)
            .map_err(pyo3::exceptions::PyRuntimeError::new_err)?;

        Ok(tetra3::python::centroids_to_numpy(py, &result.centroids))
    }

    #[pyo3(signature = (image, **kwargs))]
    /// Extracts star centroids using the highly-optimized fast sequential pipeline.
    ///
    /// Args:
    ///     image (numpy.ndarray): 2D uint8 or float32 image array.
    ///     **kwargs: Fast extraction options (downsample, max_returned, etc).
    ///
    /// Returns:
    ///     numpy.ndarray or tuple: Array of shape (N, 2) for centroids. If virtual crops
    ///     are used, returns a tuple containing (base_centroids, (crop_1_centroids, ...)).
    pub fn extract_fast<'py>(
        &self,
        py: Python<'py>,
        image: Bound<'py, PyAny>,
        kwargs: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let options = tetra3::fast_extractor::FastExtractOptions::from_kwargs(kwargs)?;

        let result = if let Ok(img_u8) = image.extract::<numpy::PyReadonlyArray2<u8>>() {
            let img_view = img_u8.as_array();
            self.inner.extract_fast(&img_view, options)
        } else if let Ok(img_f32) = image.extract::<numpy::PyReadonlyArray2<f32>>() {
            let img_view = img_f32.as_array();
            self.inner.extract_fast(&img_view, options)
        } else {
            return Err(pyo3::exceptions::PyTypeError::new_err(
                "Image must be a 2D NumPy array of u8 or f32",
            ));
        }
        .map_err(pyo3::exceptions::PyRuntimeError::new_err)?;

        let core_result = tetra3::python::fast_centroids_to_numpy(py, &result.centroids);
        if let Some(crop_results) = &result.virtual_crop_centroids {
            let mut crop_list = Vec::with_capacity(crop_results.len());
            for crop in crop_results {
                crop_list.push(tetra3::python::fast_centroids_to_numpy(py, crop));
            }
            let elements: Vec<Bound<'py, pyo3::types::PyAny>> =
                vec![core_result, PyTuple::new(py, crop_list).unwrap().into_any()];
            Ok(PyTuple::new(py, elements).unwrap().into_any())
        } else {
            Ok(core_result)
        }
    }

    #[pyo3(signature = (image, **kwargs))]
    /// Performs a full plate solve from an image using the standard pipeline.
    ///
    /// Args:
    ///     image (numpy.ndarray): 2D float32 image array.
    ///     **kwargs: Options for both extraction and solving (fov_estimate, match_radius, etc).
    ///
    /// Returns:
    ///     dict: A dictionary containing the solve results (RA, Dec, Roll, FOV, status, etc).
    pub fn solve_from_image<'py>(
        &self,
        py: Python<'py>,
        image: PyReadonlyArray2<'py, f32>,
        kwargs: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let extract_options = tetra3::extractor::ExtractOptions::from_kwargs(kwargs)?;
        let solve_options = tetra3::solver::SolveOptions::from_kwargs(kwargs)?;
        let img_view = image.as_array();

        let solution = self
            .inner
            .solve_from_image(&img_view, extract_options, solve_options, None)
            .map_err(pyo3::exceptions::PyRuntimeError::new_err)?;

        solution.to_dict(py, None)
    }

    #[pyo3(signature = (image, **kwargs))]
    /// Performs a full plate solve from an image using the highly-optimized fast sequential pipeline.
    ///
    /// Args:
    ///     image (numpy.ndarray): 2D uint8 or float32 image array.
    ///     **kwargs: Options for both fast extraction and solving.
    ///
    /// Returns:
    ///     dict: A dictionary containing the solve results (RA, Dec, Roll, FOV, status, etc).
    pub fn solve_from_image_fast<'py>(
        &self,
        py: Python<'py>,
        image: Bound<'py, PyAny>,
        kwargs: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let extract_options = tetra3::fast_extractor::FastExtractOptions::from_kwargs(kwargs)?;
        let solve_options = tetra3::solver::SolveOptions::from_kwargs(kwargs)?;

        let solution = if let Ok(img_u8) = image.extract::<numpy::PyReadonlyArray2<u8>>() {
            let img_view = img_u8.as_array();
            self.inner
                .solve_from_image_fast(&img_view, extract_options, solve_options, None)
        } else if let Ok(img_f32) = image.extract::<numpy::PyReadonlyArray2<f32>>() {
            let img_view = img_f32.as_array();
            self.inner
                .solve_from_image_fast(&img_view, extract_options, solve_options, None)
        } else {
            return Err(pyo3::exceptions::PyTypeError::new_err(
                "Image must be a 2D NumPy array of u8 or f32",
            ));
        }
        .map_err(pyo3::exceptions::PyRuntimeError::new_err)?;

        solution.to_dict(py, None)
    }

    #[pyo3(signature = (centroids, size, **kwargs))]
    /// Solves for the telescope's pointing using pre-extracted centroids.
    ///
    /// Args:
    ///     centroids (numpy.ndarray): 2D float64 array of shape (N, 2) representing (y, x) coordinates.
    ///     size (tuple): Image (height, width).
    ///     **kwargs: Solve options (fov_estimate, match_radius, solve_timeout, etc).
    ///
    /// Returns:
    ///     dict: A dictionary containing the solve results.
    pub fn solve_from_centroids<'py>(
        &self,
        py: Python<'py>,
        centroids: PyReadonlyArray2<'py, f64>,
        size: (f64, f64),
        kwargs: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let solve_options = tetra3::solver::SolveOptions::from_kwargs(kwargs)?;
        let cents_view = centroids.as_array().to_owned();
        let solution = self
            .inner
            .solve_from_centroids(&cents_view, size, solve_options, None)
            .map_err(pyo3::exceptions::PyRuntimeError::new_err)?;
        solution.to_dict(py, None)
    }

    #[pyo3(signature = (centroids_list, size, **kwargs))]
    /// Solves from multiple centroid sets sequentially, returning the first successful match.
    ///
    /// Args:
    ///     centroids_list (list): A list of numpy arrays, each of shape (N, 2).
    ///     size (tuple): Image (height, width).
    ///     **kwargs: Solve options.
    ///
    /// Returns:
    ///     dict: A dictionary containing the solve results from the successful set.
    pub fn solve_from_centroids_batch<'py>(
        &self,
        py: Python<'py>,
        centroids_list: Vec<PyReadonlyArray2<'py, f64>>,
        size: (f64, f64),
        kwargs: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let solve_options = tetra3::solver::SolveOptions::from_kwargs(kwargs)?;

        let cents_views: Vec<ndarray::ArrayView2<f64>> =
            centroids_list.iter().map(|c| c.as_array()).collect();
        let batch: Vec<(ndarray::Array2<f64>, Option<tetra3::extractor::Crop>)> =
            cents_views.iter().map(|v| (v.to_owned(), None)).collect();
        let solution = self
            .inner
            .solve_from_centroids_batch(&batch, size, solve_options, None)
            .map_err(pyo3::exceptions::PyRuntimeError::new_err)?;
        solution.to_dict(py, None)
    }

    /// Sets the observer's location, allowing the solver to compute azimuth/elevation
    /// or incorporate magnetic declination.
    ///
    /// Args:
    ///     lat (float): Latitude in degrees.
    ///     lon (float): Longitude in degrees.
    pub fn set_observer_location(&self, lat: f64, lon: f64) {
        self.inner.set_observer_location(lat, lon);
    }

    /// Attempts to initialize and start the configured IMU hardware (e.g. BNO080 or BMI160)
    /// to continually track the camera's orientation.
    ///
    /// Returns:
    ///     bool: True if IMU successfully started.
    pub fn start_imu(&self) -> PyResult<bool> {
        self.inner
            .start_imu()
            .map_err(pyo3::exceptions::PyRuntimeError::new_err)
    }
}

#[pymodule]
fn olive_solve(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyFusedSolver>()?;
    Ok(())
}
