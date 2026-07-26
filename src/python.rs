use crate::{FusedSolver, ImuType};
use ndarray::ArrayView2;
use numpy::{IntoPyArray, PyArrayMethods, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};
use tetra3::extractor::{BgSubMode, Crop, ExtractOptions, ExtractionResult, SigmaMode};
use tetra3::fast_extractor::{FastBgSubMode, FastDownsample, FastExtractOptions, FastSigmaMode};
use tetra3::solver::{Solution, SolveOptions};

// --- We will append the helpers here ---
// --- Helper Functions to Map Python kwargs to Rust Structs ---

fn centroids_to_numpy<'py>(
    py: Python<'py>,
    centroids: &[tetra3::extractor::CentroidResult],
) -> Bound<'py, pyo3::types::PyAny> {
    let num_centroids = centroids.len();
    let mut cents = Vec::with_capacity(num_centroids * 2);
    for c in centroids {
        cents.push(c.y);
        cents.push(c.x);
    }
    numpy::PyArray1::from_slice(py, &cents)
        .reshape([num_centroids, 2])
        .unwrap()
        .into_any()
}

fn fast_centroids_to_numpy<'py>(
    py: Python<'py>,
    centroids: &[tetra3::fast_extractor::FastCentroidResult],
) -> Bound<'py, pyo3::types::PyAny> {
    let num_centroids = centroids.len();
    let mut cents = Vec::with_capacity(num_centroids * 2);
    for c in centroids {
        cents.push(c.y);
        cents.push(c.x);
    }
    numpy::PyArray1::from_slice(py, &cents)
        .reshape([num_centroids, 2])
        .unwrap()
        .into_any()
}

fn solution_to_dict<'py>(
    py: Python<'py>,
    solution: tetra3::solver::Solution,
    ext_time: Option<f64>,
) -> PyResult<Bound<'py, PyDict>> {
    let out_dict = PyDict::new(py);

    // Base coordinate properties
    out_dict.set_item("RA", solution.ra)?;
    out_dict.set_item("Dec", solution.dec)?;
    out_dict.set_item("Roll", solution.roll)?;
    out_dict.set_item("FOV", solution.fov)?;
    out_dict.set_item("distortion", solution.distortion)?;

    // Metrics and statistics
    out_dict.set_item("RMSE", solution.rmse)?;
    out_dict.set_item("P90E", solution.p90e)?;
    out_dict.set_item("MAXE", solution.maxe)?;
    out_dict.set_item("Matches", solution.matches)?;
    out_dict.set_item("Prob", solution.prob)?;
    out_dict.set_item("is_mirrored", solution.is_mirrored)?;

    // Epochs & Status
    out_dict.set_item("epoch_equinox", solution.epoch_equinox)?;
    out_dict.set_item("epoch_proper_motion", solution.epoch_proper_motion)?;
    out_dict.set_item("status", format!("{:?}", solution.status))?;

    // Timings
    if let Some(et) = ext_time {
        out_dict.set_item("T_extract", et)?;
    }
    out_dict.set_item("T_solve", solution.t_solve_ms)?;

    // Target mapping (Vecs map naturally to Python lists via PyO3)
    if let Some(target_ra) = solution.target_ra {
        out_dict.set_item("RA_target", target_ra)?;
    }
    if let Some(target_dec) = solution.target_dec {
        out_dict.set_item("Dec_target", target_dec)?;
    }
    if let Some(target_y) = solution.target_y {
        out_dict.set_item("y_target", target_y)?;
    }
    if let Some(target_x) = solution.target_x {
        out_dict.set_item("x_target", target_x)?;
    }

    // Star structures mapped to lists
    if let Some(matched_centroids) = solution.matched_centroids {
        out_dict.set_item("matched_centroids", matched_centroids)?;
    }
    if let Some(matched_stars) = solution.matched_stars {
        out_dict.set_item("matched_stars", matched_stars)?;
    }
    if let Some(matched_cat_id) = solution.matched_cat_id {
        out_dict.set_item("matched_catID", matched_cat_id)?;
    }
    if let Some(catalog_stars) = solution.catalog_stars {
        out_dict.set_item("catalog_stars", catalog_stars)?;
    }

    if let Some(rm) = solution.rotation_matrix {
        let flat_slice = rm
            .as_slice()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("Matrix not contiguous"))?;
        let py_matrix = numpy::PyArray1::from_slice(py, flat_slice)
            .reshape([3, 3])
            .unwrap();
        out_dict.set_item("rotation_matrix", py_matrix)?;
    }

    Ok(out_dict)
}

fn parse_extract_options(kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<ExtractOptions> {
    let mut options = ExtractOptions::default();

    if let Some(dict) = kwargs {
        if let Some(val) = dict.get_item("sigma")? {
            options.sigma = val.extract()?;
        }
        if let Some(val) = dict.get_item("image_th")? {
            options.image_th = val.extract()?;
        }
        if let Some(val) = dict.get_item("downsample")? {
            options.downsample = val.extract()?;
        }
        if let Some(val) = dict.get_item("filtsize")? {
            options.filtsize = val.extract()?;
        }
        if let Some(val) = dict.get_item("binary_open")? {
            options.binary_open = val.extract()?;
        }
        if let Some(val) = dict.get_item("centroid_window")? {
            options.centroid_window = val.extract()?;
        }
        if let Some(val) = dict.get_item("min_area")? {
            options.min_area = val.extract()?;
        }
        if let Some(val) = dict.get_item("max_area")? {
            options.max_area = val.extract()?;
        }
        if let Some(val) = dict.get_item("min_sum")? {
            options.min_sum = val.extract()?;
        }
        if let Some(val) = dict.get_item("max_sum")? {
            options.max_sum = val.extract()?;
        }
        if let Some(val) = dict.get_item("max_axis_ratio")? {
            options.max_axis_ratio = val.extract()?;
        }
        if let Some(val) = dict.get_item("max_returned")? {
            options.max_returned = val.extract()?;
        }
        if let Some(val) = dict.get_item("return_images")? {
            options.return_images = val.extract()?;
        }

        // Background Subtraction Mode
        if let Some(val) = dict.get_item("bg_sub_mode")? {
            if val.is_none() {
                options.bg_sub_mode = None;
            } else {
                let mode_str: String = val.extract()?;
                options.bg_sub_mode = match mode_str.to_lowercase().as_str() {
                    "local_median" => Some(BgSubMode::LocalMedian),
                    "local_mean" => Some(BgSubMode::LocalMean),
                    "global_median" => Some(BgSubMode::GlobalMedian),
                    "global_mean" => Some(BgSubMode::GlobalMean),
                    "none" => None,
                    _ => {
                        return Err(pyo3::exceptions::PyValueError::new_err(format!(
                            "Invalid bg_sub_mode: {}",
                            mode_str
                        )));
                    }
                };
            }
        }

        // Sigma Threshold Mode
        if let Some(val) = dict.get_item("sigma_mode")? {
            let mode_str: String = val.extract()?;
            options.sigma_mode = match mode_str.to_lowercase().as_str() {
                "local_median_abs" => SigmaMode::LocalMedianAbs,
                "local_root_square" => SigmaMode::LocalRootSquare,
                "global_median_abs" => SigmaMode::GlobalMedianAbs,
                "global_root_square" => SigmaMode::GlobalRootSquare,
                _ => {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "Invalid sigma_mode: {}",
                        mode_str
                    )));
                }
            };
        }
    }
    Ok(options)
}

fn parse_fast_extract_options(kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<FastExtractOptions> {
    let mut options = FastExtractOptions::default();
    options.approximate_background = true; // Default to true for fast path

    if let Some(dict) = kwargs {
        if let Some(val) = dict.get_item("sigma")? {
            options.sigma = val.extract()?;
        }
        if let Some(val) = dict.get_item("downsample")? {
            let ds: Option<usize> = val.extract()?;
            options.downsample = match ds {
                None | Some(1) => FastDownsample::None,
                Some(2) => FastDownsample::X2,
                Some(4) => FastDownsample::X4,
                _ => {
                    return Err(pyo3::exceptions::PyValueError::new_err(
                        "Invalid downsample for fast path",
                    ));
                }
            };
        }
        if let Some(val) = dict.get_item("binary_open")? {
            options.binary_open = val.extract()?;
        }
        if let Some(val) = dict.get_item("centroid_window")? {
            options.centroid_window = val.extract()?;
        }
        if let Some(val) = dict.get_item("min_area")? {
            options.min_area = val.extract()?;
        }
        if let Some(val) = dict.get_item("max_area")? {
            options.max_area = val.extract()?;
        }
        if let Some(val) = dict.get_item("min_sum")? {
            options.min_sum = val.extract()?;
        }
        if let Some(val) = dict.get_item("max_sum")? {
            options.max_sum = val.extract()?;
        }
        if let Some(val) = dict.get_item("max_axis_ratio")? {
            options.max_axis_ratio = val.extract()?;
        }
        if let Some(val) = dict.get_item("approximate_background")? {
            options.approximate_background = val.extract()?;
        }

        // Background Subtraction Mode
        if let Some(val) = dict.get_item("bg_sub_mode")? {
            if val.is_none() {
                options.bg_sub_mode = None;
            } else {
                let mode_str: String = val.extract()?;
                options.bg_sub_mode = match mode_str.to_lowercase().as_str() {
                    "local_median" | "block_median" => {
                        Some(FastBgSubMode::BlockMedian { block_size: 32 })
                    }
                    "line_median" => Some(FastBgSubMode::LineMedian),
                    "global_median" => Some(FastBgSubMode::GlobalMedian),
                    "global_mean" => Some(FastBgSubMode::GlobalMean),
                    "none" => None,
                    _ => {
                        return Err(pyo3::exceptions::PyValueError::new_err(format!(
                            "Invalid bg_sub_mode for fast path: {}",
                            mode_str
                        )));
                    }
                };
            }
        }

        // Sigma Threshold Mode
        if let Some(val) = dict.get_item("sigma_mode")? {
            let mode_str: String = val.extract()?;
            options.sigma_mode = match mode_str.to_lowercase().as_str() {
                "global_median_abs" => FastSigmaMode::GlobalMedianAbs,
                "global_root_square" => FastSigmaMode::GlobalRootSquare,
                _ => {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "Invalid sigma_mode for fast path: {}",
                        mode_str
                    )));
                }
            };
        }

        // Virtual crops
        if let Some(val) = dict.get_item("virtual_crops")? {
            if !val.is_none() {
                let py_list: Vec<Bound<'_, pyo3::types::PyTuple>> = val.extract()?;
                let mut crops = Vec::new();
                for py_crop in py_list {
                    let len = py_crop.len();
                    if len == 1 {
                        let fraction: usize = py_crop.get_item(0)?.extract()?;
                        crops.push(Crop::Fraction(fraction));
                    } else if len == 2 {
                        let height: usize = py_crop.get_item(0)?.extract()?;
                        let width: usize = py_crop.get_item(1)?.extract()?;
                        crops.push(Crop::Center { height, width });
                    } else if len == 4 {
                        let height: usize = py_crop.get_item(0)?.extract()?;
                        let width: usize = py_crop.get_item(1)?.extract()?;
                        let offset_y: isize = py_crop.get_item(2)?.extract()?;
                        let offset_x: isize = py_crop.get_item(3)?.extract()?;
                        crops.push(Crop::Region {
                            height,
                            width,
                            offset_y,
                            offset_x,
                        });
                    } else {
                        return Err(pyo3::exceptions::PyValueError::new_err(
                            "Invalid virtual crop format",
                        ));
                    }
                }
                options.virtual_crops = Some(crops);
            }
        }
    }
    Ok(options)
}

fn parse_solve_options(kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<SolveOptions> {
    let mut options = SolveOptions::default();

    if let Some(dict) = kwargs {
        if let Some(val) = dict.get_item("fov_estimate")? {
            options.fov_estimate = val.extract()?;
        }
        if let Some(val) = dict.get_item("fov_max_error")? {
            options.fov_max_error = val.extract()?;
        }
        if let Some(val) = dict.get_item("match_radius")? {
            options.match_radius = val.extract()?;
        }
        if let Some(val) = dict.get_item("match_threshold")? {
            options.match_threshold = val.extract()?;
        }
        if let Some(val) = dict.get_item("solve_timeout")? {
            options.solve_timeout_ms = val.extract()?;
        }
        if let Some(val) = dict.get_item("distortion")? {
            options.distortion = val.extract()?;
        }
        if let Some(val) = dict.get_item("match_max_error")? {
            options.match_max_error = val.extract()?;
        }
        if let Some(val) = dict.get_item("return_matches")? {
            options.return_matches = val.extract()?;
        }
        if let Some(val) = dict.get_item("return_catalog")? {
            options.return_catalog = val.extract()?;
        }
        if let Some(val) = dict.get_item("return_rotation_matrix")? {
            options.return_rotation_matrix = val.extract()?;
        }

        // Target configurations (Parsing 2D arrays directly into ndarrays)
        if let Some(val) = dict.get_item("target_pixel")? {
            if !val.is_none() {
                let py_arr: numpy::PyReadonlyArray2<f64> = val.extract()?;
                options.target_pixel = Some(py_arr.as_array().to_owned());
            }
        }
        if let Some(val) = dict.get_item("target_sky_coord")? {
            if !val.is_none() {
                let py_arr: numpy::PyReadonlyArray2<f64> = val.extract()?;
                options.target_sky_coord = Some(py_arr.as_array().to_owned());
            }
        }
    }
    Ok(options)
}

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
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;
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
        let options = parse_extract_options(kwargs)?;
        let img_view = image.as_array();

        let result = self
            .inner
            .extract(&img_view, options)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;

        Ok(centroids_to_numpy(py, &result.centroids))
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
        let options = parse_fast_extract_options(kwargs)?;

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
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;

        let core_result = fast_centroids_to_numpy(py, &result.centroids);
        if let Some(crop_results) = &result.virtual_crop_centroids {
            let mut crop_list = Vec::with_capacity(crop_results.len());
            for crop in crop_results {
                crop_list.push(fast_centroids_to_numpy(py, crop));
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
        let extract_options = parse_extract_options(kwargs)?;
        let solve_options = parse_solve_options(kwargs)?;
        let img_view = image.as_array();

        let solution = self
            .inner
            .solve_from_image(&img_view, extract_options, solve_options, None)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;

        solution_to_dict(py, solution, None)
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
        let extract_options = parse_fast_extract_options(kwargs)?;
        let solve_options = parse_solve_options(kwargs)?;

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
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;

        solution_to_dict(py, solution, None)
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
        let solve_options = parse_solve_options(kwargs)?;
        let cents_view = centroids.as_array().to_owned();
        let solution = self
            .inner
            .solve_from_centroids(&cents_view, size, solve_options, None)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;
        solution_to_dict(py, solution, None)
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
        let solve_options = parse_solve_options(kwargs)?;

        let cents_views: Vec<ndarray::ArrayView2<f64>> =
            centroids_list.iter().map(|c| c.as_array()).collect();
        // FusedSolver solve_from_centroids_batch takes &[Array2<f64>], not &[ArrayView2<f64>]!
        let cents_owned: Vec<ndarray::Array2<f64>> =
            cents_views.iter().map(|v| v.to_owned()).collect();
        let solution = self
            .inner
            .solve_from_centroids_batch(&cents_owned, size, solve_options, None)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;
        solution_to_dict(py, solution, None)
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
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))
    }
}

#[pymodule]
fn olive_solve(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyFusedSolver>()?;
    Ok(())
}
