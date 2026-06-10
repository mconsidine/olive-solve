// Copyright (c) 2026 Omair Kamil
// See LICENSE file in root directory for license terms.

use ndarray::Array2;
#[cfg(feature = "extractor")]
use ndarray::{ArrayBase, Data, Ix2};
use std::path::{Path, PathBuf};
#[cfg(feature = "extractor")]
use std::time::Instant;

#[cfg(feature = "extractor")]
use crate::extractor::{
    BgSubMode, CentroidResult, Crop, ExtractOptions, ExtractionResult, Extractor, SigmaMode,
};
#[cfg(feature = "extractor")]
use crate::fast_extractor::{
    FastBgSubMode, FastDownsample, FastExtractOptions, FastExtractor, FastSigmaMode,
};
use crate::solver::{Solution, SolveOptions, Solver};

/// The main Tetra3 instance that centralizes star extraction and plate solving.
/// Holds lazy-initialized instances of the Solver and Extractor to minimize startup
/// overhead and prevent unnecessary memory allocations.
pub struct Tetra3 {
    database_path: PathBuf,
    solver: Option<Solver>,
    #[cfg(feature = "extractor")]
    extractor: Option<Extractor>,
    #[cfg(feature = "extractor")]
    fast_extractor: Option<FastExtractor>,
}

impl Tetra3 {
    /// Creates a new Tetra3 instance. The database is not loaded until the first
    /// solve operation is executed.
    pub fn new(database_path: impl AsRef<Path>) -> Self {
        Self {
            database_path: database_path.as_ref().to_path_buf(),
            solver: None,
            #[cfg(feature = "extractor")]
            extractor: None,
            #[cfg(feature = "extractor")]
            fast_extractor: None,
        }
    }

    /// Helper to lazy-initialize or retrieve the solver.
    fn get_solver(&mut self) -> Result<&mut Solver, Box<dyn std::error::Error>> {
        if self.solver.is_none() {
            self.solver = Some(Solver::load_database(&self.database_path)?);
        }
        Ok(self.solver.as_mut().unwrap())
    }

    /// Helper to lazy-initialize or retrieve the extractor.
    #[cfg(feature = "extractor")]
    fn get_extractor(&mut self) -> &mut Extractor {
        if self.extractor.is_none() {
            self.extractor = Some(Extractor::new());
        }
        self.extractor.as_mut().unwrap()
    }

    /// Solves the star pattern from pre-extracted centroids.
    pub fn solve_from_centroids(
        &mut self,
        centroids: &Array2<f64>,
        size: (f64, f64),
        options: SolveOptions,
    ) -> Result<Solution, Box<dyn std::error::Error>> {
        let solver = self.get_solver()?;
        Ok(solver.solve(centroids, size, options))
    }

    /// Extracts star centroids from an image array.
    #[cfg(feature = "extractor")]
    pub fn get_centroids_from_image<S>(
        &mut self,
        image: &ArrayBase<S, Ix2>,
        options: ExtractOptions,
    ) -> ExtractionResult
    where
        S: Data<Elem = f32>,
    {
        let extractor = self.get_extractor();
        extractor.extract(image, options)
    }

    /// Extracts star centroids from a u8 image array.
    #[cfg(feature = "extractor")]
    pub fn get_centroids_from_image_u8<S>(
        &mut self,
        image: &ArrayBase<S, Ix2>,
        options: ExtractOptions,
    ) -> ExtractionResult
    where
        S: Data<Elem = u8>,
    {
        let extractor = self.get_extractor();
        extractor.extract_u8(image, options)
    }

    /// Explicitly triggers the fast sequential extraction path.
    /// Falls back to the normal extractor if parameters are incompatible.
    #[cfg(feature = "extractor")]
    pub fn get_centroids_from_image_fast<S, T>(
        &mut self,
        image: &ArrayBase<S, Ix2>,
        options: ExtractOptions,
    ) -> ExtractionResult
    where
        S: Data<Elem = T>,
        T: FastPixel,
    {
        let (height, width) = image.dim();
        if let Some(fast_options) = try_to_fast_options(&options, width, height) {
            let reinit = match &self.fast_extractor {
                Some(fe) => {
                    fe.orig_width() != width
                        || fe.orig_height() != height
                        || fe.options() != &fast_options
                }
                None => true,
            };
            if reinit {
                self.fast_extractor = Some(FastExtractor::new(width, height, fast_options));
            }
            let fe = self.fast_extractor.as_mut().unwrap();
            let fast_centroids = T::extract_sequential(fe, image);

            // Map FastCentroidResult to CentroidResult
            let centroids = fast_centroids
                .into_iter()
                .map(|c| CentroidResult {
                    y: c.y,
                    x: c.x,
                    sum: c.sum,
                    area: c.area,
                    m2_xx: 0.0, // Fast extractor doesn't return moments currently
                    m2_yy: 0.0,
                    m2_xy: 0.0,
                    axis_ratio: c.axis_ratio,
                })
                .collect();

            ExtractionResult {
                centroids,
                debug_images: None,
            }
        } else {
            T::extract_normal(self, image, options)
        }
    }

    /// Runs the full pipeline using the fast sequential path.
    #[cfg(feature = "extractor")]
    pub fn solve_from_image_fast<S, T>(
        &mut self,
        image: &ArrayBase<S, Ix2>,
        extract_options: ExtractOptions,
        solve_options: SolveOptions,
    ) -> Result<(Solution, f64), Box<dyn std::error::Error>>
    where
        S: Data<Elem = T>,
        T: FastPixel,
    {
        let t0 = Instant::now();
        let extract_result = self.get_centroids_from_image_fast(image, extract_options);
        let extract_time_ms = t0.elapsed().as_secs_f64() * 1000.0;

        let num_centroids = extract_result.centroids.len();
        let mut centroids_arr = Array2::zeros((num_centroids, 2));
        for (i, c) in extract_result.centroids.iter().enumerate() {
            centroids_arr[[i, 0]] = c.y;
            centroids_arr[[i, 1]] = c.x;
        }

        let (height, width) = image.dim();
        let solution = self.solve_from_centroids(
            &centroids_arr,
            (height as f64, width as f64),
            solve_options,
        )?;

        Ok((solution, extract_time_ms))
    }

    /// Runs the full pipeline: extracts centroids from the image and immediately solves them.
    /// Returns the Solution alongside the extraction time in milliseconds.
    #[cfg(feature = "extractor")]
    pub fn solve_from_image<S>(
        &mut self,
        image: &ArrayBase<S, Ix2>,
        extract_options: ExtractOptions,
        solve_options: SolveOptions,
    ) -> Result<(Solution, f64), Box<dyn std::error::Error>>
    where
        S: Data<Elem = f32>,
    {
        let t0 = Instant::now();

        // 1. Extract centroids
        let extract_result = self.get_centroids_from_image(image, extract_options);
        let extract_time_ms = t0.elapsed().as_secs_f64() * 1000.0;

        // Map Vec<CentroidResult> into the Array2<f64> expected by the solver
        let num_centroids = extract_result.centroids.len();
        let mut centroids_arr = Array2::zeros((num_centroids, 2));
        for (i, c) in extract_result.centroids.iter().enumerate() {
            centroids_arr[[i, 0]] = c.y;
            centroids_arr[[i, 1]] = c.x;
        }

        // 2. Solve
        let (height, width) = image.dim();
        let solution = self.solve_from_centroids(
            &centroids_arr,
            (height as f64, width as f64),
            solve_options,
        )?;

        Ok((solution, extract_time_ms))
    }
}

/// Internal trait to unify dispatch between f32 and u8 for fast extraction.
#[cfg(feature = "extractor")]
pub trait FastPixel: Copy {
    fn extract_sequential<S>(
        fe: &mut FastExtractor,
        image: &ArrayBase<S, Ix2>,
    ) -> Vec<crate::fast_extractor::FastCentroidResult>
    where
        S: Data<Elem = Self>;
    fn extract_normal<S>(
        t3: &mut Tetra3,
        image: &ArrayBase<S, Ix2>,
        options: ExtractOptions,
    ) -> ExtractionResult
    where
        S: Data<Elem = Self>;
}

#[cfg(feature = "extractor")]
impl FastPixel for f32 {
    fn extract_sequential<S>(
        fe: &mut FastExtractor,
        image: &ArrayBase<S, Ix2>,
    ) -> Vec<crate::fast_extractor::FastCentroidResult>
    where
        S: Data<Elem = Self>,
    {
        fe.extract_sequential_f32(image)
    }
    fn extract_normal<S>(
        t3: &mut Tetra3,
        image: &ArrayBase<S, Ix2>,
        options: ExtractOptions,
    ) -> ExtractionResult
    where
        S: Data<Elem = Self>,
    {
        t3.get_centroids_from_image(image, options)
    }
}

#[cfg(feature = "extractor")]
impl FastPixel for u8 {
    fn extract_sequential<S>(
        fe: &mut FastExtractor,
        image: &ArrayBase<S, Ix2>,
    ) -> Vec<crate::fast_extractor::FastCentroidResult>
    where
        S: Data<Elem = Self>,
    {
        fe.extract_sequential(image)
    }
    fn extract_normal<S>(
        t3: &mut Tetra3,
        image: &ArrayBase<S, Ix2>,
        options: ExtractOptions,
    ) -> ExtractionResult
    where
        S: Data<Elem = Self>,
    {
        t3.get_centroids_from_image_u8(image, options)
    }
}

#[cfg(feature = "extractor")]
fn try_to_fast_options(
    options: &ExtractOptions,
    img_width: usize,
    img_height: usize,
) -> Option<FastExtractOptions> {
    // Check if options are compatible with FastExtractor
    // 1. downsample must be 1, 2, or 4
    let ds = match options.downsample {
        None | Some(1) => FastDownsample::None,
        Some(2) => FastDownsample::X2,
        Some(4) => FastDownsample::X4,
        _ => return None,
    };

    // 2. bg_sub_mode must be GlobalMedian or GlobalMean or LocalMedian
    let bg_mode = match options.bg_sub_mode {
        Some(BgSubMode::GlobalMedian) => Some(FastBgSubMode::GlobalMedian),
        Some(BgSubMode::GlobalMean) => Some(FastBgSubMode::GlobalMean),
        Some(BgSubMode::LocalMedian) => Some(FastBgSubMode::BlockMedian { block_size: 32 }),
        None => None,
        _ => return None, // LocalMean not supported by FastExtractor
    };

    // 3. sigma_mode must be GlobalMedianAbs or GlobalRootSquare
    let sigma_mode = match options.sigma_mode {
        SigmaMode::GlobalMedianAbs => FastSigmaMode::GlobalMedianAbs,
        SigmaMode::GlobalRootSquare => FastSigmaMode::GlobalRootSquare,
        _ => return None, // Local modes not supported
    };

    // 4. crop must be None or Center
    let crop = match &options.crop {
        None => None,
        Some(Crop::Center { height, width }) => Some((*width, *height)),
        Some(Crop::Fraction(f)) => Some((img_width / f, img_height / f)),
        _ => return None, // Region not supported
    };

    // 5. max_returned is not supported by FastExtractor (it returns all)
    if options.max_returned.is_some() {
        return None;
    }

    // 6. return_images is not supported
    if options.return_images {
        return None;
    }

    // 7. image_th is not supported (it uses sigma)
    if options.image_th.is_some() {
        return None;
    }

    Some(FastExtractOptions {
        sigma: options.sigma,
        downsample: ds,
        bg_sub_mode: bg_mode,
        sigma_mode,
        binary_open: options.binary_open,
        centroid_window: options.centroid_window,
        max_area: options.max_area,
        min_area: options.min_area,
        max_sum: options.max_sum,
        min_sum: options.min_sum,
        max_axis_ratio: options.max_axis_ratio,
        crop,
    })
}
