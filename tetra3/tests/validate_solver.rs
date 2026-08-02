// Copyright (c) 2026 Omair Kamil
// See LICENSE file in root directory for license terms.

use ndarray::Array2;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::Read;
use std::path::Path;
use std::time::Instant;
use zip::ZipArchive;

use tetra3::{SolveOptions, SolveStatus, Solver};

// --- Serialization Data Transfer Objects (DTOs) ---

#[derive(Serialize, Deserialize, Debug)]
pub struct SolveOptionsDto {
    pub fov_estimate: Option<f64>,
    pub fov_max_error: Option<f64>,
    pub match_radius: f64,
    pub match_threshold: f64,
    pub solve_timeout_ms: Option<f64>,
    pub distortion: Option<f64>,
    pub match_max_error: f64,
    pub return_matches: bool,
    pub return_catalog: bool,
    pub return_rotation_matrix: bool,
    pub target_pixel: Option<Vec<[f64; 2]>>,
    pub target_sky_coord: Option<Vec<[f64; 2]>>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct SolveInputDto {
    pub centroids: Vec<[f64; 2]>,
    pub image_height: f64,
    pub image_width: f64,
    pub options: SolveOptionsDto,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct SolutionDto {
    pub ra: Option<f64>,
    pub dec: Option<f64>,
    pub roll: Option<f64>,
    pub fov: Option<f64>,
    pub distortion: Option<f64>,
    pub rmse: Option<f64>,
    pub p90e: Option<f64>,
    pub maxe: Option<f64>,
    pub matches: Option<usize>,
    pub prob: Option<f64>,
    pub epoch_equinox: Option<f64>,
    pub epoch_proper_motion: Option<f64>,
    pub status: String,
    pub t_solve_ms: f64,
    pub rotation_matrix: Option<Vec<f64>>,
    pub target_ra: Option<Vec<f64>>,
    pub target_dec: Option<Vec<f64>>,
    pub target_y: Option<Vec<Option<f64>>>,
    pub target_x: Option<Vec<Option<f64>>>,
    pub matched_centroids: Option<Vec<[f64; 2]>>,
    pub matched_stars: Option<Vec<[f64; 3]>>,
    pub matched_cat_id: Option<Vec<Vec<u32>>>,
    pub catalog_stars: Option<Vec<(f64, f64, f64, f64, f64)>>,
}

// --- Tests ---

#[test]
fn test_solver_consistency_with_testdata() {
    let db_path = Path::new("tests/fixtures/default_database.npz");
    let zip_path = Path::new("tests/fixtures/solver_fixtures.zip");

    if !db_path.exists() {
        eprintln!("Skipping test: default_database.npz not found.");
        return;
    }
    if !zip_path.exists() {
        panic!(
            "Fixture zip not found! Run `cargo test generate_test_fixtures --release -- --ignored` first."
        );
    }

    let mut solver = Solver::load_database(db_path).expect("Failed to load Tetra3 database");

    let zip_file = File::open(zip_path).expect("Failed to open solver_fixtures.zip");
    let mut archive = ZipArchive::new(zip_file).expect("Failed to open zip archive");

    let mut all_failures = Vec::new();
    let mut total_solve_micros = 0;
    let samples = 738;
    let run_times = 100;
    let iterations = samples * run_times;

    // The counter in the JSON zip starts at 1
    for y in 1..=iterations {
        let x = (y % samples) + 1;
        // Read Input DTO
        let input_filename = format!("input_{}.json", x);
        let mut input_buffer = Vec::new();
        {
            let mut req_file = archive.by_name(&input_filename).unwrap();
            req_file.read_to_end(&mut input_buffer).unwrap();
        }
        let input_dto: SolveInputDto = serde_json::from_slice(&input_buffer).unwrap();

        // Read Output DTO
        let output_filename = format!("output_{}.json", x);
        let mut output_buffer = Vec::new();
        {
            let mut res_file = archive.by_name(&output_filename).unwrap();
            res_file.read_to_end(&mut output_buffer).unwrap();
        }
        let expected_dto: SolutionDto = serde_json::from_slice(&output_buffer).unwrap();

        // Map Input DTO to Array2 and SolveOptions
        let mut flat_cents = Vec::with_capacity(input_dto.centroids.len() * 2);
        for c in &input_dto.centroids {
            flat_cents.push(c[0]);
            flat_cents.push(c[1]);
        }
        let centroids_array =
            Array2::from_shape_vec((input_dto.centroids.len(), 2), flat_cents).unwrap();

        let target_pixel = input_dto.options.target_pixel.map(|tp| {
            let mut flat = Vec::with_capacity(tp.len() * 2);
            for c in &tp {
                flat.push(c[0]);
                flat.push(c[1]);
            }
            Array2::from_shape_vec((tp.len(), 2), flat).unwrap()
        });

        let target_sky_coord = input_dto.options.target_sky_coord.map(|tsc| {
            let mut flat = Vec::with_capacity(tsc.len() * 2);
            for c in &tsc {
                flat.push(c[0]);
                flat.push(c[1]);
            }
            Array2::from_shape_vec((tsc.len(), 2), flat).unwrap()
        });

        let options = SolveOptions {
            fov_estimate: input_dto.options.fov_estimate,
            fov_max_error: input_dto.options.fov_max_error,
            match_radius: input_dto.options.match_radius,
            match_threshold: input_dto.options.match_threshold,
            solve_timeout_ms: input_dto.options.solve_timeout_ms,
            distortion: input_dto.options.distortion,
            match_max_error: input_dto.options.match_max_error,
            return_matches: input_dto.options.return_matches,
            return_catalog: input_dto.options.return_catalog,
            return_rotation_matrix: input_dto.options.return_rotation_matrix,
            target_pixel,
            target_sky_coord,
            allow_out_of_bounds_target_pixel: None,
        };

        // --- Capture the execution time ---
        let start_time = Instant::now();

        let result = solver.solve(
            &centroids_array,
            (input_dto.image_height, input_dto.image_width),
            options,
        );

        let solve_duration = start_time.elapsed();
        total_solve_micros += solve_duration.as_micros();
        // -----------------------------------

        let status_str = match result.status {
            SolveStatus::MatchFound => "MatchFound",
            SolveStatus::NoMatch => "NoMatch",
            SolveStatus::Timeout => "Timeout",
            SolveStatus::Cancelled => "Cancelled",
            SolveStatus::TooFew => "TooFew",
        };

        if expected_dto.status == "MatchFound" {
            if status_str != "MatchFound" {
                all_failures.push(format!(
                    "Sample {}: Expected MatchFound but got {}",
                    x, status_str
                ));
                continue;
            }

            let epsilon = 1e-5;
            let expected_ra = expected_dto.ra.unwrap_or(0.0);
            let expected_dec = expected_dto.dec.unwrap_or(0.0);
            let expected_roll = expected_dto.roll.unwrap_or(0.0);
            let expected_fov = expected_dto.fov.unwrap_or(0.0);

            let result_ra = result.ra.unwrap_or(0.0);
            let result_dec = result.dec.unwrap_or(0.0);
            let result_roll = result.roll.unwrap_or(0.0);
            let result_fov = result.fov.unwrap_or(0.0);

            println!("--- Sample {} ---", x);
            println!("Solve time : {:.2?}", solve_duration);
            println!(
                "Expected   : RA: {:.6}, Dec: {:.6}, Roll: {:.6}, FOV: {:.6}",
                expected_ra, expected_dec, expected_roll, expected_fov
            );
            println!(
                "Actual     : RA: {:.6}, Dec: {:.6}, Roll: {:.6}, FOV: {:.6}",
                result_ra, result_dec, result_roll, result_fov
            );
            println!(
                "Diff       : RA: {:.6}, Dec: {:.6}, Roll: {:.6}, FOV: {:.6}",
                (result_ra - expected_ra).abs(),
                (result_dec - expected_dec).abs(),
                (result_roll - expected_roll).abs(),
                (result_fov - expected_fov).abs()
            );
            println!("-------------------\n");

            let mut sample_errors = Vec::new();

            if (result_ra - expected_ra).abs() >= epsilon {
                sample_errors.push(format!(
                    "RA mismatch: expected {}, got {}",
                    expected_ra, result_ra
                ));
            }
            if (result_dec - expected_dec).abs() >= epsilon {
                sample_errors.push(format!(
                    "Dec mismatch: expected {}, got {}",
                    expected_dec, result_dec
                ));
            }
            if (result_roll - expected_roll).abs() >= epsilon {
                sample_errors.push(format!(
                    "Roll mismatch: expected {}, got {}",
                    expected_roll, result_roll
                ));
            }
            if (result_fov - expected_fov).abs() >= epsilon {
                sample_errors.push(format!(
                    "FOV mismatch: expected {}, got {}",
                    expected_fov, result_fov
                ));
            }

            if !sample_errors.is_empty() {
                all_failures.push(format!(
                    "Sample {} failures:\n  {}",
                    x,
                    sample_errors.join("\n  ")
                ));
            }
        }
    }

    println!(
        "\n=== Performance Report ===\n\
         Total iterations: {}\n\
         Successful matches: {}\n\
         Pure solver time: {:.2} ms\n\
         Average time per solve: {:.2} ms\n",
        iterations,
        iterations - all_failures.len(),
        total_solve_micros as f64 / 1000.0,
        total_solve_micros as f64 / 1000.0 / (iterations as f64),
    );

    // Panic if there were any failures accumulated across ALL 738 iterations.
    if !all_failures.is_empty() {
        panic!(
            "{} of 738 test samples failed:\n\n{}",
            all_failures.len(),
            all_failures.join("\n\n")
        );
    }
}

#[test]
fn test_solver_mirrored_image() {
    let db_path = Path::new("tests/fixtures/default_database.npz");
    let zip_path = Path::new("tests/fixtures/solver_fixtures.zip");

    if !db_path.exists() {
        eprintln!("Skipping test: default_database.npz not found.");
        return;
    }

    let mut solver = Solver::load_database(db_path).expect("Failed to load Tetra3 database");

    let zip_file = File::open(zip_path).expect("Failed to open solver_fixtures.zip");
    let mut archive = ZipArchive::new(zip_file).expect("Failed to open zip archive");

    // We'll just test on the first sample
    let input_filename = "input_1.json";
    let mut input_buffer = Vec::new();
    {
        let mut req_file = archive.by_name(input_filename).unwrap();
        req_file.read_to_end(&mut input_buffer).unwrap();
    }
    let mut input_dto: SolveInputDto = serde_json::from_slice(&input_buffer).unwrap();

    let output_filename = "output_1.json";
    let mut output_buffer = Vec::new();
    {
        let mut res_file = archive.by_name(output_filename).unwrap();
        res_file.read_to_end(&mut output_buffer).unwrap();
    }
    let expected_dto: SolutionDto = serde_json::from_slice(&output_buffer).unwrap();

    // MIRROR the image centroids across the X axis
    for c in &mut input_dto.centroids {
        c[1] = input_dto.image_width - c[1];
    }

    // Set a target pixel exactly at one of the centroids to test back-projection
    let test_pixel = input_dto.centroids[0];
    input_dto.options.target_pixel = Some(vec![test_pixel]);

    let mut flat_cents = Vec::with_capacity(input_dto.centroids.len() * 2);
    for c in &input_dto.centroids {
        flat_cents.push(c[0]);
        flat_cents.push(c[1]);
    }
    let centroids_array =
        Array2::from_shape_vec((input_dto.centroids.len(), 2), flat_cents).unwrap();

    let target_pixel = input_dto.options.target_pixel.clone().map(|tp| {
        let mut flat = Vec::with_capacity(tp.len() * 2);
        for c in &tp {
            flat.push(c[0]);
            flat.push(c[1]);
        }
        Array2::from_shape_vec((tp.len(), 2), flat).unwrap()
    });

    let options = SolveOptions {
        fov_estimate: input_dto.options.fov_estimate,
        fov_max_error: input_dto.options.fov_max_error,
        match_radius: input_dto.options.match_radius,
        match_threshold: input_dto.options.match_threshold,
        solve_timeout_ms: input_dto.options.solve_timeout_ms,
        distortion: input_dto.options.distortion,
        match_max_error: input_dto.options.match_max_error,
        return_matches: input_dto.options.return_matches,
        return_catalog: input_dto.options.return_catalog,
        return_rotation_matrix: input_dto.options.return_rotation_matrix,
        target_pixel,
        target_sky_coord: None,
        allow_out_of_bounds_target_pixel: None,
    };

    let result = solver.solve(
        &centroids_array,
        (input_dto.image_height, input_dto.image_width),
        options,
    );

    assert_eq!(result.status, SolveStatus::MatchFound);
    assert!(
        result.is_mirrored,
        "Solver should have detected the image was mirrored"
    );

    let epsilon = 1e-4;

    // RA and Dec should exactly match the unmirrored original output
    assert!(
        (result.ra.unwrap() - expected_dto.ra.unwrap()).abs() < epsilon,
        "RA does not match"
    );
    assert!(
        (result.dec.unwrap() - expected_dto.dec.unwrap()).abs() < epsilon,
        "Dec does not match"
    );

    // Roll should EXACTLY MATCH the original roll, because the physical pointing has not changed
    let expected_roll = expected_dto.roll.unwrap().rem_euclid(360.0);
    let mut roll_diff = (result.roll.unwrap() - expected_roll).abs();
    if roll_diff > 180.0 {
        roll_diff = 360.0 - roll_diff;
    }
    assert!(
        roll_diff < epsilon,
        "Roll does not match original roll: got {}, expected {}",
        result.roll.unwrap(),
        expected_roll
    );

    // The target pixel back-projection should work seamlessly
    assert!(result.target_ra.is_some());
    assert!(result.target_dec.is_some());
}

#[test]
fn test_true_matches_consistency() {
    let db_path = Path::new("tests/fixtures/default_database.npz");
    let zip_path = Path::new("tests/fixtures/solver_fixtures.zip");

    if !db_path.exists() || !zip_path.exists() {
        return;
    }

    let mut solver = Solver::load_database(db_path).expect("Failed to load Tetra3 database");

    let zip_file = File::open(zip_path).expect("Failed to open solver_fixtures.zip");
    let mut archive = ZipArchive::new(zip_file).expect("Failed to open zip archive");

    // Test first 10 samples
    for x in 1..=10 {
        let input_filename = format!("input_{}.json", x);
        let mut input_buffer = Vec::new();
        {
            let mut req_file = archive.by_name(&input_filename).unwrap();
            req_file.read_to_end(&mut input_buffer).unwrap();
        }
        let input_dto: SolveInputDto = serde_json::from_slice(&input_buffer).unwrap();

        let mut flat_cents = Vec::with_capacity(input_dto.centroids.len() * 2);
        let mut array_of_arrays = Vec::with_capacity(input_dto.centroids.len());
        for c in &input_dto.centroids {
            flat_cents.push(c[0]);
            flat_cents.push(c[1]);
            array_of_arrays.push([c[0], c[1]]);
        }
        let centroids_array =
            Array2::from_shape_vec((input_dto.centroids.len(), 2), flat_cents).unwrap();

        let options = SolveOptions {
            fov_estimate: input_dto.options.fov_estimate,
            fov_max_error: input_dto.options.fov_max_error,
            match_radius: input_dto.options.match_radius,
            match_threshold: input_dto.options.match_threshold,
            solve_timeout_ms: input_dto.options.solve_timeout_ms,
            distortion: input_dto.options.distortion,
            match_max_error: input_dto.options.match_max_error,
            return_matches: true,
            return_catalog: false,
            return_rotation_matrix: true,
            target_pixel: None,
            target_sky_coord: None,
            allow_out_of_bounds_target_pixel: None,
        };

        let result = solver.solve(
            &centroids_array,
            (input_dto.image_height, input_dto.image_width),
            options.clone(),
        );

        if result.status == SolveStatus::MatchFound {
            let true_matches = solver
                .get_matches_for_centroids(
                    &result,
                    &array_of_arrays,
                    (input_dto.image_height, input_dto.image_width),
                    &options,
                )
                .map(|v| v.len());
            assert!(
                true_matches >= result.matches,
                "True match count ({:?}) is less than solver match count ({:?}) for sample {}",
                true_matches,
                result.matches,
                x
            );
        }
    }
}

#[test]
fn test_out_of_bounds_target_pixel() {
    let db_path = Path::new("tests/fixtures/default_database.npz");
    let zip_path = Path::new("tests/fixtures/solver_fixtures.zip");

    if !db_path.exists() {
        eprintln!("Skipping test: default_database.npz not found.");
        return;
    }
    if !zip_path.exists() {
        panic!(
            "Fixture zip not found! Run `cargo test generate_test_fixtures --release -- --ignored` first."
        );
    }

    let mut solver = Solver::load_database(db_path).expect("Failed to load Tetra3 database");

    let zip_file = File::open(zip_path).expect("Failed to open solver_fixtures.zip");
    let mut archive = ZipArchive::new(zip_file).expect("Failed to open zip archive");

    // Test first sample
    let input_filename = format!("input_1.json");
    let mut input_buffer = Vec::new();
    {
        let mut req_file = archive.by_name(&input_filename).unwrap();
        req_file.read_to_end(&mut input_buffer).unwrap();
    }
    let input_dto: SolveInputDto = serde_json::from_slice(&input_buffer).unwrap();

    let mut flat_cents = Vec::with_capacity(input_dto.centroids.len() * 2);
    for c in &input_dto.centroids {
        flat_cents.push(c[0]);
        flat_cents.push(c[1]);
    }
    let centroids_array =
        Array2::from_shape_vec((input_dto.centroids.len(), 2), flat_cents).unwrap();

    // Create a target sky coordinate that maps to a point far outside the image bounds.
    // Instead of computing it manually, we can just assign a target sky coordinate
    // that is opposite to the center RA/Dec.
    // Actually, to make it realistic, let's solve first without targets to get the RA/Dec.

    let base_options = SolveOptions {
        fov_estimate: input_dto.options.fov_estimate,
        fov_max_error: input_dto.options.fov_max_error,
        match_radius: input_dto.options.match_radius,
        match_threshold: input_dto.options.match_threshold,
        solve_timeout_ms: input_dto.options.solve_timeout_ms,
        distortion: input_dto.options.distortion,
        match_max_error: input_dto.options.match_max_error,
        return_matches: false,
        return_catalog: false,
        return_rotation_matrix: true,
        ..Default::default()
    };

    let result = solver.solve(
        &centroids_array,
        (input_dto.image_height, input_dto.image_width),
        base_options.clone(),
    );
    assert_eq!(result.status, SolveStatus::MatchFound);

    let ra = result.ra.unwrap();
    let dec = result.dec.unwrap();

    // Choose a target coordinate ~8 degrees away in Dec (image FOV is typically ~12 degrees, so edge is ~6 deg away)
    let target_dec = dec + 8.0;
    let target_ra = ra;

    let mut target_sky_coord = Array2::<f64>::zeros((1, 2));
    target_sky_coord[[0, 0]] = target_ra;
    target_sky_coord[[0, 1]] = target_dec;

    // Test with allow_out_of_bounds_target_pixel = false (or None)
    let mut options_strict = base_options.clone();
    options_strict.target_sky_coord = Some(target_sky_coord.clone());
    options_strict.allow_out_of_bounds_target_pixel = Some(false);

    let result_strict = solver.solve(
        &centroids_array,
        (input_dto.image_height, input_dto.image_width),
        options_strict,
    );
    assert_eq!(result_strict.status, SolveStatus::MatchFound);
    assert_eq!(result_strict.target_x.unwrap()[0], None);
    assert_eq!(result_strict.target_y.unwrap()[0], None);

    // Test with allow_out_of_bounds_target_pixel = true
    let mut options_allow = base_options.clone();
    options_allow.target_sky_coord = Some(target_sky_coord.clone());
    options_allow.allow_out_of_bounds_target_pixel = Some(true);

    let result_allow = solver.solve(
        &centroids_array,
        (input_dto.image_height, input_dto.image_width),
        options_allow,
    );
    assert_eq!(result_allow.status, SolveStatus::MatchFound);
    let x = result_allow.target_x.unwrap()[0];
    let y = result_allow.target_y.unwrap()[0];

    assert!(x.is_some());
    assert!(y.is_some());

    let x_val = x.unwrap();
    let y_val = y.unwrap();

    // Ensure the returned pixel is indeed out of bounds
    assert!(
        x_val < 0.0
            || x_val > input_dto.image_width
            || y_val < 0.0
            || y_val > input_dto.image_height,
        "Target pixel was not actually out of bounds: x={}, y={}, w={}, h={}",
        x_val,
        y_val,
        input_dto.image_width,
        input_dto.image_height
    );
}
