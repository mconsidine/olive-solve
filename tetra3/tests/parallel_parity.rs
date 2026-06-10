// Copyright (c) 2026 Omair Kamil
// See LICENSE file in root directory for license terms.
//
// Proves the parallel pattern search returns byte-identical solutions to the
// serial path. The parallel implementation preserves first-match-in-
// enumeration-order semantics, so every field of the Solution (not just the
// pointing) must agree exactly; any divergence is a determinism bug.

use ndarray::Array2;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::Read;
use std::path::Path;
use zip::ZipArchive;

use tetra3::{SolveOptions, SolveStatus, Solver};

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

fn dto_to_options(dto: &SolveOptionsDto, parallel: bool) -> SolveOptions {
    let to_arr2 = |v: &Option<Vec<[f64; 2]>>| {
        v.as_ref().map(|rows| {
            let mut flat = Vec::with_capacity(rows.len() * 2);
            for c in rows {
                flat.push(c[0]);
                flat.push(c[1]);
            }
            Array2::from_shape_vec((rows.len(), 2), flat).unwrap()
        })
    };
    SolveOptions {
        fov_estimate: dto.fov_estimate,
        fov_max_error: dto.fov_max_error,
        match_radius: dto.match_radius,
        match_threshold: dto.match_threshold,
        solve_timeout_ms: None, // no timeout: keep both runs deterministic
        distortion: dto.distortion,
        match_max_error: dto.match_max_error,
        return_matches: true,
        return_catalog: dto.return_catalog,
        return_rotation_matrix: true,
        target_pixel: to_arr2(&dto.target_pixel),
        target_sky_coord: to_arr2(&dto.target_sky_coord),
        parallel,
        ..Default::default()
    }
}

/// Wall-clock comparison of the serial vs parallel search on no-match fields
/// (scrambled centroids force a full enumeration — the blind-solve worst
/// case). Run on target hardware with:
///   cargo test -p tetra3 --release --test parallel_parity -- --ignored --nocapture
#[test]
#[ignore]
fn bench_parallel_speedup_full_enumeration() {
    let db_path = Path::new("tests/fixtures/default_database.npz");
    if !db_path.exists() {
        eprintln!("Skipping bench: default_database.npz not found.");
        return;
    }
    let mut solver = Solver::load_database(db_path).expect("Failed to load Tetra3 database");

    // Deterministic pseudo-random star field: unlikely to match the catalog,
    // so both paths enumerate every 4-star combination.
    let n_stars = 30;
    let (height, width) = (760.0_f64, 960.0_f64);
    let mut seed: u64 = 0x5DEECE66D;
    let mut next = || {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (seed >> 33) as f64 / (1u64 << 31) as f64
    };
    let mut flat = Vec::with_capacity(n_stars * 2);
    for _ in 0..n_stars {
        flat.push(next() * height);
        flat.push(next() * width);
    }
    let centroids = Array2::from_shape_vec((n_stars, 2), flat).unwrap();

    let opts = |parallel: bool| SolveOptions {
        fov_estimate: None,
        solve_timeout_ms: None,
        parallel,
        ..Default::default()
    };

    let trials = 5;
    let mut t_serial = 0.0;
    let mut t_parallel = 0.0;
    for _ in 0..trials {
        let s = solver.solve(&centroids, (height, width), opts(false));
        assert_eq!(s.status, SolveStatus::NoMatch);
        t_serial += s.t_solve_ms;
        let p = solver.solve(&centroids, (height, width), opts(true));
        assert_eq!(p.status, SolveStatus::NoMatch);
        t_parallel += p.t_solve_ms;
    }
    println!(
        "Full-enumeration NoMatch over {} trials ({} threads): serial {:.1} ms/solve, parallel {:.1} ms/solve, speedup {:.2}x",
        trials,
        rayon::current_num_threads(),
        t_serial / trials as f64,
        t_parallel / trials as f64,
        t_serial / t_parallel
    );
}

#[test]
fn test_parallel_solve_matches_serial_exactly() {
    let db_path = Path::new("tests/fixtures/default_database.npz");
    let zip_path = Path::new("tests/fixtures/solver_fixtures.zip");

    if !db_path.exists() || !zip_path.exists() {
        eprintln!("Skipping test: solver fixtures not found.");
        return;
    }

    assert!(
        rayon::current_num_threads() > 1,
        "parity test needs a multi-threaded rayon pool to exercise the parallel path"
    );

    let mut solver = Solver::load_database(db_path).expect("Failed to load Tetra3 database");

    let zip_file = File::open(zip_path).expect("Failed to open solver_fixtures.zip");
    let mut archive = ZipArchive::new(zip_file).expect("Failed to open zip archive");

    let iterations = 738;
    let mut n_matches = 0;
    let mut failures = Vec::new();

    for x in 1..=iterations {
        let mut input_buffer = Vec::new();
        {
            let mut req_file = archive.by_name(&format!("input_{}.json", x)).unwrap();
            req_file.read_to_end(&mut input_buffer).unwrap();
        }
        let input: SolveInputDto = serde_json::from_slice(&input_buffer).unwrap();

        let mut flat_cents = Vec::with_capacity(input.centroids.len() * 2);
        for c in &input.centroids {
            flat_cents.push(c[0]);
            flat_cents.push(c[1]);
        }
        let centroids = Array2::from_shape_vec((input.centroids.len(), 2), flat_cents).unwrap();
        let size = (input.image_height, input.image_width);

        let serial = solver.solve(&centroids, size, dto_to_options(&input.options, false));
        let parallel = solver.solve(&centroids, size, dto_to_options(&input.options, true));

        if serial.status != parallel.status {
            failures.push(format!(
                "Sample {}: status diverged: serial {:?} vs parallel {:?}",
                x, serial.status, parallel.status
            ));
            continue;
        }
        if serial.status == SolveStatus::MatchFound {
            n_matches += 1;
        }

        // Exact (bit-level) agreement: the same winning combination must
        // produce the same arithmetic on both paths.
        let mut diffs = Vec::new();
        let mut cmp_f64 = |name: &str, a: Option<f64>, b: Option<f64>| {
            if a.map(f64::to_bits) != b.map(f64::to_bits) {
                diffs.push(format!("{}: serial {:?} vs parallel {:?}", name, a, b));
            }
        };
        cmp_f64("ra", serial.ra, parallel.ra);
        cmp_f64("dec", serial.dec, parallel.dec);
        cmp_f64("roll", serial.roll, parallel.roll);
        cmp_f64("fov", serial.fov, parallel.fov);
        cmp_f64("distortion", serial.distortion, parallel.distortion);
        cmp_f64("rmse", serial.rmse, parallel.rmse);
        cmp_f64("p90e", serial.p90e, parallel.p90e);
        cmp_f64("maxe", serial.maxe, parallel.maxe);
        cmp_f64("prob", serial.prob, parallel.prob);

        if serial.matches != parallel.matches {
            diffs.push(format!(
                "matches: serial {:?} vs parallel {:?}",
                serial.matches, parallel.matches
            ));
        }
        if serial.is_mirrored != parallel.is_mirrored {
            diffs.push("is_mirrored diverged".to_string());
        }
        if serial.matched_centroids != parallel.matched_centroids {
            diffs.push("matched_centroids diverged".to_string());
        }
        if serial.matched_stars != parallel.matched_stars {
            diffs.push("matched_stars diverged".to_string());
        }
        if serial.rotation_matrix != parallel.rotation_matrix {
            diffs.push("rotation_matrix diverged".to_string());
        }
        if serial.quaternion.map(|q| q.map(f64::to_bits))
            != parallel.quaternion.map(|q| q.map(f64::to_bits))
        {
            diffs.push("quaternion diverged".to_string());
        }

        if !diffs.is_empty() {
            failures.push(format!("Sample {}:\n  {}", x, diffs.join("\n  ")));
        }
    }

    println!(
        "Parity over {} samples ({} MatchFound), {} divergences",
        iterations,
        n_matches,
        failures.len()
    );
    assert!(
        n_matches > 0,
        "fixture set produced no MatchFound samples; parity test is vacuous"
    );
    assert!(
        failures.is_empty(),
        "Serial/parallel divergence in {} sample(s):\n{}",
        failures.len(),
        failures.join("\n")
    );
}
