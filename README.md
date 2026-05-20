# Olive Solve - Tetra3 Solver in Rust

A fast, robust, and async-friendly Rust implementation and optimization of the [cedar-solve](https://github.com/smroid/cedar-solve) centroid extraction and plate solving algorithms. 

## Unique Features

This project is not just a straight port of the upstream Python logic. It introduces several performance optimizations and unique extraction features designed for constrained hardware and specific sensor characteristics:

### Extractor

* **Optimized `u8` Pipelines**: Highly optimized processing pipelines tailored specifically for 8-bit grayscale images, minimizing memory overhead and bandwidth.
* **Fast Extractor Implementation**: Leverages aggressive pre-allocation and `rayon`-based multi-threading for increased performance across a subset of supported extraction modes.
* **Sequential Fast Extractor**: An alternative sequential path that trades a negligible amount of accuracy for much faster single-threaded performance.
* **Hybrid Background Subtraction Modes**: Includes custom `Line Median` and `Block Median` background subtraction modes. These act as high-performance compromises between the fast (but less accurate) `Global Median` and the highly accurate (but computationally expensive) `Local Median` modes. *Note: `Line Median` is specifically designed to excel at handling cameras that exhibit horizontal banding noise.*

### Solver

* **Database Support**: Supports both `tetra3` and `cedar-solve` database formats.
* **Performance**: Blazingly fast single-threaded performance - centroids generated from clean images typically solve in under 1ms on a Raspberry Pi Zero 2W.

## Repository Structure

This workspace is divided into two primary crates:

* **`tetra3`**: The core algorithms. `solver.rs` is a Rust port of the [Tetra3](https://github.com/smroid/cedar-solve/blob/master/tetra3/tetra3.py) `solve_from_centroids` function. `extractor.rs` is a Rust port of the `get_centroids_from_image` function. `tetra3.rs` provides the standard interface corresponding to the Python project.
* **`tetra3-py`**: Python bindings for the optimized tetra3 Rust implementation.
* **`server`**: A gRPC server that exposes tetra3's algorithms as a service.

## Getting Started

### Prerequisites
* [Rust / Cargo](https://rustup.rs/)
* Python 3

### Building
To build the workspace:

```
cargo build --release
```

### Testing

A set of real-world test data is provided for validating the algorithms and the wrappers.

#### Validation Tests

From the project root, run:

```
cargo test --release -- --test-threads=1
```

Optionally add `--nocapture` to the end of the command above to print the full test ouput to `stdout`.

#### Tests for Python Bindings

From the `tetra3-py` root, run:

```
./test_python_wrapper.sh
```

#### Performance Tests

The solver tests provide a performance report at the end of the output:

```
cargo test --release test_solver_consistency_with_testdata -- --nocapture
```

To compare the extraction performance against the original `cedar-solve` implementation:

1. Clone [cedar-solve](https://github.com/smroid/cedar-solve)
2. Run `./setup.sh` in the `cedar-solve` root.
3. Source the Python activation script:
```
source ../cedar-solve/.cedar_venv/bin/activate
```
4. From the repo root run:
```
cargo test --release test_performance_vs_python -- --nocapture --test-threads=1 --ignored
```

To compare the extraction performance against `cedar-detect`, run:

```
cargo test --release test_performance_vs_python -- --nocapture --test-threads=1 --ignored
```

## FAQ

1\. Why not port the database generation function?

Database generation is a one-time operation that doesn't benefit from a port.

2\. How can I generate an appropriate database?

Refer to [cedar-solve](https://github.com/smroid/cedar-solve/blob/master/tetra3/tetra3.py) or [esa/tetra](https://github.com/esa/tetra3/blob/master/tetra3/tetra3.py) for database generation. Note that the 2 versions of tetra3 have slightly different database formats, but this Rust implementation is compatible with both.

3\. What kind of performance gain can I expect to see for the solver?

On a Raspberry Pi 5 with 4 GB RAM the Rust version ~130x faster. On a Raspberry Pi Zero 2W with 512 MB of RAM the Rust version has a similar performance gain. In both cases solves in the `cedar-server` pipeline take well under 1 ms.

4\. What kind of performance gain can I expect to see for the extractor?

Benchmarks on the Raspberry Pi 5 with 4 GB RAM show ~15x improvement over the extractor in `cedar-solve`.

5\. How does the extractor port compare to `cedar-detect`?

`cedar-detect` is up to 2x as fast. The extractor port here uses the same algorithm as `cedar-solve` and produces the same results. `cedar-detect` provides a custom algorithm.

## License

This project is licensed under Apache 2.0 license.

See LICENSE.md for full details.

## Disclaimer

All product names, trademarks and registered trademarks are property of their respective owners. All company, product and service names used in this website are for identification purposes only. Use of these names, trademarks and brands does not imply endorsement.

`olive-solve` is not affiliated with, endorsed by, or sponsored by Clear Skies Astro or the European Space Agency.

Cedar™ is a trademark of Clear Skies Astro, registered in the U.S. and other countries.
