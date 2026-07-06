# AGENTS.md — AI-agent contributor guide for olive-solve

Model-agnostic guide for any AI coding agent (or human) working in this repo
with no prior context. This repo has no CLAUDE.md; this file is the primary
onboarding document. The ecosystem-level guide lives in
`mconsidine/diofinder` → `AGENTS.md` — read that for how this crate fits into
the finder product, its release/consumption pipeline, and the incident
history that shaped the API guarantees below.

---

## 1. What this repo is

**olive-solve** is the lost-in-space plate solver used by the diofinder
electronic telescope finder. It is a Rust workspace derived from the
tetra3 / cedar-solve lineage (4-star geometric hash → SVD attitude →
verification → refinement):

| Path | Contents |
|---|---|
| `tetra3/` | The solver crate (all algorithms; `src/solver.rs` is the core) |
| `tetra3-py/` | PyO3 binding, built with maturin as an **abi3** wheel (one wheel covers every CPython ≥ 3.8). **Installs as Python package `tetra3`.** |
| `server/` | gRPC server wrapper — NOT used by diofinder; low-touch |
| `docs/decisions/` | Decision records (release-pipeline restructure etc.) |

**Naming trap:** the installed package name is `tetra3`, but this is NOT the
`mconsidine/tetra3rs` project (a separate, fuller solver on crates.io/PyPI)
nor ESA's original Python tetra3. On a diofinder device, `import tetra3` is
THIS repo's wheel.

Default branch: `main`. Work via branch → PR → squash-merge.

---

## 2. The consumer contract (do not break)

diofinder (`solver_proc.py`) is the production consumer. It installs whatever
wheel the **latest GitHub release** of this repo carries, across a fleet of
devices that update at different times. The API below is load-bearing;
diofinder capability-probes rather than version-checks, so additive,
keyword-only evolution is the rule.

```python
import tetra3
t3 = tetra3.Tetra3("/var/lib/diofinder/default_database.npz")  # positional path

soln = t3.solve_from_centroids(
    centroids,                  # Nx2 float64, (row, col), origin top-left
    (height, width),            # frame size tuple
    fov_estimate=13.54,         # deg
    fov_max_error=0.1,          # deg
    match_radius=0.01,
    match_threshold=1e-5,
    solve_timeout=1500,         # ms
    distortion=0.0,
    target_pixel=np.array([[y, x]], dtype=np.float64),   # optional, f64!
    target_sky_coord=...,       # optional Nx2 (ra, dec) f64
    return_matches=False,
    attitude_hint=(w, x, y, z), # optional; None/absent = blind
    hint_uncertainty_deg=2.0,
    strict_hint=False,          # only passed if the signature accepts it
)
# Returns a dict: RA, Dec, Roll, FOV (deg), Matches, quaternion [w,x,y,z],
# status as a Rust Debug STRING ("MatchFound"/"NoMatch"/"Timeout"/
# "Cancelled"/"TooFew"), x_target/y_target when target_sky_coord given.
# RA is None on failure — that None is diofinder's success test.
```

Behavioral guarantees diofinder depends on (each has a field incident behind
it — see diofinder AGENTS.md §4):

1. **Blind fallback after a failed hinted pass** (since v0.1.3, unless
   `strict_hint=True`). The hint-cone candidate rejection is a **geodesic**
   angle (includes roll), and diofinder's hint is built in the IMU body frame,
   so post-slew hints can sit outside their own cone. Pre-0.1.3 wheels
   hard-NoMatched in that state — a field re-acquisition deadlock. Never
   remove or weaken the fallback pass.
2. **Status strings** stay PascalCase Debug names (diofinder maps them in
   `_OLIVE_STATUS`).
3. **f64 array inputs**: the binding extracts `PyReadonlyArray2<f64>`; a
   consumer passing f32 must keep failing loudly, not silently degrade.
4. `get_centroids_from_image_fast` / `get_centroids_from_image` (the
   `extractor` cargo feature, ON by default) power diofinder's "Legacy"
   preset; centroids returned as (row, col). diofinder probes with
   `hasattr` and falls back if the feature was compiled out.
5. New kwargs must be **optional keyword args with safe defaults** — the
   binding's Python signature is `(centroids, size, **kwargs)`, so consumers
   cannot reliably introspect; diofinder gates new-kwarg use behind
   try/inspect probes and version knowledge. Removing or renaming a kwarg is
   a breaking change requiring a diofinder-side release in lockstep.

Known soft spot (documented, not load-bearing): the candidate-level FOV
window prune in `solver.rs` (`(fov2 - f_est).abs() < f_err` over
`pattern_largest_edge`) is empirically permissive on v0.1.5 — solves succeed
even when the true FOV lies outside `fov_estimate ± fov_max_error`. Treat the
window as a search accelerator, not an enforced constraint; if you tighten
it, coordinate with diofinder's calibrated-tolerance machinery first.

---

## 3. Build, test, develop

```bash
cargo build --release          # workspace
cargo test                     # solver crate tests

# Local wheel for the host arch (x86 fine — used by diofinder's offline
# debug-bundle replay):
python3 -m pip wheel ./tetra3-py --no-deps -w wheels/
pip install --force-reinstall --no-deps wheels/tetra3-*.whl
```

Validation data / integration tests live under `tetra3/tests/`. The star
databases come from `mconsidine/astro_databases` releases (`.npz` format with
`pattern_catalog`, `pattern_largest_edge`, `pattern_key_hashes`, `star_table`,
`props_packed`, `star_catalog_IDs`). Keep loader compatibility with those
assets — devices in the field do not re-download databases on wheel updates.

Performance context: the target is a Raspberry Pi Zero 2W (Cortex-A53,
`.cargo/config.toml` pins the tuning). Easy-field solves are ~2–10 ms there;
keep the lost-in-space path allocation-light.

---

## 4. Releasing

Version lives in **three files that must bump in lockstep**:
`tetra3/Cargo.toml`, `tetra3-py/Cargo.toml`, `tetra3-py/pyproject.toml`
(the last one names the wheel). Grep the old version before tagging.

The `release-wheels` workflow builds the aarch64 abi3 wheel and attaches it
to a GitHub release. Two triggers:

- tag push `vX.Y.Z`, or
- **workflow_dispatch** with input `version` — use this from environments
  whose git proxy rejects tag pushes (a known failure mode; the dispatch is
  the reliable path). Conventions: `vX.Y.Z` = full release (becomes
  "latest" — every diofinder update will pull it!), `vX.Y.Z-suffix` =
  prerelease (never "latest"; consumers opt in via diofinder's
  `OLIVE_SOLVE_TAG`), `*-dev` = build only.

**A full release is immediately fleet-visible**: diofinder images and OTA
updates install the latest release. Anything behaviorally risky goes out as
a prerelease first and gets validated via diofinder's offline bundle replay
(`diofinder/tests/diag_solve.py --bundle`) before promotion.

---

## 5. Backlog

### DONE (v0.1.7): five micro-optimizations ported from upstream `oakamil/olive-solve`

Reviewed the 3 commits the upstream project (this fork's tracking source) was
ahead by, and ported the following into `tetra3/src/solver.rs`. All are
either algebraically equivalent reformulations or provably output-equivalent
early exits — no intended behavior change, validated against the
`validate_solver` integration suite (consistency, mirrored-image, and
blind-fallback tests) after each change:

- **Early-rejection SVD pre-pass** (`try_pattern_combo`): reject a candidate
  rotation right after the SVD if the rotated catalog pattern doesn't align
  with the image pattern within a generous tolerance, before paying for the
  KD-tree verification pass.
- **Reciprocal multiplication** (`try_pattern_combo`): hoist the two
  loop-invariant largest-edge reciprocals out of the 5-edge ratio-check loop
  instead of dividing per comparison.
- **Direct 2×2 Cramer's-rule solve** (`verify_and_build_solution`): the
  distortion/FOV least-squares refinement is always a 2-unknown system;
  solve it via the normal equations + closed-form 2×2 inverse instead of a
  full `DMatrix`/`DVector` SVD pseudo-inverse.
- **`ImmutableKdTree`** (`Solver::star_kd_tree`): switched from
  `kiddo::KdTree` to `kiddo::ImmutableKdTree`, staying at `f32` (unlike
  upstream's `f64`) to preserve this fork's memory-bandwidth design intent.
  Item ids are assigned positionally in build order by `new_from_slice`,
  which already matches `star_table_flat`'s own indexing. Also swapped the
  initial nearby-star KD-tree query from `.within()` to `.within_unsorted()`,
  since the result is immediately re-sorted by star index anyway.
- **Lazy verification early-break** (`verify_and_build_solution`): replaced
  the batched rotate+project passes over every nearby catalog star with a
  single streaming per-star loop that stops once `target_crop_len`
  (`2 * num_extracted_stars`) in-frame candidates are found — same
  traversal order and in-frame predicate as before, so it selects the exact
  same star set, just without projecting stars past the cutoff in dense
  fields.

**Considered and explicitly not ported**: the 6-element sorting network
(replacing `edges.sort_unstable_by` with 12 hand-unrolled compare-swaps) and
monotonic key-space pruning (constraining the pattern-key DFS to
non-decreasing tuples). The sorting-network swap is real but sub-nanosecond
relative to the SVD/KD-tree work already dominating this function. The
monotonic pruning is provably correct (catalog pattern keys are inherently
non-decreasing, since patterns are canonicalized by sorted edge order) but
prunes a search space that's already ~1 bin wide in this fork's actual
tuning (`match_max_error`/`pattern_max_error` default 0.002, `pattern_bins`
default 50 ⇒ per-dimension window ≈ 0.2 bins) — there's essentially nothing
left to prune here, even though it might matter for upstream's own tuning.

### DONE (v0.1.6): verify-only solve entry point (`verify_attitude`)

Shipped in v0.1.6. `Solver::verify_attitude` /
`Tetra3::verify_attitude(centroids, size, attitude, options)` and the
tetra3-py binding `t3.verify_attitude(centroids, size, attitude, **kwargs)`
run the existing KD-tree + `verify_and_build_solution` machinery against a
caller-supplied (w, x, y, z) quaternion, skipping the 4-star pattern search
entirely. Same solution dict as `solve_from_centroids`; wrong attitude →
fast `NoMatch`, < 4 centroids → `TooFew`; **no blind fallback** — callers
re-acquire with `solve_from_centroids`. Additive only; diofinder
capability-probes `hasattr(t3, "verify_attitude")`. Measured on a real
bundle frame (960×760, 11 centroids, default_database): identical RA/Dec
to the full solve (0.00″ delta), median 0.01 ms vs 0.5 ms for the full
solve; wrong 5° hint rejects in 0.03 ms.

### REJECTED: caching hint-rejected candidates across the two-pass fallback

`Solver::solve_from_centroids`'s two-pass fallback (`tetra3/src/solver.rs`
~1973-2116): when `attitude_hint` is set and `strict_hint` is false, pass 0
runs hinted and pass 1 re-clears the hint and re-runs the ENTIRE nested
hash-lookup/Wahba search from scratch, even though pass 0 already computed a
rotation matrix for every candidate it hint-rejected (`try_pattern_combo`,
lines 1208-1225 — the hint check runs *after* the rotation matrix is already
in hand). Measured: a hint-rejected candidate costs 4.8×-120× a blind solve
per attempt (serial 0.042ms -> 5.078ms) because of this re-enumeration.

Investigated as a caching fix (cache hint-rejected candidates' rotation
matrix + FOV during pass 0, have pass 1 replay just that list instead of
re-running the nested search) and **rejected** — not because the fix
wouldn't work, but because the consumer-side (diofinder) usage pattern
bounds the payoff to nearly nothing: diofinder drops `attitude_hint` entirely
after 5 consecutive failed solves, so this re-enumeration can only ever
occur across at most 5 attempts per reacquisition episode, at
microsecond-to-millisecond absolute costs — tens of milliseconds of total
waste at the very worst, against reacquisition episodes that in practice ran
for minutes for unrelated reasons (a diofinder-side auto-exposure bug, since
fixed). Meanwhile the fix itself carries real risk: `image_centroids_undist`
is threaded through and refined across the whole pass by verification's
distortion correction, so replaying only the hint-rejected subset in pass 1
changes the refinement trajectory those candidates see relative to today;
and the parallel path's `find_map_first` determinism guarantee (leftmost
serial-order match wins) would need to be preserved across a cache merged
from per-chunk workers. Revisit only with a concrete measurement showing the
bound above no longer holds (e.g. a consumer that keeps a hint alive past 5
fails, or a use case where hinted attempts get much more expensive).

(no open tasks)

When you complete or add a task, update this section in the same PR — this
file is the durable task queue across sessions and agent frameworks.

## 6. Handoff protocol

Same as diofinder AGENTS.md §8: tests green before commit; PR → squash-merge
to `main`; if the public Python API changed, note explicitly in the PR what
diofinder must probe for, and coordinate the release ordering (wheel release
BEFORE the diofinder release that uses the new capability).
