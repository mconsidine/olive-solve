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

### Task: verify-only solve entry point (unlocks diofinder's tracking fast path)

diofinder's experimental tracking mode does ROI-windowed detection + a
tight-hint solve, but still pays the 4-star pattern-hash on every frame
because this API has **no verify-only path**. Add one:

- **API sketch**: `verify_attitude(centroids, size, attitude_hint,
  fov_estimate, match_radius, match_threshold, distortion=0.0, ...)` →
  same solution dict. Implementation: project the catalog neighborhood
  through the hinted attitude (the KD-tree + `verify_and_build_solution`
  machinery already exists in `solver.rs`), match, refine — skipping
  candidate enumeration entirely. Fail fast (status `NoMatch`) if the
  match fraction is below threshold; the caller falls back to the full
  solve.
- **Constraints**: additive only (new method; do not change
  `solve_from_centroids`); expose via tetra3-py with keyword-only args;
  diofinder will capability-probe with `hasattr(t3, "verify_attitude")`.
- **Acceptance**: on a synthetic field with a correct hint, verify-only
  returns the same attitude as the full solve in a fraction of the time;
  with a wrong hint (> match radius) it returns NoMatch quickly; wheel
  builds abi3 as before.

When you complete or add a task, update this section in the same PR — this
file is the durable task queue across sessions and agent frameworks.

## 6. Handoff protocol

Same as diofinder AGENTS.md §8: tests green before commit; PR → squash-merge
to `main`; if the public Python API changed, note explicitly in the PR what
diofinder must probe for, and coordinate the release ordering (wheel release
BEFORE the diofinder release that uses the new capability).
