# Decision record — release-pipeline restructure across the eFinder repos

- **Date:** 2026-06-12T14:47Z (session spanned 2026-06-10 — 2026-06-12)
- **Session:** Claude Code session "pensive-allen", ID `session_01MBjXhx3TLxkk3WEkvHWqRX`
  (https://claude.ai/code/session_01MBjXhx3TLxkk3WEkvHWqRX), working branches `claude/pensive-allen-q5iis1`
- **Repositories affected:** astro_databases, diofinder, olive-solve, sycamore-extract, tetra3rs
  (plus user-applied changes to the cedar-solve fork)

## Changes in this repository (olive-solve)

**Role:** the solver that runs on the device — its `tetra3-py` wheel installs under the import name
`tetra3` and performs every live plate solve, loading the cedar-solve-generated `.npz` (its loader
implements both the cedar-solve 828-byte and standard tetra3 props schemas).

- New `release-wheels.yml` (this repo previously had no CI): a `v*` tag or dispatch builds the abi3
  aarch64 wheel (one wheel covers Python ≥ 3.8; Cortex-A53 tuning from the committed
  `.cargo/config.toml`) and attaches it to a GitHub release (`03a6d07`).
- Version conventions: `vX.Y.Z` = full release (becomes "latest"); `vX.Y.Z-suffix` = **prerelease**,
  excluded from "latest" so devices and default image builds never pick up branch/test wheels —
  consumers opt in via diofinder's `olive_solve_tag` input or `OLIVE_SOLVE_TAG` variable; `*-dev` =
  build-only (`172847a`, cherry-picked to `olive-solve-noext` as `ac57e4b` via PR #5).
- **User decision:** the deployed flavor is the `olive-solve-noext` branch (solver without the
  extraction pipelines — sycamore does extraction on-device). Release v0.1.1 (noext) is the fleet
  "latest"; division of labor mirrors cedar-detect (extract) / cedar-solve (solve).
- Branch cleanup: both `claude/*` branches verified fully content-present in `olive-solve-noext`
  and deleted. Note: `main` does not contain the noext-branch work; treat noext as the branch of
  record or reconcile main deliberately.

## Session narrative and key decisions

1. **astro_databases assessed, then narrowed to its name** — the repo had ~52 MB of committed binaries,
   a single failed CI run, a workflow depending on a release and a sibling workflow that never existed,
   and a generation script written against an imagined cedar-solve API (`fov_range=`) that had never
   executed. Decision: data + generation + tagged releases only; the five foreign wheel-build jobs were
   dropped in favor of per-repo workflows.
2. **GitHub releases over going offline** — release assets don't count against repo quotas (2 GB/file)
   and Actions is free for public repos, so the contemplated local/'act' pipeline was unnecessary;
   'act' was specifically recommended against (token still required, runner-image drift, no benefit
   over a plain script).
3. **Tagged, immutable, manifest-verified releases as the vendoring contract** — every database release
   carries `manifest.json` (parameters, input/output SHA-256s, generator versions); consumers verify
   hashes at fetch time.
4. **One star list for everything** — the merged Gaia DR3 + Hipparcos catalog (63,491 stars to G ≈ 8.0)
   feeds both database formats. A ~100-line converter (`gaia_to_hip.py`) reformats it into hip_main
   layout so stock esa/tetra3, cedar-solve, and olive-solve parse it **unmodified** — upstream
   compatibility preserved with zero solver patches. Verified: identical pattern counts vs hip_main
   with substantially denser lattice fields (min field depth 26 → 35 stars), `.npz` cost only ~0.5 MB.
5. **Per-repo wheel workflows, not an aggregator repo** — build config lives next to the code it
   builds; the old aggregator's pathologies (cross-repo SHA bookkeeping, drifting build configs) are
   structurally avoided. diofinder pulls from three release URLs instead of one.
6. **diofinder pulls everything at build time** — no binaries in git; `vendor/wheels/` survives only as
   a gitignored staging dir so `install.sh`/chroot flow stayed untouched. OTA (`efinder-update`)
   refreshes wheels from latest releases, deliberately non-fatally.
7. **Branch-testing without fleet risk** — hyphenated tags (e.g. `v0.2.0-noext`) publish as
   prereleases everywhere, invisible to "latest"-following consumers; selected explicitly via dispatch
   inputs or repo variables.
8. **Identity clarifications** — three codebases answer to `tetra3`: cedar-solve (Python; generates the
   `.npz` in CI only), olive-solve (Rust; the wheel *named* tetra3 that does all on-device solving),
   and upstream esa/tetra3 (custom-FOV fallback only). cedar-detect and cedar-solve do not overlap:
   they are the extract/solve halves of the Cedar design, mirrored here by sycamore / olive-solve.

## Assessments performed

- Initial astro_databases audit (broken paths, phantom dependencies, dead API calls — all documented in
  commit messages).
- Catalog compatibility: committed `gaia_hipp_merged.bin` header (`GDR3` v1, 63,491 stars) matches the
  current tetra3rs parser; G-band depth exactly 8.01; converter output survives the exact hip_main
  parser logic 63,491/63,491.
- olive-solve loader verified to support the cedar-solve `.npz` schema (828-byte props branch) —
  basis for `efinder-db-update` on existing devices.
- Database A/B (hip vs Gaia): 41,394 → 63,154 stars kept; patterns ~950k in both (capped); lattice
  field density up ~20–30%; `cedar_solve_13deg.npz` 13.3 MB.
- Post-release audit of the first green diofinder image (v0.0.24, first build): found it had silently
  shipped **without** the sycamore extractor (abi3 wheel name vs cp313 fetch pattern) — fixed and
  rebuilt; the second v0.0.24 build verified complete.

## Outstanding recommendations

- **olive-solve `main` lags `olive-solve-noext`** (extractor feature-gate, rayon perf work, release
  workflow). Either declare noext the trunk or merge it into main before they drift further.
- **GitHub Actions Node 20 deprecation:** runners force Node 24 from 2026-06-16; the workflows use
  actions/checkout@v4, setup-python@v5, upload-artifact@v4 — bump majors when convenient.
- **cedar-solve fork hygiene:** the fork now carries relaxed dependency floors and version 0.6.0;
  re-apply/verify on future upstream syncs. The uv `--override` in build-databases.yml remains as a
  safety net.
- **tetra3rs `.bin` format coupling:** the database format is tied to the generating tetra3rs version
  (recorded in the manifest). Regenerate + retag astro_databases when upgrading tetra3rs on a device;
  use `TETRA3RS_REF` if the fork diverges.
- **Derived-file discipline:** rerun `scripts/gaia_to_hip.py` whenever `gaia_hipp_merged.csv` changes
  (deterministic, byte-identical output for unchanged input).
- **Existing devices** get the new database via `sudo efinder-db-update [tag]` (after an OTA update
  delivers the script) or the documented curl one-liner against
  `releases/latest/download/cedar_solve_13deg.npz`.
