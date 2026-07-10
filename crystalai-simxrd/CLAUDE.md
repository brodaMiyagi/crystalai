# crystalai-simxrd

Simulate PXRD patterns from CIFs and provide the on-the-fly augmentation pipeline that feeds training. No structure curation, no ML training in this package.

Before non-trivial work here, read `SIMXRD_ROADMAP.md` and the relevant section of `../DESIGN_DECISIONS.md` (§1 coordinate, §7 peak augmentation).

## Package-specific rules

- The simulation core is NumPy throughout. PyTorch enters only via the augmentor transforms (`torch.nn.Module`-compatible) exposed at the §5.2 public API — never inside the Bragg / profile / effects math.
- log(d) is the production/training coordinate and the encoder input; all training patterns bin uniformly in log(d) over a fixed d-window. It is the coordinate, **not** necessarily where the profile is computed: the production path builds each peak **in 2θ** (TCH-PV core ⊗ instrument-geometry axial-divergence ⊗ slit, with global zero-shift) because those instrumental effects are 2θ-native, then **resamples to the log-d grid**; stochastic augmentations (Poisson/Gaussian noise, residual background, impurity peaks) are applied natively in log-d. A single-kernel native-log-d convolution is kept only as an optional fast/approximate preset. See `SIMXRD_ROADMAP.md` §1.
- Bragg intensities use occupancy-weighted structure factors for fractional sites — never supercell-order a disordered structure before computing peaks.
- Validate against `pymatgen.analysis.diffraction.xrd.XRDCalculator` for peak positions and integrated intensities. There is no SimXRD-4M / Pysimxrd comparison path.
- Augmentations are composable, domain-aware (each reads the `DomainGrid` tag and perturbs in the active coordinate), and applied on-the-fly during training — not pre-computed into static datasets.
- Two views, two position semantics. **Full-profile view:** positions are the physical measurement — never per-peak perturbed; the only position effect is a **global zero-shift** (whole-pattern 2θ offset). **Peak-list view** (a hand-picked list at inference): augmented for human error — small selection jitter + off-center bias, and **selective low-d (high-2θ) dropping** when peak-dense, guarded by a **manufactured-absence check** (never drop a reflection whose absence would imply a higher-symmetry SG), plus additive spurious peaks. This deliberately trades some lattice precision for fidelity to the real manual input. See `SIMXRD_ROADMAP.md` §5–§6 and `DESIGN_DECISIONS.md` §7.
- Wavelength randomization, Caglioti randomization, and matched-d-range simulation are load-bearing for the wavelength-conditioning strategy — do not drop them from the `PRODUCTION` preset.

## Commands (run from this directory)

This package is a member of the root **uv workspace** (`../pyproject.toml`); `crystalai-data` resolves via the workspace, and all members share one `.venv` + `uv.lock`.

- Sync the workspace (installs this package editable): `uv sync --all-packages` (from repo root or here)
- Tests: `uv run pytest`
- Lint: `uv run ruff check .`
- Dashboard: `uv run python scripts/launch_dashboard.py`

The validation criteria in `SIMXRD_ROADMAP.md` §7 must pass before `crystalai-methods` consumes the simulator.
