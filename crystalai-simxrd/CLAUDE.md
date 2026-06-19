# crystalai-simxrd

Simulate PXRD patterns from CIFs and provide the on-the-fly augmentation pipeline that feeds training. No structure curation, no ML training in this package.

Before non-trivial work here, read `SIMXRD_ROADMAP.md` and the relevant section of `../DESIGN_DECISIONS.md` (§1 coordinate, §7 peak augmentation).

## Package-specific rules

- The simulation core is NumPy throughout. PyTorch enters only via the augmentor transforms (`torch.nn.Module`-compatible) exposed at the §5.2 public API — never inside the Bragg / profile / effects math.
- log(d) is the production domain; all training patterns bin uniformly in log(d) over a fixed d-window. The production path computes Bragg peaks in d-space and convolves natively in log-d. 2θ and linear-d paths exist for validation only (Strategy A: simulate-in-2θ-then-convert).
- Bragg intensities use occupancy-weighted structure factors for fractional sites — never supercell-order a disordered structure before computing peaks.
- Validate against `pymatgen.analysis.diffraction.xrd.XRDCalculator` for peak positions and integrated intensities. There is no SimXRD-4M / Pysimxrd comparison path.
- Augmentations are composable, domain-aware (each reads the `DomainGrid` tag and perturbs in the active coordinate), and applied on-the-fly during training — not pre-computed into static datasets.
- Peak-position augmentation is importance-aware: protect high-2θ / low-d peaks (they pin the lattice), and hard-reject any draw that fabricates a diagnostic systematic absence.
- Wavelength randomization, Caglioti randomization, and matched-d-range simulation are load-bearing for the wavelength-conditioning strategy — do not drop them from the `PRODUCTION` preset.

## Commands (run from this directory)

- Install editable: `pip install -e .`
- Tests: `pytest`
- Lint: `ruff check .`
- Dashboard: `python scripts/launch_dashboard.py`

The validation criteria in `SIMXRD_ROADMAP.md` §7 must pass before `crystalai-methods` consumes the simulator.
