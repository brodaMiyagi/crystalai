# CrystalAI-simXRD — Roadmap

**Scope:** Simulate PXRD patterns from CIFs. Provide the on-the-fly augmentation pipeline that feeds training. Provide an interactive simulation dashboard for validation and demos.

**Out of scope:** Crystal-structure curation (lives in `CrystalAI-data`). ML training (lives in `CrystalAI-methods`).

---

## 1. Design at a glance

### Output domains

The simulator can produce patterns in either **2θ-intensity** or **log(d)-intensity** space.

- **2θ-I** is the natural format for angle-dispersive lab XRD and used for sanity-checking against pymatgen and against experimental data.
- **log(d)-I is the production format for training.** All training patterns are binned uniformly in log(d) over a fixed d-window. Rationale: `DESIGN_DECISIONS.md` §1.

D-spacing without the logarithm is supported as a derived format but is not the production target.

### Production simulation path: native log-d convolution

The simulator's production path computes Bragg peaks in d-space directly, then convolves each peak with a resolution function defined in log-d. The resolution function in log-d is approximately constant FWHM across the pattern (this is the property that makes log-d attractive — see `DESIGN_DECISIONS.md` §1), with the profile shape transformed from the 2θ-domain pseudo-Voigt to a slightly asymmetric form via the Jacobian.

Two non-production strategies are retained for validation:

- **Strategy A (simulate-in-2θ-then-convert-to-log-d).** Used only as a validation tool — comparing the production native-log-d output against a 2θ-convolved-then-converted reference quantifies the profile-shape approximation error.
- **Strategy C (TOF-native, in d-space with Ikeda-Carpenter back-to-back exponential × Gaussian).** Out of scope for this submission; reserved for future neutron diffraction extensions.

### Per-peak profile shapes

Profiles are computed per-peak with position-dependent FWHM via the Caglioti relation in 2θ-space, mapped to log-d via the Jacobian for the production path. This is physically correct — peak widths broaden at higher 2θ due to instrumental and sample effects — and avoids the unrealistic uniform-kernel simplification common in ML training datasets.

### Augmentation pipeline

Augmentations are composable per-call transforms in the style of torchvision's `transforms.Compose`. Each is domain-aware: it reads the `DomainGrid` tag from the pattern and applies physically appropriate perturbations in the active coordinate. Augmentations are applied on-the-fly during training, not pre-computed into static datasets.

---

## 2. Package structure

```
crystalai-simXRD/
├── pyproject.toml
├── README.md
├── ROADMAP.md                            # This file
│
├── src/
│   └── crystalai_simxrd/
│       ├── __init__.py
│       ├── core/
│       │   ├── crystal.py                # Pymatgen Structure wrapper
│       │   ├── bragg.py                  # Bragg equation; peak positions + intensities
│       │   ├── scattering.py             # Cromer-Mann atomic scattering factors
│       │   ├── wavelengths.py            # Standard X-ray wavelengths
│       │   └── domain.py                 # Domain enum (TWO_THETA, D_SPACING, LOG_D), Jacobians, DomainGrid
│       │
│       ├── profiles/
│       │   ├── peak_shapes.py            # Gaussian, Lorentzian, pseudo-Voigt, split-PV (2θ domain)
│       │   ├── caglioti.py               # Caglioti U,V,W → FWHM(2θ); mixing parameter η(2θ)
│       │   ├── asymmetry.py              # Axial divergence asymmetry (Finger-Cox-Jephcoat)
│       │   └── convolver.py              # Domain-aware profile convolution engine
│       │
│       ├── effects/
│       │   ├── broadening.py             # Scherrer size broadening, Williamson-Hall strain
│       │   ├── thermal.py                # Debye-Waller factor
│       │   ├── preferred_orientation.py  # March-Dollase model
│       │   ├── absorption.py             # Brindley absorption
│       │   ├── lorentz_polarization.py   # LP factor
│       │   └── background.py             # Chebyshev polynomial + noise models
│       │
│       ├── augmentations/
│       │   ├── profile_augmentations.py  # Full-pattern augmentations
│       │   ├── peak_augmentations.py     # Peak-list augmentations (importance-aware)
│       │   └── presets.py                # Named augmentation presets
│       │
│       ├── simulation/
│       │   ├── simulator.py              # Main orchestrator
│       │   └── batch.py                  # High-throughput batch simulation
│       │
│       ├── io/
│       │   ├── pattern_io.py             # Read/write patterns (xy, csv, xrdml, raw)
│       │   └── experimental.py           # Loaders for experimental patterns (delegates to crystalai-data)
│       │
│       ├── comparison/
│       │   ├── metrics.py                # Rwp, Rp, cosine similarity, peak-position RMSD
│       │   ├── alignment.py              # 2θ alignment / zero-shift correction
│       │   └── overlay.py                # Plotly overlay plotting utilities
│       │
│       └── utils/
│           ├── binning.py                # Domain-aware grid construction
│           ├── domain_convert.py         # Full-profile conversion between domains with Jacobian
│           ├── normalization.py
│           └── peak_detection.py         # Simple peak finding for validation only
│
├── app/
│   └── dashboard.py                      # Gradio interactive simulation app
│
├── notebooks/
│   ├── 01_profile_effects_gallery.ipynb
│   ├── 02_sim_vs_experimental.ipynb
│   ├── 03_domain_comparison.ipynb
│   └── 04_augmentation_sweep.ipynb
│
├── tests/
│   ├── test_bragg.py
│   ├── test_profiles.py
│   ├── test_domain.py
│   ├── test_effects.py
│   ├── test_augmentations.py
│   └── test_simulator.py
│
├── data/
│   ├── example_cifs/                     # NaCl, Si, CeO2, LaB6, α-quartz, rutile + one disordered
│   └── reference/                        # Scattering factor tables, wavelength tables
│
└── scripts/
    ├── simulate_batch.py                 # CLI: batch simulate from a directory of CIFs
    ├── compare_with_experiment.py        # CLI: overlay sim vs experimental
    └── launch_dashboard.py
```

No `simxrd_compat.py`. No `Pysimxrd` integration. No `database.py` (that lives in `crystalai-data`).

---

## 3. Dependencies

```toml
[project]
name = "crystalai-simxrd"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
    "numpy>=1.24",
    "scipy>=1.10",
    "matplotlib>=3.7",
    "pymatgen>=2024.1",
    "plotly>=5.15",
    "gradio>=4.0",
    "tqdm",
    "pyyaml",
    "crystalai-data",                    # for CIF source
]

[project.optional-dependencies]
dev = ["pytest", "ruff"]
```

No `ase`. No `Pysimxrd`. No `mp-api`.

---

## 4. Implementation phases

### Phase 0: Scaffolding

- Package skeleton, `pyproject.toml`, all subpackage `__init__.py` files.
- Reference data: Cromer-Mann scattering factors (JSON), standard wavelengths (YAML).
- Test infrastructure: `pytest` config, example CIF fixtures (NaCl, Si, CeO₂, LaB6, α-quartz, rutile + one disordered structure).
- `wavelengths.py`: dictionary of standard X-ray wavelengths (Cu Kα1/2/avg, Mo Kα1, Co Kα1, Cr Kα1, Ag Kα1, plus a synchrotron wavelength range generator).

### Phase 1: Core simulation engine (2θ + log-d)

| Step | Task |
|------|------|
| 1.1 | `domain.py` — `Domain` enum (`TWO_THETA`, `D_SPACING`, `LOG_D`); coordinate transforms; Jacobians; `DomainGrid` dataclass with domain tag + optional wavelength |
| 1.2 | `crystal.py` — pymatgen `Structure` wrapper; lattice parameters, space group, atomic positions |
| 1.3 | `scattering.py` — Cromer-Mann 9-parameter atomic scattering factors `f(sinθ/λ)` |
| 1.4 | `bragg.py` — given a `Structure` + wavelength + d-window: compute all allowed Bragg reflections with `(hkl, 2θ, d, multiplicity, |F(hkl)|², integrated intensity)`. **Disorder handling**: occupancy-weighted structure factors for sites with fractional occupancies — never supercell-order disordered structures before computing Bragg peaks. Validate against pymatgen's `XRDCalculator`. |
| 1.5 | `lorentz_polarization.py` — LP factor `(1 + cos²(2θ)) / (sin²(θ)·cos(θ))` for Bragg-Brentano geometry; baked into integrated intensity in 2θ-domain before any conversion |
| 1.6 | `peak_shapes.py` — Gaussian, Lorentzian, pseudo-Voigt, Thompson-Cox-Hastings PV. Operate on a generic x-axis. |
| 1.7 | `caglioti.py` — Caglioti relation: `FWHM_G² = U·tan²(θ) + V·tan(θ) + W`; mixing parameter η(2θ) from Thompson-Cox-Hastings; defaults for STADI-P, generic Bragg-Brentano |
| 1.8 | `binning.py` — domain-aware grid construction. `log_d_grid(d_min, d_max, n_bins)` is the production grid; `two_theta_grid` for validation. |
| 1.9 | `convolver.py` — domain-aware convolution. **Production path**: native log-d convolution. For each Bragg peak, compute FWHM in 2θ via Caglioti, map to log-d via the Jacobian `FWHM_logd = FWHM_2θ · |d(logd)/d(2θ)|`, generate the profile (slightly asymmetric pseudo-Voigt approximation in log-d), sum onto the log-d grid. **Validation path**: simulate-in-2θ then convert. |
| 1.10 | `simulator.py` (v1) — orchestrator: `CIF → Structure → Bragg peaks → convolved profile`. Accepts `domain=Domain.LOG_D` (production default) or other domains. Minimal version with LP factor and convolution only. |

**Checkpoint.** Simulator produces basic patterns in 2θ and log-d. Peak positions match pymatgen to within 0.001° (2θ) or 0.001 Å (d). Round-trip: simulate in 2θ, convert to log-d, fresh simulate in log-d — integrated intensities agree within 1%.

### Phase 2: Physical effects

| Step | Task |
|------|------|
| 2.1 | `thermal.py` — Debye-Waller factor `exp(-B·sin²(θ)/λ²)` applied to structure factors |
| 2.2 | `broadening.py` — size (Scherrer Lorentzian `β_L = Kλ/(D cosθ)`), strain (Gaussian `β_G = 4ε·tanθ`); fold into Caglioti parameters or add in quadrature |
| 2.3 | `preferred_orientation.py` — March-Dollase model |
| 2.4 | `asymmetry.py` — Finger-Cox-Jephcoat axial divergence (affects low-2θ peaks) |
| 2.5 | `absorption.py` — Brindley microabsorption; flat-plate absorption |
| 2.6 | `background.py` — Chebyshev polynomial (order 5–10), linear interpolation, physically motivated (Compton + air scatter + dark current). For simulation: sample Chebyshev coefficients randomly to produce varied backgrounds. |
| 2.7 | `domain_convert.py` — full-profile domain conversion between 2θ and log-d (resampling + Jacobian correction). Validation tool only — production simulates natively in the target domain. |
| 2.8 | `simulator.py` (v2) — integrate all effects with toggleable flags; YAML / dataclass configuration |

**Checkpoint.** Patterns include realistic peak shapes, thermal effects, size/strain broadening, preferred orientation, asymmetry, backgrounds — natively in log-d. Domain conversion of a convolved pattern preserves integrated intensities within 2%. Visual comparison against a handful of experimental patterns shows qualitatively similar profile shapes.

### Phase 3: Augmentation pipeline

| Step | Task |
|------|------|
| 3.1 | `profile_augmentations.py` — domain-aware composable transforms. See §5 below. |
| 3.2 | `peak_augmentations.py` — importance-aware peak-list augmentation. See §6 below. |
| 3.3 | `presets.py` — named presets: `DIFCON_STYLE` (matching the prior paper's setup), `PRODUCTION` (wavelength + Caglioti + mild asymmetry + matched d-range), `AGGRESSIVE`, `MILD` |

**Checkpoint.** Augmentation pipeline produces visually realistic variations spanning the experimental variability envelope. Each preset is callable, composable, and respects the pattern's domain tag.

### Phase 4: Interactive dashboard

`app/dashboard.py` — Gradio app with tabs:

- **Single Pattern Simulator.** Upload CIF or pick from examples. Sliders for: domain toggle (2θ / log-d), wavelength (greyed in log-d), 2θ / d range, step size, crystallite size, microstrain, temperature, preferred orientation, Caglioti U/V/W, background type + level, zero shift. Live plot updates.
- **Effect Decomposition.** Same pattern with effects toggled on/off: stick → + LP → + thermal → + size → + strain → + background → + noise.
- **Augmentation Preview.** Select a preset, apply N random augmentations to the same base pattern, overlay all.
- **Sim vs Experimental.** Upload or select an experimental pattern + corresponding CIF. Overlay simulated vs experimental, Rwp, difference curve. Adjust sim parameters to minimize residual.

### Phase 5: Batch simulation & PyTorch integration

| Step | Task |
|------|------|
| 5.1 | `batch.py` — given a CIF iterator (typically from `crystalai_data.CrystalDatabase`) + augmentation preset + domain choice, simulate all patterns. Output: HDF5 or parquet with `[cif_id, x_axis, intensity, peak_positions_d, peak_intensities, space_group, crystal_system, wavelength]`. Multiprocessing support. |
| 5.2 | Public API for `CrystalAI-methods`: `from crystalai_simxrd import Simulator, ProfileAugmentor, PeakAugmentor, Domain`. `Simulator` returns a `SimulatedPattern` dataclass with `(x_axis, intensity, peak_positions_d, peak_intensities, domain, wavelength, metadata)`. Augmentors are `torch.nn.Module`-compatible transforms that respect the pattern's domain tag. |
| 5.3 | `normalization.py` — pattern normalization (max, area, sqrt for Poisson-like data) |
| 5.4 | `peak_detection.py` — simple `scipy.find_peaks` wrapper for validation. Not used in production. |

**Checkpoint.** `CrystalAI-methods` can import this package and use `Simulator` + augmentors directly in its DataLoader.

---

## 5. Profile augmentations

All augmentations are domain-aware: they read the `DomainGrid` tag and apply physically appropriate perturbations in the active coordinate. The `PRODUCTION` preset combines all of these.

**Required augmentations** (load-bearing for the wavelength-conditioning strategy):

1. **Wavelength randomization.** For each training CIF, sample a wavelength uniformly from ~[0.5, 1.8] Å covering Cu Kα, Mo Kα, Co Kα, Cr Kα, Ag Kα, and synchrotron values. Simulate the full 2θ pattern at that wavelength with physically correct LP, DW, absorption, then convert to log-d. Without this, the wavelength input to the encoder becomes meaningless — the model cannot learn to use it. Rationale: `DESIGN_DECISIONS.md` §1.
2. **Caglioti parameter randomization.** Sample U, V, W from physically realistic instrument-to-instrument ranges spanning sharp synchrotron sources to broadened lab diffractometers.
3. **Matched d-range simulation.** Choose the d-window first (default 0.7 Å to 8 Å); for each sampled wavelength, compute the corresponding 2θ range and simulate over exactly that range. Guarantees all training patterns cover the same log-d window after binning.

**Additional augmentations:**

4. **Zero shift.** In 2θ: `δ(2θ) ~ U(-0.02°, 0.02°)`. Mapped to the equivalent perturbation in log-d.
5. **Position noise.** Per-peak white noise on positions (scaled appropriately by domain).
6. **Peak cropping / padding.** Randomly crop tails of the pattern (low / high end of the active range).
7. **Intensity noise.** Multiplicative `I' = I · (1 + ε)`, ε ~ N(0, σ²). Domain-independent.
8. **Impurity peaks.** Insert 0–6 random peaks (random position, random small intensity).
9. **Background perturbation.** Additive polynomial + white noise + Poisson noise.
10. **Mild profile asymmetry randomization.** Small randomization of asymmetry parameters (Finger-Cox-Jephcoat). Not aggressive — with wavelength as input, the model can learn the correct asymmetry-wavelength relationship.
11. **Edge cropping / padding.** Trim or extend the range. Used in conjunction with the high-log-d truncation needed for the masked-window training (see `METHODS_ROADMAP.md`).

**Explicitly dropped:** intensity envelope perturbation (originally proposed as a defense against envelope shortcuts; counterproductive once wavelength is an explicit input — destroys real intensity information the generator needs).

---

## 6. Peak-position augmentations (importance-aware)

Peak-position augmentations respect the physical role of each peak in lattice and SG determination. See `DESIGN_DECISIONS.md` §7 for the full reasoning chain.

**Rules:**

1. **Position jitter is angle-aware.** Per-peak jitter scales with local peak FWHM in log-d (approximately uniform under W1), with a floor preventing collapse at low d.
2. **Three-component drop model:**
   - *Correlated-failure*: drop probability elevated for neighbours within a configurable log-d window when one peak is dropped.
   - *High-information protection*: peaks above a 2θ threshold (default ~60° at Cu Kα equivalent) have drop probability multiplicatively capped (default 0.2×).
   - *Baseline random*: small uniform component for true random misses (default 0.05).
3. **Manufactured-absence guard (hard reject).** After every augmentation draw, check the resulting peak list against the source CIF's extinction conditions. If any reflection removed is one whose presence is *diagnostic* for the true space group (its observation rules out a higher-symmetry alternative), reject the draw and resample.
4. **Spurious peaks unchanged from the original spec.** 0–3 random spurious peaks per pattern.

**Hyperparameter defaults are starting points to be tuned.** See `DESIGN_DECISIONS.md` §7 for the table.

---

## 7. Validation criteria

Before this package is considered ready for `CrystalAI-methods`:

1. **Peak position accuracy.** Simulated peak positions match pymatgen's `XRDCalculator` to within 0.005° (2θ) or 0.0005 Å (d) for all test CIFs.
2. **Relative intensity agreement.** Pearson correlation > 0.95 between our integrated intensities and pymatgen's for standard references (LaB6, CeO₂, Si).
3. **Disorder fidelity.** For disordered test CIFs, the full pipeline (serialize to SQLite, deserialize, run Bragg calculator) produces integrated intensities matching pymatgen's `XRDCalculator` to within 1% relative for all peaks above 1% intensity. Species occupancies and fractional coordinates round-trip at full numerical precision.
4. **Domain round-trip fidelity.** Simulate in 2θ → convert to log-d → convert back to 2θ; integrated intensity per peak preserved within 2%. Peak positions preserved within 0.01° after round-trip.
5. **Native log-d vs converted consistency.** Strategy B (native log-d) and Strategy A (simulate-in-2θ-then-convert) agree on peak positions within 0.001 Å and integrated intensities within 3% relative.
6. **Experimental realism.** For at least 5 crystals where both a CIF and an experimental pattern are available in `crystalai-data`, the simulated pattern (with appropriate broadening / background settings) achieves Rwp < 15%.
7. **Augmentation coverage.** Augmented simulated patterns visually span the variability range of experimental patterns (qualitative, assessed via t-SNE of augmented sim vs experimental distributions in feature space).
8. **Performance.** Batch simulation of 1000 CIFs under one condition completes in < 5 minutes on a single CPU core.

---

## 8. Outputs consumed by other repos

| Consumer | What it gets |
|----------|--------------|
| `CrystalAI-methods` | `Simulator` (CIF → `SimulatedPattern`), `ProfileAugmentor` (composable augmentation pipeline), `PeakAugmentor` (importance-aware peak-list augmentation), `Domain` enum, `SimulatedPattern` dataclass |
| Standalone usage | `scripts/simulate_batch.py`, `scripts/compare_with_experiment.py`, `app/dashboard.py` |
