# CrystalAI-simXRD — Roadmap

**Scope:** Simulate PXRD patterns from CIFs. Provide the on-the-fly augmentation pipeline that feeds training. Provide an interactive simulation dashboard for validation and demos.

**Out of scope:** Crystal-structure curation (lives in `CrystalAI-data`). ML training (lives in `CrystalAI-methods`).

---

## 1. Design at a glance

Full rationale for the design choices below lives in `../DESIGN_DECISIONS.md`; this section states only the load-bearing points.

**Output domains.** Patterns can be produced in **2θ** (validation / sanity-checks vs pymatgen and experimental data) or **log(d)** (the production/training format — uniform log(d) bins over a fixed d-window). Linear-d is a derived format, not a target. Rationale: `DESIGN_DECISIONS.md` §1.

**Simulation domain — build in 2θ, convert to log-d once (CW).** log(d) is the encoder coordinate, not where the physics is computed. For constant-wavelength sources (all that are in scope), the pattern is **assembled in 2θ** — where the instrument physics is defined — and converted to log-d **once at the end**; only normalization and the relative-noise augmentation follow in log-d. Native-log-d single-kernel convolution is kept only as a fast approximate preview preset; TOF/neutron get their own strategies later. Rationale: `DESIGN_DECISIONS.md` §1a.

**Per-peak profile shape.** Core is the **Thompson-Cox-Hastings pseudo-Voigt** (Gaussian FWHM from the Caglioti relation `U·tan²θ + V·tanθ + W`; Lorentzian FWHM from crystallite size / strain), convolved with the instrument kernels (axial divergence, slit) in 2θ.

**Precompute vs on-the-fly.** Per-structure Bragg cost scales as `O(N_reflections × n_sites)` and the ICSD size distribution is heavy-tailed (p99 606 sites, seconds each), so pure on-the-fly Bragg cannot feed the DataLoader. The fix exploits that the **d-space peak list `{d, |F|²}` is wavelength-independent** (everything structure-dependent depends on `sinθ/λ = 1/2d`): **precompute it once per structure** (fork-safe `bragg_peaks` blob store keyed by `cif_id`, ~2–3 GB); **everything else runs on-the-fly** from that list (wavelength, geometry, profile build, effects, noise, augmentation) at flat, tail-free cost. This preserves full wavelength/augmentation diversity. Full rationale, the ICSD long-tail figure, Track-A-vs-B, storage, and the lossless-split validation gate: `DESIGN_DECISIONS.md` §9. Code: `simulation/batch.py` + `scripts/precompute_bragg.py` (Phase 5.1); `Simulator` accepts either a `Structure` (compute Bragg) or a cached peak list (skip to the live stage).

**Augmentation pipeline.** Composable per-call transforms (torchvision-`Compose` style), domain-aware, applied on-the-fly during training — never pre-computed into static datasets. See §5.

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
│       │   ├── peak_shapes.py            # Gaussian, Lorentzian, Voigt, Thompson-Cox-Hastings PV (production core), split-PV (2θ domain)
│       │   ├── caglioti.py               # Caglioti U,V,W → Gaussian FWHM(2θ); TCH mixing η(2θ)
│       │   ├── axial_divergence.py       # Axial-divergence asymmetry, parameterized by instrument geometry (Van Laar-Yelon; FCJ alt.)
│       │   ├── slit.py                   # Rectangular slit (top-hat) instrumental-resolution response
│       │   └── convolver.py              # Per-peak 2θ composite (TCH-PV ⊗ axial-div ⊗ slit) → resample to log-d grid
│       │
│       ├── effects/
│       │   ├── broadening.py             # Scherrer size broadening (Lorentzian), Williamson-Hall strain (Gaussian)
│       │   ├── thermal.py                # Debye-Waller factor
│       │   ├── preferred_orientation.py  # March-Dollase model
│       │   ├── absorption.py             # Brindley absorption
│       │   ├── lorentz_polarization.py   # LP factor
│       │   ├── zero_shift.py             # Global 2θ zero-shift (sample-displacement instrument miscalibration)
│       │   └── background.py             # residual background (bgsub domain) + optional physical Chebyshev
│       │
│       ├── augmentations/
│       │   ├── profile_augmentations.py  # Full-pattern augmentations
│       │   ├── peak_augmentations.py     # Peak-list augmentation: human-picking model (selection jitter + off-center, guarded low-d dropping, spurious peaks)
│       │   └── presets.py                # Named augmentation presets
│       │
│       ├── simulation/
│       │   ├── simulator.py              # Main orchestrator
│       │   └── batch.py                  # High-throughput batch simulation
│       │
│       ├── io/
│       │   ├── pattern_io.py             # Read/write patterns (xy, csv, xrdml, raw)
│       │   ├── bragg_store.py            # Fork-safe {d,|F|²} peak-list store keyed by cif_id (precompute cache; §1)
│       │   └── experimental.py           # Loaders for experimental patterns (delegates to crystalai-data)
│       │
│       ├── comparison/
│       │   ├── metrics.py                # Rwp (+noise-floor, GoF), Rietveld-partition R_Bragg, Rp, cosine, peak-position RMSD
│       │   ├── alignment.py              # 2θ alignment / zero-shift correction
│       │   ├── compare.py                # sim-vs-experimental: to-raw (2θ, fitted bg — criterion #6) + to-bgsub (log-d)
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
    ├── precompute_bragg.py               # CLI: build the {d,|F|²} bragg_peaks cache from crystals.sqlite (§1, Phase 5.1)
    ├── simulate_batch.py                 # CLI: batch simulate from a directory of CIFs
    ├── validate_criterion6.py            # CLI: noise-aware sim-vs-experimental gate (GoF<4 over exp_subset)
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

*Motivation: get the skeleton, reference data, and validation fixtures in place so the core can be built and checked against pymatgen from step one.*

- Package skeleton, `pyproject.toml`, all subpackage `__init__.py` files.
- Reference data: Cromer-Mann scattering factors (JSON), standard wavelengths (YAML).
- Test infrastructure: `pytest` config, example CIF fixtures (NaCl, Si, CeO₂, LaB6, α-quartz, rutile + one disordered structure).
- `wavelengths.py`: dictionary of standard X-ray wavelengths (Cu Kα1/2/avg, Mo Kα1, Co Kα1, Cr Kα1, Ag Kα1, plus a synchrotron wavelength range generator).

### Phase 1: Core simulation engine (2θ + log-d)

*Motivation: the load-bearing path — Structure/peak-list → Bragg → per-peak 2θ profile → log-d — that every effect and augmentation builds on. Correctness here is validated against pymatgen before anything is stacked on top.*

| Step | Task |
|------|------|
| 1.1 | `domain.py` — `Domain` enum (`TWO_THETA`, `D_SPACING`, `LOG_D`); coordinate transforms; Jacobians; `DomainGrid` dataclass with domain tag + optional wavelength |
| 1.2 | `crystal.py` — pymatgen `Structure` wrapper; lattice parameters, space group, atomic positions |
| 1.3 | `scattering.py` — Cromer-Mann 9-parameter atomic scattering factors `f(sinθ/λ)` |
| 1.4 | `bragg.py` — given a `Structure` + wavelength + d-window: compute all allowed Bragg reflections with `(hkl, 2θ, d, multiplicity, |F(hkl)|², integrated intensity)`. **Disorder handling**: occupancy-weighted structure factors for sites with fractional occupancies — never supercell-order disordered structures before computing Bragg peaks. Validate against pymatgen's `XRDCalculator`. |
| 1.5 | `lorentz_polarization.py` — LP factor `(1 + cos²(2θ)) / (sin²(θ)·cos(θ))` for Bragg-Brentano geometry; baked into integrated intensity in 2θ-domain before any conversion |
| 1.6 | `peak_shapes.py` — Gaussian, Lorentzian, Voigt (`scipy.special.wofz`), **Thompson-Cox-Hastings pseudo-Voigt (production core)**, split-PV. Operate on a generic x-axis (built in 2θ). |
| 1.7 | `caglioti.py` — Caglioti relation: `FWHM_G² = U·tan²(θ) + V·tan(θ) + W`; TCH mixing η(2θ); defaults for STADI-P, generic Bragg-Brentano |
| 1.8 | `binning.py` — domain-aware grid construction. `log_d_grid(d_min, d_max, n_bins)` is the production grid; `two_theta_grid` for the local per-peak construction grid and 2θ validation. |
| 1.9 | `convolver.py` — **production path: per-peak 2θ construction → resample to log-d.** For each Bragg peak: map `d→2θ(λ)`, compute Caglioti FWHM(2θ), build the TCH-PV core on a local 2θ grid, convolve the instrument kernels (axial divergence, slit — Phase 2), then resample the local profile onto the global log-d grid with the Jacobian intensity correction and accumulate. **Fast preset**: single symmetric pseudo-Voigt convolved directly in log-d (skips instrument kernels; previews/ablations only). |
| 1.10 | `simulator.py` (v1) — orchestrator: `Structure | cached peak list → Bragg peaks → per-peak 2θ profile → log-d resample`. Accepts `domain=Domain.LOG_D` (production default) or other domains. Minimal version with LP + TCH-PV convolution only (instrument kernels arrive in Phase 2). |

**Checkpoint.** Simulator produces basic patterns in 2θ and log-d. Peak positions match pymatgen to within 0.001° (2θ) or 0.001 Å (d). Round-trip: 2θ construction → log-d resample → back to 2θ preserves integrated intensities within 1%.

### Phase 2: Physical effects

*Motivation: make simulated patterns resemble real CW measurements — realistic peak shapes, instrument geometry, sample effects, and the bgsub-domain background — so the encoder trains on physically plausible variation. All applied in 2θ before the log-d conversion (`DESIGN_DECISIONS.md` §1a).*

| Step | Task |
|------|------|
| 2.1 | `thermal.py` — Debye-Waller factor `exp(-B·sin²(θ)/λ²)` applied to structure factors |
| 2.2 | `broadening.py` — size (Scherrer Lorentzian `β_L = Kλ/(D cosθ)`), strain (Gaussian `β_G = 4ε·tanθ`); fold into Caglioti parameters or add in quadrature |
| 2.3 | `preferred_orientation.py` — March-Dollase model |
| 2.4 | `axial_divergence.py` — axial-divergence asymmetry (affects low-2θ peaks), parameterized by **instrument geometry**: detector distance `L`, slit half-height `H`, sample half-height `S` (Van Laar & Yelon 1984; Finger-Cox-Jephcoat as an equivalent alt.). Convolved into the per-peak 2θ profile (§1.9). Randomizing over realistic `(L,H,S)` is the geometry-augmentation. |
| 2.5 | `slit.py` — rectangular slit (top-hat) instrumental-resolution kernel, convolved into the per-peak 2θ profile. Optional: Caglioti already carries instrumental Gaussian broadening, so enable the explicit slit only when not double-counting (reduce Caglioti `W` accordingly). |
| 2.6 | `zero_shift.py` — global 2θ zero-shift (sample-displacement miscalibration): one offset `δ(2θ)` applied to **all** peak centres before the log-d resample. A genuine instrument effect (distinct from the removed per-peak position perturbation, `DESIGN_DECISIONS.md` §7); in log-d it becomes a smooth angle-dependent distortion. |
| 2.7 | `absorption.py` — Brindley microabsorption; flat-plate absorption |
| 2.8 | `background.py` — background models, **applied in 2θ before the log-d conversion** (`DESIGN_DECISIONS.md` §1a). (a) Full physical background (Chebyshev order 5–10; Compton + air scatter + dark current) — used only when emulating raw data or for the arPLS-emulation residual mode. (b) **Residual background** (the training default): a low-order, low-amplitude smooth baseline that may go slightly negative, since we operate in the background-subtracted domain. The counting-noise floor depends on the *pre-subtraction* level, so background level and noise are sampled jointly (§5 order). |
| 2.9 | `domain_convert.py` — full-profile domain conversion between 2θ and log-d (resampling + Jacobian correction). Validation / fast-preset tool; the production path resamples per-peak (§1.9). |
| 2.10 | `simulator.py` (v2) — integrate all effects with toggleable flags; YAML / dataclass configuration |

**Checkpoint.** Patterns include realistic peak shapes, thermal effects, size/strain broadening, preferred orientation, geometry-based axial-divergence asymmetry, slit response, zero-shift, and residual backgrounds — delivered in log-d via the per-peak 2θ→log-d resample. Round-trip 2θ↔log-d preserves integrated intensities within 2%. Visual comparison against a handful of experimental patterns shows qualitatively similar profile shapes.

### Phase 3: Augmentation pipeline

*Motivation: turn the effect set into on-the-fly, config-bounded training-time variation that spans the experimental envelope (§5), so the encoder generalizes from simulated to real patterns.*

| Step | Task |
|------|------|
| 3.1 | `profile_augmentations.py` — domain-aware composable transforms. See §5 below. |
| 3.2 | `peak_augmentations.py` — human-picking model for the manually-supplied peak list: selection jitter + off-center bias, guarded low-d (high-2θ) dropping, spurious peaks. See §6 below. |
| 3.3 | `presets.py` — named presets: `DIFCON_STYLE` (matching the prior paper's setup), `PRODUCTION` (wavelength + Caglioti + mild asymmetry + matched d-range), `AGGRESSIVE`, `MILD` |

**Checkpoint.** Augmentation pipeline produces visually realistic variations spanning the experimental variability envelope. Each preset is callable, composable, and respects the pattern's domain tag.

### Phase 4: Interactive dashboard

*Motivation: visual validation and consortium demos — see each effect and augmentation act on a live pattern, and overlay sim vs experimental.*

`app/dashboard.py` — Gradio app with tabs:

- **Single Pattern Simulator.** Upload CIF or pick from examples. Sliders for: domain toggle (2θ / log-d), wavelength (greyed in log-d), 2θ / d range, step size, crystallite size, microstrain, temperature, preferred orientation, Caglioti U/V/W, axial-divergence geometry (L/H/S), zero-shift, background type + level. Live plot updates.
- **Effect Decomposition.** Same pattern with effects toggled on/off: stick → + LP → + thermal → + size → + strain → + axial-divergence → + zero-shift → + background → + noise.
- **Augmentation Preview.** Select a preset, apply N random augmentations to the same base pattern, overlay all.
- **Sim vs Experimental.** Upload or select an experimental pattern + corresponding CIF. Overlay simulated vs experimental, Rwp, difference curve. Adjust sim parameters to minimize residual.

### Phase 5: Batch simulation & PyTorch integration

*Motivation: produce the cached `bragg_peaks` artifact (§1 / `DESIGN_DECISIONS.md` §9) and the `Simulator` + augmentor API that `crystalai-methods` drives in its DataLoader.*

| Step | Task |
|------|------|
| 5.1 | `batch.py` + `scripts/precompute_bragg.py` — the **Bragg-peak precompute** (§1; rationale `DESIGN_DECISIONS.md` §9). Given a `Structure` iterator from `crystalai_data.CrystalDatabase`, compute the wavelength-independent d-space reflection list `{d, |F|²}` (occupancy-weighted, no DW/LP/λ) down to `d_min=0.7 Å`, truncated to strong peaks (+ optional dominant `hkl` for PO), and write the fork-safe `bragg_peaks` blob store keyed by `cif_id`. Multiprocessing; ~1–2 h over 257k structures. This — not full patterns — is the cached training artifact; convolution/effects/augmentation run live in the DataLoader. |
| 5.2 | Public API for `CrystalAI-methods`: `from crystalai_simxrd import Simulator, ProfileAugmentor, PeakAugmentor, Domain`. `Simulator` returns a `SimulatedPattern` dataclass with `(x_axis, intensity, peak_positions_d, peak_intensities, domain, wavelength, noise_floor, metadata)`, where **`noise_floor = (λmax, σrel)`** — the counting/baseline-noise conditioning params actually applied (§5). The encoder is conditioned on `wavelength` *and* `noise_floor` (`DESIGN_DECISIONS.md` §4, §4a); the experimental loader computes the same `(λmax, σrel)` from a pattern's background regions so sim and real share the feature. Augmentors are `torch.nn.Module`-compatible transforms that respect the pattern's domain tag. |
| 5.3 | `normalization.py` — pattern normalization (max, area, sqrt for Poisson-like data) |
| 5.4 | `peak_detection.py` — simple `scipy.find_peaks` wrapper for validation. Not used in production. |

**Checkpoint.** `CrystalAI-methods` can import this package and use `Simulator` + augmentors directly in its DataLoader.

---

## 5. Profile augmentations

Composable, domain-aware transforms applied on-the-fly (never pre-computed into static datasets). **Parameterization:** each augmentation is `f(pattern; θ)` with `θ ~ U[lo, hi]` from the config, kept smooth in `θ` where natural (differentiable-augmentation-ready); a noise term's *level* is such a `θ` while its per-sample draw is stochastic. This section covers the **full-profile** view: its peak positions are the physical measurement — never per-peak perturbed — and the only position effect is the *global* zero-shift (whole-pattern instrument offset, preserving relative spacings). The manually-picked **peak-list** view is augmented separately for human error (§6). The `PRODUCTION` preset combines all of the below.

| Augmentation | Param θ | Range / default | Notes |
|---|---|---|---|
| **Required — wavelength-conditioning** ||||
| Wavelength | λ | U[0.5, 1.8] Å | Cu/Mo/Co/Cr/Ag Kα + synchrotron; the FiLM conditioning input (`DESIGN_DECISIONS.md` §1/§4) |
| Caglioti | U, V, W | realistic instrument spread | Gaussian FWHM(2θ) = `U·tan²θ + V·tanθ + W` |
| Matched d-range | — | window **[0.7, 18] Å**, 12000 bins (DD §1b) | pick window first; per-λ 2θ range so all patterns share one log-d window |
| **Physical effects** (Pysimxrd set; sample the effect's parameter) ||||
| Crystallite size | D | realistic range | Scherrer Lorentzian |
| Microstrain | ε | — | Williamson-Hall Gaussian |
| Debye-Waller | B | — | `exp(−B/4d²)` envelope on `|F|²` |
| Preferred orientation | r (+ axis) | ~1 | March-Dollase (needs a dominant `hkl`; §9/DD) |
| Axial divergence | L, H, S | realistic geometries | Van Laar-Yelon; low-2θ asymmetry — main gain from Pysimxrd |
| Slit (optional) | width | — | top-hat; enable only if not double-counting Caglioti |
| Global zero-shift | δ(2θ) | U(−δ₀, δ₀) | whole-pattern offset (sample-displacement); *not* per-peak (§6) |
| **Stochastic / additive** ||||
| Poisson noise | λmax | U[1, 100] | counting noise, in 2θ; formula in `DESIGN_DECISIONS.md` §4a; primary noise |
| Gaussian noise | σrel (`gaussian_noise_std`; mean=0 here) | U[1e-3, 1e-1] | full-profile, **in 2θ** before the log-d conversion — normalize→add `N(0,σrel)`→re-normalize; §5a |
| Residual background | amplitude | small, may go negative | bgsub-domain residual; optional arPLS-emulation (`PRODUCTION`) reuses `crystalai_data`'s operator |
| Impurity/spurious peaks | count | 0–N | additive; never touches/removes a true peak |
| Edge crop / pad | edge | — | hard crop for masked-window training (methods) |

`(λmax, σrel)` are emitted as conditioning metadata (§5.2) so the encoder is told its noise floor (`DESIGN_DECISIONS.md` §4a). **Explicitly dropped:** intensity-envelope perturbation (destroys real intensity information the generator needs, once wavelength is an explicit input).

### Order (single 2θ → log-d conversion)

**All CW-instrument physics *and* all noise is applied in 2θ; convert to log-d once; only normalization follows in log-d** (`DESIGN_DECISIONS.md` §1a). There is exactly one Gaussian-noise mechanism (§5a) — no separate log-d-native variant.

1. **Intensities** — DW envelope, LP, preferred orientation on `|F|²`; add impurity/spurious peaks.
2. **2θ assembly** — `d→2θ(λ)` + global zero-shift; per-peak TCH-PV (Caglioti FWHM) ⊗ axial-divergence ⊗ slit → full 2θ pattern.
3. **2θ instrument effects** — add residual background (or physical-bg-then-arPLS), then **Poisson counting noise** (on total counts), then **full-profile Gaussian noise** (normalize → add `N(mean,σrel)` → re-normalize, all in 2θ; §5a).
4. **Convert to log-d** — single resample (Jacobian-corrected).
5. **Normalize** — max, or `√` (variance-stabilizes the Poisson noise); edge crop/pad → final rescale.

Steps 1–4 are the 2θ-build→log-d flow; step 3 matches AlphaDiffract's background → Poisson → Gaussian, just kept entirely in 2θ rather than splitting the Gaussian term into log-d. The global zero-shift (step 2) is applied in 2θ before the resample, so in log-d it becomes an angle-dependent distortion, not a rigid translation.

### 5a. Full-profile Gaussian noise mechanism

`EffectConfig.gaussian_noise_mean` / `gaussian_noise_std` is `simulate()`'s Gaussian-noise effect (Phase 2 — used by the dashboard's Single Pattern tab, any direct `EffectConfig` caller, *and* `ProfileAugmentor`'s training pipeline, which samples `gaussian_noise_std` from `AugmentConfig.sigma_rel` with `mean=0` and emits it as the `σrel` half of the `(λmax, σrel)` conditioning pair, `DESIGN_DECISIONS.md` §4a). There is only this one mechanism — no separate log-d-native Gaussian noise exists anywhere in the simulation path. Its purpose: let a caller preview (or train against) what a *manually background-subtracted* pattern looks like — after an analyst subtracts background there is a low-amplitude ripple left behind (noise, not Bragg peaks), and the pattern is then renormalized to [0, 1] before further use.

**Mechanism**, applied to the assembled 2θ pattern (`pattern_tt`) immediately after Poisson counting noise, regardless of the requested output domain:

1. Max-normalize `pattern_tt` to [0, 1] (`utils/normalization.normalize(..., method="max")`).
2. Add `N(mean, std)` (`effects/noise.relative_gaussian`).
3. Max-normalize again to [0, 1].

**Always in 2θ, even for d/log-d output.** This mirrors how every other 2θ-native effect (background, Poisson) is applied before the single domain conversion (§1a): for `domain=LOG_D`, the *already noisy, already renormalized* 2θ pattern is what gets Jacobian-resampled to log-d — the noise is never added natively in log-d. Because the log-d resample is intensity-conserving (not renormalizing), the returned log-d pattern is **not** itself re-bounded to [0, 1] after conversion — same as every other effect on this path (`simulate()` never globally normalizes its output; that stays a caller concern, e.g. `ProfileAugmentor`'s final `normalize()` call or the dashboard's plot normalization). Only the 2θ-domain output is directly [0, 1]-bounded by construction.

---

## 6. Peak-list augmentation: model the human-picked list

The peak-list channel's inference input is a **manually picked** list, so its augmentations model *how a human produces it* — not arbitrary corruption. (The full-profile channel is the raw measurement and keeps exact, complete positions — §5; only the peak-list view gets the human-error model here.) Rationale and the reconciliation with the lattice/SG physics: `DESIGN_DECISIONS.md` §7.

1. **Position error (selection jitter + off-center bias).** A human clicks a point that is slightly off the true profile centroid — sometimes on the profile shoulder rather than the maximum. Model as a small per-peak Gaussian jitter plus an optional small off-center bias, **sub-FWHM**, scaled in log-d. This is why the peak-list view is *not* fed idealized-clean positions: at inference it never is.
2. **Selective low-d (high-2θ) dropping.** Humans pick the prominent low-2θ (high-d) peaks and skip much of the dense high-2θ (low-d) forest. So when a pattern is **peak-dense**, drop preferentially from the **low-d / high-2θ end**, keeping the high-d peaks. **Guarded by a manufactured-absence check**: never drop a reflection whose removal would fabricate a diagnostic systematic absence — i.e. drop only while the retained high-d peaks still determine the space group. (This implements "drop low-d peaks if the high-d peaks are enough to predict the SG.")
3. **Spurious peaks.** Insert 0–3 additive impurity/contaminant peaks (random position, small intensity).

**Tradeoff (a deliberate reversal of the earlier 'protect high-2θ' rule).** High-2θ/low-d peaks pin the lattice most tightly, so preferentially dropping them costs lattice precision in `z_lattice_peaks`. We accept it because it matches the *real* manual input; the profile view (complete, exact positions) carries the lattice-precision load in the VICReg-aligned pair, and the drop guard protects SG determination.

---

## 7. Validation criteria

Before this package is considered ready for `CrystalAI-methods`:

1. **Peak position accuracy.** Simulated peak positions match pymatgen's `XRDCalculator` to within 0.005° (2θ) or 0.0005 Å (d) for all test CIFs.
2. **Relative intensity agreement.** Pearson correlation > 0.95 between our integrated intensities and pymatgen's for standard references (LaB6, CeO₂, Si).
3. **Disorder fidelity.** For disordered test CIFs, the full pipeline (serialize to SQLite, deserialize, run Bragg calculator) produces integrated intensities matching pymatgen's `XRDCalculator` to within 1% relative for all peaks above 1% intensity. Species occupancies and fractional coordinates round-trip at full numerical precision.
4. **Domain round-trip fidelity.** Simulate in 2θ → convert to log-d → convert back to 2θ; integrated intensity per peak preserved within 2%. Peak positions preserved within 0.01° after round-trip.
5. **Production resample fidelity + fast-preset error.** The production path (per-peak 2θ construction → log-d resample) preserves integrated intensity per peak within 2% and peak positions within 0.001 Å vs. the same profile evaluated on a dense 2θ grid. Separately, the optional fast native-log-d single-kernel preset is expected to *disagree* with production only in the asymmetric instrumental tails (axial divergence) — that gap is measured and documented as the preset's known approximation error, not required to be small.
6. **Experimental realism (noise-aware).** For ≥ 5 crystals with both a CIF and an experimental pattern in `crystalai-data`, the forward-simulated pattern — compared to the **raw** counts in **2θ** with a jointly-fitted Chebyshev background + scale (`compare_structure_to_raw`) and effects refined — achieves **goodness-of-fit `GoF = Rwp / Rwp_noise_floor < 4`** (i.e. the profile fit is within ~4× the counting-noise floor). The **Rietveld-partition `R_Bragg`** is reported as a structural bug-detector (a value ≫ typical, or a gross outlier, flags a real `|F|²`/position error), **not** gated at a refinement-grade threshold: a *forward* simulator of a **fixed** ICSD structure floors at `R_Bragg ≈ 20–60%` because it does not refine atoms/thermals/occupancy or model full texture (single-axis March-Dollase only) or absorption — `R_Bragg ≤ 5%` is a post-*refinement* number, out of scope here. Raw Rwp and cosine are also reported. Rationale and the metric derivations (`Rwp = √(1−cos²_w)`, the log-d-vs-2θ background Jacobian pitfall, the noise floor) live in `DESIGN_DECISIONS.md` §10. Reproduce with `scripts/validate_criterion6.py`.
7. **Augmentation coverage.** Augmented simulated patterns visually span the variability range of experimental patterns (qualitative, assessed via t-SNE of augmented sim vs experimental distributions in feature space).
8. **Performance.** Batch simulation of 1000 CIFs under one condition completes in < 1 minute on a single CPU core.
9. **Precompute/on-the-fly split is lossless (§1 boundary).** A pattern built on-the-fly from the cached `{d, |F|²}` peak list matches a full from-scratch simulation (identical conditions, no cache) to within peak positions ≤ 0.001 Å (d) and integrated intensities ≤ 1% for all test CIFs.
10. **DataLoader throughput.** From the precomputed `bragg_peaks` store, the live path (crop → DW → LP → Caglioti → convolve → augment) sustains the per-worker rate needed to feed 8 GPUs, with latency **flat across structure size** (no heavy-tail stalls) — the property the precompute exists to guarantee.

---

## 8. Outputs consumed by other repos

| Consumer | What it gets |
|----------|--------------|
| `CrystalAI-methods` | `Simulator` (CIF → `SimulatedPattern`), `ProfileAugmentor` (composable augmentation pipeline), `PeakAugmentor` (additive spurious-peak augmentation; positions never perturbed), `Domain` enum, `SimulatedPattern` dataclass |
| Standalone usage | `scripts/simulate_batch.py`, `scripts/compare_with_experiment.py`, `app/dashboard.py` |
