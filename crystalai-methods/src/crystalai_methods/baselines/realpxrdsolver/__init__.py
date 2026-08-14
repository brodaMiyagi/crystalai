"""External-baseline eval of RealPXRDSolver (DP Technology, 2025) on our
experimental store.

RealPXRDSolver is a crystal-structure generator conditioned on a **sparse PXRD
peak list** (integer-2θ-degree positions -> ``nn.Embedding(180)`` + intensities,
via a peak-set BERT encoder) plus the **chemical formula**; a flow decoder
(DiffCSP-lineage) emits candidate structures. It has **no wavelength input**
(training patterns are Cu Kα simulated), so non-Cu patterns are Bragg-remapped to
the Cu Kα1 2θ scale — the same treatment as ``baselines/pxrdgen`` (we reuse
``pxrdgen.remap``).

This subpackage is a reference harness only — like ``baselines/alphadiffract``
and ``baselines/pxrdgen`` — it never feeds any checkpoint into our own model
(see ``crystalai-methods/CLAUDE.md``).
"""
