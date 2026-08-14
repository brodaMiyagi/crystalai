"""External-baseline eval of PXRDGen (Li et al., Nat. Commun. 2025) on our
experimental store.

PXRDGen (Code Ocean capsule 7727770) is a flow/diffusion crystal-structure
generator conditioned on a PXRD pattern + chemical formula. This subpackage is a
reference harness only — like ``baselines/alphadiffract`` — it never feeds any
PXRDGen checkpoint into our own model (see ``crystalai-methods/CLAUDE.md``).
"""
