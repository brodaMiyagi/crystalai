"""Wavelength audit over the experimental store (DATA_ROADMAP.md §2).

Accurate wavelength metadata is required by the wavelength-conditioning strategy
(DESIGN_DECISIONS.md §1/§4): the network takes wavelength as an explicit FiLM
input, so a pattern with a missing or wrong wavelength would silently distort the
sim-to-real mapping. This pass writes a per-pattern ``audit`` verdict to the index
so consumers can filter on one column instead of re-deriving the rule:

* ``pass`` — angle-dispersive with a plausible wavelength, **or** a TOF pattern
  (null wavelength is expected there, not an error; none in the pool yet).
* ``missing_wavelength`` — angle-dispersive (``two_theta`` / ``d``) with no
  wavelength. Disqualifying for wavelength-conditioned training, but the pattern
  is **kept** in the store: it stays separable by this flag and reusable for the
  wavelength-sweep experiments (an angle-dispersive profile can be re-interpreted
  against assumed Cu/Mo/Co/… Kα lines to probe wavelength robustness).
* ``suspicious_wavelength`` — a wavelength outside the plausible lab/synchrotron
  window; flagged for manual review rather than trusted.

The verdict is a separate pass (not written at ingest) so the plausible window can
be retuned and re-run without re-ingesting. Run it after the ingesters. Nothing is
deleted — the flag only partitions the store.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ..crystals._ingest import load_dotenv, repo_root
from . import database as db

__all__ = ["audit_status", "run", "WAVELENGTH_MIN_A", "WAVELENGTH_MAX_A"]

# Plausible angle-dispersive wavelength window (Å). The lower bound admits
# high-energy synchrotron radiation (opXRD includes synchrotron sources; ~0.14 Å
# ≈ 87 keV is normal there), the upper bound covers the longest common lab lines
# (V Kα ~2.50, Cr Kα ~2.29) with margin. Outside this is almost certainly a
# metadata error (or a rare long-λ source) and is flagged for manual review, not
# trusted. Widened from the roadmap's original [0.4, 2.5], which predated seeing
# opXRD's synchrotron patterns and mis-flagged ~42 legitimate short-λ ones.
WAVELENGTH_MIN_A = 0.1
WAVELENGTH_MAX_A = 2.6


def audit_status(x_coord: str, wavelength_A) -> str:
    """Verdict for one pattern from its coordinate + wavelength (see module doc)."""
    if x_coord == "tof":
        return "pass"  # TOF has no angle-dispersive wavelength; null is expected
    if wavelength_A is None or (isinstance(wavelength_A, float) and pd.isna(wavelength_A)):
        return "missing_wavelength"
    if not (WAVELENGTH_MIN_A <= float(wavelength_A) <= WAVELENGTH_MAX_A):
        return "suspicious_wavelength"
    return "pass"


def run(store: Path) -> None:
    """Compute the ``audit`` verdict for every pattern and rewrite ``index.csv``."""
    store = Path(store)
    idx = db.read_index(store)
    if idx.empty:
        raise SystemExit(f"[audit] empty/absent index at {store}")

    idx["audit"] = [
        audit_status(xc, wl)
        for xc, wl in zip(idx["x_coord"], idx["wavelength_A"], strict=True)
    ]
    db.write_index_frame(store, idx)

    counts = idx["audit"].value_counts().to_dict()
    n_pass = counts.get("pass", 0)
    print(f"[audit] store    = {store}")
    print(f"[audit] window   = [{WAVELENGTH_MIN_A}, {WAVELENGTH_MAX_A}] Å")
    print(f"[audit] patterns = {len(idx):,}")
    print(f"[audit] verdict  = {counts}")
    print(f"[audit] training-eligible (pass) = {n_pass:,}/{len(idx):,}")
    for status in ("missing_wavelength", "suspicious_wavelength"):
        sub = idx[idx["audit"] == status]
        if len(sub):
            print(f"[audit]   {status}: {len(sub):,}  (by source: {sub['source'].value_counts().to_dict()})")


def main(argv: list[str] | None = None) -> None:
    root = repo_root()
    env = load_dotenv(root / ".env")
    store_default = str(db.resolve_store(env, root))

    p = argparse.ArgumentParser(description="Wavelength audit over the experimental store")
    p.add_argument("--store", default=store_default, type=str)
    args = p.parse_args(argv)
    run(Path(args.store))


if __name__ == "__main__":
    main()
