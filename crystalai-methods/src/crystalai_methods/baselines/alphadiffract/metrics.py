"""Scoring for the OpenAlphaDiffract baseline against our experimental labels.

Label alignment:
  * Crystal system — the model head is 0-indexed [Triclinic..Cubic]; ``infer.py``
    already shifts predictions to OUR 1-indexed coding (1=triclinic..7=cubic),
    which is exactly what ``crystalai_data.crystals.symmetry.sg_to_cs`` produces.
  * Space group — compared directly as integers 1..230.

The SG->CS consistency cross-check reuses the canonical ``sg_to_cs`` table (no
spglib, no re-derivation): it maps the model's *predicted SG* to a crystal
system and compares it to the model's *CS head* — a signal independent of the
head's label ordering.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from crystalai_data.crystals.symmetry import cs_number_to_name, sg_to_cs

LP_NAMES = ["a", "b", "c", "alpha", "beta", "gamma"]
LP_UNITS = ["Å", "Å", "Å", "°", "°", "°"]


def compute_metrics(df: pd.DataFrame) -> dict:
    """Metrics for one set of predictions+labels.

    Expects columns: ``cs_true``, ``cs_pred``, ``sg_true``, ``sg_pred``,
    ``sg_top5`` (list/array of 5 ints), ``lp_pred_*`` and ``lp_true_*`` for each
    of the 6 lattice params, and ``coverage``.
    """
    n = len(df)
    if n == 0:
        return {"n": 0}

    cs_true = df["cs_true"].to_numpy()
    cs_pred = df["cs_pred"].to_numpy()
    sg_true = df["sg_true"].to_numpy()
    sg_pred = df["sg_pred"].to_numpy()

    cs_ok = cs_true == cs_pred
    sg_top1 = sg_true == sg_pred
    sg_top5 = np.array(
        [t in set(row) for t, row in zip(sg_true, df["sg_top5"])]
    )

    # SG->CS internal consistency: does the CS head agree with the system
    # implied by the predicted SG? (Independent of label ordering.)
    cs_from_sg = np.array([sg_to_cs(int(s)) for s in sg_pred])
    cs_head_vs_sg = cs_from_sg == cs_pred

    lp_mae = {}
    for name in LP_NAMES:
        t = df[f"lp_true_{name}"].to_numpy(dtype=float)
        p = df[f"lp_pred_{name}"].to_numpy(dtype=float)
        mask = ~np.isnan(t)
        lp_mae[name] = float(np.abs(p[mask] - t[mask]).mean()) if mask.any() else float("nan")

    return {
        "n": n,
        "cs_acc": float(cs_ok.mean()),
        "sg_top1": float(sg_top1.mean()),
        "sg_top5": float(sg_top5.mean()),
        "cs_head_vs_sg": float(cs_head_vs_sg.mean()),
        "lp_mae": lp_mae,
        "coverage_mean": float(df["coverage"].mean()),
        "coverage_ge_095": float((df["coverage"] >= 0.95).mean()),
    }


def _fmt_block(title: str, m: dict) -> list[str]:
    if m.get("n", 0) == 0:
        return [f"{title}: (no patterns)"]
    lp = "  ".join(
        f"{k}={m['lp_mae'][k]:.3f}{u}" for k, u in zip(LP_NAMES, LP_UNITS)
    )
    return [
        f"{title}  (n={m['n']})",
        f"  crystal system  top-1 : {m['cs_acc']:.3f}",
        f"  space group     top-1 : {m['sg_top1']:.3f}   top-5 : {m['sg_top5']:.3f}",
        f"  CS-head vs SG-implied  : {m['cs_head_vs_sg']:.3f}  (internal consistency)",
        f"  lattice MAE           : {lp}",
        f"  d-window coverage     : mean {m['coverage_mean']:.3f}   "
        f"(≥0.95: {m['coverage_ge_095']:.1%})",
    ]


def format_report(df: pd.DataFrame) -> str:
    """Full text report: per bg_set, with source and crystal-system breakdowns."""
    lines: list[str] = []
    lines.append("=" * 74)
    lines.append("OpenAlphaDiffract baseline on CrystalAI experimental patterns")
    lines.append("=" * 74)

    for bg_set in ["manual", "auto"]:
        sub = df[df["bg_set"] == bg_set]
        label = {
            "manual": "MANUAL / native background (autobg=0)",
            "auto": "AUTO background (autobg=1, pybaselines)",
        }[bg_set]
        lines.append("")
        lines.extend(_fmt_block(f"[{label}]", compute_metrics(sub)))

        # By source within the set.
        for src in sorted(sub["source"].unique()):
            lines.append("")
            lines.extend(
                _fmt_block(f"    · source={src}", compute_metrics(sub[sub["source"] == src]))
            )

        # High-coverage-only slice (fair comparison, model's window fully seen).
        hc = sub[sub["coverage"] >= 0.95]
        lines.append("")
        lines.extend(_fmt_block("    · coverage≥0.95 only", compute_metrics(hc)))

    # Crystal-system breakdown of CS accuracy (combined, high-coverage).
    lines.append("")
    lines.append("-" * 74)
    lines.append("CS top-1 accuracy by true crystal system (coverage≥0.95, both sets):")
    hc = df[df["coverage"] >= 0.95]
    for cs in range(1, 8):
        s = hc[hc["cs_true"] == cs]
        if len(s):
            acc = float((s["cs_true"] == s["cs_pred"]).mean())
            lines.append(f"    {cs_number_to_name(cs):>13s} (n={len(s):4d}) : {acc:.3f}")

    lines.append("")
    lines.append("-" * 74)
    lines.append("Caveats: single model (not the paper's 10-ensemble); trained on MP")
    lines.append("DFT-simulated 20 keV patterns, evaluated on REAL data (domain gap);")
    lines.append("model sees only d∈[1.785,7.106] Å; opXRD SGs are spglib-derived,")
    lines.append("RRUFF has no atomic CIF. This is an external baseline, not a gate.")
    return "\n".join(lines)
