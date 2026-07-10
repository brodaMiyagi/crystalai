"""CLI: build the bragg_peaks store from crystals.sqlite (Phase 5.1).

    uv run python scripts/precompute_bragg.py --limit 500        # sample
    uv run python scripts/precompute_bragg.py --workers 32       # full ICSD
"""
from __future__ import annotations

import argparse

from crystalai_data.crystals._ingest import load_dotenv, repo_root
from crystalai_simxrd.simulation.batch import precompute_bragg_store


def main() -> None:
    root = repo_root()
    env = load_dotenv(root / ".env")
    crystals_default = env.get("CRYSTALS_DB", str(root / "crystalai-data" / "crystals.sqlite"))
    store_default = env.get("BRAGG_STORE", str(root / "crystalai-data" / "bragg_peaks.sqlite"))

    p = argparse.ArgumentParser(description="Precompute the Bragg peak-list store")
    p.add_argument("--crystals-db", default=crystals_default)
    p.add_argument("--out", default=store_default)
    p.add_argument("--source", default="icsd", help="crystals source filter ('' for all)")
    p.add_argument("--d-min", type=float, default=0.7)
    p.add_argument("--max-n-sites", type=int, default=None, help="skip giant-cell tail")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--workers", type=int, default=1)
    args = p.parse_args()

    precompute_bragg_store(
        args.crystals_db, args.out, d_min=args.d_min, source=(args.source or None),
        limit=args.limit, max_n_sites=args.max_n_sites, workers=args.workers,
    )


if __name__ == "__main__":
    main()
