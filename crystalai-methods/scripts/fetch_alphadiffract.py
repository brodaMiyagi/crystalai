"""Fetch the OpenAlphaDiffract weights + config into ``models/alphadiffract/``.

External baseline (arXiv 2603.23367). Weights are NOT redistributed with this
repo; this script downloads them from the Hugging Face Hub into a gitignored
directory so ``AlphaDiffract.from_pretrained`` can load them.

    python scripts/fetch_alphadiffract.py

The vendored ``model.py`` (baselines/alphadiffract/vendor/model.py) is committed
verbatim; only the binary weights + config live under ``models/``.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO_ID = "linked-liszt/OpenAlphaDiffract"
# Files needed by AlphaDiffract.from_pretrained (weights + arch config + SG graph).
FILES = ["model.safetensors", "config.json", "maxsub.json"]

# models/alphadiffract/ relative to the crystalai-methods package root.
DEST = Path(__file__).resolve().parent.parent / "models" / "alphadiffract"


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        cached = hf_hub_download(repo_id=REPO_ID, filename=name)
        target = DEST / name
        shutil.copyfile(cached, target)
        size_mb = target.stat().st_size / 1e6
        print(f"  {name:<20s} -> {target}  ({size_mb:.1f} MB)")
    print(f"\nDone. Model dir: {DEST}")


if __name__ == "__main__":
    main()
