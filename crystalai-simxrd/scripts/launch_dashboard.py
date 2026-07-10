"""Launch the SimXRD dashboard (see app/dashboard.py).

    uv run python scripts/launch_dashboard.py [--share] [--port N]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from dashboard import build  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true")
    args = ap.parse_args()
    build().launch(server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
