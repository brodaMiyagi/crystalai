"""Load the vendored OpenAlphaDiffract model and decode its three heads.

PyTorch is confined to this module (inference boundary); the remap math is
NumPy-only in ``remap.py``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .vendor.model import AlphaDiffract

# models/alphadiffract/ relative to the crystalai-methods package root.
DEFAULT_MODEL_DIR = Path(__file__).resolve().parents[4] / "models" / "alphadiffract"


def load_model(model_dir: str | Path | None = None, device: str = "cpu") -> AlphaDiffract:
    model_dir = Path(model_dir) if model_dir is not None else DEFAULT_MODEL_DIR
    if not (model_dir / "model.safetensors").exists():
        raise FileNotFoundError(
            f"model.safetensors not found in {model_dir}. "
            "Run: python scripts/fetch_alphadiffract.py"
        )
    return AlphaDiffract.from_pretrained(str(model_dir), device=device)


@torch.no_grad()
def predict_batch(
    model: AlphaDiffract, x: np.ndarray, device: str = "cpu"
) -> dict[str, np.ndarray]:
    """Run a batch of (B, 8192) inputs through the model and decode predictions.

    Returns numpy arrays:
      cs_pred   (B,)   crystal system in OUR 1-indexed coding (1=triclinic..7=cubic)
      cs_conf   (B,)   softmax confidence of the CS prediction
      sg_pred   (B,)   space group number 1..230 (argmax + 1)
      sg_conf   (B,)   softmax confidence of the SG prediction
      sg_top5   (B,5)  top-5 space group numbers 1..230, descending confidence
      lp        (B,6)  lattice params a,b,c (Å), alpha,beta,gamma (deg)
    """
    xt = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)).to(device)
    out = model(xt)
    cs_probs = torch.softmax(out["cs_logits"], dim=-1)
    sg_probs = torch.softmax(out["sg_logits"], dim=-1)

    cs_idx = cs_probs.argmax(dim=-1)
    sg_idx = sg_probs.argmax(dim=-1)
    sg_top5 = sg_probs.topk(5, dim=-1).indices  # 0-based

    return {
        # Model CS index 0..6 (Triclinic..Cubic) -> our 1-indexed code (+1).
        "cs_pred": (cs_idx + 1).cpu().numpy(),
        "cs_conf": cs_probs.gather(-1, cs_idx[:, None]).squeeze(-1).cpu().numpy(),
        "sg_pred": (sg_idx + 1).cpu().numpy(),
        "sg_conf": sg_probs.gather(-1, sg_idx[:, None]).squeeze(-1).cpu().numpy(),
        "sg_top5": (sg_top5 + 1).cpu().numpy(),
        "lp": out["lp"].cpu().numpy(),
    }
