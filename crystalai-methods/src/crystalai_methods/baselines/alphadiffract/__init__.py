"""OpenAlphaDiffract external baseline (arXiv 2603.23367) evaluation harness."""

from .remap import remap_pattern, model_d_grid, LAMBDA_MODEL
from .infer import load_model, predict_batch

__all__ = ["remap_pattern", "model_d_grid", "LAMBDA_MODEL", "load_model", "predict_batch"]
