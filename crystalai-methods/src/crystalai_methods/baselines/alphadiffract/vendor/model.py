"""
This file is self-contained: download it alongside `model.safetensors`,
`config.json`, and `maxsub.json` to load and run the model.
"""

import json
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Utility: DropPath (Stochastic Depth)
# ---------------------------------------------------------------------------
def drop_path(
    x: torch.Tensor, drop_prob: float = 0.0, training: bool = False
) -> torch.Tensor:
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor = random_tensor.floor()
    return x.div(keep_prob) * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path(x, self.drop_prob, self.training)


# ---------------------------------------------------------------------------
# ConvNeXt 1D Block
# ---------------------------------------------------------------------------
class ConvNeXtBlock1D(nn.Module):
    def __init__(
        self,
        dim: int,
        kernel_size: int,
        drop_path: float,
        layer_scale_init_value: float,
        activation: nn.Module,
    ):
        super().__init__()
        self.dwconv = nn.Conv1d(
            dim, dim, kernel_size=kernel_size, padding="same", groups=dim
        )
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = activation() if isinstance(activation, type) else activation
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones(dim))
            if layer_scale_init_value > 0
            else None
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 1)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = x * self.gamma
        x = x.permute(0, 2, 1)
        x = shortcut + self.drop_path(x)
        return x


class ConvNextBlock1DAdaptor(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        dropout: float,
        use_batchnorm: bool,
        activation: nn.Module,
        layer_scale_init_value: float,
        drop_path_rate: float,
        block_type: str,
    ):
        super().__init__()
        if in_channels != out_channels:
            act = activation() if isinstance(activation, type) else activation
            self.pwconv = nn.Sequential(nn.Linear(in_channels, out_channels), act)
        else:
            self.pwconv = None

        if block_type == "convnext":
            self.block = ConvNeXtBlock1D(
                dim=out_channels,
                kernel_size=kernel_size,
                drop_path=drop_path_rate,
                layer_scale_init_value=layer_scale_init_value,
                activation=activation,
            )
        else:
            self.block = None

        if stride > 1:
            self.reduction_pool = nn.AvgPool1d(kernel_size=stride, stride=stride)
        else:
            self.reduction_pool = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.pwconv is not None:
            x = x.permute(0, 2, 1)
            x = self.pwconv(x)
            x = x.permute(0, 2, 1)
        if self.block is not None:
            x = self.block(x)
        if self.reduction_pool is not None:
            x = self.reduction_pool(x)
        return x


# ---------------------------------------------------------------------------
# MLP head builder
# ---------------------------------------------------------------------------
def make_mlp(
    input_dim: int,
    hidden_dims: Optional[Tuple[int, ...]],
    output_dim: int,
    dropout: float = 0.2,
    output_activation: Optional[nn.Module] = None,
) -> nn.Module:
    layers: List[nn.Module] = []
    last = input_dim
    if hidden_dims is not None and len(hidden_dims) > 0:
        for hd in hidden_dims:
            layers.extend([nn.Linear(last, hd), nn.ReLU()])
            if dropout and dropout > 0:
                layers.append(nn.Dropout(dropout))
            last = hd
    layers.append(nn.Linear(last, output_dim))
    if output_activation is not None:
        layers.append(output_activation)
    return nn.Sequential(*layers)


# ---------------------------------------------------------------------------
# Backbone
# ---------------------------------------------------------------------------
class MultiscaleCNNBackbone1D(nn.Module):
    def __init__(
        self,
        dim_in: int,
        channels: Tuple[int, ...],
        kernel_sizes: Tuple[int, ...],
        strides: Tuple[int, ...],
        dropout_rate: float,
        ramped_dropout_rate: bool,
        block_type: str,
        pooling_type: str,
        final_pool: bool,
        use_batchnorm: bool,
        activation: nn.Module,
        output_type: str,
        layer_scale_init_value: float,
        drop_path_rate: float,
    ):
        super().__init__()
        assert len(channels) == len(kernel_sizes) == len(strides)
        self.dim_in = dim_in
        self.output_type = output_type

        if ramped_dropout_rate:
            dropout_per_stage = torch.linspace(
                0.0, dropout_rate, steps=len(channels)
            ).tolist()
        else:
            dropout_per_stage = [dropout_rate] * len(channels)

        if pooling_type == "average":
            pool_cls = nn.AvgPool1d
            pool_kwargs = {"kernel_size": 3, "stride": 2}
        elif pooling_type == "max":
            pool_cls = nn.MaxPool1d
            pool_kwargs = {"kernel_size": 2, "stride": 2}
        else:
            raise ValueError(f"Invalid pooling_type '{pooling_type}'")

        layers: List[nn.Module] = []
        in_ch = 1
        for i, (out_ch, k, s) in enumerate(zip(channels, kernel_sizes, strides)):
            stage_block = ConvNextBlock1DAdaptor(
                in_channels=in_ch,
                out_channels=out_ch,
                kernel_size=k,
                stride=s,
                dropout=dropout_per_stage[i],
                use_batchnorm=use_batchnorm,
                activation=activation,
                layer_scale_init_value=layer_scale_init_value,
                drop_path_rate=drop_path_rate,
                block_type=block_type,
            )
            layers.append(stage_block)
            if i < len(channels) - 1 or final_pool:
                layers.append(pool_cls(**pool_kwargs))
            in_ch = out_ch

        self.net = nn.Sequential(*layers)

        if self.output_type == "gap":
            self.dim_output = channels[-1]
        elif self.output_type == "flatten":
            with torch.no_grad():
                dummy = torch.zeros(1, 1, self.dim_in)
                out = self.net(dummy)
                self.dim_output = int(out.shape[1] * out.shape[2])
        else:
            raise ValueError(f"Invalid output_type '{self.output_type}'")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            x = x[:, None, :]
        x = self.net(x)
        if self.output_type == "gap":
            x = x.mean(dim=-1)
        else:
            x = x.reshape(x.shape[0], -1)
        return x


# ---------------------------------------------------------------------------
# GEMD distance-matrix utilities
# ---------------------------------------------------------------------------
def _build_distance_matrix_from_maxsub_lut(
    maxsub_lut: Dict[str, List[int]],
    num_sg_classes: int,
) -> torch.Tensor:
    adjacency: List[set] = [set() for _ in range(num_sg_classes)]
    for key, neighbors in maxsub_lut.items():
        src = int(key) - 1
        for raw_dst in neighbors:
            dst = int(raw_dst) - 1
            adjacency[src].add(dst)
            adjacency[dst].add(src)

    distance_matrix = torch.zeros(
        (num_sg_classes, num_sg_classes), dtype=torch.float32
    )
    for src in range(num_sg_classes):
        dists = [-1] * num_sg_classes
        dists[src] = 0
        queue = deque([src])
        while queue:
            cur = queue.popleft()
            for nxt in adjacency[cur]:
                if dists[nxt] == -1:
                    dists[nxt] = dists[cur] + 1
                    queue.append(nxt)
        distance_matrix[src] = torch.tensor(dists, dtype=torch.float32)
    return distance_matrix


def load_gemd_distance_matrix(
    path: str, num_sg_classes: int = 230
) -> torch.Tensor:
    with open(path, "r", encoding="utf-8") as f:
        payload: Any = json.load(f)
    if isinstance(payload, dict) and all(str(k).isdigit() for k in payload.keys()):
        return _build_distance_matrix_from_maxsub_lut(payload, num_sg_classes)
    elif isinstance(payload, list):
        return torch.as_tensor(payload, dtype=torch.float32)
    raise ValueError(f"Could not parse GEMD data from {path}")


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------
class AlphaDiffract(nn.Module):
    """
    AlphaDiffract: multi-task 1D ConvNeXt for powder X-ray diffraction
    pattern analysis.

    Predicts crystal system (7 classes), space group (230 classes), and
    lattice parameters (6 values: a, b, c, alpha, beta, gamma).
    """

    CRYSTAL_SYSTEMS = [
        "Triclinic",
        "Monoclinic",
        "Orthorhombic",
        "Tetragonal",
        "Trigonal",
        "Hexagonal",
        "Cubic",
    ]

    def __init__(self, config: dict, maxsub_path: Optional[str] = None):
        super().__init__()
        bb = config["backbone"]
        heads = config["heads"]
        tasks = config["tasks"]

        activation = nn.GELU

        self.backbone = MultiscaleCNNBackbone1D(
            dim_in=bb["dim_in"],
            channels=tuple(bb["channels"]),
            kernel_sizes=tuple(bb["kernel_sizes"]),
            strides=tuple(bb["strides"]),
            dropout_rate=bb["dropout_rate"],
            ramped_dropout_rate=bb["ramped_dropout_rate"],
            block_type=bb["block_type"],
            pooling_type=bb["pooling_type"],
            final_pool=bb["final_pool"],
            use_batchnorm=bb["use_batchnorm"],
            activation=activation,
            output_type=bb["output_type"],
            layer_scale_init_value=bb["layer_scale_init_value"],
            drop_path_rate=bb["drop_path_rate"],
        )
        feat_dim = self.backbone.dim_output

        self.cs_head = make_mlp(
            feat_dim, tuple(heads["cs_hidden"]), tasks["num_cs_classes"],
            dropout=heads["head_dropout"],
        )
        self.sg_head = make_mlp(
            feat_dim, tuple(heads["sg_hidden"]), tasks["num_sg_classes"],
            dropout=heads["head_dropout"],
        )
        self.lp_head = make_mlp(
            feat_dim, tuple(heads["lp_hidden"]), tasks["num_lp_outputs"],
            dropout=heads["head_dropout"],
        )

        self.bound_lp_with_sigmoid = tasks["bound_lp_with_sigmoid"]
        self.register_buffer(
            "lp_min",
            torch.tensor(tasks["lp_bounds_min"], dtype=torch.float32),
        )
        self.register_buffer(
            "lp_max",
            torch.tensor(tasks["lp_bounds_max"], dtype=torch.float32),
        )

        if maxsub_path is not None:
            gemd = load_gemd_distance_matrix(maxsub_path)
            self.register_buffer("gemd_distance_matrix", gemd)
        else:
            self.gemd_distance_matrix = None

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: PXRD pattern tensor of shape ``(batch, 8192)`` or
               ``(batch, 1, 8192)``, intensity-normalized to [0, 100].

        Returns:
            Dict with keys ``cs_logits``, ``sg_logits``, ``lp``.
        """
        feats = self.backbone(x)
        cs_logits = self.cs_head(feats)
        sg_logits = self.sg_head(feats)
        lp = self.lp_head(feats)
        if self.bound_lp_with_sigmoid:
            lp = torch.sigmoid(lp) * (self.lp_max - self.lp_min) + self.lp_min
        return {"cs_logits": cs_logits, "sg_logits": sg_logits, "lp": lp}

    # -- convenience loaders ------------------------------------------------

    @classmethod
    def from_pretrained(
        cls,
        model_dir: str,
        device: str = "cpu",
    ) -> "AlphaDiffract":
        """Load model from a directory containing config.json,
        model.safetensors, and maxsub.json."""
        model_dir = Path(model_dir)
        with open(model_dir / "config.json", "r") as f:
            config = json.load(f)

        maxsub_path = model_dir / "maxsub.json"
        model = cls(
            config,
            maxsub_path=str(maxsub_path) if maxsub_path.exists() else None,
        )

        weights_path = model_dir / "model.safetensors"
        if weights_path.exists():
            from safetensors.torch import load_file
            state_dict = load_file(str(weights_path), device=device)
        else:
            # Fallback to PyTorch format
            pt_path = model_dir / "model.pt"
            state_dict = torch.load(str(pt_path), map_location=device, weights_only=True)

        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        return model
