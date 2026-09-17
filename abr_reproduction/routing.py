"""Routing features and optional learned routing scorer for ABR."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np


FEATURE_NAMES = (
    "f_path",
    "f_step",
    "f_left",
    "f_gap",
    "committed_count",
    "refinement_round",
)


def normalize_pool_features(features: np.ndarray) -> np.ndarray:
    """Normalize routing features within one successor pool."""
    values = np.asarray(features, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != len(FEATURE_NAMES):
        raise ValueError("routing features must have shape [N, 6]")
    mean = values.mean(axis=0, keepdims=True)
    std = values.std(axis=0, keepdims=True)
    return (values - mean) / np.maximum(std, 1e-6)


class HeuristicRoutingScorer:
    """Non-learned score used for proposal mining and a strong control."""

    expects_normalized = False

    def score(self, features: np.ndarray) -> np.ndarray:
        values = np.asarray(features, dtype=np.float32)
        return values[:, 0] + values[:, 3] - values[:, 2]


class RoutingMLPScorer:
    """Configurable two-layer routing MLP.

    The supplied paper source does not specify the original hidden width,
    activation, or dropout.  The default here is explicit: 6 -> 32 -> 1,
    GELU, no dropout.  A checkpoint can override the learned weights.
    """

    expects_normalized = True

    def __init__(self, hidden_dim: int = 32, dropout: float = 0.0) -> None:
        try:
            import torch
            from torch import nn
        except ImportError as exc:  # pragma: no cover
            raise ImportError("RoutingMLPScorer requires PyTorch") from exc
        self.torch = torch
        self.model = nn.Sequential(
            nn.Linear(6, hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_dim, 1),
        )
        for layer in self.model:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
        self.device = torch.device("cpu")

    def to(self, device: Any) -> "RoutingMLPScorer":
        self.device = self.torch.device(device)
        self.model.to(self.device)
        return self

    def eval(self) -> "RoutingMLPScorer":
        self.model.eval()
        return self

    def score(self, features: np.ndarray) -> np.ndarray:
        values = self.torch.as_tensor(
            np.asarray(features, dtype=np.float32), device=self.device
        )
        self.model.eval()
        with self.torch.no_grad():
            return self.model(values).squeeze(-1).detach().cpu().numpy()

    def state_dict(self) -> Any:
        return self.model.state_dict()

    def load_state_dict(self, state_dict: Any) -> None:
        self.model.load_state_dict(state_dict)

    @classmethod
    def from_checkpoint(
        cls,
        path: str,
        hidden_dim: int = 0,
        dropout: Optional[float] = None,
        device: str = "cpu",
    ) -> "RoutingMLPScorer":
        import torch

        payload = torch.load(Path(path), map_location=device)
        if isinstance(payload, dict) and "model" in payload:
            if hidden_dim <= 0:
                hidden_dim = int(payload.get("hidden_dim", 32))
            if dropout is None:
                dropout = float(payload.get("dropout", 0.0))
            state_dict = payload["model"]
        else:
            state_dict = payload
        if hidden_dim <= 0:
            first_weight = state_dict.get("0.weight")
            hidden_dim = int(first_weight.shape[0]) if first_weight is not None else 32
        scorer = cls(
            hidden_dim=hidden_dim,
            dropout=0.0 if dropout is None else dropout,
        ).to(device).eval()
        scorer.load_state_dict(state_dict)
        return scorer
