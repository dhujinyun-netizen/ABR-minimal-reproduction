"""Train the configurable routing MLP from mined positive/negative pairs."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Sequence, Tuple

import numpy as np
import torch

from .routing import RoutingMLPScorer


def save_mined_pairs(results: Sequence[Any], output: str | Path) -> int:
    """Serialize pool-balanced pairs collected from ``DecodeResult`` objects."""
    pairs = [pair for result in results for pair in result.route_pairs]
    weights = [weight for result in results for weight in result.route_pair_weights]
    if not pairs:
        raise ValueError("the decode results contain no positive/negative route pairs")
    if len(pairs) != len(weights):
        raise ValueError("route pairs and pool-balancing weights are inconsistent")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        positive=np.stack([positive for positive, _ in pairs]).astype(np.float32),
        negative=np.stack([negative for _, negative in pairs]).astype(np.float32),
        weight=np.asarray(weights, dtype=np.float32),
    )
    return len(pairs)


def train_pairs(
    pairs: Sequence[Tuple[np.ndarray, np.ndarray]],
    sample_weights: np.ndarray | None = None,
    hidden_dim: int = 32,
    epochs: int = 100,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 4096,
    seed: int = 3407,
) -> RoutingMLPScorer:
    if not pairs:
        raise ValueError("no routing pairs were provided")
    torch.manual_seed(seed)
    scorer = RoutingMLPScorer(hidden_dim=hidden_dim)
    optimizer = torch.optim.AdamW(
        scorer.model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    positives = torch.as_tensor(
        np.stack([positive for positive, _ in pairs]), dtype=torch.float32
    )
    negatives = torch.as_tensor(
        np.stack([negative for _, negative in pairs]), dtype=torch.float32
    )
    if sample_weights is None:
        weights = torch.ones(len(pairs), dtype=torch.float32)
    else:
        weights = torch.as_tensor(np.asarray(sample_weights), dtype=torch.float32)
        if weights.ndim != 1 or weights.shape[0] != len(pairs):
            raise ValueError("sample_weights must have shape [N]")
        if torch.any(weights < 0) or not torch.isfinite(weights).all() or weights.sum() <= 0:
            raise ValueError("sample_weights must be finite, non-negative, and non-zero")
    weights = weights / weights.mean()
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    generator = torch.Generator().manual_seed(seed)
    scorer.model.train()
    for _ in range(int(epochs)):
        order = torch.randperm(positives.shape[0], generator=generator)
        for start in range(0, positives.shape[0], batch_size):
            batch = order[start : start + batch_size]
            optimizer.zero_grad()
            positive_scores = scorer.model(positives[batch]).squeeze(-1)
            negative_scores = scorer.model(negatives[batch]).squeeze(-1)
            pair_losses = torch.nn.functional.softplus(
                -(positive_scores - negative_scores)
            )
            # Globally mean-normalized weights make uniform mini-batches an
            # unbiased estimator of the pool-balanced objective.
            loss = (pair_losses * weights[batch]).mean()
            loss.backward()
            optimizer.step()
    scorer.model.eval()
    return scorer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pairs",
        required=True,
        help=".npz with positive/negative [N,6] arrays and optional pool weights",
    )
    parser.add_argument("--output", required=True, help="output .pt checkpoint")
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=3407)
    args = parser.parse_args()

    payload = np.load(args.pairs)
    positive = np.asarray(payload["positive"], dtype=np.float32)
    negative = np.asarray(payload["negative"], dtype=np.float32)
    if positive.shape != negative.shape or positive.ndim != 2 or positive.shape[1] != 6:
        raise ValueError("positive and negative must both have shape [N, 6]")
    weights = (
        np.asarray(payload["weight"], dtype=np.float32)
        if "weight" in payload.files
        else None
    )
    scorer = train_pairs(
        list(zip(positive, negative)),
        sample_weights=weights,
        hidden_dim=args.hidden_dim,
        epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": scorer.state_dict(),
            "hidden_dim": args.hidden_dim,
            "dropout": 0.0,
            "activation": "GELU",
            "optimizer": "AdamW",
            "learning_rate": args.lr,
            "weight_decay": args.weight_decay,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "seed": args.seed,
            "feature_names": [
                "f_path",
                "f_step",
                "f_left",
                "f_gap",
                "committed_count",
                "refinement_round",
            ],
            "feature_normalization": "within_successor_pool_before_pair_mining",
            "pair_weighting": "equal_total_weight_per_successor_pool_when weight is supplied",
        },
        args.output,
    )
    print(f"saved routing checkpoint: {args.output}")


if __name__ == "__main__":
    main()
