"""Separate stage-1 recovery and stage-2 routing objectives."""

from __future__ import annotations

from typing import Dict, Optional

import torch
from torch import nn
from torch.nn import functional as F


def masked_token_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    masked_positions: torch.Tensor,
) -> torch.Tensor:
    """L_tok: recover target tokens only at corrupted positions."""
    _validate_token_tensors(logits, targets, masked_positions)
    if not torch.any(masked_positions):
        return logits.sum() * 0.0
    return F.cross_entropy(logits[masked_positions], targets[masked_positions])


def legal_choice_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    legal_token_mask: torch.Tensor,
    unresolved_positions: torch.Tensor,
) -> torch.Tensor:
    """L_loc: choose each target token after illegal tokens are removed."""
    _validate_token_tensors(logits, targets, unresolved_positions)
    if legal_token_mask.shape != logits.shape or legal_token_mask.dtype != torch.bool:
        raise ValueError("legal_token_mask must be boolean with shape [B, L, V]")
    if not torch.any(unresolved_positions):
        return logits.sum() * 0.0
    selected_legal = legal_token_mask[unresolved_positions]
    selected_targets = targets[unresolved_positions]
    if not torch.all(selected_legal.gather(1, selected_targets.unsqueeze(1))):
        raise ValueError("a target token is absent from its legal-token set")
    selected_logits = logits[unresolved_positions].masked_fill(
        ~selected_legal, torch.finfo(logits.dtype).min
    )
    return F.cross_entropy(selected_logits, selected_targets)


def routing_pairwise_loss(
    positive_scores: torch.Tensor,
    negative_scores: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """L_route: log(1 + exp(-(s_pos - s_neg)))."""
    if positive_scores.shape != negative_scores.shape:
        raise ValueError("positive and negative score tensors must have equal shape")
    if positive_scores.numel() == 0:
        return positive_scores.sum() * 0.0
    losses = F.softplus(-(positive_scores - negative_scores))
    if weights is None:
        return losses.mean()
    if weights.shape != losses.shape:
        raise ValueError("routing weights must have the same shape as the scores")
    weights = weights.to(device=losses.device, dtype=losses.dtype)
    if torch.any(weights < 0) or not torch.isfinite(weights).all():
        raise ValueError("routing weights must be finite and non-negative")
    denominator = weights.sum()
    if denominator <= 0:
        raise ValueError("routing weights must have a positive sum")
    return (losses * weights).sum() / denominator


class RecoveryTrainingLoss(nn.Module):
    """Stage-1 recovery objective: L_tok + 0.5 L_loc."""

    def __init__(self, lambda_loc: float = 0.5) -> None:
        super().__init__()
        self.lambda_loc = float(lambda_loc)

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        masked_positions: torch.Tensor,
        legal_token_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        token = masked_token_loss(logits, targets, masked_positions)
        local = logits.sum() * 0.0
        if legal_token_mask is not None:
            local = legal_choice_loss(
                logits, targets, legal_token_mask, masked_positions
            )
        total = token + self.lambda_loc * local
        return {"loss": total, "tok": token, "loc": local}


class RoutingTrainingLoss(nn.Module):
    """Stage-2 pairwise routing objective."""

    def forward(
        self,
        positive_scores: torch.Tensor,
        negative_scores: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return routing_pairwise_loss(
            positive_scores,
            negative_scores,
            weights=weights,
        )


def _validate_token_tensors(
    logits: torch.Tensor,
    targets: torch.Tensor,
    positions: torch.Tensor,
) -> None:
    if logits.ndim != 3:
        raise ValueError("logits must have shape [B, L, V]")
    if targets.shape != logits.shape[:2]:
        raise ValueError("targets must have shape [B, L]")
    if positions.shape != targets.shape or positions.dtype != torch.bool:
        raise ValueError("positions must be boolean with shape [B, L]")
