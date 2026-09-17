"""Masked-state construction from the paper's quadratic schedule."""

from __future__ import annotations

from typing import Optional, Tuple

import torch


def mask_probability(timestep: torch.Tensor, total_steps: int) -> torch.Tensor:
    """Compute rho_t = 1 - (1 - t/T)^2 for t in [1, T]."""
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    timestep = torch.as_tensor(timestep)
    if torch.any(timestep < 1) or torch.any(timestep > total_steps):
        raise ValueError("timestep must be in [1, total_steps]")
    fraction = timestep.to(dtype=torch.float32) / float(total_steps)
    return 1.0 - torch.square(1.0 - fraction)


def sample_timesteps(
    batch_size: int,
    total_steps: int,
    device: Optional[torch.device] = None,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    return torch.randint(
        1,
        total_steps + 1,
        (batch_size,),
        device=device,
        generator=generator,
    )


def corrupt_identifiers(
    targets: torch.Tensor,
    timesteps: torch.Tensor,
    total_steps: Optional[int] = None,
    mask_token: int = -1,
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Independently mask identifier positions according to rho_t.

    Returns the corrupted identifier state and a boolean mask identifying the
    positions supervised by the token-recovery loss.
    """
    if targets.ndim != 2:
        raise ValueError("targets must have shape [batch, identifier_length]")
    if timesteps.ndim != 1 or timesteps.shape[0] != targets.shape[0]:
        raise ValueError("timesteps must have shape [batch]")
    total_steps = int(total_steps or targets.shape[1])
    probability = mask_probability(timesteps, total_steps).to(targets.device)
    random_values = torch.rand(
        targets.shape,
        device=targets.device,
        generator=generator,
        dtype=torch.float32,
    )
    masked_positions = random_values < probability.unsqueeze(1)
    corrupted = targets.clone()
    corrupted[masked_positions] = int(mask_token)
    return corrupted, masked_positions
