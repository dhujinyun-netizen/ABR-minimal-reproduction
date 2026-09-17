"""Adapters from the DDCap backbone to the paper-defined ABR decoder.

``paper`` mode is normative: four identifier positions, no EOS in the retrieval
state, the quadratic mask schedule, and one conditional forward pass.
``legacy`` mode explicitly reproduces the older DDCap validation convention.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import torch
from torch.nn import functional as F


@dataclass
class PreparedDDCapQuery:
    prefix_embed: torch.Tensor
    unconditional_prefix_embed: Optional[torch.Tensor] = None


class DDCapRecoveryAdapter:
    """Expose a loaded DDCap ``ClipCaptionModel`` as an ABR recovery callable."""

    def __init__(
        self,
        model: Any,
        identifier_length: int = 4,
        codebook_size: int = 256,
        time_steps: Optional[int] = None,
        schedule_mode: str = "paper",
        include_eos_context: bool = False,
        eos_token: int = 256,
        guidance_scale: float = 1.0,
        device: Optional[str] = None,
    ) -> None:
        self.model = model.module if hasattr(model, "module") else model
        self.identifier_length = int(identifier_length)
        self.codebook_size = int(codebook_size)
        self.schedule_mode = str(schedule_mode)
        if self.schedule_mode not in {"paper", "legacy"}:
            raise ValueError("schedule_mode must be 'paper' or 'legacy'")
        default_steps = (
            self.identifier_length
            if self.schedule_mode == "paper"
            else int(getattr(self.model, "time_step", self.identifier_length))
        )
        self.time_steps = int(default_steps if time_steps is None else time_steps)
        if self.identifier_length <= 0 or self.codebook_size <= 0 or self.time_steps <= 0:
            raise ValueError("identifier_length, codebook_size, and time_steps must be positive")
        self.include_eos_context = bool(include_eos_context)
        self.eos_token = int(eos_token)
        self.guidance_scale = float(guidance_scale)
        self.device = torch.device(device or next(self.model.parameters()).device)
        self.model.eval()

    @classmethod
    def for_legacy_checkpoint(
        cls,
        model: Any,
        identifier_length: int = 4,
        codebook_size: int = 256,
        guidance_scale: float = 1.15,
        device: Optional[str] = None,
    ) -> "DDCapRecoveryAdapter":
        """Match EOS, clock, and guidance used by the legacy validation decoder."""
        module = model.module if hasattr(model, "module") else model
        if not hasattr(module, "time_step"):
            raise AttributeError("legacy DDCap model must expose time_step")
        return cls(
            model=model,
            identifier_length=identifier_length,
            codebook_size=codebook_size,
            time_steps=int(module.time_step),
            schedule_mode="legacy",
            include_eos_context=True,
            eos_token=codebook_size,
            guidance_scale=guidance_scale,
            device=device,
        )

    def prepare(self, image: torch.Tensor) -> PreparedDDCapQuery:
        image = image.to(self.device)
        if image.ndim == 3:
            image = image.unsqueeze(0)
        if image.ndim != 4 or image.shape[0] != 1:
            raise ValueError("prepare expects exactly one query image [1,C,H,W]")
        with torch.no_grad():
            prefix, _ = self.model.image_encode(image)
            prefix_embed = self.model.clip_project(prefix).detach()
        unconditional = None
        if self.guidance_scale != 1.0:
            unconditional = self._unconditional_prefix(prefix_embed)
        return PreparedDDCapQuery(prefix_embed, unconditional)

    def __call__(self, query: PreparedDDCapQuery, states: np.ndarray) -> np.ndarray:
        if not isinstance(query, PreparedDDCapQuery):
            raise TypeError("query must come from adapter.prepare()")
        state_array = np.asarray(states, dtype=np.int64)
        if state_array.ndim != 2 or state_array.shape[1] != self.identifier_length:
            raise ValueError("states must have shape [batch, identifier_length]")
        committed = state_array[state_array >= 0]
        if committed.size and committed.max() >= self.codebook_size:
            raise ValueError("a committed identifier token is outside the codebook")

        batch_size = state_array.shape[0]
        safe_tokens = np.where(state_array < 0, 0, state_array)
        unresolved = state_array < 0
        if self.include_eos_context:
            safe_tokens = np.concatenate(
                [safe_tokens, np.full((batch_size, 1), self.eos_token, dtype=np.int64)],
                axis=1,
            )
            unresolved = np.concatenate(
                [unresolved, np.zeros((batch_size, 1), dtype=bool)], axis=1
            )

        ids_t = torch.as_tensor(safe_tokens, dtype=torch.long, device=self.device)
        embeds = self.model.gpt.transformer.wte(ids_t)
        unknown = torch.as_tensor(unresolved, dtype=torch.bool, device=self.device)
        bos = self.model.bos_embedding.to(device=self.device, dtype=embeds.dtype)
        embeds = torch.where(unknown.unsqueeze(-1), bos.view(1, 1, -1), embeds)

        sequence_length = safe_tokens.shape[1]
        attention = torch.zeros(
            (batch_size, sequence_length, sequence_length),
            dtype=embeds.dtype,
            device=self.device,
        )
        attention.masked_fill_(unknown.unsqueeze(1), -1e4)
        diagonal = torch.eye(sequence_length, dtype=torch.bool, device=self.device)
        attention.masked_fill_(diagonal.unsqueeze(0), 0.0)

        timestep = self._timesteps(state_array < 0)
        conditional = self._expand_prefix(query.prefix_embed, batch_size)
        with torch.no_grad():
            output = self.model.gpt(
                timestep,
                inputs_embeds=embeds,
                attention_mask=attention,
                encoder_hidden_states=conditional,
            )
            scores = output.logits
            if self.guidance_scale != 1.0:
                if query.unconditional_prefix_embed is None:
                    raise ValueError("guided decoding requires an unconditional prefix")
                unconditional = self._expand_prefix(
                    query.unconditional_prefix_embed, batch_size
                )
                output_u = self.model.gpt(
                    timestep,
                    inputs_embeds=embeds,
                    attention_mask=attention,
                    encoder_hidden_states=unconditional,
                )
                logp_c = F.log_softmax(scores, dim=-1)
                logp_u = F.log_softmax(output_u.logits, dim=-1)
                scores = self.guidance_scale * (logp_c - logp_u) + logp_u

        scores = scores[:, : self.identifier_length, : self.codebook_size]
        return scores.detach().float().cpu().numpy()

    def _timesteps(self, unresolved: np.ndarray) -> torch.Tensor:
        count = unresolved.sum(axis=1).astype(np.float64)
        ratio = count / float(self.identifier_length)
        if self.schedule_mode == "paper":
            steps = np.arange(1, self.time_steps + 1, dtype=np.float64)
            rho = 1.0 - np.square(1.0 - steps / float(self.time_steps))
            values = np.abs(ratio[:, None] - rho[None, :]).argmin(axis=1) + 1
        else:
            denominator = self.identifier_length + int(self.include_eos_context)
            values = np.floor(count * self.time_steps / float(denominator)).astype(np.int64)
            values = np.clip(values, 0, self.time_steps - 1)
        return torch.as_tensor(values, dtype=torch.long, device=self.device)

    @staticmethod
    def _expand_prefix(prefix: torch.Tensor, batch_size: int) -> torch.Tensor:
        if prefix.shape[0] == 1:
            return prefix.expand(batch_size, -1, -1)
        if prefix.shape[0] != batch_size:
            raise ValueError("prepared query batch does not match candidate batch")
        return prefix

    def _unconditional_prefix(self, conditional: torch.Tensor) -> torch.Tensor:
        if not hasattr(self.model, "pad_embedding"):
            raise AttributeError("legacy guidance requires model.pad_embedding")
        pad = self.model.pad_embedding.to(device=self.device, dtype=conditional.dtype)
        if pad.ndim == 1:
            pad = pad.view(1, 1, -1)
        elif pad.ndim == 2:
            pad = pad.unsqueeze(0)
        elif pad.ndim != 3:
            raise ValueError("pad_embedding must have rank 1, 2, or 3")
        if pad.shape[1:] != conditional.shape[1:]:
            raise ValueError("pad_embedding does not match projected image conditioning")
        return pad[:1].detach()
