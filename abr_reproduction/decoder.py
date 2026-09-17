"""Ambiguity-aware branch routing decoder.

This module implements the inference rule described in the paper:

1. batch all active states into one recovery-model call per round;
2. choose the highest-uncertainty unresolved position;
3. keep the top legal ``K_exp`` tokens at that position;
4. merge successors from all active parents into one pool; and
5. retain one global Top-B active set.

The recovery callable must return logits with shape ``[batch, L, V]`` for a
batch of partial states whose unresolved positions are represented by ``-1``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from .identifier_index import IdentifierTable
from .routing import HeuristicRoutingScorer, normalize_pool_features


RecoveryCallable = Callable[[Any, np.ndarray], np.ndarray]


@dataclass
class Candidate:
    tokens: Tuple[int, ...]
    compatible_mask: int
    path_score: float = 0.0
    route_score: float = 0.0
    parent_index: int = -1
    branched_position: int = -1
    branched_token: int = -1
    routing_features: Optional[np.ndarray] = field(default=None, repr=False)

    @property
    def complete(self) -> bool:
        return all(int(token) >= 0 for token in self.tokens)

    @property
    def committed_count(self) -> int:
        return sum(int(token) >= 0 for token in self.tokens)


@dataclass
class DecodeResult:
    active: List[Candidate]
    traces: List[Dict[str, Any]]
    route_pairs: List[Tuple[np.ndarray, np.ndarray]] = field(default_factory=list)
    route_pair_weights: List[float] = field(default_factory=list)

    @property
    def complete(self) -> List[Candidate]:
        return [candidate for candidate in self.active if candidate.complete]

    def ranked_bucket_indices(self, table: IdentifierTable) -> List[int]:
        """Return complete candidate buckets in decoder order without duplicates."""
        ranked: List[int] = []
        seen = set()
        for candidate in self.complete:
            for bucket in table.bucket_indices(candidate.compatible_mask).tolist():
                if bucket not in seen:
                    seen.add(bucket)
                    ranked.append(int(bucket))
        return ranked

    def ranked_image_indices(
        self, table: IdentifierTable, limit: Optional[int] = None
    ) -> List[int]:
        """Expand ranked identifier buckets to deterministic gallery-image order."""
        images: List[int] = []
        for bucket in self.ranked_bucket_indices(table):
            images.extend(table.bucket_to_images[bucket])
            if limit is not None and len(images) >= limit:
                return images[:limit]
        return images


class ABRDecoder:
    def __init__(
        self,
        table: IdentifierTable,
        budget: int = 8,
        expansion_width: int = 2,
        routing_scorer: Optional[Any] = None,
        max_rounds: Optional[int] = None,
    ) -> None:
        if budget <= 0:
            raise ValueError("budget must be positive")
        if expansion_width <= 0:
            raise ValueError("expansion_width must be positive")
        self.table = table
        self.budget = int(budget)
        self.expansion_width = int(expansion_width)
        self.routing_scorer = routing_scorer or HeuristicRoutingScorer()
        self.max_rounds = max_rounds or table.identifier_length

    def decode(
        self,
        query: Any,
        recovery: RecoveryCallable,
        target_bucket: Optional[int] = None,
        target_bucket_mask: Optional[int] = None,
    ) -> DecodeResult:
        if target_bucket is not None and target_bucket_mask is not None:
            raise ValueError("provide target_bucket or target_bucket_mask, not both")
        positive_bucket_mask: Optional[int] = None
        if target_bucket is not None:
            if target_bucket < 0 or target_bucket >= self.table.num_buckets:
                raise ValueError("target_bucket is out of range")
            positive_bucket_mask = 1 << int(target_bucket)
        elif target_bucket_mask is not None:
            positive_bucket_mask = int(target_bucket_mask) & self.table.all_buckets
            if positive_bucket_mask == 0:
                raise ValueError("target_bucket_mask contains no gallery bucket")
        initial = Candidate(
            tokens=tuple([-1] * self.table.identifier_length),
            compatible_mask=self.table.all_buckets,
        )
        active: List[Candidate] = [initial]
        traces: List[Dict[str, Any]] = []
        route_pairs: List[Tuple[np.ndarray, np.ndarray]] = []
        route_pair_weights: List[float] = []

        for refinement_round in range(self.max_rounds):
            if all(candidate.complete for candidate in active):
                break

            pool: List[Candidate] = []
            parent_diagnostics: List[Dict[str, Any]] = []
            incomplete = [
                (parent_index, parent)
                for parent_index, parent in enumerate(active)
                if not parent.complete
            ]
            for parent in active:
                if parent.complete:
                    if parent.routing_features is None:
                        parent.routing_features = np.asarray(
                            [
                                parent.path_score,
                                0.0,
                                0.0,
                                0.0,
                                parent.committed_count,
                                refinement_round,
                            ],
                            dtype=np.float32,
                        )
                    pool.append(parent)

            if incomplete:
                states = np.asarray(
                    [parent.tokens for _, parent in incomplete], dtype=np.int64
                )
                batched_logits = np.asarray(recovery(query, states), dtype=np.float64)
                if batched_logits.ndim == 2 and len(incomplete) == 1:
                    batched_logits = batched_logits[None, ...]
                self._validate_logits(batched_logits, len(incomplete))
                for row, (parent_index, parent) in enumerate(incomplete):
                    successors, diagnostic = self._expand_parent_from_logits(
                        parent,
                        batched_logits[row],
                        parent_index,
                        refinement_round,
                    )
                    pool.extend(successors)
                    parent_diagnostics.append(diagnostic)

            if not pool:
                break

            raw_pool_size = len(pool)
            pool = self._deduplicate_pool(pool)

            features = np.stack(
                [candidate.routing_features for candidate in pool], axis=0
            )
            normalized_features = normalize_pool_features(features)
            if getattr(self.routing_scorer, "expects_normalized", False):
                scorer_input = normalized_features
            else:
                scorer_input = features
            scores = np.asarray(self.routing_scorer.score(scorer_input)).reshape(-1)
            if scores.shape[0] != len(pool):
                raise ValueError("routing scorer returned the wrong number of scores")
            for candidate, score in zip(pool, scores.tolist()):
                candidate.route_score = float(score)

            round_pair_count = 0
            if positive_bucket_mask is not None:
                positive = [
                    normalized_features[index]
                    for index, candidate in enumerate(pool)
                    if candidate.compatible_mask & positive_bucket_mask
                ]
                negative = [
                    normalized_features[index]
                    for index, candidate in enumerate(pool)
                    if (candidate.compatible_mask & positive_bucket_mask) == 0
                ]
                pair_count = len(positive) * len(negative)
                if pair_count:
                    round_pair_count = pair_count
                    pair_weight = 1.0 / float(pair_count)
                    route_pairs.extend(
                        (np.asarray(pos, dtype=np.float32), np.asarray(neg, dtype=np.float32))
                        for pos in positive
                        for neg in negative
                    )
                    route_pair_weights.extend([pair_weight] * pair_count)

            ranked = sorted(
                enumerate(pool),
                key=lambda item: (-item[1].route_score, -item[1].path_score, item[0]),
            )
            active = [candidate for _, candidate in ranked[: self.budget]]
            traces.append(
                {
                    "round": refinement_round,
                    "raw_pool_size": raw_pool_size,
                    "pool_size": len(pool),
                    "active_size": len(active),
                    "active_tokens": [list(candidate.tokens) for candidate in active],
                    "parent_diagnostics": parent_diagnostics,
                    "route_scores": [candidate.route_score for candidate in active],
                    "route_pair_count": round_pair_count,
                }
            )

        return DecodeResult(
            active=active,
            traces=traces,
            route_pairs=route_pairs,
            route_pair_weights=route_pair_weights,
        )

    def _validate_logits(self, logits: np.ndarray, batch_size: int) -> None:
        if (
            logits.ndim != 3
            or logits.shape[0] != batch_size
            or logits.shape[1] < self.table.identifier_length
            or logits.shape[2] < self.table.codebook_size
        ):
            raise ValueError(
                "recovery callable must return [batch, identifier_length, codebook_size] logits; "
                f"got {tuple(logits.shape)} for batch={batch_size}"
            )

    @staticmethod
    def _deduplicate_pool(pool: List[Candidate]) -> List[Candidate]:
        """Merge identical partial identifiers before they consume beam slots."""
        best_by_tokens: Dict[Tuple[int, ...], Candidate] = {}
        insertion_order: List[Tuple[int, ...]] = []
        for candidate in pool:
            previous = best_by_tokens.get(candidate.tokens)
            if previous is None:
                best_by_tokens[candidate.tokens] = candidate
                insertion_order.append(candidate.tokens)
            elif candidate.path_score > previous.path_score:
                best_by_tokens[candidate.tokens] = candidate
        return [best_by_tokens[tokens] for tokens in insertion_order]

    def _expand_parent_from_logits(
        self,
        parent: Candidate,
        logits: np.ndarray,
        parent_index: int,
        refinement_round: int,
    ) -> Tuple[List[Candidate], Dict[str, Any]]:
        uncertainty: Dict[int, float] = {}
        legal_log_probs: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
        for position, token in enumerate(parent.tokens):
            if int(token) >= 0:
                continue
            legal = self.table.valid_tokens(
                parent.tokens,
                position,
                compatible_mask=parent.compatible_mask,
            )
            if legal.size == 0:
                continue
            values = logits[position, legal]
            log_probs = _log_softmax(values)
            probabilities = np.exp(log_probs)
            if legal.size > 1:
                entropy = float(
                    -np.sum(probabilities * log_probs) / np.log(float(legal.size))
                )
            else:
                entropy = 0.0
            uncertainty[position] = entropy
            legal_log_probs[position] = (legal, log_probs)

        if not uncertainty:
            return [], {
                "parent": list(parent.tokens),
                "selected_position": None,
                "legal": False,
            }

        selected_position = max(
            uncertainty,
            key=lambda position: (uncertainty[position], -position),
        )
        legal, log_probs = legal_log_probs[selected_position]
        order = np.argsort(-log_probs, kind="stable")[: self.expansion_width]
        selected_tokens = legal[order]
        selected_log_probs = log_probs[order]
        successors: List[Candidate] = []
        for token, step_score in zip(selected_tokens.tolist(), selected_log_probs.tolist()):
            child_tokens = list(parent.tokens)
            child_tokens[selected_position] = int(token)
            child_mask = parent.compatible_mask & self.table.token_bucket_mask(
                selected_position, int(token)
            )
            if child_mask == 0:
                continue
            remaining = [
                uncertainty[position]
                for position, current in enumerate(child_tokens)
                if int(current) < 0 and position in uncertainty
            ]
            f_left = float(np.mean(remaining)) if remaining else 0.0
            f_gap = float(uncertainty[selected_position] - f_left)
            child_path = float(parent.path_score + step_score)
            features = np.asarray(
                [
                    child_path,
                    float(step_score),
                    f_left,
                    f_gap,
                    sum(int(value) >= 0 for value in child_tokens),
                    refinement_round,
                ],
                dtype=np.float32,
            )
            successors.append(
                Candidate(
                    tokens=tuple(child_tokens),
                    compatible_mask=int(child_mask),
                    path_score=child_path,
                    parent_index=parent_index,
                    branched_position=selected_position,
                    branched_token=int(token),
                    routing_features=features,
                )
            )

        return successors, {
            "parent": list(parent.tokens),
            "selected_position": selected_position,
            "uncertainty": {str(k): float(v) for k, v in uncertainty.items()},
            "selected_tokens": [int(x) for x in selected_tokens.tolist()],
        }


def _log_softmax(values: np.ndarray) -> np.ndarray:
    maximum = np.max(values)
    shifted = values - maximum
    log_normalizer = maximum + np.log(np.exp(shifted).sum())
    return values - log_normalizer
