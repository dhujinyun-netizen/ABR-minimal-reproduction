"""Run a dependency-light ABR smoke test."""

from __future__ import annotations

import argparse
from typing import Any

import numpy as np

from .decoder import ABRDecoder
from .identifier_index import IdentifierTable


class ToyRecovery:
    """Deterministic masked-token scorer used only for smoke testing."""

    def __init__(self, target: np.ndarray, codebook_size: int) -> None:
        self.target = np.asarray(target, dtype=np.int64)
        self.codebook_size = int(codebook_size)

    def __call__(self, _query: Any, states: np.ndarray) -> np.ndarray:
        states = np.asarray(states, dtype=np.int64)
        logits = np.full(
            (states.shape[0], states.shape[1], self.codebook_size),
            -5.0,
            dtype=np.float64,
        )
        for batch_index, state in enumerate(states):
            for position, token in enumerate(state):
                if token < 0:
                    logits[batch_index, position, self.target[position]] = 4.0
                    competing = (self.target[position] + 1) % self.codebook_size
                    logits[batch_index, position, competing] = 3.0
                else:
                    logits[batch_index, position, token] = 4.0
        return logits


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--identifiers", default="", help="optional IRGen identifier pickle")
    parser.add_argument("--budget", type=int, default=8)
    parser.add_argument("--k-exp", type=int, default=2)
    args = parser.parse_args()

    if args.identifiers:
        table = IdentifierTable.from_pickle(args.identifiers, codebook_size=256)
        target = table.bucket_identifiers[0]
    else:
        mapping = np.asarray(
            [
                [0, 0, 0, 0],
                [0, 0, 0, 1],
                [0, 1, 0, 0],
                [1, 0, 0, 0],
                [1, 0, 1, 0],
                [1, 0, 1, 0],
            ],
            dtype=np.int64,
        )
        table = IdentifierTable.from_mapping(mapping, codebook_size=4)
        target = np.asarray([0, 0, 0, 0], dtype=np.int64)

    decoder = ABRDecoder(
        table,
        budget=args.budget,
        expansion_width=args.k_exp,
    )
    result = decoder.decode("toy-query", ToyRecovery(target, table.codebook_size))
    print("identifier_length:", table.identifier_length)
    print("num_images:", len(table.image_to_bucket))
    print("num_buckets:", table.num_buckets)
    print("rounds:", len(result.traces))
    print("final_candidates:")
    for candidate in result.active:
        buckets = table.bucket_indices(candidate.compatible_mask).tolist()
        print("  tokens=%s route=%.4f buckets=%s" % (candidate.tokens, candidate.route_score, buckets))


if __name__ == "__main__":
    main()
