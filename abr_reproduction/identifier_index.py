"""Gallery identifier table and legal-set queries used by ABR."""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np


TokenState = Sequence[int]


@dataclass
class IdentifierTable:
    """Deduplicated identifier buckets plus position/token bitsets."""

    bucket_identifiers: np.ndarray
    image_to_bucket: np.ndarray
    bucket_to_images: Tuple[Tuple[int, ...], ...]
    bitsets: Tuple[Tuple[int, ...], ...]
    codebook_size: int

    @property
    def num_buckets(self) -> int:
        return int(self.bucket_identifiers.shape[0])

    @property
    def identifier_length(self) -> int:
        return int(self.bucket_identifiers.shape[1])

    @property
    def all_buckets(self) -> int:
        return (1 << self.num_buckets) - 1

    @classmethod
    def from_mapping(
        cls,
        mapping: Union[np.ndarray, Sequence[Sequence[int]]],
        codebook_size: Optional[int] = None,
    ) -> "IdentifierTable":
        array = np.asarray(mapping)
        if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] == 0:
            raise ValueError("mapping must be a non-empty [num_images, length] array")
        if not np.issubdtype(array.dtype, np.integer):
            if not np.all(np.equal(array, np.floor(array))):
                raise ValueError("identifier tokens must be integers")
        array = array.astype(np.int64, copy=False)
        if np.any(array < 0):
            raise ValueError("identifier tokens must be non-negative")

        if codebook_size is None:
            codebook_size = int(array.max()) + 1
        if codebook_size <= int(array.max()):
            raise ValueError("codebook_size is smaller than an identifier token")

        bucket_identifiers, image_to_bucket = np.unique(
            array, axis=0, return_inverse=True
        )
        image_to_bucket = image_to_bucket.astype(np.int64, copy=False)

        images_by_bucket: List[List[int]] = [
            [] for _ in range(bucket_identifiers.shape[0])
        ]
        for image_index, bucket_index in enumerate(image_to_bucket.tolist()):
            images_by_bucket[int(bucket_index)].append(image_index)

        bitsets: List[List[int]] = [
            [0 for _ in range(codebook_size)]
            for _ in range(array.shape[1])
        ]
        for bucket_index, identifier in enumerate(bucket_identifiers.tolist()):
            bit = 1 << int(bucket_index)
            for position, token in enumerate(identifier):
                bitsets[position][int(token)] |= bit

        return cls(
            bucket_identifiers=bucket_identifiers,
            image_to_bucket=image_to_bucket,
            bucket_to_images=tuple(tuple(x) for x in images_by_bucket),
            bitsets=tuple(tuple(x) for x in bitsets),
            codebook_size=int(codebook_size),
        )

    @classmethod
    def from_pickle(
        cls,
        path: Union[str, Path],
        codebook_size: Optional[int] = None,
    ) -> "IdentifierTable":
        with Path(path).open("rb") as handle:
            payload = pickle.load(handle)
        if isinstance(payload, dict):
            if "mapping" not in payload:
                raise KeyError("identifier pickle does not contain 'mapping'")
            mapping = payload["mapping"]
        else:
            mapping = payload
        return cls.from_mapping(mapping, codebook_size=codebook_size)

    def token_bucket_mask(self, position: int, token: int) -> int:
        self._check_position(position)
        if token < 0 or token >= self.codebook_size:
            return 0
        return int(self.bitsets[position][token])

    def compatible_mask(
        self,
        tokens: TokenState,
        initial_mask: Optional[int] = None,
    ) -> int:
        if len(tokens) != self.identifier_length:
            raise ValueError("token state has the wrong identifier length")
        compatible = self.all_buckets if initial_mask is None else int(initial_mask)
        for position, token in enumerate(tokens):
            if int(token) < 0:
                continue
            compatible &= self.token_bucket_mask(position, int(token))
            if compatible == 0:
                break
        return compatible

    def valid_tokens(
        self,
        tokens: TokenState,
        position: int,
        compatible_mask: Optional[int] = None,
    ) -> np.ndarray:
        """Return tokens that preserve at least one legal gallery bucket."""
        self._check_position(position)
        if len(tokens) != self.identifier_length:
            raise ValueError("token state has the wrong identifier length")
        compatible = (
            self.compatible_mask(tokens)
            if compatible_mask is None
            else int(compatible_mask)
        )
        if compatible == 0:
            return np.empty((0,), dtype=np.int64)
        return np.asarray(
            [
                token
                for token in range(self.codebook_size)
                if compatible & self.bitsets[position][token]
            ],
            dtype=np.int64,
        )

    def bucket_indices(self, compatible_mask: int) -> np.ndarray:
        """Decode an integer bitset to sorted bucket indices."""
        mask = int(compatible_mask)
        result: List[int] = []
        while mask:
            lowest = mask & -mask
            result.append(lowest.bit_length() - 1)
            mask ^= lowest
        return np.asarray(result, dtype=np.int64)

    def images_for_bucket_mask(self, compatible_mask: int) -> Tuple[int, ...]:
        images: List[int] = []
        for bucket_index in self.bucket_indices(compatible_mask).tolist():
            images.extend(self.bucket_to_images[int(bucket_index)])
        return tuple(images)

    def bucket_mask_for_label(
        self,
        image_labels: Sequence[object],
        target_label: object,
        exclude_image_index: Optional[int] = None,
    ) -> int:
        """Return buckets containing at least one relevant image with target_label.

        If ``exclude_image_index`` is provided, that image is excluded when
        constructing the relevance mask, matching self-exclusion retrieval
        protocols.
        """
        if len(image_labels) != len(self.image_to_bucket):
            raise ValueError("image_labels must contain one label per gallery image")
        if exclude_image_index is not None:
            if (
                exclude_image_index < 0
                or exclude_image_index >= len(self.image_to_bucket)
            ):
                raise IndexError("exclude_image_index is out of range")
        mask = 0
        for image_index, label in enumerate(image_labels):
            if (
                exclude_image_index is not None
                and image_index == exclude_image_index
            ):
                continue
            if label == target_label:
                mask |= 1 << int(self.image_to_bucket[image_index])
        if mask == 0:
            raise ValueError("no retrieval-relevant image remains for target_label")
        return mask

    def _check_position(self, position: int) -> None:
        if position < 0 or position >= self.identifier_length:
            raise IndexError("identifier position out of range")
