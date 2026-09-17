from __future__ import annotations

import unittest
import tempfile
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import torch

from abr_reproduction.corruption import mask_probability
from abr_reproduction.ddcap_adapter import DDCapRecoveryAdapter
from abr_reproduction.decoder import ABRDecoder
from abr_reproduction.identifier_index import IdentifierTable
from abr_reproduction.losses import ABRTrainingLoss, legal_choice_loss
from abr_reproduction.smoke_test import ToyRecovery
from abr_reproduction.train_routing import save_mined_pairs, train_pairs


class CountingRecovery(ToyRecovery):
    def __init__(self, target: np.ndarray, codebook_size: int) -> None:
        super().__init__(target, codebook_size)
        self.batch_sizes = []

    def __call__(self, query, states: np.ndarray) -> np.ndarray:
        self.batch_sizes.append(int(states.shape[0]))
        return super().__call__(query, states)


class RecordingNormalizedScorer:
    expects_normalized = True

    def __init__(self) -> None:
        self.inputs = []

    def score(self, features: np.ndarray) -> np.ndarray:
        self.inputs.append(features.copy())
        return features[:, 0]


class FakeGPT(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.transformer = torch.nn.Module()
        self.transformer.wte = torch.nn.Embedding(300, 8)
        self.calls = []

    def forward(
        self, timestep, inputs_embeds, attention_mask, encoder_hidden_states
    ):
        self.calls.append(
            {
                "timestep": timestep.detach().cpu().clone(),
                "sequence_length": inputs_embeds.shape[1],
                "attention_shape": tuple(attention_mask.shape),
                "prefix": encoder_hidden_states.detach().cpu().clone(),
            }
        )
        batch, length, _ = inputs_embeds.shape
        logits = torch.zeros(batch, length, 257, device=inputs_embeds.device)
        logits[..., 0] = encoder_hidden_states.mean(dim=(1, 2)).unsqueeze(1)
        return SimpleNamespace(logits=logits)


class FakeDDCap(torch.nn.Module):
    def __init__(self, time_step: int = 128) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.gpt = FakeGPT()
        self.bos_embedding = torch.nn.Parameter(torch.zeros(8))
        self.pad_embedding = torch.nn.Parameter(torch.full((2, 8), -1.0))
        self.time_step = time_step

    def image_encode(self, image):
        prefix = torch.ones(image.shape[0], 2, 8, device=image.device)
        return prefix, torch.zeros(image.shape[0], 1, device=image.device)

    @staticmethod
    def clip_project(prefix):
        return prefix


class ABRReproductionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.mapping = np.asarray(
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
        self.table = IdentifierTable.from_mapping(self.mapping, codebook_size=4)

    def test_duplicate_identifiers_share_one_bucket(self) -> None:
        self.assertEqual(self.table.num_buckets, 5)
        self.assertEqual(
            int(self.table.image_to_bucket[4]), int(self.table.image_to_bucket[5])
        )

    def test_legal_set_intersection(self) -> None:
        state = (0, -1, -1, -1)
        valid = self.table.valid_tokens(state, 1)
        self.assertEqual(valid.tolist(), [0, 1])
        compatible = self.table.compatible_mask((0, 1, -1, -1))
        self.assertEqual(self.table.bucket_indices(compatible).tolist(), [2])

    def test_abr_returns_legal_complete_candidates(self) -> None:
        target = self.table.bucket_identifiers[0]
        decoder = ABRDecoder(self.table, budget=3, expansion_width=2)
        result = decoder.decode("query", ToyRecovery(target, self.table.codebook_size))
        self.assertTrue(result.traces)
        self.assertTrue(result.active)
        for candidate in result.active:
            self.assertTrue(candidate.complete)
            self.assertNotEqual(candidate.compatible_mask, 0)
            first_bucket = self.table.bucket_indices(candidate.compatible_mask)[0]
            self.assertEqual(
                tuple(self.table.bucket_identifiers[first_bucket]), candidate.tokens
            )

    def test_recovery_is_batched_once_per_refinement_round(self) -> None:
        target = self.table.bucket_identifiers[0]
        recovery = CountingRecovery(target, self.table.codebook_size)
        decoder = ABRDecoder(self.table, budget=3, expansion_width=2)
        result = decoder.decode("query", recovery)
        self.assertEqual(len(recovery.batch_sizes), len(result.traces))
        self.assertEqual(recovery.batch_sizes[0], 1)
        self.assertTrue(any(batch_size > 1 for batch_size in recovery.batch_sizes[1:]))

    def test_mlp_receives_pool_normalized_features(self) -> None:
        target = self.table.bucket_identifiers[0]
        scorer = RecordingNormalizedScorer()
        decoder = ABRDecoder(
            self.table, budget=3, expansion_width=2, routing_scorer=scorer
        )
        result = decoder.decode("query", ToyRecovery(target, self.table.codebook_size))
        self.assertEqual(len(scorer.inputs), len(result.traces))
        for values in scorer.inputs:
            np.testing.assert_allclose(values.mean(axis=0), 0.0, atol=1e-5)

    def test_paper_mask_schedule(self) -> None:
        probabilities = mask_probability(torch.tensor([1, 4]), total_steps=4)
        torch.testing.assert_close(probabilities, torch.tensor([0.4375, 1.0]))

    def test_legal_loss_ignores_illegal_high_logit(self) -> None:
        logits = torch.tensor([[[2.0, 1.0, 20.0], [0.0, 3.0, 0.0]]])
        targets = torch.tensor([[0, 1]])
        unresolved = torch.tensor([[True, False]])
        legal = torch.tensor([[[True, True, False], [True, True, True]]])
        loss = legal_choice_loss(logits, targets, legal, unresolved)
        expected = torch.nn.functional.cross_entropy(
            torch.tensor([[2.0, 1.0]]), torch.tensor([0])
        )
        torch.testing.assert_close(loss, expected)

    def test_weighted_training_objective(self) -> None:
        logits = torch.tensor([[[3.0, 0.0], [0.0, 3.0]]], requires_grad=True)
        targets = torch.tensor([[0, 1]])
        masked = torch.tensor([[True, True]])
        legal = torch.ones_like(logits, dtype=torch.bool)
        positive = torch.tensor([2.0])
        negative = torch.tensor([0.0])
        losses = ABRTrainingLoss()(
            logits,
            targets,
            masked,
            legal_token_mask=legal,
            positive_route_scores=positive,
            negative_route_scores=negative,
        )
        expected = losses["tok"] + 0.5 * losses["loc"] + 0.1 * losses["route"]
        torch.testing.assert_close(losses["loss"], expected)
        losses["loss"].backward()
        self.assertIsNotNone(logits.grad)

    def test_paper_ddcap_adapter_contract(self) -> None:
        model = FakeDDCap(time_step=128)
        adapter = DDCapRecoveryAdapter(model)
        query = adapter.prepare(torch.zeros(3, 4, 4))
        states = np.asarray(
            [
                [-1, -1, -1, -1],
                [0, -1, -1, -1],
                [0, 0, -1, -1],
                [0, 0, 0, -1],
            ]
        )
        logits = adapter(query, states)
        self.assertEqual(logits.shape, (4, 4, 256))
        self.assertEqual(len(model.gpt.calls), 1)
        self.assertEqual(model.gpt.calls[0]["sequence_length"], 4)
        self.assertEqual(model.gpt.calls[0]["attention_shape"], (4, 4, 4))
        self.assertEqual(model.gpt.calls[0]["timestep"].tolist(), [4, 2, 1, 1])
        self.assertFalse(model.training)

    def test_legacy_ddcap_adapter_is_explicit(self) -> None:
        model = FakeDDCap(time_step=128)
        adapter = DDCapRecoveryAdapter.for_legacy_checkpoint(model)
        query = adapter.prepare(torch.zeros(1, 3, 4, 4))
        logits = adapter(query, np.full((1, 4), -1, dtype=np.int64))
        self.assertEqual(logits.shape, (1, 4, 256))
        self.assertEqual(len(model.gpt.calls), 2)
        self.assertEqual(model.gpt.calls[0]["sequence_length"], 5)
        self.assertEqual(model.gpt.calls[0]["timestep"].tolist(), [102])

    def test_route_pairs_have_equal_pool_mass(self) -> None:
        target_bucket = int(self.table.image_to_bucket[0])
        target = self.table.bucket_identifiers[target_bucket]
        result = ABRDecoder(self.table, budget=3, expansion_width=2).decode(
            "query",
            ToyRecovery(target, self.table.codebook_size),
            target_bucket=target_bucket,
        )
        nonempty_pools = sum(trace["route_pair_count"] > 0 for trace in result.traces)
        self.assertEqual(len(result.route_pairs), len(result.route_pair_weights))
        self.assertAlmostEqual(sum(result.route_pair_weights), nonempty_pools)

    def test_label_consistent_route_supervision(self) -> None:
        labels = [7, 7, 8, 9, 10, 10]
        positive_mask = self.table.bucket_mask_for_label(labels, target_label=7)
        expected = {
            int(self.table.image_to_bucket[0]),
            int(self.table.image_to_bucket[1]),
        }
        self.assertEqual(set(self.table.bucket_indices(positive_mask).tolist()), expected)
        target = self.mapping[0]
        result = ABRDecoder(self.table, budget=3, expansion_width=2).decode(
            "query",
            ToyRecovery(target, self.table.codebook_size),
            target_bucket_mask=positive_mask,
        )
        self.assertTrue(result.route_pairs)
        self.assertEqual(len(result.route_pairs), len(result.route_pair_weights))

    def test_ranked_image_expansion(self) -> None:
        target = self.table.bucket_identifiers[0]
        result = ABRDecoder(self.table, budget=3, expansion_width=2).decode(
            "query", ToyRecovery(target, self.table.codebook_size)
        )
        images = result.ranked_image_indices(self.table)
        self.assertTrue(images)
        self.assertEqual(len(images), len(set(images)))

    def test_weighted_pair_serialization_and_training(self) -> None:
        target_bucket = int(self.table.image_to_bucket[0])
        target = self.table.bucket_identifiers[target_bucket]
        result = ABRDecoder(self.table, budget=3, expansion_width=2).decode(
            "query",
            ToyRecovery(target, self.table.codebook_size),
            target_bucket=target_bucket,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pairs.npz"
            count = save_mined_pairs([result], path)
            payload = np.load(path)
            self.assertEqual(payload["positive"].shape, (count, 6))
            self.assertEqual(payload["negative"].shape, (count, 6))
            self.assertEqual(payload["weight"].shape, (count,))
            scorer = train_pairs(
                list(zip(payload["positive"], payload["negative"])),
                sample_weights=payload["weight"],
                epochs=2,
                batch_size=4,
            )
            self.assertEqual(tuple(scorer.model[0].weight.shape), (32, 6))


if __name__ == "__main__":
    unittest.main()
