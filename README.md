# ABR: minimal reproducible implementation

This repository contains a compact reference implementation of ambiguity-aware
branch routing (ABR) for non-autoregressive identifier-based image retrieval.
It reproduces the algorithmic components needed to inspect and test candidate
generation, legality filtering, shared-pool selection, routing supervision,
and routing-head training without requiring a dataset or checkpoint.

## What is included

- identifier buckets and arbitrary-position legal-set queries;
- uncertainty-based branch-position selection;
- legal top-`K_exp` expansion and global top-`B` allocation;
- duplicate-state merging and deterministic tie breaking;
- six routing features and the `6 -> 32 -> 1` GELU routing MLP;
- fixed-score offline route-pool mining and pairwise routing loss;
- exact-target routing supervision used by the reported method;
- optional label-consistent supervision for diagnostic experiments;
- quadratic corruption and the weighted recovery/routing training objective;
- a DDCap adapter plus an explicit legacy-checkpoint compatibility mode;
- a synthetic smoke test and 14 unit tests.

## Installation

Python 3.9 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

## Reproduce the minimal example

No external files are required:

```bash
python -m abr_reproduction.smoke_test
python -m unittest discover -s abr_reproduction/tests -v
```

The smoke test builds a small identifier table with a duplicate bucket, runs
four rounds of legal ABR decoding, and prints the final legal candidates. The
test suite checks legality, batching, normalization, deterministic image
expansion, routing-pair construction, the routing trainer, and the DDCap
adapter contract.

An existing IRGen-style identifier mapping can also be inspected:

```bash
python -m abr_reproduction.smoke_test \
  --identifiers /path/to/generated_identifier_mapping.pkl
```

## Exact-target routing supervision

For a paired training image, the reported method labels a successor positive
when it can still reach that image's assigned identifier bucket:

```python
target_bucket = int(table.image_to_bucket[target_image_index])
mined = decoder.decode(
    prepared_query,
    recovery_model,
    target_bucket=target_bucket,
)
```

This is a conservative surrogate for label-level Recall@K: preserving the
paired target bucket is sufficient, but not necessary, for a successful
same-label retrieval. It gives every partial route one unambiguous target.

For diagnostic comparisons only, a set of same-label buckets can be supplied
with `target_bucket_mask`:

```python
positive_mask = table.bucket_mask_for_label(gallery_labels, query_label)
diagnostic = decoder.decode(
    prepared_query,
    recovery_model,
    target_bucket_mask=positive_mask,
)
```

Both modes expose mined pairs in `DecodeResult.route_pairs`. Every non-empty
successor pool receives equal total weight through
`DecodeResult.route_pair_weights`.

## Train the routing MLP

After collecting decode results from training queries:

```python
from abr_reproduction.train_routing import save_mined_pairs

save_mined_pairs(mined_results, "route_pairs.npz")
```

Train and save the routing head:

```bash
python -m abr_reproduction.train_routing \
  --pairs route_pairs.npz \
  --output abr_routing_mlp.pt
```

The default optimizer configuration is AdamW for 100 epochs, learning rate
`1e-3`, weight decay `1e-4`, pair batch size 4096, and seed 3407.

## Connect a DDCap checkpoint

`DDCapRecoveryAdapter` converts a loaded recovery backbone to the callable
interface expected by `ABRDecoder`. The default adapter uses four identifier
positions, no EOS state, and one conditional forward pass. Old checkpoints
that require EOS context and classifier-free guidance must opt in explicitly
through `DDCapRecoveryAdapter.for_legacy_checkpoint(...)`.

The repository intentionally does not contain datasets or unpublished model
weights. Consequently, the synthetic example verifies the implementation and
interfaces; reproducing paper-level retrieval numbers additionally requires
the corresponding trained recovery checkpoint and exported identifier table.

