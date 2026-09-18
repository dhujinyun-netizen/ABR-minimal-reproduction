# ABR: minimal reproducible implementation

This repository contains the public reference implementation of
Ambiguity-Aware Branch Routing (ABR) for non-autoregressive identifier-based
image retrieval.

## What is included

- identifier buckets and arbitrary-position legal-set queries;
- ambiguity-based branch-position selection;
- legal top-`K_exp` expansion and global top-`B` allocation;
- duplicate-state merging and deterministic tie breaking;
- six routing features and the `6 -> 32 -> 1` GELU routing MLP;
- fixed-score offline successor-pool mining;
- label-consistent routing supervision used by the revised method;
- exact-target supervision retained as a controlled diagnostic baseline;
- separate recovery-stage and routing-stage training objectives following the
  revised two-stage training procedure;
- quadratic corruption for recovery-model training;
- a DDCap adapter plus an explicit legacy-checkpoint compatibility mode;
- synthetic smoke tests and unit tests.

No test-gallery labels are used during routing-head training or inference.

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

## Label-consistent routing supervision

The revised method uses label-consistent supervision for routing-head training.

For a training query with relevance label `query_label`, we first construct the
set of training-side identifier buckets that contain at least one
retrieval-relevant image with the same label. When the benchmark protocol
excludes self-matches, the query image itself is excluded from this set.

```python
positive_mask = table.bucket_mask_for_label(
    training_labels,
    query_label,
    exclude_image_index=query_image_index,
)

mined = decoder.decode(
    prepared_query,
    recovery_model,
    target_bucket_mask=positive_mask,
)
```

A mined successor is positive if its compatible-bucket set intersects this
label-consistent bucket mask, and negative otherwise.

Only training-split labels are used to construct routing supervision. Test
gallery labels are never used for routing-head training, and test-time routing
requires no relevance label.

### Exact-target diagnostic

For the controlled exact-target ablation reported in the paper, a successor is
positive only when it can still reach the training image's assigned identifier
bucket:

```python
target_bucket = int(table.image_to_bucket[target_image_index])

diagnostic = decoder.decode(
    prepared_query,
    recovery_model,
    target_bucket=target_bucket,
)
```

Thus, label-consistent supervision is the default revised method, while
exact-target supervision is retained only as a controlled comparison.

Both modes expose mined pairs in `DecodeResult.route_pairs`. Every non-empty
successor pool receives equal total weight through
`DecodeResult.route_pair_weights`.

## Two-stage training

Training is explicitly separated into two stages.

### Stage 1: recovery model

The recovery model is trained with

\[
L_{\mathrm{rec}} = L_{\mathrm{tok}} + 0.5 L_{\mathrm{loc}}.
\]

After this stage, the recovery model is frozen.

### Stage 2: routing head

A single offline mining pass generates successor pools using the fixed
bootstrap score

\[
s_{\mathrm{boot}} = f_{\mathrm{path}} + f_{\mathrm{gap}} - f_{\mathrm{left}}.
\]

Label-consistent positive/negative successor pairs are then constructed from
the fixed mined pools. Only the `6 -> 32 -> 1` routing MLP is optimized with
the pairwise routing loss.

The pools are not re-mined after fitting the routing MLP, and there is no joint
recovery/routing optimization.

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

The four benchmark datasets are third-party public datasets and are therefore
not redistributed here. This repository provides the ABR-specific reference
implementation and tests. Paper-level retrieval numbers additionally require
the corresponding pretrained recovery model and exported identifier mapping.
