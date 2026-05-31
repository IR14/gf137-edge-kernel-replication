# HYP-007: Repair-Aware GF(137) Checkpoint Baseline

Status: pre-freeze checkpoint-repair audit

## Claim Under Test

The GF(137) repair layer from HYP-006 can be applied to a real checkpoint from
the small GF(137) inference kernel.  If each 16-symbol model block is encoded
as `RS(26,16)` over GF(137), the model weights can be recovered exactly after
10 erased axes per block, and the repaired model produces the same predictions
as the original checkpoint.

This is an engineering claim about recoverable finite-field checkpoints.  It is
not a claim about compression, cryptography, or fundamental physics.

## Why This Exists

HYP-006 proved the erasure-repair property on isolated payload blocks.  HYP-007
tests whether that property survives contact with the actual model artifacts
used by the GF(137) edge-kernel benchmark: `w1`, `b1`, `w2`, and `b2`.

## Baselines

Repair scheme:

- GF(137) `RS(26,16)` checkpoint blocks.

Controls:

- raw checkpoint blocks with no parity;
- 2x direct repetition blocks.

The controls are intentionally simple.  They test whether the algebraic repair
layer is doing work beyond ordinary storage or duplication.

## Metrics

For each model shape:

- raw checkpoint size in symbols;
- number of 16-symbol blocks;
- storage symbols for GF(137) repair and 2x repetition;
- exact recovery under deterministic erasure patterns;
- exact recovery under random erasure trials;
- prediction mismatches after GF(137) repair;
- whether raw and repetition controls fail under the same erasure budget.

## Pass / Downgrade Criteria

The audit passes if:

- every GF(137)-repaired checkpoint matches the original model symbols exactly;
- every GF(137)-repaired checkpoint produces zero prediction mismatches;
- raw no-parity checkpoint storage fails under at least one tested erasure
  condition;
- 2x repetition is marked as larger than GF(137) repair and fails under paired
  adversarial erasures.

The claim is downgraded if:

- any GF(137)-repaired model differs from the original checkpoint;
- any GF(137)-repaired model changes predictions;
- a simpler equal-or-smaller control gives the same tested guarantee;
- the repair layer requires post-hoc shape-specific tuning.

## Non-Claims

This audit does not claim:

- faster inference;
- semantic compression;
- cryptographic security;
- improved model accuracy;
- a fundamental law of nature.

It claims only exact checkpoint repair for the tested GF(137) model blocks.
