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

## Energy-Cost Accounting Scope

HYP-007 also reports a limited energy-cost proxy.  The proxy compares the
storage-bit lower bound for FP32 checkpoints, byte-valued GF(137) checkpoints,
and `RS(26,16)`-protected GF(137) checkpoints by using Landauer's bound

```text
E_min >= k_B T ln(2)
```

per erased bit at room temperature.  This is a lower-bound bit-accounting
calculation only.  It is not a measured hardware power result, and it does not
prove that physical systems implement GF(137).

For the dense kernels in this repository the computational complexity is
reported as follows:

- a general square dense matrix multiply scales as `O(N^3)`;
- the tested two-layer edge kernel scales as `O(R K H + R H)`;
- GF(137) arithmetic does not change that asymptotic shape by itself, but it
  changes the representation from FP32 values to byte-valued residues and
  replaces floating-point modular arithmetic with integer residue operations;
- `mac137` denotes the inference hot-path operation: integer multiply-add plus
  Barrett reduction modulo 137;
- `inv137` denotes multiplicative inversion in GF(137), used by repair
  interpolation and not by the inference hot path.

The Schwinger marker

```text
s = alpha / (2 pi)
```

is reported as the dimensionless factor `1-s`.  It is not used as a pass
criterion and is not treated as evidence for an exact physical joule reduction.

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
- FP32, raw GF(137), repaired GF(137), and repetition storage in bits;
- Landauer lower-bound storage proxy for FP32 and repaired GF(137);
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
- measured joule savings on real hardware;
- an exact `1-s` reduction in physical energy;
- improved model accuracy;
- a fundamental law of nature.

It claims only exact checkpoint repair for the tested GF(137) model blocks.
