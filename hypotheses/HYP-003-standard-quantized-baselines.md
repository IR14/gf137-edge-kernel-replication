# HYP-003: Standard Quantized Baseline Sweep

Status: pre-freeze engineering audit

## Claim Under Test

The GF(137) edge kernel should keep the `4x` model-storage advantage relative
to an equivalent `float32` modular kernel across several small dense inference
sizes.  Runtime may improve relative to equivalent `float32` modular arithmetic,
but it is not expected to dominate every plain `uint8_t` kernel that removes the
field-residue operation.

This is an engineering audit, not a fundamental-physics claim.

## Why This Exists

`HYP-002` passed on Apple ARM, Linux VPS x86_64, and GitHub Actions Ubuntu.
However, it used one fixed matrix shape.  `HYP-003` asks whether that result is
stable across a small shape sweep and whether the advantage survives a stricter
comparison against a C++ `float32` modular baseline.

## Baselines

Equivalent baselines:

- NumPy `float32` modular inference;
- NumPy `uint32` GF(137) reference;
- C++ native `float32` modular inference;
- C++ native GF(137) inference.

Non-equivalent control:

- C++ native plain `uint8_t` threshold inference without GF(137) residue
  reductions.

The plain `uint8_t` row is not allowed to satisfy output-agreement checks
because it computes a different function.  It is included only to measure how
much runtime is spent on the field operation rather than on loop and memory
overhead.

## Initial Sweep Grid

The initial sweep uses deterministic vectors with seed `5137`:

| Label | Rows | Input width | Hidden width |
|---|---:|---:|---:|
| small | 128 | 32 | 16 |
| hyp002 | 256 | 64 | 32 |
| medium | 512 | 128 | 64 |

## Metrics

For each shape:

- model storage bytes for `float32` and GF(137);
- runtime for each backend;
- exact mismatch count against the GF(137) reference for equivalent rows;
- storage ratio `float32 / GF(137)`;
- speedup `C++ float32 modular / C++ GF(137)`;
- runtime ratio `plain uint8 / GF(137)`.

## Pass / Downgrade Criteria

The preliminary sweep is considered supportive if:

- storage ratio is at least `3.9x` for every shape;
- equivalent rows have zero mismatches for every shape;
- C++ GF(137) is faster than C++ `float32` modular inference for at least two
  of the three initial shapes.

The claim is downgraded if:

- exact agreement fails on any equivalent row;
- GF(137) loses the storage advantage;
- a later standard int8/TFLite/ONNX backend dominates GF(137) on both runtime
  and deployment simplicity for the relevant target devices.

## Next Extension

If this proxy sweep is stable, add optional baselines:

- ONNX Runtime dynamic/static quantized dense layers;
- TFLite or TFLite Micro style int8 kernels;
- a hand-written int8 dot-product kernel with saturated arithmetic.
