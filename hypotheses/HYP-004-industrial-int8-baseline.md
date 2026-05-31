# HYP-004: Industrial Int8 Baseline Proxy

Status: pre-freeze engineering audit

## Claim Under Test

GF(137) should not be described as a general replacement for standard int8
inference.  Its defensible niche is compact modular inference with exact
finite-field semantics.  A standard int8 dense kernel with int32 accumulation is
expected to be a strong raw-speed baseline, even when GF(137) keeps a compact
model representation.

This is an engineering audit, not a fundamental-physics claim.

## Why This Exists

`HYP-002` showed a reproducible speed and storage advantage over equivalent
`float32` modular arithmetic.  `HYP-003` showed that result across three matrix
sizes and recorded the plain `uint8_t` limitation.  `HYP-004` moves one step
closer to deployment practice by adding a hand-written int8 dense baseline with
int32 bias and accumulation.

## Baselines

Equivalent GF(137) rows:

- C++ native `float32` modular inference;
- C++ native GF(137) inference.

Industrial proxy row:

- C++ native int8 dense inference with int32 bias and accumulator.

The int8 row computes a different function.  It is therefore not part of the
exact-output agreement gate.  It measures a standard quantized execution style:
signed int8 operands, int32 accumulation, thresholded activations, and byte-size
weights.

## Initial Sweep Grid

The sweep uses the same deterministic shape grid as `HYP-003`:

| Label | Rows | Input width | Hidden width |
|---|---:|---:|---:|
| small | 128 | 32 | 16 |
| hyp002 | 256 | 64 | 32 |
| medium | 512 | 128 | 64 |

## Metrics

For each shape:

- GF(137) model bytes;
- int8 model bytes including int32 bias terms;
- C++ GF(137) runtime;
- C++ int8 runtime;
- exact mismatch count for equivalent GF(137) rows;
- runtime ratio `int8 / GF(137)`;
- memory ratio `int8 / GF(137)`.

## Pass / Downgrade Criteria

The audit passes if:

- equivalent GF(137) rows have zero mismatches;
- all int8 timings are finite and positive;
- int8 model storage stays within `1.25x` of GF(137) model storage for each
  tested shape.

The GF(137) speed claim is downgraded if:

- int8 is faster on all shapes.  In that case the result should be described as
  a compact modular-arithmetic kernel, not a fastest-general-int8 kernel.

## Next Extension

If this proxy result is useful, add optional external baselines:

- ONNX Runtime quantized linear layers;
- TFLite / TFLite Micro int8 kernels;
- ARM NEON and x86 SIMD dot-product intrinsics for the int8 proxy.
