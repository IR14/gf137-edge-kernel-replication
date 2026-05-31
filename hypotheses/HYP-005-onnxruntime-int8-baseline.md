# HYP-005: ONNX Runtime Int8 Baseline

Status: pre-freeze external-runtime audit

## Claim Under Test

GF(137) should be evaluated against a real external inference runtime before
any deployment claim is made.  HYP-005 adds ONNX Runtime with an int8
`MatMulInteger` graph as the first external baseline.

This is an engineering audit, not a fundamental-physics claim.

## Why This Exists

`HYP-004` introduced a hand-written int8 dense proxy.  That is useful for a
dependency-light local check, but it is not an industry runtime.  ONNX Runtime
is a better next boundary because it includes graph execution overhead, runtime
initialization choices, and provider-specific CPU kernels.

## Baselines

The audit compares:

- C++ native GF(137);
- C++ native int8 dense proxy with int32 accumulation;
- ONNX Runtime CPU `MatMulInteger` int8 graph.

The ONNX row computes a standard int8 dense-threshold function.  It is not
functionally equivalent to GF(137), so it is not part of the exact-output
agreement gate.

## Initial Sweep Grid

The sweep uses the same deterministic shape grid as HYP-003 and HYP-004:

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
- C++ int8 proxy runtime;
- ONNX Runtime int8 runtime;
- exact mismatch count for equivalent GF(137) rows;
- runtime ratio `ONNX int8 / GF(137)`;
- runtime ratio `ONNX int8 / C++ int8`.

## Pass / Downgrade Criteria

The audit passes if:

- ONNX Runtime imports and executes the graph;
- all timings are finite and positive;
- equivalent GF(137) rows have zero mismatches;
- int8 model storage stays within `1.25x` of GF(137) model storage.

The GF(137) deployment-speed claim is narrowed if:

- ONNX Runtime int8 is faster than GF(137) on most or all shapes.

The ONNX baseline is not a failure condition by itself.  It is a boundary test.

## Next Extension

If ONNX Runtime runs cleanly on Apple ARM, Linux VPS, and GitHub Actions, the
next step is a TFLite or TFLite Micro baseline.  If ONNX installation becomes
fragile across platforms, HYP-005 should remain an optional external-runtime
audit rather than a required core gate.
