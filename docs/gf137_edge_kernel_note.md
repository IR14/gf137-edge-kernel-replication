# GF(137) Edge-Kernel Replication Note

Date: 2026-05-31

## Scope

This note summarizes the current engineering replication track for the GF(137)
edge-inference kernel.  The result is limited to small dense kernels and fixed
deterministic test vectors.  It is not a fundamental-physics result and it is
not a claim that GF(137) replaces standard int8 inference.

The tested claim is narrower:

> A byte-valued GF(137) inference kernel can reduce model storage relative to an
> equivalent `float32` modular implementation, preserve exact output agreement,
> and in these small kernels often run faster than equivalent modular
> `float32` arithmetic.  Against standard int8-style inference, the speed result
> is hardware- and shape-dependent.

## Repository Evidence

The current evidence is split into five hypothesis tracks:

| Track | Purpose | Main output |
|---|---|---|
| HYP-002 | Single-shape replication against equivalent `float32` modular baselines | `outputs/edge_kernel_replication.md` |
| HYP-003 | Three-shape sweep with plain `uint8_t` control | `outputs/quantized_baseline_sweep.md` |
| HYP-004 | Industrial-style int8 proxy with int32 accumulation | `outputs/industrial_int8_baseline.md` |
| HYP-005 | ONNX Runtime int8 `MatMulInteger` external-runtime baseline | `outputs/onnxruntime_int8_baseline.md` |
| HYP-006 | GF(137) `RS(26,16)` erasure-repair audit | `outputs/gf137_erasure_repair.md` |

External checks:

| Environment | Evidence |
|---|---|
| GitHub Actions Ubuntu, HYP-002 | <https://github.com/IR14/gf137-edge-kernel-replication/actions/runs/26698801881> |
| GitHub Actions Ubuntu, HYP-003 | <https://github.com/IR14/gf137-edge-kernel-replication/actions/runs/26698321050> |
| GitHub Actions Ubuntu, HYP-004 | <https://github.com/IR14/gf137-edge-kernel-replication/actions/runs/26698801892> |
| GitHub Actions Ubuntu, HYP-005 | <https://github.com/IR14/gf137-edge-kernel-replication/actions/runs/26699031559> |
| GitHub Actions Ubuntu, HYP-006 | pending |
| Linux VPS x86_64, HYP-002 | `outputs/vps_vds2640757_expanded_edge_kernel_replication.md` |
| Linux VPS x86_64, HYP-003 | `outputs/vps_vds2640757_hyp003_quantized_baseline_sweep.md` |
| Linux VPS x86_64, HYP-004 | `outputs/vps_vds2640757_hyp004_industrial_int8_baseline.md` |
| Linux VPS x86_64, HYP-005 | `outputs/vps_vds2640757_hyp005_onnxruntime_int8_baseline.md` |

## HYP-002: Single-Shape Replication

HYP-002 uses one deterministic kernel shape:

| Rows | Input width | Hidden width | Seed | Field |
|---:|---:|---:|---:|---:|
| 256 | 64 | 32 | 5137 | 137 |

The expanded HYP-002 benchmark compares:

- NumPy `float32` modular inference;
- NumPy `uint32` GF(137) reference;
- C++ portable/native `float32` modular inference;
- C++ portable/native GF(137);
- C++ portable/native plain `uint8_t` threshold control.

The plain `uint8_t` control computes a different function and is not part of
the exact-agreement gate.

### HYP-002 Summary

| Environment | Storage ratio `float32/GF137` | Speedup `NumPy float32/GF137` | Speedup `C++ float32/GF137` | Exact agreement |
|---|---:|---:|---:|---|
| Apple ARM local | 4.000x | 4.585x | 5.198x | pass |
| Linux VPS x86_64 | 4.000x | 1.295x | 1.061x | pass |

HYP-002 passes on both local Apple ARM and Linux VPS x86_64.  The speedup is
large on Apple ARM and small but positive on the VPS.

## HYP-003: Three-Shape Quantized Sweep

HYP-003 repeats the audit across three deterministic shapes:

| Label | Rows | Input width | Hidden width |
|---|---:|---:|---:|
| small | 128 | 32 | 16 |
| hyp002 | 256 | 64 | 32 |
| medium | 512 | 128 | 64 |

The gate is intentionally modest:

- storage ratio at least `3.9x`;
- zero mismatches for equivalent rows;
- GF(137) faster than equivalent C++ `float32` modular inference on at least two
  of the three shapes.

### HYP-003 Summary

| Environment | Storage pass | Agreement pass | GF(137) wins vs C++ `float32` | Plain `uint8_t` faster |
|---|---|---|---:|---:|
| Apple ARM local | pass | pass | 3/3 | 1/3 |
| Linux VPS x86_64 | pass | pass | 3/3 | 3/3 |

This narrows the engineering claim.  GF(137) is stable against equivalent
`float32` modular inference across the tested shapes.  Plain `uint8_t` often
wins when the field-residue operation is removed, especially on the VPS.  That
is not a contradiction because the plain `uint8_t` row computes a different
function.

## HYP-004: Industrial Int8 Proxy

HYP-004 adds a hand-written int8 dense baseline with:

- signed int8 inputs and weights;
- int32 bias terms;
- int32 accumulation;
- thresholded hidden and output activations.

This is still not ONNX Runtime or TFLite.  It is a dependency-light proxy for a
standard quantized execution style.

### HYP-004 Summary

| Environment | Measurement pass | Agreement pass | Int8 memory profile pass | Standard int8 faster | GF(137) faster |
|---|---|---|---|---:|---:|
| Apple ARM local | pass | pass | pass | 0/3 | 3/3 |
| Linux VPS x86_64 | pass | pass | pass | 2/3 | 1/3 |

The HYP-004 result is the current boundary of the claim.  On Apple ARM, the
GF(137) implementation is faster than the hand-written int8 proxy in the three
tested shapes.  On the Linux VPS, the standard int8 proxy is faster in two of
the three shapes.

The conservative interpretation is:

> GF(137) is a compact exact modular-inference representation.  It should not
> be presented as a universal replacement for standard int8 kernels.  Its speed
> advantage depends on the machine, compiler, shape, and baseline.

## HYP-005: ONNX Runtime Int8 Baseline

HYP-005 adds an external runtime baseline using ONNX Runtime CPU execution with
an int8 `MatMulInteger` graph.  The ONNX row is not functionally equivalent to
GF(137), so it is not part of the exact-output agreement gate.  It is included
to test the deployment boundary against a real graph runtime.

### HYP-005 Summary

| Environment | Runtime available | Measurement pass | Agreement pass | ONNX int8 faster | GF(137) faster |
|---|---|---|---|---:|---:|
| Apple ARM local | pass | pass | pass | 2/3 | 1/3 |
| Linux VPS x86_64 | pass | pass | pass | 2/3 | 1/3 |

In the current local run, ONNX Runtime int8 is faster than GF(137) in two of
the three shapes and GF(137) is faster in one shape.  ONNX Runtime is also
faster than the hand-written C++ int8 proxy in all three local shapes.  The
Linux VPS run confirms the same ONNX-vs-GF(137) split, with ONNX faster than the
hand-written C++ int8 proxy in two of the three shapes.  This is the clearest
current evidence that deployment-speed claims must be limited and
shape-specific.

## HYP-006: GF(137) Erasure Repair

HYP-006 tests a different property from runtime speed.  It implements a small
Reed-Solomon-style `RS(26,16)` code over GF(137):

- 16 payload symbols;
- 26 encoded axes;
- 10 erased axes;
- polynomial interpolation over GF(137) for exact recovery.

Because 137 is prime and the 26 evaluation points are distinct, any 16
surviving axes determine the original degree `< 16` polynomial.  The audit
compares this against raw no-parity storage and 2x direct repetition.

### HYP-006 Summary

| Environment | GF(137) random repair | GF(137) adversarial repair | Raw random repair | 2x repetition random repair | Storage comparison |
|---|---:|---:|---:|---:|---|
| Apple ARM local | 2000/2000 | 6/6 | 0/2000 | 263/2000 | RS uses 26 symbols, repetition uses 32 |

This result supports an exact finite-field erasure-repair property.  It does
not show semantic compression, cryptographic security, or a physical law.

## Current Claim

The claim supported by the current artifacts is:

1. GF(137) model storage is `4x` smaller than equivalent `float32` modular
   storage for the tested kernel layouts.
2. Equivalent GF(137) rows match the frozen reference outputs exactly in the
   tested runs.
3. GF(137) beats equivalent C++ `float32` modular arithmetic in HYP-002 and in
   the three-shape HYP-003 sweep on both tested machines.
4. Against standard int8-style inference, runtime is not uniformly better.
   HYP-004 shows a shape- and hardware-dependent boundary.
5. Against ONNX Runtime int8, the local HYP-005 result is also shape-dependent,
   with ONNX faster on most tested shapes.
6. GF(137) supports exact erasure repair in the tested `RS(26,16)` construction
   after any 10 erased axes, with lower storage than 2x direct repetition.

## What This Does Not Show

This repository does not show that:

- GF(137) is faster than all int8 inference implementations;
- GF(137) is better than ONNX Runtime, TFLite, or vendor-optimized kernels;
- the result scales to large neural networks;
- the erasure-repair result is semantic compression or cryptographic security;
- the result has any direct implication for fundamental physics.

## Kill Conditions

The claim should be narrowed further if:

- an ONNX Runtime or TFLite int8 baseline dominates GF(137) on runtime,
  deployment simplicity, and memory on the target devices;
- exact output agreement fails under frozen vectors;
- GF(137) `RS(26,16)` recovery fails under any 10-axis erasure pattern;
- the speed advantage over equivalent `float32` modular arithmetic disappears
  on additional external machines;
- the benchmark depends on manual machine-specific tuning.

## Next Experiment

HYP-007 should connect the two engineering tracks:

1. Apply the `RS(26,16)` repair layer to model weights or intermediate symbols.
2. Measure accuracy and runtime after controlled weight erasures.
3. Compare against standard Reed-Solomon libraries and ordinary checkpoint
   replication.

The goal is not to protect the GF(137) claim.  The goal is to find whether
finite-field repair gives a useful engineering property that standard int8
deployment baselines do not already provide more simply.
