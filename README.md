# Falsifiable Physics Lab

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20469661.svg)](https://doi.org/10.5281/zenodo.20469661)

This directory is an intentionally separate workspace for high-risk,
high-standard research ideas.

The goal is not to advertise a grand claim.  The goal is to make every strong
idea precise enough to fail, survive external tests, and be compared against
standard models without rhetorical padding.

## North Star

Find one falsifiable prediction that is:

- new relative to the archived e-5-137 / GF(137) notes;
- specified before looking at the target validation data;
- numerically precise enough to fail;
- cheaper to test than to argue about;
- reproducible by an external reader from public data or a simple lab setup.

## Current Best Candidates

The strongest physics path is not a broad "theory of everything" claim.  It is
a blind residual prediction:

> Define one finite-field-derived correction, freeze it, and test whether it
> predicts an independent cosmological or particle-data residual that standard
> baselines do not explain.

If the prediction fails, record the failure.  If it survives, narrow the claim
and repeat on an independent dataset.

The strongest engineering path is separate: test whether GF(137) gives a useful
recoverability property under erasures, independent of raw runtime speed.  That
track is now represented by `HYP-006`.

## Technical Note

The current GF(137) edge-kernel replication result is summarized in
`docs/gf137_edge_kernel_note.md`.  It is the best entry point for the engineering
track because it separates supported claims, limitations, kill conditions, and
the next finite-field repair baseline.

## Release Package

This repository is ready for a `v0.1.0` release as:

```text
v0.1.0 - GF(137) Edge-Kernel Replication Baselines
```

Release files:

- `CITATION.cff` - citation metadata;
- `RELEASE_NOTES.md` - release scope and supported claims;
- `REPRODUCIBILITY.md` - local, VPS, and CI reproduction checklist.

## Current Fast Replication Track

`HYP-002` tests whether the GF(137) edge-inference storage/runtime claim
survives a one-command reproduction.  The benchmark now includes equivalent
NumPy and C++ `float32` modular baselines, plus a non-equivalent plain `uint8_t`
control for loop overhead.

Run:

```bash
./scripts/run_hyp002.sh
```

Outputs:

```text
outputs/edge_kernel_replication.md
outputs/edge_kernel_replication.json
```

The current local Apple ARM run reports `4.000x` storage reduction,
`4.585x` speedup against NumPy `float32` modular inference, and `5.198x`
speedup against C++ `float32` modular inference, with zero mismatches in all
equivalent rows.

The local result is not enough.  A Linux/x86_64 VPS replication also passed
under the expanded baseline matrix: `4.000x` storage reduction, `1.295x`
speedup against NumPy `float32` modular inference, `1.061x` speedup against
C++ `float32` modular inference, and zero mismatches in all equivalent rows.
See `outputs/vps_vds2640757_expanded_edge_kernel_replication.md`.

GitHub Actions also passed on Ubuntu for the expanded baseline matrix:
[`run 26698037019`](https://github.com/IR14/gf137-edge-kernel-replication/actions/runs/26698037019).

Old x86 note:

```bash
NUMPY_SPEC="numpy==1.26.4" ./scripts/run_hyp002.sh
```

Use this fallback if a newer NumPy wheel fails because the CPU lacks X86_V2
baseline support.

CI:

```text
.github/workflows/hyp002-replication.yml
```

The CI job runs the same benchmark on `ubuntu-24.04`, asserts the three pass
flags, and uploads the JSON/Markdown report as an artifact.

## Current Baseline Expansion

`HYP-003` expands the engineering audit from one matrix size to a three-shape
sweep against stricter quantized baselines.  It keeps the equivalent C++
`float32` modular baseline and adds a non-equivalent plain `uint8_t` control to
show the cost of the GF(137) residue operation.

Run:

```bash
./scripts/run_hyp003.sh
python scripts/assert_hyp003_pass.py
```

Outputs:

```text
outputs/quantized_baseline_sweep.md
outputs/quantized_baseline_sweep.json
```

The current Apple ARM sweep reports `4.000x` storage reduction for all shapes,
zero mismatches in equivalent rows, and GF(137) speedups over C++ `float32`
modular inference on `3/3` shapes.  Plain `uint8_t` is faster on `1/3` shapes,
which is recorded as an expected limitation rather than hidden.

The Ubuntu CI sweep also passed:
[`run 26698321050`](https://github.com/IR14/gf137-edge-kernel-replication/actions/runs/26698321050).

## Current Industrial Baseline

`HYP-004` adds a hand-written C++ int8 dense-inference proxy with int32 bias and
accumulation.  This row is not functionally equivalent to GF(137); it is a
standard quantized baseline for raw deployment-style inference.

Run:

```bash
./scripts/run_hyp004.sh
python scripts/assert_hyp004_pass.py
```

Outputs:

```text
outputs/industrial_int8_baseline.md
outputs/industrial_int8_baseline.json
```

The current Apple ARM audit reports valid measurements, zero mismatches on the
equivalent GF(137) rows, and int8 model storage within `1.25x` of GF(137) on all
three shapes.  GF(137) is faster than the hand-written int8 proxy on `3/3`
shapes in this local run, but this is explicitly marked for external
replication before any stronger speed claim.

The Ubuntu CI audit also passed:
[`run 26698801892`](https://github.com/IR14/gf137-edge-kernel-replication/actions/runs/26698801892).

The Linux/x86_64 VPS audit reports the same measurement and agreement passes,
with a shape-dependent speed result: the int8 proxy is faster on `2/3` shapes
and GF(137) is faster on `1/3` shapes.  This narrows the claim: GF(137) remains
compact and exact for modular inference, but it should not be presented as a
universal replacement for standard int8 kernels.

## Current External Runtime Baseline

`HYP-005` adds ONNX Runtime with a CPU `MatMulInteger` int8 graph.  This is the
first external-runtime baseline; it includes runtime graph execution overhead
and provider-specific CPU kernels.

Run:

```bash
./scripts/run_hyp005.sh
python scripts/assert_hyp005_pass.py
```

Outputs:

```text
outputs/onnxruntime_int8_baseline.md
outputs/onnxruntime_int8_baseline.json
```

The current Apple ARM audit reports valid measurements, zero mismatches on the
equivalent GF(137) rows, and int8 model storage within `1.25x` of GF(137).  ONNX
Runtime int8 is faster than GF(137) on `2/3` shapes and GF(137) is faster on
`1/3` shapes.  ONNX Runtime also beats the hand-written C++ int8 proxy on `3/3`
shapes in this run.  This further narrows the speed claim to finite-field
semantics and shape-specific performance, not general int8 deployment speed.

The Ubuntu CI audit also passed:
[`run 26699031559`](https://github.com/IR14/gf137-edge-kernel-replication/actions/runs/26699031559).

The Linux/x86_64 VPS audit confirms the same main boundary: ONNX Runtime int8
is faster than GF(137) on `2/3` shapes and GF(137) is faster on `1/3` shapes.
ONNX Runtime is faster than the hand-written C++ int8 proxy on `2/3` shapes on
that VPS run.

## Current Repair Baseline

`HYP-006` moves away from raw speed claims and tests a different finite-field
property: exact erasure repair.  It implements a small Reed-Solomon-style
`RS(26,16)` code over GF(137), where 16 payload symbols are encoded into 26
axes and recovered after 10 erased axes.

Run:

```bash
./scripts/run_hyp006.sh
python scripts/assert_hyp006_pass.py
```

Outputs:

```text
outputs/gf137_erasure_repair.md
outputs/gf137_erasure_repair.json
```

The current Apple ARM audit reports exact GF(137) recovery on `2000/2000`
random erasure trials and `6/6` deterministic adversarial erasure patterns.
The raw no-parity control fails under the same erasure budget, and 2x direct
repetition uses more storage while remaining unsafe against paired adversarial
erasures.  This supports an erasure-repair claim only; it is not semantic
compression, cryptography, or a physics result.

The Ubuntu CI audit also passed:
[`run 26710817193`](https://github.com/IR14/gf137-edge-kernel-replication/actions/runs/26710817193).

## Current Checkpoint Repair Track

`HYP-007` applies the same `RS(26,16)` repair layer to actual GF(137) model
checkpoints from the edge-kernel benchmark.  It flattens `w1`, `b1`, `w2`, and
`b2` into 16-symbol blocks, encodes each block into 26 axes, erases 10 axes per
block, repairs the checkpoint, and verifies that predictions remain identical.

Run:

```bash
NUMPY_SPEC="numpy==1.26.4" ./scripts/run_hyp007.sh
python scripts/assert_hyp007_pass.py
```

Outputs:

```text
outputs/repair_aware_checkpoint.md
outputs/repair_aware_checkpoint.json
```

The current Apple ARM audit reports byte-exact GF(137) checkpoint repair on all
three shapes.  The repaired models have `0` prediction mismatches.  Raw storage
and 2x direct repetition are included only as controls: raw storage fails under
the erasure budget, and repetition uses more storage while failing paired
adversarial erasures.

## Ground Rules

- No post-hoc fitting after seeing the validation result.
- No new constants unless they are fixed before the test.
- No rhetorical claims stronger than the residual table.
- Every hypothesis must have a kill condition.
- Every positive result must include the boring null checks.

## Directory Map

- `hypotheses/` - frozen hypotheses and their failure criteria.
- `protocols/` - replication and blind-test rules.
- `notes/` - lab journal and decision history.
- `data/` - small public-data pointers or manifests, not bulk data.
- `outputs/` - generated figures, reports, and result tables.
- `scripts/` - scripts that run one frozen test at a time.

## First Rule

The first serious milestone is not publication.  It is a result that someone
else can reproduce and criticize without asking what the author meant.
