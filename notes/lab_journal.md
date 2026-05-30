# Lab Journal

## 2026-05-31

Created a separate workspace for high-standard falsifiable work.

Decision:

- do not use broad claims as starting points;
- start from one frozen, externally testable residual prediction;
- require a kill condition before any validation run.

Open question:

- choose the first target dataset or laboratory benchmark.

## 2026-05-31: First Track Selection

Selected two tracks:

- `HYP-001`: CMB `N_eff` invariant as the physics track.
- `HYP-002`: GF(137) edge-kernel replication as the near-term engineering
  discipline check.

Reasoning:

- `N_eff` gives a fixed physical number and an explicit baseline.
- The effect size is small, so it is not a near-term discovery claim.
- The edge-kernel track is faster to replicate and tests whether the project's
  computational claims survive outside the author's machine.

## 2026-05-31: HYP-002 Local Run

Frozen `HYP-002` with `outputs/HYP-002-freeze_manifest.json`.

Initial scalar implementation:

- storage: passed;
- agreement: passed;
- speed: failed.

The failed scalar run is preserved under:

- `outputs/edge_kernel_replication_initial_scalar_fail.md`
- `outputs/edge_kernel_replication_initial_scalar_fail.json`

Optimized implementation:

- changed only the benchmark implementation, not the hypothesis;
- used row-wise hidden accumulation and ARM-friendly 32-hidden-unit path;
- storage ratio: `4.000x`;
- runtime speedup over NumPy float32 modular baseline: `4.796x` in the latest
  local run;
- mismatches: `0`.

Status:

- local pass;
- external replication still required.

## 2026-05-31: HYP-002 VPS Replication

Ran `HYP-002` on VPS `vds2640757`.

Environment:

- Linux x86_64;
- Python 3.12.3;
- g++ 13.3.0;
- NumPy 1.26.4.

NumPy 2.4.6 did not import on the VPS because the wheel requires X86_V2 CPU
baseline support.  The run was repeated with `numpy==1.26.4`.

Result:

- storage ratio: `4.000x`;
- runtime speedup over NumPy float32 modular baseline: `1.154x`;
- mismatches: `0`;
- pass flags: storage, agreement, and speed all true.

Conclusion:

- `HYP-002` passes the second-machine replication gate.
- The effect is hardware-sensitive: strong on Apple ARM, modest on the VPS.
- Next gate is CI or a third independent machine with dependency pinning.
