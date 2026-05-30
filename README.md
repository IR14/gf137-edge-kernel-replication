# Falsifiable Physics Lab

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

## Current Best Candidate

The strongest path is not a broad "theory of everything" claim.  The strongest
path is a blind residual prediction:

> Define one finite-field-derived correction, freeze it, and test whether it
> predicts an independent cosmological or particle-data residual that standard
> baselines do not explain.

If the prediction fails, record the failure.  If it survives, narrow the claim
and repeat on an independent dataset.

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
