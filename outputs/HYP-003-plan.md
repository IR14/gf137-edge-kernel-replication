# HYP-003 Plan: Standard Quantized Baseline Sweep

This plan extends the `HYP-002` replication track without changing its result.
The purpose is to compare GF(137) against stricter engineering baselines across
multiple matrix sizes.

## Scope

The first pass remains dependency-light:

- reuse the deterministic vector generator from `scripts/edge_kernel_replication.py`;
- reuse the same C++ shared library;
- run three shapes: `small`, `hyp002`, and `medium`;
- compare equivalent GF(137) and `float32` modular kernels;
- include a non-equivalent plain `uint8_t` threshold control.

## Non-Goals

- Do not claim a physics result.
- Do not claim GF(137) is the fastest possible quantized inference format.
- Do not treat the plain `uint8_t` control as functionally equivalent.
- Do not tune constants after seeing the sweep.

## Expected Outcomes

Possible supportive result:

- `4x` storage reduction across all shapes;
- zero mismatches for equivalent rows;
- GF(137) faster than C++ `float32` modular arithmetic on most tested shapes.

Possible limiting result:

- plain `uint8_t` is faster than GF(137), because it removes residue reduction;
- GF(137) has a narrower role: compact modular/repair-friendly inference, not
  general-purpose fastest int8 inference.

## Output Files

The runner writes:

```text
outputs/quantized_baseline_sweep.json
outputs/quantized_baseline_sweep.md
```

The result should be reviewed before freezing HYP-003 as a CI gate.
