# HYP-002 Result Summary

Status: local pass, second-machine replication pass

## Freeze

Freeze manifest:

```text
outputs/HYP-002-freeze_manifest.json
```

Combined SHA-256:

```text
76456d36b6a85fec36fd40bb570043cc24971f95a160e5effece1f5f7d3f9855
```

## One-Command Runner

```bash
./scripts/run_hyp002.sh
```

Equivalent:

```bash
uv run --with numpy python scripts/edge_kernel_replication.py
```

## Local Result

Final report:

```text
outputs/edge_kernel_replication.md
outputs/edge_kernel_replication.json
```

Summary:

```text
Storage ratio float32/native GF(137): 4.000x
Runtime speedup float32/native GF(137): 4.796x
Mismatch count: 0
```

Pass flags:

```text
pass_storage = true
pass_agreement = true
pass_speed = true
```

## Failed Development Run

The first scalar implementation passed storage and agreement but failed speed.
That result is preserved:

```text
outputs/edge_kernel_replication_initial_scalar_fail.md
outputs/edge_kernel_replication_initial_scalar_fail.json
```

This matters because the claim is implementation-sensitive.  The passing result
requires the optimized row layout and ARM-friendly path in
`scripts/gf137_edge_kernel.cpp`.

## Interpretation

This result supports only the stated engineering replication claim on the local
machine.  It is not a fundamental-physics result.

The next required step is replication on a second machine or CI runner.

## VPS Replication

Second-machine result:

```text
outputs/vps_vds2640757_edge_kernel_replication.md
outputs/vps_vds2640757_edge_kernel_replication.json
```

Summary:

```text
Storage ratio float32/native GF(137): 4.000x
Runtime speedup float32/native GF(137): 1.154x
Mismatch count: 0
pass_storage = true
pass_agreement = true
pass_speed = true
```

The VPS result used Linux/x86_64, Python 3.12.3, NumPy 1.26.4, and g++ 13.3.0.
The newer NumPy 2.4.6 wheel failed on this host because the CPU did not support
the required X86_V2 baseline.  This is an environment caveat, not a benchmark
failure.

Next gate:

```text
CI or third-machine replication with a pinned fallback dependency set.
```

CI workflow:

```text
.github/workflows/hyp002-replication.yml
```

The workflow is intentionally strict.  It fails if any of the frozen pass flags
are false.
