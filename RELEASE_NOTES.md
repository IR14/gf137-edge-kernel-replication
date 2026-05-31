# Release Notes

## v0.1.0 - GF(137) Edge-Kernel Replication Baselines

Release date: 2026-05-31

DOI: <https://doi.org/10.5281/zenodo.20469661>

This is the first release-ready replication package for the GF(137)
edge-kernel audit.

## Included Tracks

| Track | Description | Primary report |
|---|---|---|
| HYP-002 | Single-shape GF(137) replication against equivalent `float32` modular baselines | `outputs/edge_kernel_replication.md` |
| HYP-003 | Three-shape sweep against `float32` modular and plain `uint8_t` controls | `outputs/quantized_baseline_sweep.md` |
| HYP-004 | Hand-written C++ int8 dense baseline with int32 accumulation | `outputs/industrial_int8_baseline.md` |
| HYP-005 | ONNX Runtime CPU `MatMulInteger` int8 external-runtime baseline | `outputs/onnxruntime_int8_baseline.md` |

## External Replication Artifacts

| Environment | Reports |
|---|---|
| Apple ARM local | `outputs/edge_kernel_replication.md`, `outputs/quantized_baseline_sweep.md`, `outputs/industrial_int8_baseline.md`, `outputs/onnxruntime_int8_baseline.md` |
| Linux VPS x86_64 | `outputs/vps_vds2640757_expanded_edge_kernel_replication.md`, `outputs/vps_vds2640757_hyp003_quantized_baseline_sweep.md`, `outputs/vps_vds2640757_hyp004_industrial_int8_baseline.md`, `outputs/vps_vds2640757_hyp005_onnxruntime_int8_baseline.md` |
| GitHub Actions Ubuntu | HYP-002, HYP-003, HYP-004, and HYP-005 workflow runs linked from `README.md` and `docs/gf137_edge_kernel_note.md` |

## Supported Claim

The artifacts support a narrow engineering claim:

1. GF(137) model storage is `4x` smaller than equivalent `float32` modular
   storage for the tested dense kernel layouts.
2. Equivalent GF(137) rows match the frozen reference outputs exactly in the
   tested runs.
3. GF(137) is consistently faster than equivalent C++ `float32` modular
   arithmetic in the tested HYP-002 and HYP-003 layouts.
4. Against standard int8-style inference, speed is shape- and
   hardware-dependent.  ONNX Runtime int8 is faster than GF(137) on `2/3`
   tested shapes in both Apple ARM and Linux VPS runs.

## Explicit Non-Claims

This release does not claim that:

- GF(137) is faster than all int8 inference implementations;
- GF(137) replaces ONNX Runtime, TFLite, or vendor-optimized kernels;
- the result scales to large neural networks;
- the benchmark establishes any fundamental-physics result.

## Reproduction

Run the full local audit:

```bash
./scripts/run_hyp002.sh
python scripts/assert_hyp002_pass.py

./scripts/run_hyp003.sh
python scripts/assert_hyp003_pass.py

./scripts/run_hyp004.sh
python scripts/assert_hyp004_pass.py

./scripts/run_hyp005.sh
python scripts/assert_hyp005_pass.py
```

For older Linux x86_64 CPUs whose NumPy wheels may lack the required CPU
baseline, use:

```bash
NUMPY_SPEC="numpy==1.26.4" ./scripts/run_hyp002.sh
NUMPY_SPEC="numpy==1.26.4" ./scripts/run_hyp003.sh
NUMPY_SPEC="numpy==1.26.4" ./scripts/run_hyp004.sh
NUMPY_SPEC="numpy==1.26.4" ONNX_SPEC="onnx" ONNXRUNTIME_SPEC="onnxruntime" ./scripts/run_hyp005.sh
```

## Next Work

The next audit should add a TFLite or TFLite Micro baseline.  That should be
treated as a boundary test, not as an attempt to preserve the GF(137) speed
claim.
