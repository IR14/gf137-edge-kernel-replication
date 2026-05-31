# Reproducibility Checklist

This checklist describes how to reproduce the `v0.1.0` GF(137) edge-kernel
baseline package.

## Requirements

Required:

- Python 3.12 or newer;
- `numpy`;
- C++17 compiler (`clang++` or `g++`);
- POSIX shell for runner scripts.

Required for HYP-005:

- `onnx`;
- `onnxruntime`;
- ONNX Runtime CPU provider.

The runner scripts install Python dependencies through `uv` when available.
Without `uv`, they create or reuse `.venv` and install dependencies with `pip`.

## One-Command Runs

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

## Linux x86_64 Compatibility Run

Some VPS CPUs do not support newer NumPy wheel CPU baselines.  Use NumPy
`1.26.4` if import fails with an `X86_V2` baseline error:

```bash
NUMPY_SPEC="numpy==1.26.4" ./scripts/run_hyp002.sh
NUMPY_SPEC="numpy==1.26.4" ./scripts/run_hyp003.sh
NUMPY_SPEC="numpy==1.26.4" ./scripts/run_hyp004.sh
NUMPY_SPEC="numpy==1.26.4" ONNX_SPEC="onnx" ONNXRUNTIME_SPEC="onnxruntime" ./scripts/run_hyp005.sh
```

## Expected Output Files

| Track | JSON | Markdown |
|---|---|---|
| HYP-002 | `outputs/edge_kernel_replication.json` | `outputs/edge_kernel_replication.md` |
| HYP-003 | `outputs/quantized_baseline_sweep.json` | `outputs/quantized_baseline_sweep.md` |
| HYP-004 | `outputs/industrial_int8_baseline.json` | `outputs/industrial_int8_baseline.md` |
| HYP-005 | `outputs/onnxruntime_int8_baseline.json` | `outputs/onnxruntime_int8_baseline.md` |

## Pass Flags

HYP-002 must report:

```json
{
  "pass_storage": true,
  "pass_agreement": true,
  "pass_speed": true
}
```

HYP-003 must report:

```json
{
  "pass_storage": true,
  "pass_agreement": true,
  "pass_speed_proxy": true
}
```

HYP-004 must report:

```json
{
  "pass_measurement": true,
  "pass_agreement": true,
  "pass_int8_memory_profile": true
}
```

HYP-005 must report:

```json
{
  "pass_runtime_available": true,
  "pass_measurement": true,
  "pass_agreement": true,
  "pass_int8_memory_profile": true
}
```

Runtime ratios are expected to vary by machine.  The release claim does not
require GF(137) to beat int8 or ONNX Runtime on every shape.

## External Replication

The repository includes previously recorded Apple ARM, Linux VPS x86_64, and
GitHub Actions Ubuntu artifacts.  Use the files under `outputs/` for comparison
only; rerunning the scripts will overwrite the generic output files.

## Integrity Rules

- Do not edit generated result files by hand.
- Do not tune constants after seeing a benchmark result.
- Treat faster standard int8 or ONNX Runtime results as claim-narrowing
  evidence, not as a failure to hide.
- Preserve machine metadata in JSON and Markdown outputs.
