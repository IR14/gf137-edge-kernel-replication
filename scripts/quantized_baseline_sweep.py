#!/usr/bin/env python3
"""Run the HYP-003 quantized baseline sweep.

This script reuses the HYP-002 GF(137) kernel and adds a small deterministic
shape sweep.  The plain uint8 row is intentionally non-equivalent; it is a
runtime control, not a correctness baseline.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from edge_kernel_replication import (
    P_FIELD,
    ROOT,
    compile_library,
    load_library,
    make_vectors,
    model_bytes,
    predict_numpy_float32_modular,
    predict_numpy_gf137,
    run_cpp_float32_modular,
    run_cpp_gf137,
    run_cpp_plain_uint8,
    system_metadata,
)


RESULT_JSON = ROOT / "outputs" / "quantized_baseline_sweep.json"
RESULT_MD = ROOT / "outputs" / "quantized_baseline_sweep.md"


@dataclass(frozen=True)
class ShapeConfig:
    label: str
    rows: int
    input_width: int
    hidden_width: int


@dataclass(frozen=True)
class SweepRow:
    shape: str
    backend: str
    memory_bytes: int
    time_ms: float
    mismatch_count: int | None
    agreement_required: bool
    notes: str


SHAPES = [
    ShapeConfig("small", rows=128, input_width=32, hidden_width=16),
    ShapeConfig("hyp002", rows=256, input_width=64, hidden_width=32),
    ShapeConfig("medium", rows=512, input_width=128, hidden_width=64),
]


def median_runtime(fn, trials: int) -> tuple[float, np.ndarray]:
    timings = []
    result = None
    for _ in range(trials):
        start = time.perf_counter()
        result = fn()
        timings.append((time.perf_counter() - start) * 1000.0)
    assert result is not None
    return float(np.median(timings)), np.asarray(result, dtype=np.uint8)


def time_python_repeated(fn, repeats: int, trials: int) -> tuple[float, np.ndarray]:
    def repeated() -> np.ndarray:
        result = None
        for _ in range(repeats):
            result = fn()
        assert result is not None
        return result

    return median_runtime(repeated, trials)


def time_cpp_repeated(runner, lib, vectors, repeats: int, trials: int) -> tuple[float, np.ndarray]:
    timings = []
    result = None
    for _ in range(trials):
        elapsed, result = runner(lib, vectors, repeats)
        timings.append(elapsed)
    assert result is not None
    return float(np.median(timings)), result


def run_shape(shape: ShapeConfig, lib, seed: int, repeats: int, trials: int) -> dict:
    vectors = make_vectors(seed, shape.rows, shape.input_width, shape.hidden_width)
    reference = predict_numpy_gf137(vectors)
    rows: list[SweepRow] = []

    numpy_float_time, numpy_float_y = time_python_repeated(
        lambda: predict_numpy_float32_modular(vectors), repeats, trials
    )
    rows.append(
        SweepRow(
            shape=shape.label,
            backend="NumPy float32 modular",
            memory_bytes=model_bytes(vectors, 4),
            time_ms=numpy_float_time,
            mismatch_count=int(np.count_nonzero(numpy_float_y != reference)),
            agreement_required=True,
            notes="equivalent modular float32 reference",
        )
    )

    numpy_gf_time, numpy_gf_y = time_python_repeated(
        lambda: predict_numpy_gf137(vectors), repeats, trials
    )
    rows.append(
        SweepRow(
            shape=shape.label,
            backend="NumPy uint32 GF(137) reference",
            memory_bytes=model_bytes(vectors, 1),
            time_ms=numpy_gf_time,
            mismatch_count=int(np.count_nonzero(numpy_gf_y != reference)),
            agreement_required=True,
            notes="vectorized Python GF(137), uint8 model storage",
        )
    )

    cpp_float_time, cpp_float_y = time_cpp_repeated(
        run_cpp_float32_modular, lib, vectors, repeats, trials
    )
    rows.append(
        SweepRow(
            shape=shape.label,
            backend="C++ native O3 float32 modular",
            memory_bytes=model_bytes(vectors, 4),
            time_ms=cpp_float_time,
            mismatch_count=int(np.count_nonzero(cpp_float_y != reference)),
            agreement_required=True,
            notes="equivalent C++ float32 modular kernel",
        )
    )

    plain_time, plain_y = time_cpp_repeated(
        run_cpp_plain_uint8, lib, vectors, repeats, trials
    )
    rows.append(
        SweepRow(
            shape=shape.label,
            backend="C++ native O3 uint8 plain threshold",
            memory_bytes=model_bytes(vectors, 1),
            time_ms=plain_time,
            mismatch_count=None,
            agreement_required=False,
            notes="non-equivalent uint8 control without residue reduction",
        )
    )

    gf_time, gf_y = time_cpp_repeated(run_cpp_gf137, lib, vectors, repeats, trials)
    rows.append(
        SweepRow(
            shape=shape.label,
            backend="C++ native O3 GF(137)",
            memory_bytes=model_bytes(vectors, 1),
            time_ms=gf_time,
            mismatch_count=int(np.count_nonzero(gf_y != reference)),
            agreement_required=True,
            notes="optimized GF(137) kernel",
        )
    )

    by_backend = {row.backend: row for row in rows}
    float_cpp = by_backend["C++ native O3 float32 modular"]
    plain_cpp = by_backend["C++ native O3 uint8 plain threshold"]
    gf_cpp = by_backend["C++ native O3 GF(137)"]

    shape_summary = {
        "shape": asdict(shape),
        "storage_ratio_float32_to_gf137": float_cpp.memory_bytes / gf_cpp.memory_bytes,
        "speedup_cpp_float32_to_gf137": float_cpp.time_ms / gf_cpp.time_ms,
        "speed_ratio_plain_uint8_to_gf137": plain_cpp.time_ms / gf_cpp.time_ms,
        "equivalent_mismatches_zero": all(
            row.mismatch_count == 0 for row in rows if row.agreement_required
        ),
    }
    return {
        "rows": [asdict(row) for row in rows],
        "summary": shape_summary,
    }


def write_markdown(payload: dict) -> None:
    lines = [
        "# HYP-003 Quantized Baseline Sweep",
        "",
        "This report is generated by `scripts/quantized_baseline_sweep.py`.",
        "",
        "## Machine",
        "",
    ]
    for key, value in payload["metadata"].items():
        lines.append(f"- `{key}`: `{value}`")

    lines.extend(
        [
            "",
            "## Configuration",
            "",
            f"- `seed`: `{payload['config']['seed']}`",
            f"- `repeats`: `{payload['config']['repeats']}`",
            f"- `trials`: `{payload['config']['trials']}`",
            f"- `field`: `{payload['config']['field']}`",
            "",
            "## Results",
            "",
            "| Shape | Backend | Memory bytes | Runtime ms | Mismatches | Notes |",
            "|---|---|---:|---:|---:|---|",
        ]
    )
    for row in payload["rows"]:
        mismatches = "n/a" if row["mismatch_count"] is None else str(row["mismatch_count"])
        lines.append(
            f"| {row['shape']} | {row['backend']} | {row['memory_bytes']} | "
            f"{row['time_ms']:.6f} | {mismatches} | {row['notes']} |"
        )

    lines.extend(
        [
            "",
            "## Shape Summary",
            "",
            "| Shape | Storage ratio | C++ float32/GF137 speedup | Plain uint8/GF137 ratio | Agreement |",
            "|---|---:|---:|---:|---|",
        ]
    )
    for item in payload["shape_summaries"]:
        shape = item["shape"]["label"]
        agreement = "pass" if item["equivalent_mismatches_zero"] else "fail"
        lines.append(
            f"| {shape} | {item['storage_ratio_float32_to_gf137']:.3f}x | "
            f"{item['speedup_cpp_float32_to_gf137']:.3f}x | "
            f"{item['speed_ratio_plain_uint8_to_gf137']:.3f}x | {agreement} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            payload["interpretation"],
            "",
        ]
    )
    RESULT_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=5137)
    parser.add_argument("--repeats", type=int, default=500)
    parser.add_argument("--trials", type=int, default=3)
    args = parser.parse_args()

    native_flags = ["-O3", "-DNDEBUG", "-ffast-math", "-funroll-loops"]
    native_flags.append("-mcpu=native" if sys.platform == "darwin" else "-march=native")
    native_path = compile_library("hyp003_native_o3", native_flags)
    lib = load_library(native_path)

    all_rows = []
    shape_summaries = []
    for shape in SHAPES:
        result = run_shape(shape, lib, args.seed, args.repeats, args.trials)
        all_rows.extend(result["rows"])
        shape_summaries.append(result["summary"])

    pass_storage = all(item["storage_ratio_float32_to_gf137"] >= 3.9 for item in shape_summaries)
    pass_agreement = all(item["equivalent_mismatches_zero"] for item in shape_summaries)
    speed_wins = sum(item["speedup_cpp_float32_to_gf137"] > 1.0 for item in shape_summaries)
    pass_speed_proxy = speed_wins >= 2
    plain_faster_count = sum(item["speed_ratio_plain_uint8_to_gf137"] < 1.0 for item in shape_summaries)

    interpretation = (
        f"Storage pass={pass_storage}. Agreement pass={pass_agreement}. "
        f"GF(137) beats equivalent C++ float32 modular inference on {speed_wins}/"
        f"{len(shape_summaries)} shapes. Plain uint8 is faster on {plain_faster_count}/"
        f"{len(shape_summaries)} shapes, which is expected because it removes the "
        "GF(137) residue operation and computes a non-equivalent function. "
        "This sweep is a pre-freeze engineering audit and does not establish a "
        "fundamental-physics result."
    )

    payload = {
        "metadata": system_metadata(),
        "config": {
            "seed": args.seed,
            "repeats": args.repeats,
            "trials": args.trials,
            "field": P_FIELD,
        },
        "rows": all_rows,
        "shape_summaries": shape_summaries,
        "summary": {
            "pass_storage": pass_storage,
            "pass_agreement": pass_agreement,
            "pass_speed_proxy": pass_speed_proxy,
            "speed_wins_cpp_float32_shapes": speed_wins,
            "plain_uint8_faster_shapes": plain_faster_count,
        },
        "interpretation": interpretation,
    }

    RESULT_JSON.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    write_markdown(payload)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
