#!/usr/bin/env python3
"""Run the HYP-004 industrial int8 baseline proxy.

The int8 baseline is intentionally non-equivalent to GF(137).  It measures a
standard quantized dense execution style: int8 operands, int32 bias and
accumulation, and thresholded hidden/output activations.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from edge_kernel_replication import (
    ROOT,
    compile_library,
    load_library,
    make_vectors,
    model_bytes,
    predict_numpy_gf137,
    run_cpp_float32_modular,
    run_cpp_gf137,
    system_metadata,
)
from quantized_baseline_sweep import SHAPES, ShapeConfig, time_cpp_repeated


INT8_SOURCE = ROOT / "scripts" / "standard_int8_kernel.cpp"
BUILD_DIR = ROOT / "outputs" / "build"
RESULT_JSON = ROOT / "outputs" / "industrial_int8_baseline.json"
RESULT_MD = ROOT / "outputs" / "industrial_int8_baseline.md"


@dataclass(frozen=True)
class Int8Vectors:
    x: np.ndarray
    w1: np.ndarray
    b1: np.ndarray
    w2: np.ndarray
    b2: np.ndarray


@dataclass(frozen=True)
class BaselineRow:
    shape: str
    backend: str
    memory_bytes: int
    time_ms: float
    mismatch_count: int | None
    agreement_required: bool
    notes: str


def library_name(label: str) -> str:
    suffix = ".dylib" if sys.platform == "darwin" else ".so"
    return f"libstandard_int8_{label}{suffix}"


def compile_int8_library(label: str, flags: list[str]) -> Path:
    compiler = shutil.which("clang++") or shutil.which("g++")
    if compiler is None:
        raise RuntimeError("No C++ compiler found. Install clang++ or g++.")
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    output = BUILD_DIR / library_name(label)
    shared_flag = "-dynamiclib" if sys.platform == "darwin" else "-shared"
    command = [
        compiler,
        "-std=c++17",
        "-fPIC",
        shared_flag,
        *flags,
        str(INT8_SOURCE),
        "-o",
        str(output),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    (BUILD_DIR / f"compile_{label}.log").write_text(
        "\n".join(
            [
                "command: " + " ".join(command),
                f"returncode: {completed.returncode}",
                "",
                "[stdout]",
                completed.stdout,
                "[stderr]",
                completed.stderr,
            ]
        ),
        encoding="utf-8",
    )
    if completed.returncode != 0:
        raise RuntimeError(f"C++ compile failed for {label}")
    return output


def load_int8_library(path: Path):
    lib = ctypes.CDLL(str(path))
    i8 = ctypes.POINTER(ctypes.c_int8)
    i32 = ctypes.POINTER(ctypes.c_int32)
    u8 = ctypes.POINTER(ctypes.c_uint8)
    lib.standard_int8_predict_repeated.argtypes = [
        i8,
        i8,
        i32,
        i8,
        i32,
        u8,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_int,
    ]
    lib.standard_int8_predict_repeated.restype = None
    return lib


def ptr_i8(array: np.ndarray):
    return array.ctypes.data_as(ctypes.POINTER(ctypes.c_int8))


def ptr_i32(array: np.ndarray):
    return array.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))


def ptr_u8(array: np.ndarray):
    return array.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))


def make_int8_vectors(seed: int, rows: int, k_width: int, h_width: int) -> Int8Vectors:
    rng = np.random.default_rng(seed + 4004)
    return Int8Vectors(
        x=rng.integers(-8, 8, size=(rows, k_width), dtype=np.int8),
        w1=rng.integers(-64, 64, size=(k_width, h_width), dtype=np.int8),
        b1=rng.integers(-512, 512, size=(h_width,), dtype=np.int32),
        w2=rng.integers(-64, 64, size=(h_width,), dtype=np.int8),
        b2=rng.integers(-512, 512, size=(1,), dtype=np.int32),
    )


def int8_model_bytes(vectors: Int8Vectors) -> int:
    return int(
        vectors.w1.size
        + vectors.w2.size
        + 4 * vectors.b1.size
        + 4 * vectors.b2.size
    )


def int8_thresholds(k_width: int, h_width: int) -> tuple[int, int]:
    return 0, 0


def run_int8_cpp(lib, vectors: Int8Vectors, repeats: int) -> tuple[float, np.ndarray]:
    x = np.ascontiguousarray(vectors.x, dtype=np.int8)
    w1 = np.ascontiguousarray(vectors.w1, dtype=np.int8)
    b1 = np.ascontiguousarray(vectors.b1, dtype=np.int32)
    w2 = np.ascontiguousarray(vectors.w2, dtype=np.int8)
    b2 = np.ascontiguousarray(vectors.b2, dtype=np.int32)
    y = np.empty(x.shape[0], dtype=np.uint8)
    hidden_threshold, output_threshold = int8_thresholds(x.shape[1], w1.shape[1])
    start = time.perf_counter()
    lib.standard_int8_predict_repeated(
        ptr_i8(x),
        ptr_i8(w1),
        ptr_i32(b1),
        ptr_i8(w2),
        ptr_i32(b2),
        ptr_u8(y),
        x.shape[0],
        x.shape[1],
        w1.shape[1],
        hidden_threshold,
        output_threshold,
        repeats,
    )
    return (time.perf_counter() - start) * 1000.0, y.copy()


def run_shape(
    shape: ShapeConfig,
    gf_lib,
    int8_lib,
    seed: int,
    repeats: int,
    trials: int,
) -> dict:
    gf_vectors = make_vectors(seed, shape.rows, shape.input_width, shape.hidden_width)
    gf_reference = predict_numpy_gf137(gf_vectors)
    int8_vectors = make_int8_vectors(seed, shape.rows, shape.input_width, shape.hidden_width)

    rows: list[BaselineRow] = []

    float_time, float_y = time_cpp_repeated(
        run_cpp_float32_modular, gf_lib, gf_vectors, repeats, trials
    )
    rows.append(
        BaselineRow(
            shape=shape.label,
            backend="C++ native O3 float32 modular",
            memory_bytes=model_bytes(gf_vectors, 4),
            time_ms=float_time,
            mismatch_count=int(np.count_nonzero(float_y != gf_reference)),
            agreement_required=True,
            notes="equivalent modular float32 kernel",
        )
    )

    gf_time, gf_y = time_cpp_repeated(run_cpp_gf137, gf_lib, gf_vectors, repeats, trials)
    rows.append(
        BaselineRow(
            shape=shape.label,
            backend="C++ native O3 GF(137)",
            memory_bytes=model_bytes(gf_vectors, 1),
            time_ms=gf_time,
            mismatch_count=int(np.count_nonzero(gf_y != gf_reference)),
            agreement_required=True,
            notes="equivalent finite-field kernel",
        )
    )

    int8_time, _int8_y = time_cpp_repeated(
        run_int8_cpp, int8_lib, int8_vectors, repeats, trials
    )
    rows.append(
        BaselineRow(
            shape=shape.label,
            backend="C++ native O3 standard int8 proxy",
            memory_bytes=int8_model_bytes(int8_vectors),
            time_ms=int8_time,
            mismatch_count=None,
            agreement_required=False,
            notes="non-equivalent int8 dense kernel with int32 accumulation",
        )
    )

    by_backend = {row.backend: row for row in rows}
    gf_row = by_backend["C++ native O3 GF(137)"]
    int8_row = by_backend["C++ native O3 standard int8 proxy"]
    float_row = by_backend["C++ native O3 float32 modular"]

    summary = {
        "shape": asdict(shape),
        "storage_ratio_float32_to_gf137": float_row.memory_bytes / gf_row.memory_bytes,
        "memory_ratio_int8_to_gf137": int8_row.memory_bytes / gf_row.memory_bytes,
        "speedup_float32_to_gf137": float_row.time_ms / gf_row.time_ms,
        "speed_ratio_int8_to_gf137": int8_row.time_ms / gf_row.time_ms,
        "equivalent_mismatches_zero": all(
            row.mismatch_count == 0 for row in rows if row.agreement_required
        ),
    }
    return {
        "rows": [asdict(row) for row in rows],
        "summary": summary,
    }


def write_markdown(payload: dict) -> None:
    lines = [
        "# HYP-004 Industrial Int8 Baseline Proxy",
        "",
        "This report is generated by `scripts/industrial_int8_baseline.py`.",
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
            "| Shape | float32/GF137 storage | int8/GF137 memory | float32/GF137 speedup | int8/GF137 runtime | Agreement |",
            "|---|---:|---:|---:|---:|---|",
        ]
    )
    for item in payload["shape_summaries"]:
        shape = item["shape"]["label"]
        agreement = "pass" if item["equivalent_mismatches_zero"] else "fail"
        lines.append(
            f"| {shape} | {item['storage_ratio_float32_to_gf137']:.3f}x | "
            f"{item['memory_ratio_int8_to_gf137']:.3f}x | "
            f"{item['speedup_float32_to_gf137']:.3f}x | "
            f"{item['speed_ratio_int8_to_gf137']:.3f}x | {agreement} |"
        )

    lines.extend(["", "## Interpretation", "", payload["interpretation"], ""])
    RESULT_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=5137)
    parser.add_argument("--repeats", type=int, default=500)
    parser.add_argument("--trials", type=int, default=3)
    args = parser.parse_args()

    native_flags = ["-O3", "-DNDEBUG", "-ffast-math", "-funroll-loops"]
    native_flags.append("-mcpu=native" if sys.platform == "darwin" else "-march=native")
    gf_path = compile_library("hyp004_gf_native_o3", native_flags)
    int8_path = compile_int8_library("hyp004_int8_native_o3", native_flags)
    gf_lib = load_library(gf_path)
    int8_lib = load_int8_library(int8_path)

    all_rows = []
    shape_summaries = []
    for shape in SHAPES:
        result = run_shape(shape, gf_lib, int8_lib, args.seed, args.repeats, args.trials)
        all_rows.extend(result["rows"])
        shape_summaries.append(result["summary"])

    pass_agreement = all(item["equivalent_mismatches_zero"] for item in shape_summaries)
    pass_measurement = all(
        row["time_ms"] > 0 and row["memory_bytes"] > 0 for row in all_rows
    )
    pass_int8_memory_profile = all(
        item["memory_ratio_int8_to_gf137"] <= 1.25 for item in shape_summaries
    )
    int8_faster_shapes = sum(
        item["speed_ratio_int8_to_gf137"] < 1.0 for item in shape_summaries
    )
    gf137_faster_shapes = len(shape_summaries) - int8_faster_shapes

    if int8_faster_shapes == len(shape_summaries):
        speed_note = (
            "The standard int8 proxy is faster on every tested shape. GF(137) "
            "should therefore be presented as compact modular inference, not "
            "as the fastest general int8 path."
        )
    elif int8_faster_shapes == 0:
        speed_note = (
            "GF(137) is faster than the standard int8 proxy on every tested "
            "shape in this run. This should be rechecked on external machines."
        )
    else:
        speed_note = (
            "The speed result is shape-dependent: the standard int8 proxy wins "
            "on some shapes and GF(137) wins on others."
        )

    interpretation = (
        f"Measurement pass={pass_measurement}. Agreement pass={pass_agreement}. "
        f"Int8 memory profile pass={pass_int8_memory_profile}. Standard int8 "
        f"is faster on {int8_faster_shapes}/{len(shape_summaries)} shapes; "
        f"GF(137) is faster on {gf137_faster_shapes}/{len(shape_summaries)} shapes. "
        f"{speed_note} This audit does not establish a fundamental-physics result."
    )

    payload = {
        "metadata": system_metadata(),
        "config": {
            "seed": args.seed,
            "repeats": args.repeats,
            "trials": args.trials,
        },
        "rows": all_rows,
        "shape_summaries": shape_summaries,
        "summary": {
            "pass_measurement": pass_measurement,
            "pass_agreement": pass_agreement,
            "pass_int8_memory_profile": pass_int8_memory_profile,
            "standard_int8_faster_shapes": int8_faster_shapes,
            "gf137_faster_shapes": gf137_faster_shapes,
        },
        "interpretation": interpretation,
    }

    RESULT_JSON.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    write_markdown(payload)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
