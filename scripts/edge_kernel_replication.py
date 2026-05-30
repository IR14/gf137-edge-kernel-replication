#!/usr/bin/env python3
"""Run the HYP-002 GF(137) edge-kernel replication benchmark.

One-command usage:

    uv run --with numpy python scripts/edge_kernel_replication.py

The benchmark is intentionally small and deterministic.  It compares storage,
runtime, and output agreement for a fixed two-layer modular network.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import platform
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CPP_SOURCE = ROOT / "scripts" / "gf137_edge_kernel.cpp"
BUILD_DIR = ROOT / "outputs" / "build"
RESULT_JSON = ROOT / "outputs" / "edge_kernel_replication.json"
RESULT_MD = ROOT / "outputs" / "edge_kernel_replication.md"
P_FIELD = 137
THRESHOLD = 42


@dataclass(frozen=True)
class BenchmarkRow:
    backend: str
    memory_bytes: int
    time_1000_passes_ms: float
    mismatch_count: int | None
    agreement_required: bool
    notes: str


def system_metadata() -> dict[str, str]:
    compiler = shutil.which("clang++") or shutil.which("g++") or ""
    compiler_version = ""
    if compiler:
        completed = subprocess.run(
            [compiler, "--version"],
            capture_output=True,
            text=True,
            check=False,
        )
        compiler_version = completed.stdout.splitlines()[0] if completed.stdout else ""
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": sys.version.replace("\n", " "),
        "numpy": np.__version__,
        "compiler": compiler,
        "compiler_version": compiler_version,
    }


def make_vectors(seed: int, rows: int, k_width: int, h_width: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {
        "x": rng.integers(0, 3, size=(rows, k_width), dtype=np.uint8),
        "w1": rng.integers(0, P_FIELD, size=(k_width, h_width), dtype=np.uint8),
        "b1": rng.integers(0, P_FIELD, size=(h_width,), dtype=np.uint8),
        "w2": rng.integers(0, P_FIELD, size=(h_width,), dtype=np.uint8),
        "b2": rng.integers(0, P_FIELD, size=(1,), dtype=np.uint8),
    }


def predict_numpy_gf137(vectors: dict[str, np.ndarray]) -> np.ndarray:
    x = vectors["x"].astype(np.uint32)
    w1 = vectors["w1"].astype(np.uint32)
    b1 = vectors["b1"].astype(np.uint32)
    w2 = vectors["w2"].astype(np.uint32)
    b2 = vectors["b2"].astype(np.uint32)
    hidden = ((x @ w1 + b1) % P_FIELD >= THRESHOLD).astype(np.uint32)
    return ((hidden @ w2 + b2[0]) % P_FIELD >= THRESHOLD).astype(np.uint8)


def predict_numpy_float32_modular(vectors: dict[str, np.ndarray]) -> np.ndarray:
    x = vectors["x"].astype(np.float32)
    w1 = vectors["w1"].astype(np.float32)
    b1 = vectors["b1"].astype(np.float32)
    w2 = vectors["w2"].astype(np.float32)
    b2 = vectors["b2"].astype(np.float32)
    hidden = (np.mod(x @ w1 + b1, np.float32(P_FIELD)) >= np.float32(THRESHOLD)).astype(
        np.float32
    )
    return (
        np.mod(hidden @ w2 + b2[0], np.float32(P_FIELD)) >= np.float32(THRESHOLD)
    ).astype(np.uint8)


def time_callable(fn, repeats: int, trials: int) -> tuple[float, np.ndarray]:
    timings = []
    result = fn()
    for _ in range(trials):
        start = time.perf_counter()
        for _repeat in range(repeats):
            result = fn()
        timings.append((time.perf_counter() - start) * 1000.0)
    return float(np.median(timings)), np.asarray(result, dtype=np.uint8)


def library_name(label: str) -> str:
    suffix = ".dylib" if sys.platform == "darwin" else ".so"
    return f"libgf137_edge_{label}{suffix}"


def compile_library(label: str, flags: list[str]) -> Path:
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
        str(CPP_SOURCE),
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


def load_library(path: Path):
    lib = ctypes.CDLL(str(path))
    u8 = ctypes.POINTER(ctypes.c_uint8)
    f32 = ctypes.POINTER(ctypes.c_float)
    lib.gf137_predict.argtypes = [
        u8,
        u8,
        u8,
        u8,
        u8,
        u8,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_uint8,
        ctypes.c_uint8,
    ]
    lib.gf137_predict.restype = None
    lib.gf137_predict_repeated.argtypes = [
        u8,
        u8,
        u8,
        u8,
        u8,
        u8,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_uint8,
        ctypes.c_uint8,
        ctypes.c_int,
    ]
    lib.gf137_predict_repeated.restype = None
    lib.float32_modular_predict_repeated.argtypes = [
        u8,
        f32,
        f32,
        f32,
        f32,
        u8,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_int,
    ]
    lib.float32_modular_predict_repeated.restype = None
    lib.plain_uint8_predict_repeated.argtypes = [
        u8,
        u8,
        u8,
        u8,
        u8,
        u8,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_int,
    ]
    lib.plain_uint8_predict_repeated.restype = None
    return lib


def ptr_u8(array: np.ndarray):
    return array.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))


def ptr_f32(array: np.ndarray):
    return array.ctypes.data_as(ctypes.POINTER(ctypes.c_float))


def run_cpp_gf137(lib, vectors: dict[str, np.ndarray], repeats: int) -> tuple[float, np.ndarray]:
    x = np.ascontiguousarray(vectors["x"], dtype=np.uint8)
    w1 = np.ascontiguousarray(vectors["w1"], dtype=np.uint8)
    b1 = np.ascontiguousarray(vectors["b1"], dtype=np.uint8)
    w2 = np.ascontiguousarray(vectors["w2"], dtype=np.uint8)
    b2 = np.ascontiguousarray(vectors["b2"], dtype=np.uint8)
    y = np.empty(x.shape[0], dtype=np.uint8)
    start = time.perf_counter()
    lib.gf137_predict_repeated(
        ptr_u8(x),
        ptr_u8(w1),
        ptr_u8(b1),
        ptr_u8(w2),
        ptr_u8(b2),
        ptr_u8(y),
        x.shape[0],
        x.shape[1],
        w1.shape[1],
        THRESHOLD,
        THRESHOLD,
        repeats,
    )
    return (time.perf_counter() - start) * 1000.0, y.copy()


def run_cpp_float32_modular(
    lib, vectors: dict[str, np.ndarray], repeats: int
) -> tuple[float, np.ndarray]:
    x = np.ascontiguousarray(vectors["x"], dtype=np.uint8)
    w1 = np.ascontiguousarray(vectors["w1"], dtype=np.float32)
    b1 = np.ascontiguousarray(vectors["b1"], dtype=np.float32)
    w2 = np.ascontiguousarray(vectors["w2"], dtype=np.float32)
    b2 = np.ascontiguousarray(vectors["b2"], dtype=np.float32)
    y = np.empty(x.shape[0], dtype=np.uint8)
    start = time.perf_counter()
    lib.float32_modular_predict_repeated(
        ptr_u8(x),
        ptr_f32(w1),
        ptr_f32(b1),
        ptr_f32(w2),
        ptr_f32(b2),
        ptr_u8(y),
        x.shape[0],
        x.shape[1],
        w1.shape[1],
        float(THRESHOLD),
        float(THRESHOLD),
        repeats,
    )
    return (time.perf_counter() - start) * 1000.0, y.copy()


def plain_uint8_thresholds(k_width: int, h_width: int) -> tuple[int, int]:
    hidden_threshold = k_width * (P_FIELD // 2)
    output_threshold = h_width * (P_FIELD // 2)
    return hidden_threshold, output_threshold


def run_cpp_plain_uint8(
    lib, vectors: dict[str, np.ndarray], repeats: int
) -> tuple[float, np.ndarray]:
    x = np.ascontiguousarray(vectors["x"], dtype=np.uint8)
    w1 = np.ascontiguousarray(vectors["w1"], dtype=np.uint8)
    b1 = np.ascontiguousarray(vectors["b1"], dtype=np.uint8)
    w2 = np.ascontiguousarray(vectors["w2"], dtype=np.uint8)
    b2 = np.ascontiguousarray(vectors["b2"], dtype=np.uint8)
    y = np.empty(x.shape[0], dtype=np.uint8)
    hidden_threshold, output_threshold = plain_uint8_thresholds(x.shape[1], w1.shape[1])
    start = time.perf_counter()
    lib.plain_uint8_predict_repeated(
        ptr_u8(x),
        ptr_u8(w1),
        ptr_u8(b1),
        ptr_u8(w2),
        ptr_u8(b2),
        ptr_u8(y),
        x.shape[0],
        x.shape[1],
        w1.shape[1],
        hidden_threshold,
        output_threshold,
        repeats,
    )
    return (time.perf_counter() - start) * 1000.0, y.copy()


def model_bytes(vectors: dict[str, np.ndarray], dtype_size: int) -> int:
    count = (
        vectors["w1"].size
        + vectors["b1"].size
        + vectors["w2"].size
        + vectors["b2"].size
    )
    return int(count * dtype_size)


def write_markdown(payload: dict) -> None:
    rows = payload["rows"]
    lines = [
        "# HYP-002 Edge-Kernel Replication Result",
        "",
        "This report is generated by `scripts/edge_kernel_replication.py`.",
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
        ]
    )
    for key, value in payload["config"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(
        [
            "",
            "## Results",
            "",
            "| Backend | Memory bytes | 1000 passes ms | Mismatches | Notes |",
            "|---|---:|---:|---:|---|",
        ]
    )
    for row in rows:
        mismatches = "n/a" if row["mismatch_count"] is None else str(row["mismatch_count"])
        lines.append(
            f"| {row['backend']} | {row['memory_bytes']} | "
            f"{row['time_1000_passes_ms']:.6f} | {mismatches} | "
            f"{row['notes']} |"
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
    parser.add_argument("--rows", type=int, default=256)
    parser.add_argument("--input-width", type=int, default=64)
    parser.add_argument("--hidden-width", type=int, default=32)
    parser.add_argument("--seed", type=int, default=5137)
    parser.add_argument("--repeats", type=int, default=1000)
    parser.add_argument("--trials", type=int, default=5)
    args = parser.parse_args()

    vectors = make_vectors(args.seed, args.rows, args.input_width, args.hidden_width)
    reference = predict_numpy_gf137(vectors)

    rows: list[BenchmarkRow] = []

    numpy_float_time, numpy_float_y = time_callable(
        lambda: predict_numpy_float32_modular(vectors),
        repeats=args.repeats,
        trials=args.trials,
    )
    rows.append(
        BenchmarkRow(
            backend="NumPy float32 modular",
            memory_bytes=model_bytes(vectors, 4),
            time_1000_passes_ms=numpy_float_time,
            mismatch_count=int(np.count_nonzero(numpy_float_y != reference)),
            agreement_required=True,
            notes="same modular operation expressed with float32 arrays",
        )
    )

    numpy_uint_time, numpy_uint_y = time_callable(
        lambda: predict_numpy_gf137(vectors),
        repeats=args.repeats,
        trials=args.trials,
    )
    rows.append(
        BenchmarkRow(
            backend="NumPy uint32 GF(137) reference",
            memory_bytes=model_bytes(vectors, 1),
            time_1000_passes_ms=numpy_uint_time,
            mismatch_count=int(np.count_nonzero(numpy_uint_y != reference)),
            agreement_required=True,
            notes="vectorized Python reference, uint8 model storage",
        )
    )

    portable_path = compile_library("portable_o3", ["-O3", "-DNDEBUG"])
    native_flags = ["-O3", "-DNDEBUG", "-ffast-math", "-funroll-loops"]
    native_flags.append("-mcpu=native" if sys.platform == "darwin" else "-march=native")
    native_path = compile_library("native_o3", native_flags)

    for label, path, note in [
        ("portable", portable_path, "portable scalar flags"),
        ("native", native_path, "native compiler flags"),
    ]:
        lib = load_library(path)
        for backend_name, runner, dtype_size, agreement_required, row_note in [
            (
                f"C++ {label} O3 float32 modular",
                run_cpp_float32_modular,
                4,
                True,
                f"same modular arithmetic with float32 model storage, {note}",
            ),
            (
                f"C++ {label} O3 uint8 plain threshold",
                run_cpp_plain_uint8,
                1,
                False,
                f"same topology without GF(137) residue reductions, {note}",
            ),
            (
                f"C++ {label} O3 GF(137)",
                run_cpp_gf137,
                1,
                True,
                f"optimized GF(137) kernel, {note}",
            ),
        ]:
            timings = []
            cpp_y = None
            for _ in range(args.trials):
                elapsed, cpp_y = runner(lib, vectors, args.repeats)
                timings.append(elapsed)
            assert cpp_y is not None
            mismatch_count = (
                int(np.count_nonzero(cpp_y != reference)) if agreement_required else None
            )
            rows.append(
                BenchmarkRow(
                    backend=backend_name,
                    memory_bytes=model_bytes(vectors, dtype_size),
                    time_1000_passes_ms=float(np.median(timings)),
                    mismatch_count=mismatch_count,
                    agreement_required=agreement_required,
                    notes=row_note,
                )
            )

    rows_by_backend = {row.backend: row for row in rows}
    float_row = rows_by_backend["NumPy float32 modular"]
    native_float_row = rows_by_backend["C++ native O3 float32 modular"]
    native_plain_row = rows_by_backend["C++ native O3 uint8 plain threshold"]
    native_gf_row = rows_by_backend["C++ native O3 GF(137)"]
    storage_ratio = float_row.memory_bytes / native_gf_row.memory_bytes
    speedup_numpy_float = float_row.time_1000_passes_ms / native_gf_row.time_1000_passes_ms
    speedup_cpp_float = (
        native_float_row.time_1000_passes_ms / native_gf_row.time_1000_passes_ms
    )
    speed_ratio_plain_uint8 = (
        native_plain_row.time_1000_passes_ms / native_gf_row.time_1000_passes_ms
    )
    pass_storage = storage_ratio >= 3.9
    pass_agreement = all(
        row.mismatch_count == 0 for row in rows if row.agreement_required
    )
    pass_speed = speedup_numpy_float > 1.0 and speedup_cpp_float > 1.0
    interpretation = (
        f"Storage ratio float32/native GF(137) = {storage_ratio:.3f}x. "
        f"Runtime speedup NumPy float32/native GF(137) = {speedup_numpy_float:.3f}x. "
        f"Runtime speedup C++ float32/native GF(137) = {speedup_cpp_float:.3f}x. "
        f"Native plain uint8/native GF(137) runtime ratio = {speed_ratio_plain_uint8:.3f}x. "
        f"Pass storage={pass_storage}, pass agreement={pass_agreement}, "
        f"pass speed={pass_speed}. "
        "The plain uint8 row is intentionally non-equivalent and is included only as a "
        "lower-level control for loop and storage overhead. This benchmark supports only "
        "the stated engineering replication claim; it does not establish a "
        "fundamental-physics result."
    )

    payload = {
        "metadata": system_metadata(),
        "config": {
            "rows": args.rows,
            "input_width": args.input_width,
            "hidden_width": args.hidden_width,
            "seed": args.seed,
            "repeats": args.repeats,
            "trials": args.trials,
            "field": P_FIELD,
            "threshold": THRESHOLD,
        },
        "rows": [asdict(row) for row in rows],
        "summary": {
            "storage_ratio_float32_to_native": storage_ratio,
            "speedup_numpy_float32_to_native_gf137": speedup_numpy_float,
            "speedup_cpp_float32_to_native_gf137": speedup_cpp_float,
            "speed_ratio_native_plain_uint8_to_native_gf137": speed_ratio_plain_uint8,
            "pass_storage": pass_storage,
            "pass_agreement": pass_agreement,
            "pass_speed": pass_speed,
        },
        "interpretation": interpretation,
    }
    RESULT_JSON.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    write_markdown(payload)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
