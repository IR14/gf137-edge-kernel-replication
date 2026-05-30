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
    mismatch_count: int
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
    return lib


def ptr(array: np.ndarray):
    return array.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))


def run_cpp(lib, vectors: dict[str, np.ndarray], repeats: int) -> tuple[float, np.ndarray]:
    x = np.ascontiguousarray(vectors["x"], dtype=np.uint8)
    w1 = np.ascontiguousarray(vectors["w1"], dtype=np.uint8)
    b1 = np.ascontiguousarray(vectors["b1"], dtype=np.uint8)
    w2 = np.ascontiguousarray(vectors["w2"], dtype=np.uint8)
    b2 = np.ascontiguousarray(vectors["b2"], dtype=np.uint8)
    y = np.empty(x.shape[0], dtype=np.uint8)
    start = time.perf_counter()
    lib.gf137_predict_repeated(
        ptr(x),
        ptr(w1),
        ptr(b1),
        ptr(w2),
        ptr(b2),
        ptr(y),
        x.shape[0],
        x.shape[1],
        w1.shape[1],
        THRESHOLD,
        THRESHOLD,
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
        lines.append(
            f"| {row['backend']} | {row['memory_bytes']} | "
            f"{row['time_1000_passes_ms']:.6f} | {row['mismatch_count']} | "
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
            notes="vectorized Python reference, uint8 model storage",
        )
    )

    portable_path = compile_library("portable_o3", ["-O3", "-DNDEBUG"])
    native_flags = ["-O3", "-DNDEBUG", "-ffast-math", "-funroll-loops"]
    native_flags.append("-mcpu=native" if sys.platform == "darwin" else "-march=native")
    native_path = compile_library("native_o3", native_flags)

    for label, path, note in [
        ("C++ portable O3 GF(137)", portable_path, "optimized portable scalar loop"),
        ("C++ native O3 GF(137)", native_path, "optimized native loop"),
    ]:
        lib = load_library(path)
        timings = []
        cpp_y = None
        for _ in range(args.trials):
            elapsed, cpp_y = run_cpp(lib, vectors, args.repeats)
            timings.append(elapsed)
        assert cpp_y is not None
        rows.append(
            BenchmarkRow(
                backend=label,
                memory_bytes=model_bytes(vectors, 1),
                time_1000_passes_ms=float(np.median(timings)),
                mismatch_count=int(np.count_nonzero(cpp_y != reference)),
                notes=note,
            )
        )

    float_row = rows[0]
    native_row = rows[-1]
    storage_ratio = float_row.memory_bytes / native_row.memory_bytes
    speedup = float_row.time_1000_passes_ms / native_row.time_1000_passes_ms
    pass_storage = storage_ratio >= 3.9
    pass_agreement = all(row.mismatch_count == 0 for row in rows)
    pass_speed = speedup > 1.0
    interpretation = (
        f"Storage ratio float32/native GF(137) = {storage_ratio:.3f}x. "
        f"Runtime speedup float32/native GF(137) = {speedup:.3f}x. "
        f"Pass storage={pass_storage}, pass agreement={pass_agreement}, "
        f"pass speed={pass_speed}. "
        "This benchmark supports only the stated engineering replication claim; "
        "it does not establish a fundamental-physics result."
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
            "speedup_float32_to_native": speedup,
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
