#!/usr/bin/env python3
"""Run the HYP-005 ONNX Runtime int8 baseline audit."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
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
    run_cpp_gf137,
    system_metadata,
)
from industrial_int8_baseline import (
    compile_int8_library,
    int8_model_bytes,
    int8_thresholds,
    load_int8_library,
    make_int8_vectors,
    run_int8_cpp,
)
from quantized_baseline_sweep import SHAPES, ShapeConfig, time_cpp_repeated


RESULT_JSON = ROOT / "outputs" / "onnxruntime_int8_baseline.json"
RESULT_MD = ROOT / "outputs" / "onnxruntime_int8_baseline.md"


@dataclass(frozen=True)
class BaselineRow:
    shape: str
    backend: str
    memory_bytes: int
    time_ms: float
    mismatch_count: int | None
    agreement_required: bool
    notes: str


def require_onnx_modules():
    try:
        import onnx  # noqa: PLC0415
        import onnxruntime as ort  # noqa: PLC0415
    except Exception as exc:  # pragma: no cover - exercised only on missing deps
        raise RuntimeError(
            "HYP-005 requires `onnx` and `onnxruntime`. "
            "Run with `./scripts/run_hyp005.sh` or install both packages."
        ) from exc
    return onnx, ort


def tensor_from_array(onnx, name: str, array: np.ndarray):
    return onnx.numpy_helper.from_array(np.ascontiguousarray(array), name=name)


def create_onnx_int8_model(onnx, vectors, path: Path) -> None:
    from onnx import TensorProto, helper  # noqa: PLC0415

    rows, k_width = vectors.x.shape
    h_width = vectors.w1.shape[1]
    hidden_threshold, output_threshold = int8_thresholds(k_width, h_width)

    w2 = np.ascontiguousarray(vectors.w2.reshape(h_width, 1), dtype=np.int8)
    b2 = np.ascontiguousarray(vectors.b2.reshape(1), dtype=np.int32)

    initializers = [
        tensor_from_array(onnx, "w1", vectors.w1),
        tensor_from_array(onnx, "b1", vectors.b1),
        tensor_from_array(onnx, "w2", w2),
        tensor_from_array(onnx, "b2", b2),
        tensor_from_array(onnx, "x_zero_point", np.array(0, dtype=np.int8)),
        tensor_from_array(onnx, "w1_zero_point", np.array(0, dtype=np.int8)),
        tensor_from_array(onnx, "h_zero_point", np.array(0, dtype=np.int8)),
        tensor_from_array(onnx, "w2_zero_point", np.array(0, dtype=np.int8)),
        tensor_from_array(
            onnx,
            "hidden_threshold",
            np.full((h_width,), hidden_threshold, dtype=np.int32),
        ),
        tensor_from_array(
            onnx,
            "output_threshold",
            np.full((1,), output_threshold, dtype=np.int32),
        ),
    ]

    nodes = [
        helper.make_node(
            "MatMulInteger",
            ["x", "w1", "x_zero_point", "w1_zero_point"],
            ["hidden_mm"],
        ),
        helper.make_node("Add", ["hidden_mm", "b1"], ["hidden_acc"]),
        helper.make_node("Greater", ["hidden_acc", "hidden_threshold"], ["hidden_bool"]),
        helper.make_node(
            "Cast",
            ["hidden_bool"],
            ["hidden_i8"],
            to=TensorProto.INT8,
        ),
        helper.make_node(
            "MatMulInteger",
            ["hidden_i8", "w2", "h_zero_point", "w2_zero_point"],
            ["out_mm"],
        ),
        helper.make_node("Add", ["out_mm", "b2"], ["out_acc"]),
        helper.make_node("Greater", ["out_acc", "output_threshold"], ["out_bool"]),
        helper.make_node("Cast", ["out_bool"], ["y"], to=TensorProto.UINT8),
    ]

    graph = helper.make_graph(
        nodes,
        "hyp005_standard_int8_proxy",
        [helper.make_tensor_value_info("x", TensorProto.INT8, [rows, k_width])],
        [helper.make_tensor_value_info("y", TensorProto.UINT8, [rows, 1])],
        initializer=initializers,
    )
    model = helper.make_model(
        graph,
        producer_name="gf137-hyp005",
        opset_imports=[helper.make_operatorsetid("", 18)],
    )
    model.ir_version = 10
    onnx.checker.check_model(model)
    onnx.save(model, path)


def make_session(ort, model_path: Path):
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(
        str(model_path),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )


def run_onnx_repeated(session, x: np.ndarray, repeats: int) -> tuple[float, np.ndarray]:
    x_input = np.ascontiguousarray(x, dtype=np.int8)
    result = None
    start = time.perf_counter()
    for _ in range(repeats):
        result = session.run(["y"], {"x": x_input})[0]
    assert result is not None
    return (time.perf_counter() - start) * 1000.0, np.asarray(result, dtype=np.uint8)


def time_onnx(session, x: np.ndarray, repeats: int, trials: int) -> tuple[float, np.ndarray]:
    timings = []
    result = None
    for _ in range(trials):
        elapsed, result = run_onnx_repeated(session, x, repeats)
        timings.append(elapsed)
    assert result is not None
    return float(np.median(timings)), result


def run_shape(
    shape: ShapeConfig,
    gf_lib,
    int8_lib,
    seed: int,
    repeats: int,
    trials: int,
    temp_dir: Path,
    onnx,
    ort,
) -> dict:
    gf_vectors = make_vectors(seed, shape.rows, shape.input_width, shape.hidden_width)
    gf_reference = predict_numpy_gf137(gf_vectors)
    int8_vectors = make_int8_vectors(seed, shape.rows, shape.input_width, shape.hidden_width)
    model_path = temp_dir / f"hyp005_{shape.label}.onnx"
    create_onnx_int8_model(onnx, int8_vectors, model_path)
    session = make_session(ort, model_path)

    rows: list[BaselineRow] = []

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

    cpp_int8_time, _cpp_int8_y = time_cpp_repeated(
        run_int8_cpp, int8_lib, int8_vectors, repeats, trials
    )
    rows.append(
        BaselineRow(
            shape=shape.label,
            backend="C++ native O3 standard int8 proxy",
            memory_bytes=int8_model_bytes(int8_vectors),
            time_ms=cpp_int8_time,
            mismatch_count=None,
            agreement_required=False,
            notes="non-equivalent C++ int8 dense kernel",
        )
    )

    onnx_time, _onnx_y = time_onnx(session, int8_vectors.x, repeats, trials)
    rows.append(
        BaselineRow(
            shape=shape.label,
            backend="ONNX Runtime CPU MatMulInteger int8",
            memory_bytes=int8_model_bytes(int8_vectors),
            time_ms=onnx_time,
            mismatch_count=None,
            agreement_required=False,
            notes="external runtime int8 graph, single-thread CPU provider",
        )
    )

    by_backend = {row.backend: row for row in rows}
    gf_row = by_backend["C++ native O3 GF(137)"]
    cpp_int8_row = by_backend["C++ native O3 standard int8 proxy"]
    onnx_row = by_backend["ONNX Runtime CPU MatMulInteger int8"]

    summary = {
        "shape": asdict(shape),
        "memory_ratio_int8_to_gf137": onnx_row.memory_bytes / gf_row.memory_bytes,
        "speed_ratio_cpp_int8_to_gf137": cpp_int8_row.time_ms / gf_row.time_ms,
        "speed_ratio_onnx_int8_to_gf137": onnx_row.time_ms / gf_row.time_ms,
        "speed_ratio_onnx_int8_to_cpp_int8": onnx_row.time_ms / cpp_int8_row.time_ms,
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
        "# HYP-005 ONNX Runtime Int8 Baseline",
        "",
        "This report is generated by `scripts/onnxruntime_int8_baseline.py`.",
        "",
        "## Machine",
        "",
    ]
    for key, value in payload["metadata"].items():
        lines.append(f"- `{key}`: `{value}`")

    lines.extend(
        [
            "",
            "## Runtime",
            "",
            f"- `onnx`: `{payload['runtime']['onnx']}`",
            f"- `onnxruntime`: `{payload['runtime']['onnxruntime']}`",
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
            "| Shape | int8/GF137 memory | C++ int8/GF137 runtime | ONNX int8/GF137 runtime | ONNX/C++ int8 runtime | Agreement |",
            "|---|---:|---:|---:|---:|---|",
        ]
    )
    for item in payload["shape_summaries"]:
        shape = item["shape"]["label"]
        agreement = "pass" if item["equivalent_mismatches_zero"] else "fail"
        lines.append(
            f"| {shape} | {item['memory_ratio_int8_to_gf137']:.3f}x | "
            f"{item['speed_ratio_cpp_int8_to_gf137']:.3f}x | "
            f"{item['speed_ratio_onnx_int8_to_gf137']:.3f}x | "
            f"{item['speed_ratio_onnx_int8_to_cpp_int8']:.3f}x | {agreement} |"
        )

    lines.extend(["", "## Interpretation", "", payload["interpretation"], ""])
    RESULT_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=5137)
    parser.add_argument("--repeats", type=int, default=200)
    parser.add_argument("--trials", type=int, default=3)
    args = parser.parse_args()

    onnx, ort = require_onnx_modules()

    native_flags = ["-O3", "-DNDEBUG", "-ffast-math", "-funroll-loops"]
    native_flags.append("-mcpu=native" if sys.platform == "darwin" else "-march=native")
    gf_path = compile_library("hyp005_gf_native_o3", native_flags)
    int8_path = compile_int8_library("hyp005_int8_native_o3", native_flags)
    gf_lib = load_library(gf_path)
    int8_lib = load_int8_library(int8_path)

    all_rows = []
    shape_summaries = []
    with tempfile.TemporaryDirectory() as temp_name:
        temp_dir = Path(temp_name)
        for shape in SHAPES:
            result = run_shape(
                shape,
                gf_lib,
                int8_lib,
                args.seed,
                args.repeats,
                args.trials,
                temp_dir,
                onnx,
                ort,
            )
            all_rows.extend(result["rows"])
            shape_summaries.append(result["summary"])

    pass_runtime_available = True
    pass_measurement = all(row["time_ms"] > 0 and row["memory_bytes"] > 0 for row in all_rows)
    pass_agreement = all(item["equivalent_mismatches_zero"] for item in shape_summaries)
    pass_int8_memory_profile = all(
        item["memory_ratio_int8_to_gf137"] <= 1.25 for item in shape_summaries
    )
    onnx_faster_than_gf137 = sum(
        item["speed_ratio_onnx_int8_to_gf137"] < 1.0 for item in shape_summaries
    )
    gf137_faster_than_onnx = len(shape_summaries) - onnx_faster_than_gf137
    onnx_faster_than_cpp_int8 = sum(
        item["speed_ratio_onnx_int8_to_cpp_int8"] < 1.0 for item in shape_summaries
    )

    if onnx_faster_than_gf137 == len(shape_summaries):
        speed_note = (
            "ONNX Runtime int8 is faster than GF(137) on every tested shape. "
            "This strongly narrows the GF(137) deployment-speed claim."
        )
    elif onnx_faster_than_gf137 == 0:
        speed_note = (
            "GF(137) is faster than ONNX Runtime int8 on every tested shape in "
            "this run. This should be checked on external machines."
        )
    else:
        speed_note = (
            "The ONNX Runtime int8 comparison is shape-dependent."
        )

    interpretation = (
        f"Runtime available={pass_runtime_available}. Measurement pass={pass_measurement}. "
        f"Agreement pass={pass_agreement}. Int8 memory profile pass={pass_int8_memory_profile}. "
        f"ONNX Runtime int8 is faster than GF(137) on {onnx_faster_than_gf137}/"
        f"{len(shape_summaries)} shapes; GF(137) is faster on {gf137_faster_than_onnx}/"
        f"{len(shape_summaries)} shapes. ONNX Runtime int8 is faster than the C++ "
        f"int8 proxy on {onnx_faster_than_cpp_int8}/{len(shape_summaries)} shapes. "
        f"{speed_note} This audit does not establish a fundamental-physics result."
    )

    payload = {
        "metadata": system_metadata(),
        "runtime": {
            "onnx": onnx.__version__,
            "onnxruntime": ort.__version__,
        },
        "config": {
            "seed": args.seed,
            "repeats": args.repeats,
            "trials": args.trials,
        },
        "rows": all_rows,
        "shape_summaries": shape_summaries,
        "summary": {
            "pass_runtime_available": pass_runtime_available,
            "pass_measurement": pass_measurement,
            "pass_agreement": pass_agreement,
            "pass_int8_memory_profile": pass_int8_memory_profile,
            "onnx_int8_faster_than_gf137_shapes": onnx_faster_than_gf137,
            "gf137_faster_than_onnx_int8_shapes": gf137_faster_than_onnx,
            "onnx_int8_faster_than_cpp_int8_shapes": onnx_faster_than_cpp_int8,
        },
        "interpretation": interpretation,
    }

    RESULT_JSON.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    write_markdown(payload)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
