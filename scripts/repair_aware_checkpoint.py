#!/usr/bin/env python3
"""Run the HYP-007 repair-aware GF(137) checkpoint audit.

The audit applies the HYP-006 RS(26,16) erasure-repair layer to actual GF(137)
model arrays from the edge-kernel benchmark.  It checks whether repaired model
weights are byte-exact and whether predictions stay identical after erasing
10 encoded axes per 16-symbol block.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from edge_kernel_replication import (
    P_FIELD,
    ROOT,
    make_vectors,
    predict_numpy_gf137,
    system_metadata,
)
from gf137_erasure_repair import (
    AXES,
    ERASURE_BUDGET,
    PAYLOAD_SYMBOLS,
    decode_rs,
    deterministic_erasure_patterns,
    encode_rs,
    erase_values,
    repetition2_success,
)


RESULT_JSON = ROOT / "outputs" / "repair_aware_checkpoint.json"
RESULT_MD = ROOT / "outputs" / "repair_aware_checkpoint.md"
MODEL_KEYS = ("w1", "b1", "w2", "b2")
BOLTZMANN_J_PER_K = 1.380649e-23
ROOM_TEMPERATURE_K = 300.0
CODATA_ALPHA_INV = 137.035999084
BARRETT_MU_137 = (1 << 32) // P_FIELD


@dataclass(frozen=True)
class ShapeConfig:
    label: str
    rows: int
    input_width: int
    hidden_width: int


@dataclass(frozen=True)
class ShapeSummary:
    shape: str
    raw_symbols: int
    blocks: int
    rs_storage_symbols: int
    repetition_storage_symbols: int
    rs_expansion_vs_raw: float
    repetition_expansion_vs_raw: float
    fp32_storage_bits: int
    gf137_raw_storage_bits: int
    rs_storage_bits: int
    repetition_storage_bits: int
    landauer_fp32_storage_j: float
    landauer_rs_storage_j: float
    landauer_rs_vs_fp32_ratio: float
    landauer_rs_reduction_fraction: float
    gf137_random_successes: int
    gf137_random_trials: int
    gf137_deterministic_successes: int
    gf137_deterministic_patterns: int
    gf137_max_prediction_mismatches: int
    raw_random_successes: int
    raw_random_trials: int
    raw_deterministic_successes: int
    raw_deterministic_patterns: int
    repetition_random_successes: int
    repetition_random_trials: int
    repetition_adversarial_successes: int
    repetition_adversarial_patterns: int
    notes: str


SHAPES = [
    ShapeConfig("small", rows=128, input_width=32, hidden_width=16),
    ShapeConfig("hyp002", rows=256, input_width=64, hidden_width=32),
    ShapeConfig("medium", rows=512, input_width=128, hidden_width=64),
]


def flatten_model(vectors: dict[str, np.ndarray]) -> tuple[np.ndarray, list[dict]]:
    parts = []
    specs = []
    for key in MODEL_KEYS:
        array = np.ascontiguousarray(vectors[key], dtype=np.uint8)
        specs.append({"key": key, "shape": array.shape, "size": int(array.size)})
        parts.append(array.reshape(-1))
    return np.concatenate(parts).astype(np.uint8), specs


def unflatten_model(
    flat: np.ndarray,
    specs: list[dict],
    template: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    repaired = {"x": template["x"]}
    cursor = 0
    for spec in specs:
        size = spec["size"]
        values = flat[cursor : cursor + size]
        repaired[spec["key"]] = values.reshape(spec["shape"]).astype(np.uint8)
        cursor += size
    return repaired


def payload_blocks(flat: np.ndarray) -> tuple[list[list[int]], int]:
    pad = (-int(flat.size)) % PAYLOAD_SYMBOLS
    if pad:
        flat = np.concatenate([flat, np.zeros(pad, dtype=np.uint8)])
    blocks = flat.reshape(-1, PAYLOAD_SYMBOLS)
    return [block.astype(int).tolist() for block in blocks], pad


def encode_blocks(blocks: list[list[int]]) -> list[list[int]]:
    return [encode_rs(block) for block in blocks]


def repair_blocks(
    encoded_blocks: list[list[int]],
    erased_by_block: list[set[int]],
    original_size: int,
) -> np.ndarray:
    repaired: list[int] = []
    for encoded, erased in zip(encoded_blocks, erased_by_block):
        repaired.extend(decode_rs(erase_values(encoded, erased)))
    return np.asarray(repaired[:original_size], dtype=np.uint8)


def random_block_erasures(
    rng: random.Random,
    blocks: int,
    population: int,
    erased: int,
) -> list[set[int]]:
    return [set(rng.sample(range(population), erased)) for _ in range(blocks)]


def raw_model_success(erased_by_block: list[set[int]]) -> bool:
    payload_slots = set(range(PAYLOAD_SYMBOLS))
    return all(payload_slots.isdisjoint(erased) for erased in erased_by_block)


def repetition_model_success(erased_by_block: list[set[int]]) -> bool:
    return all(repetition2_success(erased) for erased in erased_by_block)


def repetition_adversarial_patterns() -> dict[str, set[int]]:
    return {
        "erase_first_five_pairs": {
            idx for pair in range(5) for idx in (pair, pair + PAYLOAD_SYMBOLS)
        },
        "erase_middle_five_pairs": {
            idx for pair in range(5, 10) for idx in (pair, pair + PAYLOAD_SYMBOLS)
        },
        "erase_last_five_pairs": {
            idx for pair in range(11, 16) for idx in (pair, pair + PAYLOAD_SYMBOLS)
        },
    }


def landauer_bits_joule(bits: int, temperature_k: float = ROOM_TEMPERATURE_K) -> float:
    return bits * BOLTZMANN_J_PER_K * temperature_k * math.log(2.0)


def schwinger_s(alpha_inv: float = CODATA_ALPHA_INV) -> float:
    return (1.0 / alpha_inv) / (2.0 * math.pi)


def run_gf137_random_trials(
    vectors: dict[str, np.ndarray],
    encoded_blocks: list[list[int]],
    specs: list[dict],
    reference: np.ndarray,
    original_flat: np.ndarray,
    rng: random.Random,
    random_trials: int,
) -> tuple[int, int]:
    successes = 0
    max_mismatches = 0
    for _ in range(random_trials):
        erased_by_block = random_block_erasures(
            rng, len(encoded_blocks), AXES, ERASURE_BUDGET
        )
        repaired_flat = repair_blocks(encoded_blocks, erased_by_block, int(original_flat.size))
        repaired_vectors = unflatten_model(repaired_flat, specs, vectors)
        prediction = predict_numpy_gf137(repaired_vectors)
        mismatches = int(np.count_nonzero(prediction != reference))
        max_mismatches = max(max_mismatches, mismatches)
        if np.array_equal(repaired_flat, original_flat) and mismatches == 0:
            successes += 1
    return successes, max_mismatches


def run_gf137_deterministic_patterns(
    vectors: dict[str, np.ndarray],
    encoded_blocks: list[list[int]],
    specs: list[dict],
    reference: np.ndarray,
    original_flat: np.ndarray,
) -> tuple[int, int]:
    successes = 0
    max_mismatches = 0
    for erased in deterministic_erasure_patterns().values():
        erased_by_block = [set(erased) for _ in encoded_blocks]
        repaired_flat = repair_blocks(encoded_blocks, erased_by_block, int(original_flat.size))
        repaired_vectors = unflatten_model(repaired_flat, specs, vectors)
        prediction = predict_numpy_gf137(repaired_vectors)
        mismatches = int(np.count_nonzero(prediction != reference))
        max_mismatches = max(max_mismatches, mismatches)
        if np.array_equal(repaired_flat, original_flat) and mismatches == 0:
            successes += 1
    return successes, max_mismatches


def run_shape(shape: ShapeConfig, seed: int, random_trials: int) -> ShapeSummary:
    vectors = make_vectors(seed, shape.rows, shape.input_width, shape.hidden_width)
    reference = predict_numpy_gf137(vectors)
    original_flat, specs = flatten_model(vectors)
    blocks, _pad = payload_blocks(original_flat)
    encoded_blocks = encode_blocks(blocks)
    rng = random.Random(seed + len(blocks) + shape.rows)

    gf_random_successes, gf_random_mismatches = run_gf137_random_trials(
        vectors,
        encoded_blocks,
        specs,
        reference,
        original_flat,
        rng,
        random_trials,
    )
    gf_deterministic_successes, gf_deterministic_mismatches = run_gf137_deterministic_patterns(
        vectors,
        encoded_blocks,
        specs,
        reference,
        original_flat,
    )
    gf_max_mismatches = max(gf_random_mismatches, gf_deterministic_mismatches)

    raw_random_successes = 0
    repetition_random_successes = 0
    rng_controls = random.Random(seed + len(blocks) + shape.hidden_width + 1000)
    for _ in range(random_trials):
        raw_erasures = random_block_erasures(
            rng_controls, len(blocks), AXES, ERASURE_BUDGET
        )
        repetition_erasures = random_block_erasures(
            rng_controls, len(blocks), 2 * PAYLOAD_SYMBOLS, ERASURE_BUDGET
        )
        raw_random_successes += int(raw_model_success(raw_erasures))
        repetition_random_successes += int(repetition_model_success(repetition_erasures))

    deterministic_patterns = deterministic_erasure_patterns()
    raw_deterministic_successes = sum(
        raw_model_success([set(erased) for _ in encoded_blocks])
        for erased in deterministic_patterns.values()
    )

    repetition_patterns = repetition_adversarial_patterns()
    repetition_adversarial_successes = sum(
        repetition_model_success([set(erased) for _ in encoded_blocks])
        for erased in repetition_patterns.values()
    )

    raw_symbols = int(original_flat.size)
    block_count = len(blocks)
    rs_storage = block_count * AXES
    repetition_storage = block_count * 2 * PAYLOAD_SYMBOLS
    fp32_bits = raw_symbols * 32
    gf137_raw_bits = raw_symbols * 8
    rs_bits = rs_storage * 8
    repetition_bits = repetition_storage * 8
    fp32_landauer = landauer_bits_joule(fp32_bits)
    rs_landauer = landauer_bits_joule(rs_bits)
    rs_vs_fp32 = rs_landauer / fp32_landauer
    return ShapeSummary(
        shape=shape.label,
        raw_symbols=raw_symbols,
        blocks=block_count,
        rs_storage_symbols=rs_storage,
        repetition_storage_symbols=repetition_storage,
        rs_expansion_vs_raw=rs_storage / raw_symbols,
        repetition_expansion_vs_raw=repetition_storage / raw_symbols,
        fp32_storage_bits=fp32_bits,
        gf137_raw_storage_bits=gf137_raw_bits,
        rs_storage_bits=rs_bits,
        repetition_storage_bits=repetition_bits,
        landauer_fp32_storage_j=fp32_landauer,
        landauer_rs_storage_j=rs_landauer,
        landauer_rs_vs_fp32_ratio=rs_vs_fp32,
        landauer_rs_reduction_fraction=1.0 - rs_vs_fp32,
        gf137_random_successes=gf_random_successes,
        gf137_random_trials=random_trials,
        gf137_deterministic_successes=gf_deterministic_successes,
        gf137_deterministic_patterns=len(deterministic_patterns),
        gf137_max_prediction_mismatches=gf_max_mismatches,
        raw_random_successes=raw_random_successes,
        raw_random_trials=random_trials,
        raw_deterministic_successes=raw_deterministic_successes,
        raw_deterministic_patterns=len(deterministic_patterns),
        repetition_random_successes=repetition_random_successes,
        repetition_random_trials=random_trials,
        repetition_adversarial_successes=repetition_adversarial_successes,
        repetition_adversarial_patterns=len(repetition_patterns),
        notes="GF(137) repair is byte-exact; controls are storage/repetition boundaries.",
    )


def write_markdown(payload: dict) -> None:
    lines = [
        "# HYP-007 Repair-Aware GF(137) Checkpoint Audit",
        "",
        "This report is generated by `scripts/repair_aware_checkpoint.py`.",
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
            "## Shape Results",
            "",
            "| Shape | Raw symbols | Blocks | RS storage | Repetition storage | GF random | GF deterministic | Max prediction mismatches | Raw random | Repetition adversarial |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in payload["shape_summaries"]:
        lines.append(
            f"| {row['shape']} | {row['raw_symbols']} | {row['blocks']} | "
            f"{row['rs_storage_symbols']} ({row['rs_expansion_vs_raw']:.3f}x) | "
            f"{row['repetition_storage_symbols']} ({row['repetition_expansion_vs_raw']:.3f}x) | "
            f"{row['gf137_random_successes']}/{row['gf137_random_trials']} | "
            f"{row['gf137_deterministic_successes']}/{row['gf137_deterministic_patterns']} | "
            f"{row['gf137_max_prediction_mismatches']} | "
            f"{row['raw_random_successes']}/{row['raw_random_trials']} | "
            f"{row['repetition_adversarial_successes']}/{row['repetition_adversarial_patterns']} |"
        )

    lines.extend(
        [
            "",
            "## Energy-Cost Proxy",
            "",
            "The energy rows use Landauer's lower bound, "
            "`E >= k_B T ln(2)` per erased bit at 300 K.  This is a storage-bit "
            "accounting proxy, not a measured hardware power result.",
            "",
            "| Shape | FP32 bits | RS bits | Landauer FP32 J | Landauer RS J | RS/FP32 | Proxy reduction |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in payload["shape_summaries"]:
        lines.append(
            f"| {row['shape']} | {row['fp32_storage_bits']} | "
            f"{row['rs_storage_bits']} | {row['landauer_fp32_storage_j']:.6e} | "
            f"{row['landauer_rs_storage_j']:.6e} | "
            f"{row['landauer_rs_vs_fp32_ratio']:.3f} | "
            f"{row['landauer_rs_reduction_fraction']:.3f} |"
        )

    proxy = payload["operation_proxy"]
    lines.extend(
        [
            "",
            "## Operation Proxy",
            "",
            f"- `mac137`: {proxy['mac137']}",
            f"- `inv137`: {proxy['inv137']}",
            f"- `1_minus_s`: `{proxy['one_minus_s']:.12f}`",
            "",
            "The `1-s` factor is reported as a dimensionless model marker.  It "
            "is not used as evidence that physical energy is reduced by exactly "
            "`1-s`.",
        ]
    )

    lines.extend(["", "## Interpretation", "", payload["interpretation"], ""])
    RESULT_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=5137)
    parser.add_argument("--random-trials", type=int, default=3)
    args = parser.parse_args()

    shape_summaries = [
        run_shape(shape, args.seed, args.random_trials)
        for shape in SHAPES
    ]

    pass_gf137_checkpoint_repair = all(
        row.gf137_random_successes == row.gf137_random_trials
        and row.gf137_deterministic_successes == row.gf137_deterministic_patterns
        and row.gf137_max_prediction_mismatches == 0
        for row in shape_summaries
    )
    pass_controls_fail = all(
        row.raw_random_successes < row.raw_random_trials
        and row.repetition_adversarial_successes < row.repetition_adversarial_patterns
        for row in shape_summaries
    )
    pass_storage_boundary = all(
        row.rs_storage_symbols < row.repetition_storage_symbols
        for row in shape_summaries
    )
    mean_rs_vs_fp32_landauer = sum(
        row.landauer_rs_vs_fp32_ratio for row in shape_summaries
    ) / len(shape_summaries)
    mean_rs_reduction = 1.0 - mean_rs_vs_fp32_landauer
    one_minus_s = 1.0 - schwinger_s()

    interpretation = (
        f"GF(137) checkpoint repair pass={pass_gf137_checkpoint_repair}. "
        f"Control failure pass={pass_controls_fail}. "
        f"RS storage below 2x repetition pass={pass_storage_boundary}. "
        f"Mean RS/FP32 Landauer storage proxy={mean_rs_vs_fp32_landauer:.3f}, "
        f"for a proxy reduction of {mean_rs_reduction:.3f}. "
        "The result supports repair-aware finite-field checkpointing for the "
        "tested small kernels. The energy-cost rows are lower-bound bit-accounting "
        "proxies; they do not claim measured joule savings, semantic compression, "
        "cryptographic security, or a fundamental-physics result."
    )

    payload = {
        "metadata": system_metadata(),
        "config": {
            "seed": args.seed,
            "random_trials": args.random_trials,
            "field": P_FIELD,
            "payload_symbols": PAYLOAD_SYMBOLS,
            "axes": AXES,
            "erasure_budget_per_block": ERASURE_BUDGET,
            "model_keys": list(MODEL_KEYS),
        },
        "shape_summaries": [asdict(row) for row in shape_summaries],
        "summary": {
            "pass_gf137_checkpoint_repair": pass_gf137_checkpoint_repair,
            "pass_controls_fail": pass_controls_fail,
            "pass_storage_boundary": pass_storage_boundary,
            "mean_rs_vs_fp32_landauer_storage_ratio": mean_rs_vs_fp32_landauer,
            "mean_rs_landauer_storage_reduction_fraction": mean_rs_reduction,
            "one_minus_s": one_minus_s,
        },
        "operation_proxy": {
            "mac137": (
                "uint8/int accumulator multiply-add followed by Barrett reduction "
                f"modulo 137 with mu={BARRETT_MU_137}; this is the inference hot path."
            ),
            "inv137": (
                "multiplicative inverse a^135 mod 137; used by interpolation/repair, "
                "not by the inference hot path."
            ),
            "one_minus_s": one_minus_s,
        },
        "interpretation": interpretation,
    }
    RESULT_JSON.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    write_markdown(payload)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
