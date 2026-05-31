#!/usr/bin/env python3
"""Run the HYP-006 GF(137) erasure-repair audit.

The audit implements a small Reed-Solomon-style RS(26,16) code over GF(137).
Payload symbols are polynomial coefficients.  Encoded axes are evaluations at
the nonzero points 1..26.  Any 16 surviving axes determine the original degree
< 16 polynomial, so the payload is recoverable after any 10 erasures.
"""

from __future__ import annotations

import argparse
import itertools
import json
import platform
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT_JSON = ROOT / "outputs" / "gf137_erasure_repair.json"
RESULT_MD = ROOT / "outputs" / "gf137_erasure_repair.md"

FIELD = 137
AXES = 26
PAYLOAD_SYMBOLS = 16
ERASURE_BUDGET = AXES - PAYLOAD_SYMBOLS


@dataclass(frozen=True)
class RepairConfig:
    field: int = FIELD
    axes: int = AXES
    payload_symbols: int = PAYLOAD_SYMBOLS
    erasure_budget: int = ERASURE_BUDGET
    seed: int = 5137
    random_trials: int = 2000


@dataclass(frozen=True)
class SchemeSummary:
    scheme: str
    storage_symbols: int
    expansion_vs_raw: float
    random_trials: int
    random_successes: int
    random_success_rate: float
    adversarial_patterns: int
    adversarial_successes: int
    adversarial_success_rate: float
    worst_case_guarantee_under_budget: bool
    notes: str


def inv_mod(value: int, field: int = FIELD) -> int:
    return pow(value % field, field - 2, field)


def poly_eval(coefficients: list[int], x_value: int, field: int = FIELD) -> int:
    acc = 0
    power = 1
    x_mod = x_value % field
    for coefficient in coefficients:
        acc = (acc + coefficient * power) % field
        power = (power * x_mod) % field
    return acc


def encode_rs(payload: list[int], axes: int = AXES, field: int = FIELD) -> list[int]:
    if len(payload) != PAYLOAD_SYMBOLS:
        raise ValueError(f"payload must have {PAYLOAD_SYMBOLS} symbols")
    if axes >= field:
        raise ValueError("axis count must be smaller than field size")
    return [poly_eval(payload, x_value, field) for x_value in range(1, axes + 1)]


def solve_vandermonde(points: list[tuple[int, int]], field: int = FIELD) -> list[int]:
    size = len(points)
    matrix = []
    for x_value, y_value in points:
        row = []
        power = 1
        x_mod = x_value % field
        for _ in range(size):
            row.append(power)
            power = (power * x_mod) % field
        matrix.append(row + [y_value % field])

    for col in range(size):
        pivot = None
        for row in range(col, size):
            if matrix[row][col] % field:
                pivot = row
                break
        if pivot is None:
            raise ValueError("singular interpolation matrix")
        if pivot != col:
            matrix[col], matrix[pivot] = matrix[pivot], matrix[col]

        pivot_inv = inv_mod(matrix[col][col], field)
        matrix[col] = [(value * pivot_inv) % field for value in matrix[col]]

        for row in range(size):
            if row == col:
                continue
            factor = matrix[row][col] % field
            if factor == 0:
                continue
            matrix[row] = [
                (matrix[row][idx] - factor * matrix[col][idx]) % field
                for idx in range(size + 1)
            ]

    return [matrix[row][-1] % field for row in range(size)]


def decode_rs(encoded: list[int | None], payload_symbols: int = PAYLOAD_SYMBOLS) -> list[int]:
    survivors = [
        (idx + 1, value)
        for idx, value in enumerate(encoded)
        if value is not None
    ]
    if len(survivors) < payload_symbols:
        raise ValueError("not enough surviving axes to decode")
    return solve_vandermonde(survivors[:payload_symbols])[:payload_symbols]


def erase_values(values: list[int], erased_indices: set[int]) -> list[int | None]:
    return [None if idx in erased_indices else value for idx, value in enumerate(values)]


def deterministic_erasure_patterns() -> dict[str, set[int]]:
    return {
        "first_10": set(range(10)),
        "last_10": set(range(AXES - 10, AXES)),
        "even_10": set(range(0, 20, 2)),
        "odd_10": set(range(1, 21, 2)),
        "middle_10": set(range(8, 18)),
        "spread_10": {0, 3, 5, 8, 11, 14, 17, 20, 23, 25},
    }


def random_erasure_sets(rng: random.Random, population: int, erased: int, trials: int):
    for _ in range(trials):
        yield set(rng.sample(range(population), erased))


def raw_no_parity_success(erased_indices: set[int]) -> bool:
    payload_slots = set(range(PAYLOAD_SYMBOLS))
    return payload_slots.isdisjoint(erased_indices)


def repetition2_success(erased_indices: set[int]) -> bool:
    # Slots 0..15 hold copy A; slots 16..31 hold copy B.
    for symbol in range(PAYLOAD_SYMBOLS):
        if symbol in erased_indices and symbol + PAYLOAD_SYMBOLS in erased_indices:
            return False
    return True


def run_rs_trials(config: RepairConfig) -> SchemeSummary:
    rng = random.Random(config.seed)
    random_successes = 0
    start = time.perf_counter()
    for erased in random_erasure_sets(
        rng, config.axes, config.erasure_budget, config.random_trials
    ):
        payload = [rng.randrange(config.field) for _ in range(config.payload_symbols)]
        encoded = encode_rs(payload, config.axes, config.field)
        decoded = decode_rs(erase_values(encoded, erased), config.payload_symbols)
        if decoded == payload:
            random_successes += 1

    adversarial_successes = 0
    for name, erased in deterministic_erasure_patterns().items():
        payload = [((idx + 1) * 17 + len(name)) % config.field for idx in range(config.payload_symbols)]
        encoded = encode_rs(payload, config.axes, config.field)
        decoded = decode_rs(erase_values(encoded, erased), config.payload_symbols)
        if decoded == payload:
            adversarial_successes += 1

    _elapsed = time.perf_counter() - start
    pattern_count = len(deterministic_erasure_patterns())
    return SchemeSummary(
        scheme="GF(137) RS(26,16)",
        storage_symbols=config.axes,
        expansion_vs_raw=config.axes / config.payload_symbols,
        random_trials=config.random_trials,
        random_successes=random_successes,
        random_success_rate=random_successes / config.random_trials,
        adversarial_patterns=pattern_count,
        adversarial_successes=adversarial_successes,
        adversarial_success_rate=adversarial_successes / pattern_count,
        worst_case_guarantee_under_budget=True,
        notes="MDS polynomial code over GF(137); any 16 surviving axes recover payload.",
    )


def run_raw_trials(config: RepairConfig) -> SchemeSummary:
    rng = random.Random(config.seed)
    successes = sum(
        1
        for erased in random_erasure_sets(
            rng, config.axes, config.erasure_budget, config.random_trials
        )
        if raw_no_parity_success(erased)
    )
    adversarial = deterministic_erasure_patterns()
    adversarial_successes = sum(1 for erased in adversarial.values() if raw_no_parity_success(erased))
    return SchemeSummary(
        scheme="Raw 16-symbol payload",
        storage_symbols=config.payload_symbols,
        expansion_vs_raw=1.0,
        random_trials=config.random_trials,
        random_successes=successes,
        random_success_rate=successes / config.random_trials,
        adversarial_patterns=len(adversarial),
        adversarial_successes=adversarial_successes,
        adversarial_success_rate=adversarial_successes / len(adversarial),
        worst_case_guarantee_under_budget=False,
        notes="Compact storage without parity; any erased payload slot loses information.",
    )


def run_repetition_trials(config: RepairConfig) -> SchemeSummary:
    rng = random.Random(config.seed)
    storage_symbols = 2 * config.payload_symbols
    successes = sum(
        1
        for erased in random_erasure_sets(
            rng, storage_symbols, config.erasure_budget, config.random_trials
        )
        if repetition2_success(erased)
    )
    adversarial_patterns = {
        "erase_first_five_pairs": set(itertools.chain.from_iterable((idx, idx + PAYLOAD_SYMBOLS) for idx in range(5))),
        "erase_middle_five_pairs": set(itertools.chain.from_iterable((idx, idx + PAYLOAD_SYMBOLS) for idx in range(5, 10))),
        "erase_last_five_pairs": set(itertools.chain.from_iterable((idx, idx + PAYLOAD_SYMBOLS) for idx in range(11, 16))),
    }
    adversarial_successes = sum(
        1 for erased in adversarial_patterns.values() if repetition2_success(erased)
    )
    return SchemeSummary(
        scheme="2x direct repetition",
        storage_symbols=storage_symbols,
        expansion_vs_raw=storage_symbols / config.payload_symbols,
        random_trials=config.random_trials,
        random_successes=successes,
        random_success_rate=successes / config.random_trials,
        adversarial_patterns=len(adversarial_patterns),
        adversarial_successes=adversarial_successes,
        adversarial_success_rate=adversarial_successes / len(adversarial_patterns),
        worst_case_guarantee_under_budget=False,
        notes="Simple duplicate storage; larger than RS(26,16) and not safe against adversarial paired erasures.",
    )


def system_metadata() -> dict[str, str]:
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": sys.version.replace("\n", " "),
    }


def write_markdown(payload: dict) -> None:
    lines = [
        "# HYP-006 GF(137) Erasure-Repair Audit",
        "",
        "This report is generated by `scripts/gf137_erasure_repair.py`.",
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
            "| Scheme | Storage symbols | Expansion | Random success | Adversarial success | Worst-case guarantee | Notes |",
            "|---|---:|---:|---:|---:|---|---|",
        ]
    )
    for row in payload["schemes"]:
        guarantee = "yes" if row["worst_case_guarantee_under_budget"] else "no"
        lines.append(
            f"| {row['scheme']} | {row['storage_symbols']} | "
            f"{row['expansion_vs_raw']:.3f}x | "
            f"{row['random_successes']}/{row['random_trials']} "
            f"({row['random_success_rate']:.3f}) | "
            f"{row['adversarial_successes']}/{row['adversarial_patterns']} "
            f"({row['adversarial_success_rate']:.3f}) | "
            f"{guarantee} | {row['notes']} |"
        )

    lines.extend(["", "## Interpretation", "", payload["interpretation"], ""])
    RESULT_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=5137)
    parser.add_argument("--random-trials", type=int, default=2000)
    args = parser.parse_args()

    config = RepairConfig(seed=args.seed, random_trials=args.random_trials)
    schemes = [
        run_rs_trials(config),
        run_raw_trials(config),
        run_repetition_trials(config),
    ]

    rs = schemes[0]
    raw = schemes[1]
    repetition = schemes[2]
    pass_rs_repair = (
        rs.random_successes == rs.random_trials
        and rs.adversarial_successes == rs.adversarial_patterns
    )
    pass_controls = raw.random_success_rate < 1.0 and repetition.adversarial_success_rate < 1.0
    pass_storage_boundary = rs.storage_symbols < repetition.storage_symbols

    interpretation = (
        f"GF(137) RS(26,16) repair pass={pass_rs_repair}. "
        f"Control limitation pass={pass_controls}. "
        f"RS storage is {rs.storage_symbols} symbols versus {repetition.storage_symbols} "
        f"symbols for 2x repetition, so storage-boundary pass={pass_storage_boundary}. "
        "The result supports an exact finite-field erasure-repair property, not "
        "semantic compression, cryptography, or a fundamental-physics claim."
    )

    payload = {
        "metadata": system_metadata(),
        "config": asdict(config),
        "schemes": [asdict(scheme) for scheme in schemes],
        "summary": {
            "pass_rs_repair": pass_rs_repair,
            "pass_controls": pass_controls,
            "pass_storage_boundary": pass_storage_boundary,
            "rs_random_success_rate": rs.random_success_rate,
            "raw_random_success_rate": raw.random_success_rate,
            "repetition_random_success_rate": repetition.random_success_rate,
        },
        "interpretation": interpretation,
    }
    RESULT_JSON.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    write_markdown(payload)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
