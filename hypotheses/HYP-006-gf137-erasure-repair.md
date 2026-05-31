# HYP-006: GF(137) Erasure-Repair Baseline

Status: pre-freeze repair audit

## Claim Under Test

GF(137) is not only a compact inference arithmetic.  It also supports exact
finite-field repair semantics.  A Reed-Solomon-style `RS(26,16)` code over
GF(137) can encode 16 byte-valued payload symbols into 26 axes and recover the
payload exactly after any 10 erased axes.

This is an information-repair engineering claim, not a fundamental-physics
claim.

## Why This Exists

HYP-002 through HYP-005 narrowed the speed claim.  ONNX Runtime int8 is often
faster than GF(137), so raw runtime is not the strongest differentiator.  The
next defensible direction is exact finite-field recovery under data loss.

## Baselines

Repair scheme:

- GF(137) `RS(26,16)` polynomial evaluation code.

Controls:

- raw 16-symbol payload with no parity;
- 2x direct repetition of 16 symbols, stored in 32 slots.

The controls are intentionally simple.  They show the difference between
compact storage, repetition, and algebraic erasure repair.

## Metrics

For each run:

- payload symbols: 16;
- encoded axes: 26;
- erasure budget: 10 axes;
- exact recovery success rate for random erasure sets;
- exact recovery on deterministic adversarial erasure patterns;
- storage expansion relative to the raw payload;
- whether the scheme has a worst-case guarantee under the tested erasure
  budget.

## Pass / Downgrade Criteria

The audit passes if:

- GF(137) `RS(26,16)` recovers exactly in all deterministic erasure patterns;
- GF(137) `RS(26,16)` recovers exactly in all random erasure trials;
- raw no-parity storage fails under the same erasure budget in random trials;
- 2x repetition is explicitly marked as larger and not worst-case safe against
  10 adversarial erasures.

The claim is downgraded if:

- GF(137) decoding fails for any tested erasure pattern within the 10-axis
  erasure budget;
- the implementation requires post-hoc tuning of the field, axis count, or
  payload length;
- a simpler code with comparable storage has the same worst-case guarantee and
  simpler implementation.

## Non-Claims

This audit does not claim:

- semantic compression;
- cryptographic security;
- biological or physical immortality;
- a fundamental law of nature.

It claims only exact finite-field erasure repair in a small reproducible code.
