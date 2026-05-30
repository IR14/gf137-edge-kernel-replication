# HYP-002: GF(137) Edge-Kernel Replication

Status: active-replication-track

## Claim

For small dense inference kernels, a GF(137)-bounded `uint8` representation can
reduce model storage relative to `float32` and can improve runtime on
integer-friendly hardware under fixed modular arithmetic constraints.

This is an engineering claim, not a fundamental-physics claim.

## Frozen Metrics

Primary metrics:

- model storage in bytes;
- time for 1000 forward passes;
- exact output agreement with the frozen byte-valued kernel.

## Baselines

Required baselines:

- NumPy `float32`;
- NumPy `uint32` GF(137) reference;
- C++ `float32` modular kernel with the same field operation;
- C++ `uint8_t` plain-threshold kernel without residue reduction;
- C++ portable and native GF(137) loops with compiler optimizations;
- optional ARM NEON path when hardware is available.

## Pass Condition

The replication passes if an external machine reproduces:

- approximately `4x` storage reduction relative to `float32`;
- no output mismatch against the frozen byte-valued reference;
- speedup over the NumPy `float32` modular baseline on the same machine;
- speedup over the C++ `float32` modular baseline on the same machine.

The plain `uint8_t` row is a non-equivalent control. It measures loop and
storage overhead without the GF(137) residue operation, so it is reported but
not included in the exact-agreement gate.

## Kill Condition

The claim is downgraded if:

- results require a machine-specific manual tweak;
- a standard int8/TFLite/ONNX baseline beats the GF(137) path on all relevant
  metrics;
- output agreement fails under the frozen test vectors;
- the benchmark cannot be reproduced by a fresh clone and one command.

## Next Step

Build a standalone benchmark package with:

- fixed test vectors;
- fixed compiler flags;
- raw timing logs;
- machine metadata;
- one-command runner.
