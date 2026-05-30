# Roadmap

## Phase 0: Freeze the Question

Choose exactly one question.

Bad question:

> Can e-5-137 explain modern physics?

Good question:

> Does a fixed e-5-137 correction predict a pre-registered residual in an
> independent dataset better than the null baseline?

Deliverable:

- one hypothesis file in `hypotheses/`;
- one validation dataset named before analysis;
- one scalar pass/fail metric.

Current first target:

- `HYP-002`: GF(137) edge-kernel replication, used as a fast discipline check
  before larger physical claims are pursued.

## Phase 1: Derive Before Measuring

Write the derivation without using the validation data.

Deliverable:

- frozen equations;
- parameter table;
- uncertainty budget;
- kill condition.

## Phase 2: Blind Test

Run the validation once.

Deliverable:

- raw result;
- null baseline;
- ablation table;
- scripts with exact command lines.

## Phase 3: External Replication Package

Make it easy for another person to reproduce or refute.

Deliverable:

- minimal repository;
- README with one command;
- public data manifest;
- expected result hash or table.

## Phase 4: Submission Target

Only after a blind pass:

- short technical note;
- arXiv or field-appropriate preprint;
- clear statement of what was not shown.

## Stop Conditions

Stop or downgrade the hypothesis if:

- the result appears only after adding a new tuning parameter;
- the sign of the effect changes under a standard systematic check;
- the effect disappears on the first independent dataset;
- the result cannot be stated in one sentence with a number.
