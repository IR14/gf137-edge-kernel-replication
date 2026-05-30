# Replication Gate

A result cannot leave this directory as a "main claim" until it passes this
gate.

## Minimum Package

- one command that runs the test;
- public data source or exact data manifest;
- frozen constants;
- no hidden manual edits;
- machine-readable result table;
- short negative-result section.

## Reviewer Questions

Before sharing, answer:

1. What would make this result false?
2. Which data were not used during derivation?
3. What is the null baseline?
4. What is the smallest code path that reproduces the number?
5. Which checks failed?

## Evidence Levels

Level 0:

- idea only.

Level 1:

- derivation frozen.

Level 2:

- blind test run once.

Level 3:

- independent data or machine replication.

Level 4:

- external person reproduces the result.

Level 5:

- field experts cannot explain it away with standard systematics.
