# Freeze Protocol

Before validation, a hypothesis must be frozen.

## Freeze Inputs

Freeze at least:

- hypothesis markdown file;
- validation protocol;
- constants table;
- primary metric;
- pass condition;
- kill condition.

## Freeze Manifest

The freeze manifest records:

- UTC timestamp;
- file paths;
- SHA-256 hash for each file;
- combined SHA-256 hash over all file hashes.

After the manifest is written, changing the hypothesis means making a new
version, not silently editing the old one.

## Allowed Edits After Freeze

Allowed:

- typo fixes that do not change equations, metrics, data, or thresholds;
- adding a result section after validation;
- adding failed checks.

Not allowed:

- changing the validation dataset after seeing results;
- changing thresholds;
- adding constants;
- changing the primary metric.
