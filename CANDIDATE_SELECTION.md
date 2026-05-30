# Candidate Selection

The first target should maximize falsifiability, not rhetorical scope.

## Candidate Matrix

| Candidate | Type | Primary number | Near-term testability | Risk | Decision |
|---|---|---:|---|---|---|
| CMB `N_eff` invariant | physics | `N_eff = 3.0570989176262793` | medium/long | effect is small | active physics track |
| GF(137) edge-kernel replication | engineering | storage and runtime ratios | high | not fundamental physics | active replication track |
| DESI directional residual | cosmology | fixed residual amplitude | medium | prior analyses already inspected related data | parked until a clean holdout is named |
| particle mass coincidences | particle phenomenology | ppm residuals | high for tables, low for new prediction | mostly retrospective | not first target |

## Selected Physics Track

The selected physics track is `HYP-001`: a frozen CMB effective-neutrino-count
prediction.

It has a small effect size, which is a weakness.  Its advantage is that it is
fully specified before any new CMB validation result is used.

## Selected Replication Track

The selected near-term replication track is `HYP-002`: reproduce the GF(137)
integer-kernel memory and runtime claims on at least two machines.

This is not a fundamental-physics claim.  It is a discipline check: if the
small engineering result cannot replicate cleanly, the larger physical claims
should not be trusted.
