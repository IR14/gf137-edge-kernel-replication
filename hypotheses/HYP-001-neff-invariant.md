# HYP-001: CMB N_eff Invariant

Status: frozen-candidate

## Claim

The finite-field parameterization predicts a fixed effective relativistic
degree-of-freedom value

```text
N_eff* = 3.0570989176262793
```

under the following frozen equations:

```text
N = 5
D = 26
p = 137
q = N(N - 2) / (e^4 pi^3)
delta_phi = cos(1/N) - cos(2/N)
Delta N_eff = (D / p) * delta_phi * (1 - q)
N_eff* = 3.046 + Delta N_eff
```

Numerically:

```text
q = 0.008860611874290873
delta_phi = 0.05900558383835652
Delta N_eff = 0.011098917626279356
N_eff* = 3.0570989176262793
```

## Baseline

The baseline prediction is the standard value

```text
N_eff_LCDM = 3.046
```

This hypothesis does not claim a large deviation.  The predicted displacement
from the baseline is

```text
N_eff* - N_eff_LCDM = 0.011098917626279356
```

## Validation Dataset

The validation dataset must be an external CMB analysis that reports a
posterior constraint on `N_eff` under a standard `LambdaCDM + N_eff` extension.

The validation dataset must be named before its central value is inspected.

Allowed examples:

- a future public CMB-S4 `N_eff` constraint;
- a future Simons Observatory `N_eff` constraint;
- a new official combined CMB analysis released after this hypothesis file.

Current or already inspected constraints may be used only as context, not as
the primary validation test.

## Primary Metric

Let an external result report

```text
N_obs +/- sigma_obs
```

with approximately Gaussian uncertainty.

The primary metric is the fixed-model chi-square difference:

```text
Delta_chi2 =
    ((N_obs - 3.046)^2 - (N_obs - N_eff*)^2) / sigma_obs^2
```

Positive `Delta_chi2` favors `N_eff*` over the baseline.  Negative
`Delta_chi2` favors the baseline.

## Pass Condition

This hypothesis passes only if

```text
Delta_chi2 >= 4
```

on the named validation dataset, with no additional fitted parameters.

## Kill Condition

This hypothesis is downgraded or killed if any of the following occurs:

- the validation dataset is chosen after inspecting its `N_eff` central value;
- a new constant or free parameter is introduced after this file is frozen;
- `Delta_chi2 <= -4` on the named validation dataset;
- the published uncertainty becomes small enough to exclude `N_eff*` at
  greater than `3 sigma`;
- the result is explainable by a known systematic or prior choice in the
  validation analysis.

## Interpretation Rules

- A consistency result is not a discovery.
- A weak preference is not a discovery.
- A pass requires a pre-registered dataset and `Delta_chi2 >= 4`.
- A discovery-level claim requires independent replication beyond this gate.
