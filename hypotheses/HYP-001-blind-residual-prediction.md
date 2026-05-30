# HYP-001: Blind Residual Prediction

Status: draft

## Claim

A fixed finite-field-derived correction may predict a small residual in an
independent physical dataset after standard baselines are applied.

This is not yet a claim about new physics.  It is a test harness for whether
the archived e-5-137/GF(137) structures can produce a prediction that was not
chosen after seeing the validation result.

## Frozen Inputs

To be filled before running any validation:

- constants:
  - `N = 5`
  - `D = 26`
  - `p = 137`
  - `I5 = 42`
- derived correction:
  - pending
- target observable:
  - pending
- validation dataset:
  - pending

## Pass Metric

The hypothesis must define exactly one primary scalar metric before analysis.

Examples:

- signed residual amplitude;
- chi-square improvement with fixed degrees of freedom;
- held-out likelihood difference;
- pre-registered bin-to-bin trend.

## Kill Condition

This hypothesis fails if any of the following happens:

- the validation dataset is inspected before the correction is frozen;
- an extra free parameter is added after the first result;
- the effect is smaller than the stated systematic uncertainty;
- the effect cannot beat the null baseline on the primary metric.

## Required Null Checks

- shuffled labels or randomized sky positions, if cosmology;
- alternate binning;
- jackknife or block-null split;
- comparison to a standard baseline model;
- explicit report of failed variants.

## Next Decision

Pick one concrete target:

1. cosmological residual in public DESI/Planck/ACT products;
2. particle-data residual in PDG tables;
3. laboratory-scale integer-kernel prediction that can be reproduced on a
   second machine.
