# Gradient Cosine Diagnostic

Date: 2026-07-02

## Purpose

Several microprobes suggested that auxiliary terms can change representation losses without improving primary hard/soft losses. This diagnostic checks whether the loss terms are directly fighting output learning or mostly consuming gradient budget through global clipping.

This is a one-batch diagnostic only. No repository code was changed.

## Setup

- Code baseline: `main` at `c842c92`
- Batch: first FineWeb-Edu batch, `seq=512`, `bs=2`
- Time steps: 4
- Precision: bf16 autocast
- Initialization: loop16-style teacher token/position embedding initialization
- Losses: paper weights `EA=0.2`, `SAA=0.1`, `SFA=0.1`, `STA=0.3`, `HTA=0.3`

## Weighted Loss Values

| Term | Raw loss | Weighted loss |
| --- | ---: | ---: |
| EA | 3.2569 | 0.6514 |
| SAA | 0.6284 | 0.0628 |
| SFA | 0.6286 | 0.0629 |
| STA | 7.9541 | 2.3862 |
| HTA | 11.5777 | 3.4733 |

## Cosines Against Output Losses

Cosines are computed within parameter groups. Missing gradients for a parameter are treated as zero so groups remain aligned by parameter name.

| Group | STA vs EA | STA vs SAA | STA vs SFA | STA vs HTA | HTA vs EA | HTA vs SAA | HTA vs SFA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| tied token/head | -0.0057 | -0.0056 | -0.0101 | +0.6941 | -0.0091 | -0.0038 | -0.0078 |
| attention weights | NA | +0.0000 | +0.0753 | +0.9548 | NA | +0.0000 | +0.0699 |
| MLP weights | NA | +0.0002 | +0.0664 | +0.9603 | NA | +0.0001 | +0.0620 |
| final norm | NA | NA | NA | +0.9836 | NA | NA | NA |
| readout scale | NA | NA | NA | +1.0000 | NA | NA | NA |

## Interpretation

STA and HTA are strongly aligned on the main output path. EA/SAA/SFA are not strongly opposite to the output losses; they are mostly near-orthogonal on the shared groups inspected here. This supports the current policy of not changing the paper loss weights directly: the issue is not a simple sign conflict.

SAA still has a large gradient norm relative to its scalar weighted loss, so global gradient clipping can allocate update budget to a near-orthogonal auxiliary direction. However, prior raw/direct SAA microprobes and gates did not improve primary hard/soft losses, so there is no supported SAA-only candidate to promote.

## Decision

No code loop is created from this diagnostic. Keep loop16 as the current best baseline and avoid loss-weight or clip-threshold changes unless the user explicitly relaxes the paper-faithful objective.
