# Loop Iteration 62 - Step40 Gradient Conflict Diagnostic

Date: 2026-07-02

## Purpose

After loop56-61, most candidates produced the same pattern: one of hard/soft improved while the other regressed, or short-window gains disappeared in broader windows. Loop62 checks whether the bottleneck at the current 40-step baseline is still auxiliary-loss pressure or whether the output losses themselves have become conflicting.

This is a diagnostic only. No source code was changed and no training update was applied.

## Setup

- Checkpoint: `output/loop61-zero-mlp-bias-screen40-base/checkpoint-last.pt`
- Checkpoint step: 40
- Diagnostic batch: first FineWeb-Edu streaming batch, `seq=512`, `bs=2`
- Time steps: 4
- Precision: bf16 autocast
- Losses: paper weights `EA=0.2`, `SAA=0.1`, `SFA=0.1`, `STA=0.3`, `HTA=0.3`
- Command log: `logs/loop62-step40-gradient-diagnostic/diagnostic.log`

## Weighted Loss Values

| Term | Raw loss | Weighted loss |
| --- | ---: | ---: |
| EA | 2.8806 | 0.5761 |
| SAA | 0.5868 | 0.0587 |
| SFA | 0.4747 | 0.0475 |
| STA | 4.5278 | 1.3583 |
| HTA | 7.4951 | 2.2485 |

## Group Gradient Norms

| Group | EA | SAA | SFA | STA | HTA |
| --- | ---: | ---: | ---: | ---: | ---: |
| tied token/head | 0.0770 | 0.0010 | 0.0001 | 0.2655 | 0.4298 |
| position embeddings | 0.0250 | 0.0007 | 0.0001 | 0.0000 | 0.0000 |
| attention weights | 0.0000 | 0.1316 | 0.0151 | 0.0915 | 0.1143 |
| MLP weights | 0.0000 | 0.0539 | 0.0115 | 0.3212 | 0.2634 |
| block norms | 0.0000 | 0.0043 | 0.0006 | 0.0035 | 0.0038 |
| final norm | 0.0000 | 0.0000 | 0.0000 | 0.0781 | 0.0493 |
| readout scale | 0.0000 | 0.0000 | 0.0000 | 0.1998 | 0.1038 |

## Output-Loss Cosines

Cosines are computed within each parameter group. `NA` means at least one side had zero norm.

| Group | STA vs EA | STA vs SAA | STA vs SFA | STA vs HTA | HTA vs EA | HTA vs SAA | HTA vs SFA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| tied token/head | -0.0526 | -0.0034 | -0.0151 | +0.2449 | +0.0277 | +0.0024 | -0.0069 |
| position embeddings | +0.0113 | -0.0187 | -0.0498 | +0.4728 | +0.0043 | +0.0069 | -0.0952 |
| attention weights | NA | +0.0000 | +0.0185 | +0.2605 | NA | +0.0000 | +0.0415 |
| MLP weights | NA | -0.0000 | +0.0968 | +0.1482 | NA | +0.0000 | +0.0533 |
| block norms | NA | +0.0000 | +0.0208 | +0.1234 | NA | +0.0000 | +0.0322 |
| final norm | NA | NA | NA | -0.2937 | NA | NA | NA |
| readout scale | NA | NA | NA | -1.0000 | NA | NA | NA |

## Interpretation

At initialization, the previous gradient-cosine diagnostic showed STA and HTA were strongly aligned on the output path. By the loop61 baseline step40 checkpoint, that is no longer true:

- STA and HTA are only weakly aligned on the tied token/head group.
- STA and HTA are negative on the final readout norm.
- STA and HTA are exactly opposite on the scalar readout scale.
- EA/SAA/SFA remain near-orthogonal or small relative to output gradients on most shared groups.

This explains the repeated loop pattern where a candidate improves soft loss but regresses hard loss, or improves hard loss without a stable soft-window win. The current bottleneck is not a simple auxiliary-loss conflict.

## Decision

No promotion candidate is created from loop62 directly. Keep loop16 as the official small-gate metric baseline and loop14 as the best historical long-run behavior.

Next candidates should avoid more EA/SAA/SFA projector/input/reset variants unless they also address the STA/HTA output-path disagreement. Changing SpAD loss weights remains out of scope under the paper-faithful requirement, so any candidate must preserve the five loss definitions and weights.
