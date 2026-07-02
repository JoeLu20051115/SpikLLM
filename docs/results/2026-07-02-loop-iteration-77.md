# Loop Iteration 77 - Token-Random Gradient Conflict Diagnostic

Date: 2026-07-02

## Purpose

Loop72 token-random improved last25 hard/soft means and step80 hard loss, but failed the small-gate promotion rule because step80 soft loss remained worse than the loop16 official baseline and the recent soft slope was not clearly descending. Loop77 checks whether token-random actually fixes the loop62 STA/HTA output-path conflict, or whether it only improves auxiliary alignment while leaving the output losses opposed.

This is a diagnostic only. No source code was changed, no optimizer step was applied, and no candidate was promoted.

## Setup

- GPU: 1x H200, GPU0
- Batch: first FineWeb-Edu streaming batch from the loop72 geometry
- Sequence length: 512
- Batch size: 2
- Time steps: 4
- Precision: bf16 autocast
- Losses: paper SpAD weights `EA=0.2`, `SAA=0.1`, `SFA=0.1`, `STA=0.3`, `HTA=0.3`
- Checkpoints:
  - base: `output/loop61-zero-mlp-bias-screen40-base/checkpoint-last.pt`
  - token-random step40: `output/loop71-embedding-init-split-screen40-token-random/checkpoint-last.pt`
  - token-random step80: `output/loop72-token-random-gate80/checkpoint-last.pt`
- Logs:
  - initial failed script attempt: `logs/loop77-token-random-gradient-diagnostic/diagnostic.log`
  - successful rerun: `logs/loop77-token-random-gradient-diagnostic/diagnostic-rerun.log`

The first script attempt failed because parameters with no gradient were omitted from some group vectors, which made cosine vector lengths inconsistent. The rerun fixed the diagnostic by inserting zero gradients for grouped parameters with no gradient.

## Weighted Loss Values

| Variant | EA raw / weighted | SAA raw / weighted | SFA raw / weighted | STA raw / weighted | HTA raw / weighted |
| --- | ---: | ---: | ---: | ---: | ---: |
| base step40 | 2.8806 / 0.5761 | 0.5868 / 0.0587 | 0.4747 / 0.0475 | 4.5278 / 1.3583 | 7.4951 / 2.2485 |
| token-random step40 | 0.5729 / 0.1146 | 0.5457 / 0.0546 | 0.4045 / 0.0404 | 4.5320 / 1.3596 | 7.5273 / 2.2582 |
| token-random step80 | 0.5377 / 0.1075 | 0.5530 / 0.0553 | 0.3600 / 0.0360 | 4.5416 / 1.3625 | 7.5335 / 2.2601 |

On the same diagnostic batch, token-random strongly improves EA/SFA but does not improve STA/HTA values. This matches loop72's pattern: useful auxiliary alignment and some training-window gains, but no clean step80 soft-loss win.

## STA vs HTA Gradient Cosines

| Group | base step40 | token-random step40 | token-random step80 |
| --- | ---: | ---: | ---: |
| tied token/head | +0.2449 | +0.2676 | +0.2898 |
| position embeddings | +0.4728 | +0.1469 | +0.3060 |
| attention weights | +0.2605 | +0.2689 | +0.2329 |
| MLP weights | +0.1482 | -0.1868 | +0.0368 |
| block norms | +0.1234 | +0.1294 | +0.2548 |
| final norm | -0.2937 | -0.2824 | -0.2151 |
| readout scale | -1.0000 | -1.0000 | -1.0000 |

## STA / HTA Gradient Norms

| Group | base STA / HTA | token-random step40 STA / HTA | token-random step80 STA / HTA |
| --- | ---: | ---: | ---: |
| tied token/head | 0.2655 / 0.4298 | 0.2749 / 0.4678 | 0.2737 / 0.4387 |
| attention weights | 0.0915 / 0.1143 | 0.0202 / 0.0159 | 0.0231 / 0.0210 |
| MLP weights | 0.3212 / 0.2634 | 0.0613 / 0.0512 | 0.0970 / 0.0404 |
| final norm | 0.0781 / 0.0493 | 0.0139 / 0.0150 | 0.0175 / 0.0135 |
| readout scale | 0.1998 / 0.1038 | 0.1619 / 0.1325 | 0.2114 / 0.1057 |

## Interpretation

Token-random changes the training geometry, but it does not remove the output bottleneck:

- EA and SFA are much lower, so token-random genuinely helps auxiliary alignment.
- STA/HTA on the diagnostic batch are slightly worse than base at both step40 and step80.
- The tied token/head STA/HTA cosine improves modestly, which explains why hard-loss windows can improve.
- The scalar readout remains exactly opposite for STA and HTA.
- The final layer norm remains negative, although less negative by step80.
- MLP STA/HTA alignment becomes worse under token-random, especially at step40.

This supports loop72's decision not to promote token-random to long training. The unresolved issue is still the output-path conflict between soft and hard losses, not EA/SAA/SFA alone.

## Decision

Do not promote loop77; it is diagnostic only.

For the next candidate, prefer a small-screen output-path change that reduces the final-norm/readout conflict while preserving the paper's five SpAD losses and weights. Because loop73 already showed that freezing the readout scale alone fails, the next test should avoid another readout-only variant unless it is paired with a mechanism that directly targets the final output normalization conflict.
