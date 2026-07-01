# SAA Gradient and Raw+Direct Microprobe - No Loop Promotion

Date: 2026-07-02

## Purpose

After loop35 failed, a single-batch gradient decomposition showed that the attention-alignment path was a plausible source of update imbalance. This diagnostic checks whether a more paper-literal SAA variant should become the next loop:

- use raw MSE for SAA instead of distribution-normalized MSE;
- feed teacher attention probabilities directly into the rate encoder instead of row-max scaling to the spike threshold.

This exact combination had not been tested by the earlier loops: loop26 tested raw SAA only, and loop27 tested direct teacher attention drive only.

No repository code was changed for these diagnostics.

## Gradient Decomposition

Setup: `main` at `c39c3db`, one FineWeb-Edu batch, `seq=512`, `bs=2`, `T=4`, bf16 autocast, loop16-style initialization.

Weighted per-loss gradient norms:

| Term | Raw loss | Weight | Weighted loss | Total grad norm |
| --- | ---: | ---: | ---: | ---: |
| EA | 3.2569 | 0.2 | 0.6514 | 0.0898 |
| SAA | 0.6284 | 0.1 | 0.0628 | 24.1417 |
| SFA | 0.6286 | 0.1 | 0.0629 | 0.0247 |
| STA | 7.9541 | 0.3 | 2.3862 | 7.1945 |
| HTA | 11.5777 | 0.3 | 3.4733 | 9.9647 |
| Total | 6.6366 | 1.0 | 6.6366 | 29.5142 |

The current SAA term has a small scalar contribution but a large gradient contribution, mainly through attention and MLP weights. This explains why an SAA-form microprobe was worth testing before another code loop.

## 20-Step Microprobe

Matched setup: first 320 prefetched FineWeb micro-batches reused across variants, `seq=512`, `bs=2`, `GA=16`, `T=4`, bf16, seed 0.

| Variant | Step 20 hard | Step 20 soft | Last10 hard | Last10 soft | Step20 delta vs base | Last10 delta vs base |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| base | 7.9564 | 4.5028 | 7.8951 | 4.4773 | 0.0000 / 0.0000 | 0.0000 / 0.0000 |
| raw+direct SAA | 7.9366 | 4.4995 | 7.8895 | 4.4737 | -0.0198 / -0.0033 | -0.0056 / -0.0036 |

The 20-step screen was only slightly positive, below the "clearly better" threshold.

## 40-Step Microprobe

Because the 20-step result was the first same-direction SAA signal, it was extended to 40 steps before considering a numbered loop. This run used first 640 prefetched FineWeb micro-batches reused across variants. Warmup differs from the 20-step probe because the scheduler uses the configured max step count, so compare only within this table.

| Variant | Step 40 hard | Step 40 soft | Last10 hard | Last10 soft | Last20 hard | Last20 soft |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| base | 7.6659 | 4.3908 | 7.7851 | 4.3733 | 7.7640 | 4.3830 |
| raw+direct SAA | 7.7252 | 4.4344 | 7.8284 | 4.4039 | 7.7766 | 4.3911 |

Deltas for raw+direct SAA vs base:

- Step 40 hard/soft: +0.0594 / +0.0437
- Last10 hard/soft: +0.0433 / +0.0306
- Last20 hard/soft: +0.0126 / +0.0080

## Decision

Do not create a loop36 candidate for raw+direct SAA. It slightly improved the 20-step screen but failed the 40-step confirmation on every primary hard/soft metric. Do not run an 80-step gate.

Keep loop16 as the current best baseline. The next candidate should not be another attention-loss-only variant unless it has a stronger mechanistic reason and passes the fixed streaming micro-screen first.
