# Loop Iteration 67 - SFFN Second LIF Screen

Date: 2026-07-02

## Hypothesis

The paper defines the spiking feed-forward layer as:

- `FC(x) = SN(Wx + b)`
- `SFFN(x) = FC2(FC1(x))`

The current local implementation applies a LIF neuron after `fc1`, but returns the raw analog output of `fc2`:

- current: `fc2(LIF(fc1(x)))`
- loop67 candidate: `LIF(fc2(LIF(fc1(x))))`

Loop67 tests whether adding the missing second SFFN spike neuron improves early hard/soft loss while keeping the rest of the training setup unchanged:

- keep teacher-copied token/position embeddings, tied LM head, SFSA, all five SpAD losses, paper weights, optimizer, scheduler, labels, and teacher logits unchanged;
- attach one `fc2_lif` to every student MLP block;
- patch `BiSpikMLP.forward` at runtime only.

This was a runtime monkeypatch only. No source code was changed.

## Setup

- GPU: 1x H200, GPU0
- Dataset: FineWeb-Edu streaming
- Sequence length: 512
- Time steps: 4
- Batch size: 2
- Gradient accumulation: 16
- Optimizer steps: 40
- Scheduler horizon: 80 steps
- Precision: bf16
- Seed: Python, NumPy, and Torch set to 0 before the run

## Runs

- Base run: `loop61-zero-mlp-bias-screen40-base`
- Base W&B: `wandb/offline-run-20260702_070958-2k9fzgtc/run-2k9fzgtc.wandb`
- Candidate run: `loop67-sffn-second-lif-screen40`
- Candidate W&B: `wandb/offline-run-20260702_074039-n5dtnhc4/run-n5dtnhc4.wandb`

## Result

| Variant | Step 40 hard | Step 40 soft | Last10 hard | Last10 soft | Last20 hard | Last20 soft | Last25 hard | Last25 soft | Full40 hard | Full40 soft |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| base | 7.6710 | 4.4294 | 7.8349 | 4.4474 | 7.7931 | 4.4301 | 7.8150 | 4.4569 | 8.2558 | 4.8073 |
| SFFN second LIF | 8.5531 | 4.5171 | 8.7588 | 4.5266 | 8.8099 | 4.5583 | 8.8587 | 4.5918 | 9.2868 | 4.9944 |

Deltas for candidate vs base:

- Step 40 hard/soft: +0.8821 / +0.0877
- Last10 hard/soft: +0.9239 / +0.0792
- Last20 hard/soft: +1.0168 / +0.1283
- Last25 hard/soft: +1.0436 / +0.1349
- Full40 hard/soft: +1.0311 / +0.1870

Secondary diagnostics at step 40:

| Variant | Token acc | Teacher agreement | Top5 acc | Target rank | Target margin | Spike rate | Hidden std | Logit std | Grad norm | Readout |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| base | 4.40% | 5.28% | 13.60% | 4285.7 | -5.1024 | 33.16% | 0.4711 | 1.5517 | 0.4328 | 0.9885 |
| SFFN second LIF | 0.39% | 0.78% | 9.78% | 3869.3 | -7.8523 | 62.44% | 0.3577 | 1.8915 | 1.3287 | 0.9843 |

Window slopes:

| Variant | Last10 hard slope/100 | Last10 soft slope/100 | Last25 hard slope/100 | Last25 soft slope/100 | Full40 hard slope/100 | Full40 soft slope/100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| base | -0.1364 | +0.5694 | -0.1509 | -0.4970 | -5.3884 | -4.3982 |
| SFFN second LIF | -1.2329 | +0.1515 | -1.5551 | -1.0179 | -5.2333 | -4.9960 |

## Decision

Fail loop67. Do not extend to an 80-step gate:

- It is worse than the matched base on every hard/soft metric window.
- Step40 hard loss regresses by +0.8821, and last25 hard regresses by +1.0436.
- Token accuracy collapses from 4.40% to 0.39%.
- Teacher top1 agreement collapses from 5.28% to 0.78%.
- Spike rate jumps from 33.16% to 62.44%, suggesting the extra SFFN spike neuron over-discretizes the MLP path instead of improving alignment.
- The target margin becomes much worse despite a lower target rank, which is consistent with poorer logit calibration.

Keep loop16 as the official small-gate metric baseline and loop14 as the best historical long-run behavior. A naive second LIF after `fc2` is not a promotion candidate.

