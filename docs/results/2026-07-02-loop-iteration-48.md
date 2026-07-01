# Loop Iteration 48 - Isolated Internal Norm Removal Screen

Date: 2026-07-02

## Hypothesis

Loop47 showed a strong 20-step win when both internal pre-LayerNorms were removed, but the effect reversed by 40 steps. This screen isolates which internal norm contributed to the early signal:

- remove only `attention_norm`;
- remove only `mlp_norm`.

Both candidates were runtime monkeypatches only. No source code was changed.

## Setup

- Base code: stable gradient clipping and scheduler-horizon support.
- GPU: 1x H200, GPU0
- Dataset: FineWeb-Edu streaming, first 320 prefetched micro-batches reused by all variants
- Sequence length: 512
- Time steps: 4
- Batch size: 2
- Gradient accumulation: 16
- Optimizer steps: 20
- Precision: bf16
- Seed: Python, NumPy, and Torch set to 0 before each variant

## Result

| Variant | Step 20 hard | Step 20 soft | Last10 hard | Last10 soft | Token acc | Teacher agreement | Target rank | Target margin | Spike rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| base | 7.9454 | 4.5078 | 7.9000 | 4.4800 | 4.01% | 6.07% | 5081.1 | -5.0093 | 45.11% |
| no `attention_norm` | 7.9593 | 4.4963 | 7.8842 | 4.4684 | 6.16% | 7.14% | 5383.6 | -5.0734 | 59.23% |
| no `mlp_norm` | 7.9640 | 4.4950 | 7.9117 | 4.4842 | 3.42% | 5.09% | 4945.8 | -5.0497 | 42.17% |

Deltas for candidates vs base:

| Variant | Step20 hard | Step20 soft | Last10 hard | Last10 soft |
| --- | ---: | ---: | ---: | ---: |
| no `attention_norm` | +0.0139 | -0.0114 | -0.0158 | -0.0117 |
| no `mlp_norm` | +0.0186 | -0.0127 | +0.0117 | +0.0041 |

## Decision

Do not promote either partial-norm candidate.

Removing only `attention_norm` improves soft loss, last10 hard/soft, token accuracy, and teacher agreement, but step20 hard loss, target rank, and target margin regress. Removing only `mlp_norm` regresses hard loss and recent-window losses. Neither is a clear 20-step small-screen win, so no 40-step or 80-step gate is justified.

Keep loop16 as the official small-gate metric baseline. Loop47's early gain appears to require the coupled removal, but that coupled variant failed the 40-step confirmation.
