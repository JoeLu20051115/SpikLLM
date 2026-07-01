# Loop Iteration 35 - Attention Spike Threshold Floor

Date: 2026-07-02

## Hypothesis

Several recent loops changed output calibration, temporal input, and EA scale without beating loop16. A direct SFSA diagnostic showed that `SN_Attn` is extremely dense under the current threshold: on a fixed `seq=128` batch, early layers fired on roughly 75-85% of allowed causal attention entries. Because `attn_int` is an integer spike-overlap count, using the global threshold `0.7` means any single overlapping Q/K spike can activate attention. This loop tests a single structural change: keep all model thresholds at `0.7`, but require at least one full integer overlap for the attention-map neuron by using `max(spike_threshold, 1.0)` only for `attn_lif`.

## Code Change

- Branch: `loop35-attn-threshold-floor`
- Commit: `d9ad9f8` (`fix: require integer overlap for attention spikes`)
- Only `BiSpikAttention.attn_lif` uses threshold floor `1.0`.
- Q/K/V, attention-output, final attention output, MLP, and block LIF thresholds remain `config.spike_threshold`.

## Verification

- RED test: `test_sfsa_attention_neuron_uses_integer_overlap_threshold_floor` failed before implementation because `attn_lif.v_threshold` was `0.7`.
- Targeted GREEN: `test_sfsa_attention_neuron_uses_integer_overlap_threshold_floor` passed.
- Full smoke before gate: `44 passed, 58 warnings`.
- Fixed-batch pre-gate screen was positive:
  - Loop16 reference from the same script: final hard/soft 5.2881 / 1.0851, last-8 hard/soft 5.3367 / 1.1523.
  - Loop35 candidate: final hard/soft 4.8727 / 0.9481, last-8 hard/soft 4.9065 / 0.9936.

## Small-Batch Gate

Matched geometry:
- GPU: 1x H200
- Sequence length: 512
- Time steps: 4
- Per-GPU batch size: 2
- Gradient accumulation: 16
- Max optimizer steps: 80
- Precision: bf16
- Seed wrapper: Python, NumPy, and Torch seeds set to 0 before entering the training CLI.

Current baseline:
- Code state: loop16 identity-projector candidate, commit `9c0d64f`
- W&B: `cvxuw267`
- Step 80 hard/soft: 7.4532 / 4.5195
- Last 25-step hard/soft means: 7.7798 / 4.2932
- Token accuracy at step 80: 4.79%
- Teacher top-1 agreement at step 80: 6.07%
- Target rank mean at step 80: 4162.4
- Target margin mean at step 80: -4.6015

Loop35 candidate:
- Run: `loop35-attnthr1-small-seq512-bs2-ga16-1xh200-20260702-001333`
- W&B: https://wandb.ai/luenqiao2005-agency-for-science-technology-and-research/bispikclm/runs/y5utwr7o
- Step 80 hard/soft: 7.4586 / 4.5449
- Last 25-step hard/soft means: 7.7739 / 4.2949
- Token accuracy at step 80: 4.70%
- Teacher top-1 agreement at step 80: 5.68%
- Target rank mean at step 80: 4143.8
- Target margin mean at step 80: -4.6489
- Spike rate mean at step 80: 22.55%
- Readout scale at step 80: 0.9838
- Logit std at step 80: 1.5048

## Decision

Fail the loop35 candidate. The fixed-batch screen improved, and last-25 hard loss is slightly better than loop16, but step-80 hard and soft losses are both worse, last-25 soft is worse, and token accuracy / teacher agreement / target margin regress. This is not a clear small-batch win.

Do not merge loop35 code into `main`, and do not launch a long probe. Keep loop16 as the current small-batch best baseline.

## Baseline Refresh

Because loop35 was close on last-25 hard loss, reran the loop16 baseline with the same seed, code, and gate geometry:

- Run: `loop16-refresh-small-seq512-bs2-ga16-1xh200-20260702-001801`
- W&B: https://wandb.ai/luenqiao2005-agency-for-science-technology-and-research/bispikclm/runs/1f8plmtv
- Step 80 hard/soft: 7.45319 / 4.51955
- Last 25-step hard/soft means: 7.77983 / 4.29321
- Token accuracy at step 80: 4.79%
- Teacher top-1 agreement at step 80: 6.07%
- Target rank mean at step 80: 4162.4
- Target margin mean at step 80: -4.6015

This reproduces the original loop16 gate (`cvxuw267`) to rounding precision, so the loop35 failure is not explained by streaming-data variance.

Next direction: attention sparsity alone is insufficient. The next loop should inspect why fixed-batch improvements are not transferring to the streaming gate, with special attention to whether the streaming gate variance or dataset order is masking small candidate effects.
