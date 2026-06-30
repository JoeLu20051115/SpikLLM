# BiSpikCLM Paper-Faithful Loop Engineering Design

## Goal

Iterate on the BiSpikCLM training implementation until a paper-faithful 3x H200 probe shows healthy early optimization. A probe is healthy when hard-label and soft-target losses either fall below 5 within 2000 optimizer steps or show a strong enough downward trend to justify one additional 2000-step confirmation window.

Once a probe passes, continue the accepted run into full training instead of repeatedly restarting.

## Fixed Constraints

The loop must stay faithful to the BiSpikCLM offline SpAD setup.

- Teacher: OPT-family causal LM, starting with `facebook/opt-125m`.
- Data: FineWeb-Edu streaming corpus.
- Student: BiSpikCLM causal LM path, not an ANN substitute.
- Loss terms: embedding alignment, attention alignment, feature alignment, soft-target distillation, hard-label CE.
- Loss weights: `lambda_emb=0.2`, `lambda_attn=0.1`, `lambda_feat=0.1`, `lambda_soft=0.3`, `lambda_hard=0.3`.
- Distillation temperature: `2.0`.
- Optimizer/schedule defaults: Adam, cosine schedule, warmup ratio `0.2`, gradient clip `0.7`.

The loop must not use shortcuts that make the training unfair:

- Do not train on a fixed batch for probe acceptance.
- Do not leak labels into logits.
- Do not replace student logits with teacher logits.
- Do not disable or down-weight hard loss or soft loss.
- Do not change the paper loss weights to make curves look better.
- Do not count a dummy-batch overfit as full-training evidence.

## Current Baseline Evidence

The previous GA=1 run was stopped because it was both non-faithful and unhealthy:

- Command used `batch_size=8`, `gradient_accumulation_steps=1`, `world_size=3`.
- Effective batch was 24 sequences, far below the paper target.
- At step 4338, hard and soft losses still averaged about 14.72 and 12.43 over the last 1000 steps.
- Representation losses decreased, but output losses did not show a passing trend.

This run is evidence for debugging, not an acceptable full-training run.

## Probe Acceptance Rule

Each candidate implementation gets one 2000-step paper-faithful probe.

Pass immediately if:

- hard loss is below 5, and
- soft loss is below 5.

Allow one additional 2000-step confirmation window if:

- hard and soft losses have not both reached 5, but
- their last-window slopes are strongly negative, and
- their last-window means are clearly lower than their early-window means.

Fail and stop the run if:

- hard or soft loss remains in high-level oscillation,
- either loss mean rises over the last 500 to 1000 steps,
- gradients become unstable or non-finite,
- the run OOMs under a configuration that should fit,
- the code or command violates the fixed constraints.

## Loop Procedure

Each loop iteration follows the same sequence:

1. Gather evidence from the previous run: process state, W&B/local history, checkpoints, loss windows, learning rate, gradient norm, and memory.
2. State one root-cause hypothesis.
3. Make the smallest implementation change that tests that hypothesis.
4. Run local smoke tests.
5. Request code review of the implementation diff, with specific attention to paper faithfulness and unfair shortcuts.
6. Fix critical or important review findings before launching.
7. Launch a 3x H200 warmup/probe run with W&B enabled.
8. Monitor until 2000 optimizer steps or earlier failure.
9. Apply the probe acceptance rule.
10. Keep the change only if evidence improves; otherwise revert the change and form a new hypothesis.

## Candidate Root-Cause Areas

The first iterations should prioritize implementation issues that can explain hard/soft loss stagnation while representation losses improve:

- Student output path: hidden-state scale, LIF output range, LM head tying, and logit magnitude.
- Teacher/student initialization: whether any paper-faithful transfer from teacher embeddings or output head is missing.
- Causal LM plumbing: shift, mask, pad handling, and token accuracy calculation.
- Distillation alignment: sequence alignment, layer pairing, rate encoding, and whether projector behavior hides student failure.
- Schedule/effective batch: whether the launched probe matches the paper target closely enough under H200 memory limits.
- Numerical stability: bf16 autocast boundaries, gradient clipping target modules, and non-finite checks.

Only one major hypothesis should be tested per iteration.

## Run Configuration Policy

Prefer the closest paper-faithful run that fits in memory:

- Target sequence length: 2048 when feasible.
- If 2048 OOMs, use 1024 as a documented hardware-constrained probe.
- Target per-GPU batch and gradient accumulation should approximate the paper effective batch.
- If a target setting crashes, run a short preheat first and record the exact failure.

The run command must record:

- W&B URL and run name.
- output directory.
- sequence length, batch size, gradient accumulation, world size.
- resolved max steps and warmup steps.
- target tokens for the full continuation.

## Review Requirements

Every implementation change that survives smoke tests must receive a code review before a probe launch.

The reviewer must check:

- loss weights and temperature were not changed,
- the change does not leak teacher or label information into the student logits,
- masks and shifts still implement standard causal LM training,
- DDP and checkpoint behavior remain valid,
- tests cover the changed behavior where practical.

Critical and important findings block the next probe.

## Artifacts

Each loop iteration should leave a concise record under `docs/results/` containing:

- hypothesis,
- code diff summary,
- test command and result,
- review outcome,
- launch command,
- W&B URL,
- first/last/windowed hard and soft loss,
- decision: continue, extend, stop, keep, or revert.

## Success State

The loop succeeds when a paper-faithful probe satisfies the acceptance rule and can continue into the full 1B-token run without restarting from an unfair or diagnostic-only state.
