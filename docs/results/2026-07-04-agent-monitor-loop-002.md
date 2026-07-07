# Agent Monitor Loop 002

Date: 2026-07-04 07:10:24 +08

Monitor scope: observe the other agent/worktree state and evaluate the reported loop evidence. No training, retry, code edit, or experiment launch was performed as part of this monitor record.

## Current Machine State

GPU state at monitor time:

| GPU | Util | Memory | Assessment |
| ---: | ---: | ---: | --- |
| 0 | 0% | 0 / 143771 MiB | free |
| 1 | 0% | 0 / 143771 MiB | free |
| 2 | 0% | 0 / 143771 MiB | free |

No active GPU compute process was reported by `nvidia-smi`.

## Worktree Progress

Observed worktree:

```text
/mnt/data3/data_xingrui/lueq/SpikLLM_OPD/SpikLLM/.worktrees/loop-opd-sa-topd
```

Latest observed head:

```text
e8ff366 docs: record SA-TOPD lean gate results
```

New result documents observed in that worktree:

```text
docs/results/2026-07-04-sa-topd-gate1-gate2.md
docs/results/2026-07-04-sa-topd-gate3.md
docs/results/2026-07-04-sa-topd-gate4-lean-topk.md
```

Remaining uncommitted items in the monitored worktree:

```text
M  .gitignore
M  scripts/analyze_wandb_probe.py
M  tests/smoke/test_probe_analysis.py
?? data/sa-topd-step1000-fineweb-prompts.pt
```

Evaluation: the other agent has progressed from preflight implementation to concrete Gate 1-4 evidence. The prompt buffer is still untracked, so this loop is not fully reproducibility-clean until that artifact is either committed, checksummed and archived, or regenerated deterministically.

## Evidence Summary

Fixed setup reported by the other agent:

| Field | Value |
| --- | --- |
| Starting checkpoint | loop14 step1000 SpAD checkpoint |
| Prompt buffer | `data/sa-topd-step1000-fineweb-prompts.pt` |
| Prompt buffer sha256 | `720f391ac0d777e9813e877e568b34e3d90b19af34ea849f0aaee3603061342a` |
| Seed | `1000` |
| Gate 3 budget | 80 optimizer steps |
| Gate 4 budget | 40-step top-k canaries and 80-step lean comparisons |

Gate outcomes:

| Gate | Result | Monitor Evaluation |
| --- | --- | --- |
| Gate 1 drift diagnostic | pass, `drift_observed` | The step1000 SpAD checkpoint has measurable student-generated drift; OPD direction is justified for testing. |
| Gate 2 canary | default S0 failed, calibrated S0 passed | Default spike interval was too tight and produced silent/overactive layer failures; calibrated interval fixed health without OOM/NaN. |
| Gate 3 matched 80-step | B1/S0/S1/S3 beat B0 on hard loss, top1 agreement, rank, and margin | On-policy distillation direction is viable, but the full module stack is not accepted as-is. |
| Gate 4 lean top-k sweep | K96 selected as balanced candidate | Lean TOPD with `top_k=96` is the best current trade-off under this single-seed evidence. |

Key Gate 4 80-step recent-window metrics:

| Variant | Hard | Soft | Mass | Top1 | Top5 | Rank | Margin | KL | Entropy | Silent | Overactive |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| K64 beta_topk=1.0 | 1.6945 | 2.4580 | 0.8910 | 0.6993 | 0.8015 | 1.0814 | 1.9949 | 0.8354 | 2.0595 | 0.0 | 0.0 |
| K96 beta_topk=1.0 | 1.4970 | 2.5932 | 0.9097 | 0.7088 | 0.8056 | 1.0543 | 2.0818 | 0.8894 | 2.2160 | 0.0 | 0.0 |
| K128 beta_topk=1.0 | 1.4007 | 2.6880 | 0.9213 | 0.7067 | 0.8072 | 1.0570 | 2.1763 | 0.9143 | 2.3019 | 0.0 | 0.0 |
| K128 beta_topk=1.5 | 1.4854 | 2.7071 | 0.9214 | 0.7073 | 0.8034 | 1.0483 | 2.1032 | 0.9283 | 2.3974 | 0.0 | 0.0 |
| K128 no-topk | 0.1844 | 3.9269 | 0.9198 | 0.7099 | 0.7714 | 1.0099 | 4.2617 | 2.1353 | 0.5791 | 0.0 | 0.0 |

## Failure And Improvement Reasons

Default full S0 failure reason:

- The original spike interval thresholds triggered persistent silent/overactive layer failures in Gate 2.
- This was not an OOM, NaN, runaway grad norm, or low top-k-mass failure.

Accepted improvement:

- Calibrating the spike health interval to `spike_p_min=0.01`, `spike_p_max=0.60` removed the health failure.
- Active spike loss still showed no optimization contribution in Gate 3, so it should remain disabled and used only as a diagnostic guard.

Rejected or demoted modules:

- Rate bridge: remove or set `beta_feat=0.0`; it did not improve the primary metrics enough to justify keeping it.
- Active spike regularization: set `lambda_spike_init=0.0`, `lambda_spike_final=0.0`; keep only calibrated health monitoring.
- `beta_topk=1.5`: reject for now because it worsened soft/KL trade-off and increased grad norm without a clear primary-metric win.

Kept module:

- Top-k local-support distillation should stay. The no-topk ablation wins hard/rank/margin, but it collapses local-support behavior: KL rises strongly, soft/top-k loss worsens, entropy collapses, and top5 agreement is worse than top-k variants.

## Feasibility Verdict

Current verdict: direction is feasible, original full SA-TOPD is not accepted as the final method.

The viable method candidate is lean TOPD:

```text
top_k=96
beta_feat=0.0
beta_topk=1.0
beta_anchor=0.05
lambda_spike_init=0.0
lambda_spike_final=0.0
spike_p_min=0.01
spike_p_max=0.60
```

Reasoning:

- Compared with B0, OPD variants substantially improve response hard/anchor loss, teacher top1 agreement, target rank, and target margin under matched 80-step evidence.
- Compared with K64, K96 improves hard loss, teacher top1/top5 agreement, rank, margin, and teacher top-k mass while keeping spike health clean.
- Compared with K128, K96 has a better balanced trade-off because it pays less soft/KL cost while keeping nearly the same teacher-agreement and rank gains.
- Compared with no-topk, K96 preserves teacher local-support behavior instead of collapsing entropy and KL.

Confidence level: preliminary positive, not final. The evidence is still one checkpoint, one prompt buffer, one seed, and short 80-step gates. It is enough to continue the method direction, but not enough to claim full training will succeed.

## Monitor Decision

Do not declare the method complete yet.

Next valid loop:

1. Run K96 lean TOPD on one additional seed or one additional prompt-buffer shard for 80 steps.
2. If the trade-off repeats, run a 160-step stability gate against K64 and no-topk.
3. Only after those pass should the method be written as the final second-stage recipe and considered for a larger multi-GPU run.

Hard acceptance condition: the next loop must preserve or improve response hard/soft loss trade-off, teacher agreement, target rank/margin, top-k KL/entropy, and spike health relative to B0 and no-topk under matched checkpoint, seed or declared seed, prompt buffer, and token budget.
