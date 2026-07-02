# SA-TOPD Final Experimental Paradigm and Theory Positioning

Date: 2026-07-02

Status: consensus draft after simulated reviewer debate.

This document records the final research design for SA-TOPD after adversarial review. It is intended to guide the next implementation and manuscript-writing steps without drifting away from the core direction: spike-aware two-stage on-policy distillation for binary spiking causal language models.

## 1. Reviewer Convergence

The reviewer panel converged on the following decision.

1. The core direction is worth keeping: SpAD cold start followed by spike-aware on-policy post-training is a coherent paper story.
2. The contribution must be framed as a system-level adaptation for binary spiking causal language models, not as a claim that top-k distillation, feature MSE, or firing-rate regularization are individually new.
3. The theory section must not claim absolute immunity to gradient explosion or gradient vanishing. It should give a bounded-assumption analysis that explains why local rate-domain supervision shortens gradient paths and can reduce reliance on long surrogate-gradient Jacobian chains.
4. The reliability product weighting mechanism should not be a main contribution. It can be treated as an optional stabilization heuristic only if ablations show it helps.
5. The minimum publishable experiment package must include SpAD-only continuation, naive OPD, and SA-TOPD ablations. Without these, the method may look like an untested engineering combination.

## 2. Final Thesis

SA-TOPD argues that offline ANN-to-SNN distillation is a necessary but incomplete training paradigm for spiking causal language models. SpAD-style offline distillation can cold-start a binary spiking student into a useful region, but autoregressive inference still exposes the student to its own generated histories. SA-TOPD addresses this by adding a second on-policy post-training stage in which the teacher supervises states induced by the student itself.

The paper's defensible central thesis is:

> For binary spiking causal language models, on-policy distillation becomes practical only when it is adapted to spike-domain representation mismatch, online output-supervision cost, and firing-rate dynamics. SA-TOPD combines these adaptations into a two-stage training framework and improves over SpAD-only continuation under matched experimental budgets.

Before the full experiment is complete, the last clause should be read as the empirical success criterion rather than a completed result: SA-TOPD must be judged by whether it improves over SpAD-only continuation under matched budgets. This thesis intentionally does not claim that the original BiSpikCLM result is false, that OPD is newly invented, or that gradient pathologies are mathematically impossible.

## 3. Method Overview

Let `T` be a frozen ANN causal language-model teacher and `S_theta` be a binary spiking causal language-model student. SA-TOPD has two stages.

### 3.1 Stage 1: SpAD Cold Start

Stage 1 follows BiSpikCLM-style Spike-Aware Alignment Distillation. The student is trained on real text prefixes with the five offline losses:

```text
L_SpAD =
  lambda_EA  * L_EA
+ lambda_SAA * L_SAA
+ lambda_SFA * L_SFA
+ lambda_STA * L_STA
+ lambda_HTA * L_HTA
```

The paper-faithful default weights are:

```text
lambda_EA  = 0.2
lambda_SAA = 0.1
lambda_SFA = 0.1
lambda_STA = 0.3
lambda_HTA = 0.3
```

Stage 1 is not presented as a new contribution. Its role is to provide a stable student checkpoint for online post-training. In the current repository, the best empirical baseline is the loop16/current-source SpAD implementation, with loop91b as the strongest observed small-configuration run:

```text
loop91b: seq512, batch2, GA16, T=4
step 2249, hard=6.2209, soft=2.6782, total=3.0423
```

For the paper, the stage-1 checkpoint must be reported as the best available SpAD-only reproduction checkpoint under the declared training budget. If an official BiSpikCLM-scale checkpoint is unavailable, the claim must be phrased as improvement over the paper-faithful reproduction baseline, not over the original authors' unreproduced number.

### 3.2 Stage 2: Spike-Aware On-Policy Post-Training

For each training example, a real prompt `x` is given to the student. The student samples or decodes a response trajectory:

```text
y_1:K ~ pi_theta(. | x)
```

At each generated position `k`, the teacher is queried on the student-induced state:

```text
s_k = (x, y_<k)
```

The teacher does not sample the trajectory. It only provides detached supervision on states that the student actually visits.

The stage-2 loss is computed only on the generated response region:

```text
L_TOPD =
  1 / sum_k m_k
  * sum_k m_k [
      beta_feat * L_bridge(k)
    + beta_topk * L_topk(k)
    + beta_anchor * L_anchor(k)
    ]
  + lambda_spike(t) * sum_k L_spike(k)
```

where `m_k = 1` for response tokens and `0` for prompt tokens. `L_anchor` is optional and should be used only when mixed offline replay or teacher-forced anchors are included to reduce forgetting. The core SA-TOPD claim depends on `L_bridge`, `L_topk`, and `L_spike`.

## 4. Core Modules

### 4.1 Rate-Domain Feature Bridge

The student layer `l` emits binary spikes over `T_snn` internal timesteps. Its rate-domain feature at generated token step `k` is:

```text
H_S,l(k) = (1 / T_snn) * sum_t S_l(k, t)
```

with `H_S,l(k)` in `[0, 1]^d`.

The teacher hidden state `H_T,l(k)` is continuous. To bridge the spike-rate domain to the teacher hidden domain without adding a heavy projector, SA-TOPD uses a lightweight channel affine adapter:

```text
A_l(H_S,l) = gamma_l * H_S,l + beta_l
```

with `gamma_l` initialized to `1` and `beta_l` initialized to `0`.

The feature bridge loss is:

```text
L_bridge(k) =
  1 / |L_sel|
  * sum_{l in L_sel} mean_j || stopgrad(H_T,l,j(k)) - A_l(H_S,l,j(k)) ||_2^2
```

`L_sel` should be selected-layer or layer-group supervision, not necessarily all layers. A cost-aware default is early/middle/late layers, for example:

```text
L_sel = {25%, 50%, 75%, 100% depth}
```

Full-layer alignment can be reported as an ablation. It should not be mandatory in the main method because it may be expensive and over-constraining.

### 4.2 Top-k Local-Support Output Distillation

At each student-induced state, the teacher returns top-k indices:

```text
I_k = TopK(Z_T(k), k_top)
```

Teacher and student logits are gathered only on `I_k`:

```text
z_T^I(k) = gather(Z_T(k), I_k)
z_S^I(k) = gather(Z_S(k), I_k)
```

The local distributions are:

```text
p_T^I = softmax(z_T^I / tau)
p_S^I = softmax(z_S^I / tau)
```

The local-support distillation loss is:

```text
L_topk(k) = - sum_i p_T^I(i) * log p_S^I(i)
```

This is a local forward-KL cross-entropy up to the teacher entropy constant. It should be described as optimizing the teacher's high-confidence local support, not as an exact full-vocabulary KL.

Important complexity boundary:

1. If the implementation computes the full student LM head and then gathers top-k logits, the method reduces KL/storage/teacher-supervision density, but the student LM-head forward cost remains `O(V)`.
2. A true `O(k)` or near-`O(k)` output-layer claim requires indexed output projection or sampled-head implementation. Without that implementation, the manuscript must not claim that LM-head compute is reduced from `O(V)` to `O(k)`.

The experiment should report teacher top-k mass:

```text
mass_topk(k) = sum_{i in I_k} softmax(Z_T(k) / tau)_i
```

This guards against the criticism that local-support KL discards too much teacher probability mass.

### 4.3 Dynamic Spike-Rate Regularization

For each selected layer or all layers, compute the mean firing rate:

```text
FR_l(k) = mean_{tokens, channels, timesteps} S_l(k, t)
```

The interval penalty is:

```text
L_spike(k) =
  sum_l [
    max(0, FR_l(k) - p_max)^2
  + max(0, p_min - FR_l(k))^2
  ]
```

The coefficient is annealed:

```text
lambda_spike(t) =
  lambda_final
  + (lambda_init - lambda_final) * decay(t)
```

with `lambda_init > lambda_final`. The early phase prioritizes keeping the spiking dynamics alive and not overactive. The late phase relaxes the constraint to allow representation fitting.

This module should be framed as a dynamics regularizer, not as a proof that firing rates are always bounded inside `[p_min, p_max]`.

### 4.4 Optional Reliability Gate

The previous product-form SOD gate is not recommended as a main method:

```text
w_k = product_u (d_u + eps) / (d_{u+1} + eps)
```

It can suppress gradients exactly where the student needs correction. If a reliability gate is tested, use a clipped per-token version with a nonzero floor:

```text
d_k = L_topk(k)
r_k = clip(exp(-alpha * max(0, d_k - EMA(d))), r_min, 1.0)
```

Then replace:

```text
L_bridge(k) + L_topk(k)
```

with:

```text
r_k * (L_bridge(k) + L_topk(k))
```

This gate is optional. It must be ablated before being included in the main paper.

## 5. Theory Positioning

The theory section should provide conditional stability analysis, not absolute guarantees.

### 5.1 Assumptions

Use explicit assumptions:

1. The network has finite depth `L` and finite spiking unroll length `T_snn`.
2. Layer weights have bounded operator norms during the analyzed interval:

```text
||W_l||_2 <= B_W
```

3. Surrogate derivatives are bounded:

```text
0 <= sigma'(u) <= B_SG
```

4. Adapter parameters are bounded, by regularization or clipping:

```text
||gamma_l||_infty <= B_gamma
||beta_l||_infty <= B_beta
```

5. Teacher hidden states and student rate features are bounded:

```text
||H_T,l||_2 <= B_T
H_S,l in [0, 1]^d
```

6. Loss weights are finite and nonnegative.

These assumptions are not cosmetic. They are required for the upper-bound analysis.

### 5.2 Proposition 1: Conditional Upper Bound on Gradient Norms

Under the assumptions above, each local loss source has bounded derivative.

For top-k local support:

```text
d L_topk / d z_S^I = (1 / tau) * (p_S^I - p_T^I)
```

Since `p_S^I` and `p_T^I` are probability vectors over `k_top` entries:

```text
||d L_topk / d z_S^I||_2 <= sqrt(2) / tau
```

For the feature bridge:

```text
d L_bridge,l / d H_S,l
  = (2 / d) * gamma_l * (A_l(H_S,l) - stopgrad(H_T,l))
```

which is bounded by the bounded teacher states, bounded student rates, and bounded adapter parameters.

For the spike-rate interval penalty, the derivative is piecewise linear outside the interval and zero inside. It is bounded if firing rates remain in `[0, 1]` and the penalty coefficients are finite.

The BPTT derivative through a leaky spiking layer is bounded when leak `kappa < 1`, surrogate derivative is bounded, and inputs are bounded. Therefore, for each layer:

```text
||d L_TOPD / d W_l|| <= C_l < infinity
```

where `C_l` depends on `B_W`, `B_SG`, `B_gamma`, `tau`, loss weights, depth, and sequence length.

Correct manuscript wording:

> Under bounded weights, bounded adapters, bounded teacher states, and bounded surrogate derivatives, SA-TOPD has finite gradient-norm upper bounds over a finite unroll interval.

Forbidden wording:

> SA-TOPD absolutely prevents gradient explosion.

### 5.3 Proposition 2: Local Feature Supervision Shortens Gradient Paths

With output-only OPD, a shallow layer `l` receives output supervision through the product:

```text
d L_output / d W_l
  = d L_output / d H_L
    * product_{i=l}^{L-1} J_i
    * d H_l / d W_l
```

where `J_i = d H_{i+1} / d H_i`.

If many `||J_i|| < 1`, this path can decay quickly with depth.

With layer-wise or selected-layer rate bridge, layer `l` receives an additional local path when `l in L_sel`:

```text
d L_bridge,l / d W_l
  = d L_bridge,l / d H_l
    * d H_l / d W_l
```

This path does not include the product of all later-layer Jacobians. Therefore, selected-layer rate bridge reduces dependence on deep surrogate-gradient chains.

Correct manuscript wording:

> The bridge loss introduces local supervision paths whose length is independent of the number of subsequent layers, reducing reliance on products of surrogate-gradient Jacobians.

Forbidden wording:

> The bridge loss mathematically eliminates gradient vanishing.

Why the forbidden wording is wrong:

1. The local feature error may already be near zero.
2. The surrogate derivative can be zero or near zero in saturated regimes.
3. Different gradient components can cancel.
4. Adapter parameters may absorb mismatch without updating the spiking backbone.

The defensible claim is path shortening and risk reduction, not a universal positive lower bound.

### 5.4 Proposition 3: Spike-Rate Regularization Penalizes Dynamics Drift

The spike-rate regularizer creates a restoring penalty when a layer becomes too silent or too active:

```text
FR_l < p_min or FR_l > p_max
```

This gives a direct objective-level signal against silent collapse and overactivation. It does not guarantee all rates stay within the interval for every optimizer step, but it makes such drift measurable and penalized.

Correct manuscript wording:

> The firing-rate interval penalty turns spike-dynamics drift into an explicit optimization cost and provides diagnostics for silent or overactive collapse.

### 5.5 Proposition 4: Top-k Local Support Is a Biased but Controlled Approximation

Top-k local-support distillation is not equal to full-vocabulary forward KL unless the teacher's probability mass outside top-k is negligible. Define:

```text
M_k = sum_{i in I_k} p_T_full(i)
```

When `M_k` is high, local-support KL captures most teacher probability mass. When `M_k` is low, the approximation is biased and may ignore meaningful alternatives.

Therefore the experiment must log:

```text
teacher_topk_mass
teacher_topk_entropy
student_topk_entropy
```

The claim should be:

> Top-k local-support distillation trades exact full-vocabulary matching for lower online supervision cost and reduced long-tail noise, with the retained teacher mass reported as a diagnostic.

## 6. Final Experimental Paradigm

### 6.1 Research Questions

RQ1: Does SA-TOPD improve over SpAD-only continuation under the same checkpoint, token budget, and hardware budget?

RQ2: Does naive OPD destabilize or underperform when applied directly to a spiking causal LM?

RQ3: Which spike-aware components are responsible for gains: rate bridge, top-k local support, or spike-rate regularization?

RQ4: Does SA-TOPD improve language-model quality without destroying spike-efficiency proxies?

RQ5: Are gains robust across at least two seeds or two checkpoints?

### 6.2 Baselines and Ablations

Minimum required table:

| ID | Method | Purpose |
| --- | --- | --- |
| B0 | SpAD-only continuation | Tests whether gains are just more training |
| B1 | Naive OPD | Tests whether ANN-style OPD transfers directly |
| B2 | SA-TOPD without rate bridge | Tests feature bridge contribution |
| B3 | SA-TOPD without spike-rate regularization | Tests dynamics-control contribution |
| B4 | SA-TOPD without top-k, full-vocab where feasible | Tests local-support tradeoff |
| B5 | SA-TOPD selected-layer bridge | Main method |
| B6 | SA-TOPD full-layer bridge | Cost/overconstraint ablation |
| B7 | SA-TOPD plus optional reliability gate | Optional appendix only |

If full-vocabulary online distillation is infeasible, B4 should be replaced by:

```text
top-k sweep: k in {16, 32, 64, 128}
```

and the paper must explicitly state that full-vocabulary online KL was computationally infeasible under the available budget.

### 6.3 Small-Batch Gate

Every new method must beat the current SpAD-only baseline on the same small configuration before any full run:

```text
seq_len = 512
batch_size = 2
gradient_accumulation = 16
T_snn = 4
```

The comparison must start from the same stage-1 checkpoint. A candidate is considered promising only if, over a matched token budget:

1. final hard loss is lower than SpAD-only continuation;
2. final soft loss is lower than SpAD-only continuation;
3. teacher top-1/top-5 agreement improves or remains stable;
4. spike-rate metrics do not collapse;
5. no NaN, runaway grad norm, or sustained silent/overactive layers appear.

The current empirical reference from prior runs is:

```text
loop91b:
hard=6.2209
soft=2.6782
total=3.0423
token_accuracy=13.11%
teacher_top1_agreement=20.74%
tokens_seen=36.8M
```

For a fair SA-TOPD experiment, this value is a historical guide, not a substitute for a same-checkpoint matched continuation control.

### 6.4 Full-Run Gate

Only after passing the small-batch gate should the method run the full baseline geometry:

```text
3 x H200
seq_len = 1024
batch_size_per_rank = 4
gradient_accumulation = 64
T_snn = 4
precision = bf16
```

This is the loop14/loop16 long-run geometry already used in the project records.

### 6.5 Metrics

Report training metrics:

```text
hard_loss
soft_loss
total_loss
feature_bridge_loss
topk_loss
spike_rate_loss
grad_norm
learning_rate
tokens_seen
```

Report quality metrics:

```text
validation_ppl
rollout_ppl or continuation negative log-likelihood
token_accuracy
top5_accuracy
teacher_top1_agreement
teacher_top5_agreement
target_rank_mean
target_margin_mean
```

Report spike and efficiency metrics:

```text
layerwise spike_rate_mean
layerwise spike_rate_std
silent_layer_count
overactive_layer_count
SynOps proxy
activation sparsity
peak_memory_gb
wall_clock_tokens_per_second
teacher_query_cost
```

Report top-k diagnostics:

```text
teacher_topk_mass
teacher_topk_entropy
student_topk_entropy
KL_on_topk
```

### 6.6 Stop and Continue Rules

For small-batch screens:

Continue if both hard and soft losses are below the SpAD-only continuation curve or both show clearly stronger downward trends over 100+ optimizer steps.

Reject if the method improves one auxiliary metric but worsens hard/soft loss or destabilizes spike dynamics.

For full runs:

Continue when hard and soft losses are both below 5, or both maintain clear downward trends without spike collapse.

Stop and diagnose when:

1. hard loss plateaus above the matched SpAD-only continuation;
2. soft loss rises for a sustained window;
3. spike rates collapse to near-zero or saturate;
4. grad norm or loss shows repeated instability;
5. teacher top-k mass is too low for the chosen `k`.

## 7. Manuscript Claim Boundaries

### Allowed Claims

Use these claims if experiments support them:

1. To the best of our knowledge, SA-TOPD is the first systematic study of spike-aware on-policy post-training for binary spiking causal language models.
2. SA-TOPD improves over our SpAD-only reproduction baseline under matched checkpoint and training budgets.
3. Rate-domain bridge, local-support output distillation, and spike-rate regularization address SNN-specific failure modes in online distillation.
4. Under bounded assumptions, local feature supervision provides shorter gradient paths than output-only OPD.
5. SA-TOPD can improve language modeling quality while monitoring and preserving spike-efficiency proxies.

### Forbidden Claims

Do not use these:

1. SA-TOPD proves BiSpikCLM is wrong or fabricated.
2. SA-TOPD absolutely prevents gradient explosion.
3. SA-TOPD absolutely prevents gradient vanishing.
4. Top-k distillation reduces LM-head compute from `O(V)` to `O(k)` unless indexed output projection is implemented.
5. SA-TOPD outperforms official BiSpikCLM unless the official setup is exactly reproduced.
6. Full-layer OPRD is always better than selected-layer supervision.

## 8. Recommended Paper Structure

1. Introduction: SNN causal LM promise, SpAD cold start, offline exposure bias, need for spike-aware OPD.
2. Related Work: spiking LMs, offline ANN-to-SNN distillation, OPD/GKD, top-k OPD and efficient distillation.
3. Method: two-stage SA-TOPD, rate bridge, top-k local support, spike-rate regularization.
4. Theory: bounded gradient analysis and local supervision path shortening.
5. Experiments: SpAD reproduction baseline, small-batch gates, full runs, ablations, efficiency/stability.
6. Limitations: dependence on teacher access, top-k approximation bias, reproduction gap from official BiSpikCLM, hardware budget.

## 9. Source Anchors

Use these as external anchors in the paper:

1. BiSpikCLM arXiv page: https://arxiv.org/abs/2605.13859
   - Use for the claim that BiSpikCLM introduces SFSA and SpAD and aligns ANN teacher and SNN student across multiple levels.
2. GKD / On-Policy Distillation of Language Models: https://arxiv.org/abs/2306.13649
   - Use for the claim that OPD trains students on self-generated output sequences with teacher feedback.
3. OPD survey: https://arxiv.org/abs/2604.00626
   - Use for broader OPD framing and exposure-bias motivation.
4. Verl OPD documentation: https://verl.readthedocs.io/en/latest/algo/opd.html
   - Use cautiously to show top-k forward-KL OPD exists, so SA-TOPD must not claim top-k OPD as a standalone invention.

## 10. Next Implementation Plan

The next implementation should be scoped as a separate plan:

1. Add stage-2 rollout data flow from a fixed SpAD checkpoint.
2. Implement teacher inference on student-induced states with detached top-k outputs.
3. Add selected-layer rate bridge adapters.
4. Add dynamic spike-rate regularization and logging.
5. Add SpAD-only continuation and naive OPD controls.
6. Run small-batch gate against the same checkpoint before any full three-GPU run.

The implementation plan must preserve the user's loop rule: each candidate first competes against the current best baseline on the small batch; only clear small-batch wins proceed to full training.
