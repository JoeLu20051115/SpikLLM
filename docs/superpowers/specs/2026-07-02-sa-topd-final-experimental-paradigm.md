# SA-TOPD 最终实验范式与理论定位

日期：2026-07-02

状态：经过模拟审稿人辩论后的共识草案。

本文档记录 SA-TOPD 经过对抗式审查后的最终研究设计。它用于指导后续实现和论文写作，同时避免偏离核心方向：面向二值脉冲因果语言模型的脉冲感知两阶段 on-policy distillation。

## 1. 审稿人共识

模拟 reviewer panel 最终收敛到以下判断。

1. 核心方向应该保留：SpAD 冷启动之后接 spike-aware on-policy post-training，是一条完整且可信的论文故事线。
2. 贡献必须被表述为“面向二值脉冲因果语言模型的系统级适配”，而不是声称 top-k distillation、feature MSE 或 firing-rate regularization 这些模块各自都是全新发明。
3. 理论部分不能声称 SA-TOPD 对梯度爆炸或梯度消失具有绝对免疫。应改为有界假设下的分析，用来解释局部 rate-domain 监督为什么能缩短梯度路径，并减少对长链 surrogate-gradient Jacobian 乘积的依赖。
4. 可靠性乘积加权机制不应该作为主贡献。只有当 ablation 证明它有效时，才把它作为可选稳定化 heuristic。
5. 最小可发表实验包必须包含 SpAD-only continuation、naive OPD 和 SA-TOPD ablations。没有这些对照，方法会被认为只是未经验证的工程组合。

## 2. 最终中心论点

SA-TOPD 的核心观点是：offline ANN-to-SNN distillation 对脉冲因果语言模型是必要的，但不是完整的训练范式。SpAD 类离线蒸馏可以把二值脉冲学生模型 cold-start 到一个可用区域，但自回归推理仍然会让学生暴露在自身生成的历史状态上。SA-TOPD 通过加入第二阶段 on-policy post-training，让教师在学生自身诱导出的状态上提供监督，从而处理这一问题。

本文可以防守的中心论点是：

> 对于二值脉冲因果语言模型，on-policy distillation 只有在适配脉冲域表征错配、在线输出监督开销和发放率动力学之后，才真正具备可操作性。SA-TOPD 将这些适配组合成一个两阶段训练框架，并以 matched budget 下优于 SpAD-only continuation 作为经验成功标准。

在完整实验完成之前，最后一句必须被理解为经验判定标准，而不是已经完成的结果 claim：SA-TOPD 必须通过 matched budget 下是否优于 SpAD-only continuation 来判断。这个论点不声称原始 BiSpikCLM 结果是假的，不声称 OPD 是本文新发明的，也不声称梯度病态在数学上不可能发生。

## 3. 方法总览

设 `T` 为冻结的 ANN 因果语言模型教师，`S_theta` 为二值脉冲因果语言模型学生。SA-TOPD 包含两个阶段。

### 3.1 阶段一：SpAD 冷启动

阶段一遵循 BiSpikCLM 风格的 Spike-Aware Alignment Distillation。学生模型在真实文本前缀上训练，并优化五个离线损失：

```text
L_SpAD =
  lambda_EA  * L_EA
+ lambda_SAA * L_SAA
+ lambda_SFA * L_SFA
+ lambda_STA * L_STA
+ lambda_HTA * L_HTA
```

论文复现默认权重为：

```text
lambda_EA  = 0.2
lambda_SAA = 0.1
lambda_SFA = 0.1
lambda_STA = 0.3
lambda_HTA = 0.3
```

阶段一不作为本文新贡献。它的作用是为在线后训练提供稳定的学生 checkpoint。在当前仓库中，最佳实证 baseline 是 loop16/current-source SpAD 实现，其中 loop91b 是观察到的最强小配置 run：

```text
loop91b: seq512, batch2, GA16, T=4
step 2249, hard=6.2209, soft=2.6782, total=3.0423
```

论文中应把阶段一 checkpoint 表述为“在声明训练预算下获得的最佳 SpAD-only reproduction checkpoint”。如果无法得到官方 BiSpikCLM 规模的 checkpoint，则只能声称相对于 paper-faithful reproduction baseline 的提升，不能声称优于原作者未复现的官方数值。

### 3.2 阶段二：Spike-Aware On-Policy Post-Training

对每个训练样本，给学生一个真实 prompt `x`。学生从该 prompt 出发采样或解码 response trajectory：

```text
y_1:K ~ pi_theta(. | x)
```

在每个生成位置 `k`，教师在学生诱导出的状态上被查询：

```text
s_k = (x, y_<k)
```

教师不参与采样 trajectory。教师只在学生实际访问到的状态上提供 detached supervision。

阶段二的损失只在生成 response 区域计算：

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

其中 `m_k = 1` 表示 response token，`m_k = 0` 表示 prompt token。`L_anchor` 是可选项，只在加入 mixed offline replay 或 teacher-forced anchor 以降低遗忘风险时使用。SA-TOPD 的核心 claim 依赖 `L_bridge`、`L_topk` 和 `L_spike`。

## 4. 核心模块

### 4.1 Rate-Domain Feature Bridge

学生第 `l` 层在 `T_snn` 个内部时间步上发出二值脉冲。其在生成 token step `k` 的 rate-domain feature 为：

```text
H_S,l(k) = (1 / T_snn) * sum_t S_l(k, t)
```

其中 `H_S,l(k)` 位于 `[0, 1]^d`。

教师隐状态 `H_T,l(k)` 是连续实值表示。为了在不引入重型 projector 的情况下桥接 spike-rate domain 和 teacher hidden domain，SA-TOPD 使用轻量通道仿射 adapter：

```text
A_l(H_S,l) = gamma_l * H_S,l + beta_l
```

其中 `gamma_l` 初始化为 `1`，`beta_l` 初始化为 `0`。

feature bridge loss 为：

```text
L_bridge(k) =
  1 / |L_sel|
  * sum_{l in L_sel} mean_j || stopgrad(H_T,l,j(k)) - A_l(H_S,l,j(k)) ||_2^2
```

`L_sel` 应该是 selected-layer 或 layer-group supervision，不必强制覆盖所有层。一个 cost-aware 默认设置是 early/middle/late layers，例如：

```text
L_sel = {25%, 50%, 75%, 100% depth}
```

全层对齐可以作为 ablation 报告。它不应成为主方法的强制配置，因为全层对齐可能开销过大，也可能对表示空间造成过约束。

### 4.2 Top-k Local-Support Output Distillation

在每个学生诱导状态上，教师返回 top-k indices：

```text
I_k = TopK(Z_T(k), k_top)
```

教师和学生 logits 只在 `I_k` 上 gather：

```text
z_T^I(k) = gather(Z_T(k), I_k)
z_S^I(k) = gather(Z_S(k), I_k)
```

局部分布定义为：

```text
p_T^I = softmax(z_T^I / tau)
p_S^I = softmax(z_S^I / tau)
```

local-support distillation loss 为：

```text
L_topk(k) = - sum_i p_T^I(i) * log p_S^I(i)
```

这是一个局部 forward-KL cross-entropy，差一个教师 entropy 常数项。它应被描述为在教师高置信局部支撑集上优化，而不是精确的 full-vocabulary KL。

重要复杂度边界：

1. 如果实现仍然先计算完整 student LM head，再 gather top-k logits，那么该方法降低的是 KL/storage/teacher-supervision density，但 student LM-head forward cost 仍然是 `O(V)`。
2. 只有实现 indexed output projection 或 sampled-head 时，才能声称输出层开销达到真正的 `O(k)` 或近似 `O(k)`。没有这种实现时，论文不能声称 LM-head compute 从 `O(V)` 降到 `O(k)`。

实验应报告 teacher top-k mass：

```text
mass_topk(k) = sum_{i in I_k} softmax(Z_T(k) / tau)_i
```

这个指标用于回应一个关键质疑：local-support KL 是否丢弃了过多教师概率质量。

### 4.3 动态 Spike-Rate Regularization

对每个 selected layer 或所有层，计算平均发放率：

```text
FR_l(k) = mean_{tokens, channels, timesteps} S_l(k, t)
```

区间惩罚项为：

```text
L_spike(k) =
  sum_l [
    max(0, FR_l(k) - p_max)^2
  + max(0, p_min - FR_l(k))^2
  ]
```

系数随训练进程退火：

```text
lambda_spike(t) =
  lambda_final
  + (lambda_init - lambda_final) * decay(t)
```

其中 `lambda_init > lambda_final`。早期阶段优先保证脉冲动力学“活着且不过激”；后期阶段逐步放松约束，释放表示拟合自由度。

该模块应被表述为 dynamics regularizer，而不是证明发放率一定始终处在 `[p_min, p_max]` 内。

### 4.4 可选 Reliability Gate

之前的 product-form SOD gate 不建议作为主方法：

```text
w_k = product_u (d_u + eps) / (d_{u+1} + eps)
```

它可能在学生最需要纠正的位置压低梯度。如果要测试 reliability gate，应使用带非零下界的 clipped per-token version：

```text
d_k = L_topk(k)
r_k = clip(exp(-alpha * max(0, d_k - EMA(d))), r_min, 1.0)
```

然后把：

```text
L_bridge(k) + L_topk(k)
```

替换为：

```text
r_k * (L_bridge(k) + L_topk(k))
```

这个 gate 是可选项。只有 ablation 证明它有效后，才可以被纳入主论文。

## 5. 理论定位

理论部分应提供条件稳定性分析，而不是绝对保证。

### 5.1 假设

需要显式写出以下假设：

1. 网络深度 `L` 有限，spiking unroll length `T_snn` 有限。
2. 在分析区间内，每层权重的 operator norm 有界：

```text
||W_l||_2 <= B_W
```

3. surrogate derivative 有界：

```text
0 <= sigma'(u) <= B_SG
```

4. adapter 参数通过 regularization 或 clipping 保持有界：

```text
||gamma_l||_infty <= B_gamma
||beta_l||_infty <= B_beta
```

5. 教师 hidden states 和学生 rate features 有界：

```text
||H_T,l||_2 <= B_T
H_S,l in [0, 1]^d
```

6. 所有 loss weights 都是有限且非负的。

这些假设不是装饰性文字。它们是 upper-bound analysis 成立的必要条件。

### 5.2 命题一：梯度范数的条件上界

在上述假设下，每个局部 loss source 都具有有界导数。

对于 top-k local support：

```text
d L_topk / d z_S^I = (1 / tau) * (p_S^I - p_T^I)
```

由于 `p_S^I` 和 `p_T^I` 都是 `k_top` 个元素上的概率向量：

```text
||d L_topk / d z_S^I||_2 <= sqrt(2) / tau
```

对于 feature bridge：

```text
d L_bridge,l / d H_S,l
  = (2 / d) * gamma_l * (A_l(H_S,l) - stopgrad(H_T,l))
```

该项由有界教师状态、有界学生 rates 和有界 adapter 参数共同约束，因此有界。

对于 spike-rate interval penalty，导数在区间外是 piecewise linear，在区间内为零。由于 firing rates 位于 `[0, 1]` 且惩罚系数有限，该项导数有界。

当 leak `kappa < 1`、surrogate derivative 有界且输入有界时，穿过 leaky spiking layer 的 BPTT derivative 有界。因此对每层都有：

```text
||d L_TOPD / d W_l|| <= C_l < infinity
```

其中 `C_l` 依赖 `B_W`、`B_SG`、`B_gamma`、`tau`、loss weights、depth 和 sequence length。

正确论文表述：

> 在 bounded weights、bounded adapters、bounded teacher states 和 bounded surrogate derivatives 的假设下，SA-TOPD 在有限 unroll interval 内具有有限的 gradient-norm upper bounds。

禁止表述：

> SA-TOPD absolutely prevents gradient explosion.

### 5.3 命题二：局部特征监督缩短梯度路径

在 output-only OPD 中，浅层 `l` 接收到输出监督需要经过：

```text
d L_output / d W_l
  = d L_output / d H_L
    * product_{i=l}^{L-1} J_i
    * d H_l / d W_l
```

其中 `J_i = d H_{i+1} / d H_i`。

如果许多 `||J_i|| < 1`，这条路径会随深度快速衰减。

引入 layer-wise 或 selected-layer rate bridge 后，当 `l in L_sel` 时，第 `l` 层得到一条额外局部路径：

```text
d L_bridge,l / d W_l
  = d L_bridge,l / d H_l
    * d H_l / d W_l
```

这条路径不包含后续所有层 Jacobian 的乘积。因此，selected-layer rate bridge 降低了训练对深层 surrogate-gradient chains 的依赖。

正确论文表述：

> bridge loss 引入了局部监督路径，其长度不依赖后续层数，从而降低了训练对 surrogate-gradient Jacobian 乘积的依赖。

禁止表述：

> The bridge loss mathematically eliminates gradient vanishing.

禁止这样写的原因：

1. 局部 feature error 可能已经接近零。
2. surrogate derivative 在饱和区域可能为零或接近零。
3. 不同梯度分量可能相互抵消。
4. adapter 参数可能吸收错配，而不更新 spiking backbone。

可防守 claim 是路径缩短和风险降低，不是 universal positive lower bound。

### 5.4 命题三：Spike-Rate Regularization 惩罚动力学漂移

当某层变得过于静默或过度激活时：

```text
FR_l < p_min or FR_l > p_max
```

spike-rate regularizer 会产生恢复性惩罚。它给 silent collapse 和 overactivation 提供直接目标信号。它不保证每个 optimizer step 后所有 firing rates 都处于区间内，但它让这种漂移变成可测量、可惩罚的优化成本。

正确论文表述：

> firing-rate interval penalty 将 spike-dynamics drift 转化为显式优化成本，并为 silent 或 overactive collapse 提供诊断信号。

### 5.5 命题四：Top-k Local Support 是有偏但可诊断的近似

除非教师在 top-k 之外的概率质量可以忽略，否则 top-k local-support distillation 不等价于 full-vocabulary forward KL。定义：

```text
M_k = sum_{i in I_k} p_T_full(i)
```

当 `M_k` 较高时，local-support KL 捕获了大部分教师概率质量。当 `M_k` 较低时，这个近似存在明显 bias，可能忽略有意义的替代 token。

因此实验必须记录：

```text
teacher_topk_mass
teacher_topk_entropy
student_topk_entropy
```

论文 claim 应该写成：

> Top-k local-support distillation 在精确 full-vocabulary matching 与较低在线监督开销、较少长尾噪声之间做了权衡，并通过 retained teacher mass 作为诊断指标。

## 6. 最终实验范式

### 6.1 研究问题

RQ1：在相同 checkpoint、token budget 和 hardware budget 下，SA-TOPD 是否优于 SpAD-only continuation？

RQ2：naive OPD 直接应用到 spiking causal LM 时，是否会不稳定或表现不足？

RQ3：SA-TOPD 的收益分别来自哪些 spike-aware components：rate bridge、top-k local support 还是 spike-rate regularization？

RQ4：SA-TOPD 是否能在不破坏 spike-efficiency proxies 的情况下提升语言建模质量？

RQ5：收益是否能在至少两个 seeds 或两个 checkpoints 上保持稳健？

### 6.2 Baselines 与 Ablations

最小必需表格：

| ID | Method | Purpose |
| --- | --- | --- |
| B0 | SpAD-only continuation | 检查收益是否只是来自更多训练 |
| B1 | Naive OPD | 检查 ANN-style OPD 是否能直接迁移 |
| B2 | SA-TOPD without rate bridge | 检查 feature bridge 贡献 |
| B3 | SA-TOPD without spike-rate regularization | 检查 dynamics-control 贡献 |
| B4 | SA-TOPD without top-k, full-vocab where feasible | 检查 local-support tradeoff |
| B5 | SA-TOPD selected-layer bridge | 主方法 |
| B6 | SA-TOPD full-layer bridge | cost/overconstraint ablation |
| B7 | SA-TOPD plus optional reliability gate | 仅作为可选 appendix |

如果 full-vocabulary online distillation 不可行，则 B4 应替换为：

```text
top-k sweep: k in {16, 32, 64, 128}
```

并且论文必须明确说明：在当前可用预算下，full-vocabulary online KL 在计算上不可行。

### 6.3 小批量 Gate

每个新方法在进入 full run 之前，必须先在同一小配置上击败当前 SpAD-only baseline：

```text
seq_len = 512
batch_size = 2
gradient_accumulation = 16
T_snn = 4
```

比较必须从同一个 stage-1 checkpoint 开始。在 matched token budget 下，candidate 只有满足以下条件才算 promising：

1. final hard loss 低于 SpAD-only continuation；
2. final soft loss 低于 SpAD-only continuation；
3. teacher top-1/top-5 agreement 提升或保持稳定；
4. spike-rate metrics 不发生 collapse；
5. 不出现 NaN、runaway grad norm 或持续 silent/overactive layers。

当前历史实证参考为：

```text
loop91b:
hard=6.2209
soft=2.6782
total=3.0423
token_accuracy=13.11%
teacher_top1_agreement=20.74%
tokens_seen=36.8M
```

对于公平的 SA-TOPD 实验，这个数值只是历史参考，不能替代 same-checkpoint matched continuation control。

### 6.4 Full-Run Gate

只有通过小批量 gate 后，方法才进入完整 baseline geometry：

```text
3 x H200
seq_len = 1024
batch_size_per_rank = 4
gradient_accumulation = 64
T_snn = 4
precision = bf16
```

这是项目记录中已经使用过的 loop14/loop16 long-run geometry。

### 6.5 指标

报告训练指标：

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

报告质量指标：

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

报告 spike 与 efficiency 指标：

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

报告 top-k diagnostics：

```text
teacher_topk_mass
teacher_topk_entropy
student_topk_entropy
KL_on_topk
```

### 6.6 Stop 与 Continue Rules

对于 small-batch screens：

如果 hard 和 soft losses 都低于 SpAD-only continuation 曲线，或二者在 100+ optimizer steps 窗口内展现出明显更强下降趋势，则继续。

如果方法只改善一个辅助指标，但恶化 hard/soft loss 或破坏 spike dynamics，则拒绝。

对于 full runs：

当 hard 和 soft losses 都低于 5，或二者在无 spike collapse 的情况下保持清晰下降趋势时继续。

满足以下条件时停止并诊断：

1. hard loss 在 matched SpAD-only continuation 之上 plateau；
2. soft loss 在持续窗口内上升；
3. spike rates collapse 到接近零或发生饱和；
4. grad norm 或 loss 重复不稳定；
5. teacher top-k mass 对当前 `k` 来说过低。

## 7. Manuscript Claim 边界

### 允许的 Claims

如果实验支持，可以使用以下 claim：

1. To the best of our knowledge，SA-TOPD 是第一个系统研究 binary spiking causal language models 的 spike-aware on-policy post-training 的工作。
2. SA-TOPD 在 matched checkpoint 和 training budget 下优于我们的 SpAD-only reproduction baseline。
3. Rate-domain bridge、local-support output distillation 与 spike-rate regularization 共同处理了在线蒸馏中的 SNN-specific failure modes。
4. 在有界假设下，local feature supervision 相比 output-only OPD 提供更短的梯度路径。
5. SA-TOPD 可以在监控并保留 spike-efficiency proxies 的同时提升语言建模质量。

### 禁止的 Claims

不要使用以下 claim：

1. SA-TOPD 证明 BiSpikCLM 是错的或造假的。
2. SA-TOPD absolutely prevents gradient explosion.
3. SA-TOPD absolutely prevents gradient vanishing.
4. 除非实现 indexed output projection，否则不要声称 top-k distillation 将 LM-head compute 从 `O(V)` 降到 `O(k)`。
5. 除非完全复现官方设置，否则不要声称 SA-TOPD 优于 official BiSpikCLM。
6. 不要声称 full-layer OPRD 总是优于 selected-layer supervision。

## 8. 推荐论文结构

1. Introduction：SNN causal LM 的价值、SpAD cold start、offline exposure bias，以及 spike-aware OPD 的必要性。
2. Related Work：spiking LMs、offline ANN-to-SNN distillation、OPD/GKD、top-k OPD 与 efficient distillation。
3. Method：two-stage SA-TOPD、rate bridge、top-k local support、spike-rate regularization。
4. Theory：bounded gradient analysis 与 local supervision path shortening。
5. Experiments：SpAD reproduction baseline、small-batch gates、full runs、ablations、efficiency/stability。
6. Limitations：teacher access 依赖、top-k approximation bias、与 official BiSpikCLM 的 reproduction gap、hardware budget。

## 9. Source Anchors

论文中可使用以下外部 anchors：

1. BiSpikCLM arXiv page: https://arxiv.org/abs/2605.13859
   - 用于支持 BiSpikCLM 提出 SFSA 和 SpAD，并在多个层级对齐 ANN teacher 与 SNN student。
2. GKD / On-Policy Distillation of Language Models: https://arxiv.org/abs/2306.13649
   - 用于支持 OPD 在 self-generated output sequences 上使用 teacher feedback 训练学生。
3. OPD survey: https://arxiv.org/abs/2604.00626
   - 用于 broader OPD framing 和 exposure-bias motivation。
4. Verl OPD documentation: https://verl.readthedocs.io/en/latest/algo/opd.html
   - 谨慎使用，用来说明 top-k forward-KL OPD 已经存在，因此 SA-TOPD 不能把 top-k OPD 作为 standalone invention。

## 10. 下一步实现计划

下一步实现应作为单独计划展开：

1. 从固定 SpAD checkpoint 加入 stage-2 rollout data flow。
2. 在 student-induced states 上实现 teacher inference，并 detached top-k outputs。
3. 加入 selected-layer rate bridge adapters。
4. 加入 dynamic spike-rate regularization 与对应 logging。
5. 加入 SpAD-only continuation 和 naive OPD controls。
6. 在任何 full three-GPU run 前，先运行 small-batch gate against the same checkpoint。

实现计划必须保留用户的 loop rule：每个 candidate 先在小批量上和当前最佳 baseline 比较；只有明显小批量胜出后，才进入 full training。
