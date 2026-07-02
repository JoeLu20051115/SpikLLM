# SA-TOPD：面向二值脉冲因果语言模型的脉冲感知两阶段 On-Policy 蒸馏

> 中文顶会风格主文草稿，结果待填版。本文档用于组织论文主线和实验协议；所有尚未完成的实验结论均显式标为“待实验验证”，不得在投稿版本中保留为已完成结果。

## 摘要

二值脉冲因果语言模型为低功耗自回归推理提供了一条有吸引力的路径，但其训练仍高度依赖离线 ANN-to-SNN 蒸馏。近期 BiSpikCLM 通过 Softmax-Free Spiking Attention 与 Spike-Aware Alignment Distillation 证明了全二值脉冲因果语言模型的可行性，但该范式主要在真实前缀或离线语料状态上对齐教师与学生，仍未直接处理学生在推理时由自身生成历史诱导出的状态分布。与此同时，ANN 语言模型中的 on-policy distillation 已经表明，在学生自身轨迹上接受教师反馈可以缓解自回归蒸馏的 exposure bias；然而，将这一思想直接迁移到 SNN 会遇到脉冲速率表征与连续教师隐空间错配、在线词表监督开销、以及发放率动力学漂移三类额外约束。

本文提出 **SA-TOPD**，一个面向二值脉冲因果语言模型的脉冲感知两阶段 on-policy 蒸馏框架。SA-TOPD 首先使用 SpAD 进行离线冷启动，将学生模型带入可训练的教师分布邻域；随后在学生自生成轨迹上执行在线后训练，并通过三类设计使 OPD 适配脉冲模型：rate-domain feature bridge 将时间平均脉冲率与教师连续隐状态对齐，top-k local-support distillation 在教师高置信局部支撑集上提供输出监督，动态退火发放率正则显式约束在线训练中的脉冲动力学漂移。本文的贡献不是把 top-k 蒸馏、特征 MSE 或发放率正则各自包装成单点新颖性，而是将它们组织为一个针对 SNN 表征、计算和动力学约束的系统级 OPD 适配。理论上，我们在有界权重、有界 surrogate gradient 和受控 adapter 的假设下给出有限时间梯度上界，说明 selected-layer 局部特征监督如何缩短从 loss 到浅层脉冲参数的监督路径，并显式刻画 top-k 局部蒸馏相对于 full-vocabulary KL 的有偏性。实验上，本文将以 SpAD-only continuation 和 naive OPD 为核心对照，验证 SA-TOPD 是否能在 matched checkpoint 和 matched budget 下改善语言建模质量，同时维持 spike-rate 与能效代理指标的稳定性。

## 1. 引言

脉冲神经网络为因果语言模型的低功耗推理提供了不同于稠密浮点 Transformer 的计算范式。传统 ANN 语言模型依赖连续激活、矩阵乘法和 softmax 归一化来建模大词表条件分布，而 SNN 通过离散脉冲、膜电位积分和事件驱动传播表达信息。这种表示方式在神经形态硬件和低能耗部署场景中具有潜在优势，但也带来了更困难的优化问题：因果语言模型不仅需要在长上下文中保持 token 依赖，还需要在自回归展开中维持稳定概率分布，而二值脉冲、替代梯度和多时间步动力学都会放大训练不稳定性。

现有脉冲语言模型的关键突破在于证明离线分层蒸馏可以完成脉冲学生的冷启动。BiSpikCLM 提出 Softmax-Free Spiking Attention 与 Spike-Aware Alignment Distillation，将 ANN 教师与 SNN 学生在 embedding、attention、intermediate features 和 logits 等层级上进行对齐，从而训练 fully binary spiking causal language model。这个结果说明，离线 ANN-to-SNN 蒸馏是构建脉冲因果语言模型的有效入口。然而，自回归推理的训练目标并不只是在真实前缀上拟合教师分布；测试时，学生必须在自己生成的历史上持续展开，而早期 token 的偏差会改变后续状态分布并导致误差累积。换言之，SpAD 能解决冷启动问题，但 offline-only 蒸馏本身并不消除 student-induced trajectory 上的 exposure bias。

On-policy distillation 为这一分布失配提供了自然动机。GKD 与后续 OPD 工作将学生自身生成的序列纳入训练，让教师在学生实际访问的状态上提供反馈，从而把训练分布向推理分布靠近。对 ANN 语言模型而言，这一思想主要涉及如何选择 divergence、如何查询教师、以及如何稳定长序列后训练；但对二值脉冲因果语言模型而言，直接套用 ANN OPD 并不充分。首先，学生内部表征是时间平均脉冲率或膜电位相关状态，而教师隐状态处在连续浮点空间，在线阶段的层级监督存在表示域错配。其次，在线 rollout 会显著增加教师查询与输出分布监督成本，SNN 的多时间步展开进一步放大显存和计算压力。最后，学生在自生成轨迹上更新会改变层级发放率分布，缺少约束时容易出现近静默或过激活的动力学漂移。

本文提出 **Spike-Aware Two-stage On-Policy Distillation (SA-TOPD)**，以“离线冷启动 + 在线脉冲感知后训练”的分工处理上述问题。第一阶段完整保留 SpAD 的作用：在固定语料和教师监督下获得结构稳定、分布合理的脉冲学生初始化。第二阶段将训练状态切换到学生自身诱导的轨迹，并在该轨迹上施加三类 spike-aware 约束：rate-domain bridge 为脉冲速率特征提供局部层级监督，top-k local-support distillation 将输出蒸馏限制在教师高置信局部支撑集上，annealed spike-rate regularization 将在线训练中的动力学漂移转化为显式优化成本。这一设计不是把 ANN OPD 简单迁移到 SNN，而是围绕 SNN 的表征、计算和动力学约束重写在线蒸馏。

本文贡献可以概括为三点。第一，我们系统提出二值脉冲因果语言模型的 spike-aware on-policy post-training 问题，并将其与 SpAD 冷启动形成两阶段训练范式。第二，我们设计了面向 SNN 在线蒸馏的三类适配机制，分别处理脉冲-连续表征错配、在线输出监督开销与发放率动力学漂移；这些机制的价值来自它们在 SNN 在线后训练中的组合约束，而不是任何单个模块的孤立新颖性。第三，我们给出有界假设下的梯度路径分析和 top-k 近似偏差分析，并提出一个以 SpAD-only continuation、naive OPD 和模块 ablation 为核心的实验协议，用于检验 SA-TOPD 是否真正优于离线蒸馏后的继续训练。

## 2. 相关工作

### 2.1 脉冲语言模型与离线 ANN-to-SNN 蒸馏

脉冲神经网络长期被视为低功耗序列建模的一种候选架构，但将其扩展到因果语言模型需要同时处理大词表概率建模、长上下文依赖和自回归稳定性。BiSpikCLM 是这一方向中最接近本文的工作：它通过 Softmax-Free Spiking Attention 去除传统注意力中的 softmax，并通过 Spike-Aware Alignment Distillation 在多个层级对齐 ANN 教师和二值脉冲学生。本文不重新发明 SpAD，也不否认离线蒸馏的重要性；相反，SA-TOPD 将 SpAD 作为第一阶段冷启动，并研究一个后续问题：当脉冲学生已经被拉入教师邻域后，如何在学生自身生成轨迹上继续训练。

### 2.2 自回归语言模型中的 On-Policy Distillation

标准知识蒸馏通常在固定数据或教师生成数据上训练学生，因此学生训练时看到的状态不一定等于推理时自身诱导出的状态。GKD 和后续 OPD 研究将学生生成的序列纳入训练，并在这些 self-generated states 上查询教师反馈，从而缓解 train-test distribution mismatch。近期 OPD 文献也开始讨论不同反馈信号、teacher access 和 token-level loss granularity 的统一框架。本文继承的是 OPD 的分布匹配动机，而不是声称 OPD 本身是新发明；我们的差异在于目标模型是二值脉冲因果语言模型，其在线后训练必须同时控制脉冲表示域和发放率动力学。

### 2.3 高效局部输出蒸馏

在线蒸馏的输出层监督往往受限于词表规模和教师查询成本。已有工程系统和 OPD 实现已经包含 top-k forward KL 或 teacher top-k sparse supervision，用于降低 full-vocabulary soft target 的存储和计算压力。因此，top-k 局部蒸馏不能作为本文的单点新颖性。SA-TOPD 对这一技术的使用是更受约束的：top-k local support 被放入 SNN 在线训练框架中，并与 rate-domain feature bridge 和 spike-rate regularization 联合使用，以应对脉冲模型在 rollout 过程中的表示错配和动力学漂移。

## 3. 方法

### 3.1 问题设定与框架总览

SA-TOPD 研究冻结 ANN 教师 `T` 和二值脉冲因果语言模型学生 `S_theta` 之间的两阶段蒸馏。给定真实 prompt `x`，学生在第二阶段自回归生成 response trajectory `y_1:K ~ pi_theta(. | x)`，教师仅在学生已经访问到的状态 `s_k = (x, y_<k)` 上提供 detached supervision。训练目标只在 response 区域激活，prompt 区域仅用于条件上下文。这个设定使在线阶段更接近推理时的状态分布，同时避免教师参与采样而改变学生轨迹。

SA-TOPD 的总体目标为：

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

其中 `m_k` 为 response mask，`L_anchor` 是可选的离线回放或 teacher-forced anchor 项。主方法只依赖三个核心项：`L_bridge`、`L_topk` 和 `L_spike`。这些项分别对应 SNN 在线蒸馏中的三个失败模式：内部表征跨域错配、输出层在线监督成本过高、以及脉冲发放率在学生轨迹上漂移。

### 3.2 阶段一：SpAD 冷启动

第一阶段使用 BiSpikCLM 风格的 SpAD 完成学生冷启动。学生在真实语料前缀上联合优化 embedding alignment、spiking attention alignment、spiking feature alignment、soft target alignment 和 hard target alignment：

```text
L_SpAD =
  lambda_EA  * L_EA
+ lambda_SAA * L_SAA
+ lambda_SFA * L_SFA
+ lambda_STA * L_STA
+ lambda_HTA * L_HTA
```

本文使用 paper-faithful 权重 `lambda_EA = 0.2`、`lambda_SAA = 0.1`、`lambda_SFA = 0.1`、`lambda_STA = 0.3`、`lambda_HTA = 0.3`。阶段一的目标不是作为本文方法创新，而是获得可进入在线后训练的脉冲学生 checkpoint。若无法完全复现官方 BiSpikCLM 训练规模，所有后续结论均应表述为相对于本文 SpAD reproduction baseline 的改进，而不是相对于官方结果的改进。

### 3.3 Rate-Domain Feature Bridge

Rate-domain feature bridge 为脉冲学生提供与教师连续隐空间对齐的局部监督路径。学生第 `l` 层在 `T_snn` 个内部时间步上的二值脉冲为 `S_l(k,t)`，其时间平均速率表示为：

```text
H_S,l(k) = (1 / T_snn) * sum_t S_l(k, t)
```

由于 `H_S,l(k)` 位于 `[0,1]^d`，而教师隐状态 `H_T,l(k)` 位于连续实数空间，直接最小化二者 MSE 会混合表示域差异和语义差异。SA-TOPD 因此使用逐通道仿射 adapter：

```text
A_l(H_S,l) = gamma_l * H_S,l + beta_l
```

并以 `gamma_l = 1`、`beta_l = 0` 初始化，使在线阶段开始时不破坏阶段一已学习的表示流形。对 selected layers `L_sel`，feature bridge loss 为：

```text
L_bridge(k) =
  1 / |L_sel|
  * sum_{l in L_sel} mean_j || stopgrad(H_T,l,j(k)) - A_l(H_S,l,j(k)) ||_2^2
```

主实验优先使用 selected-layer bridge，例如覆盖网络深度的 25%、50%、75% 和 100% 位置。全层 bridge 可以作为 ablation，而不作为默认主方法，因为它可能带来过高教师 hidden-state 存储成本，并且可能过度约束学生内部表示。

### 3.4 Top-k Local-Support Output Distillation

Top-k local-support distillation 将在线输出监督限制在教师高置信局部支撑集上。对学生诱导状态 `s_k`，教师给出 logits `Z_T(k)`，并取：

```text
I_k = TopK(Z_T(k), k_top)
```

教师和学生只在 `I_k` 上 gather logits：

```text
z_T^I(k) = gather(Z_T(k), I_k)
z_S^I(k) = gather(Z_S(k), I_k)
```

再用温度 `tau` 得到局部分布：

```text
p_T^I = softmax(z_T^I / tau)
p_S^I = softmax(z_S^I / tau)
```

输出蒸馏目标为：

```text
L_topk(k) = - sum_i p_T^I(i) * log p_S^I(i)
```

该目标是局部 forward-KL 的交叉熵形式，而不是 exact full-vocabulary KL。若实现仍计算完整学生 LM head 后再 gather top-k logits，则它只降低 soft target 存储、KL 计算和监督密度，不降低学生 LM-head forward 的 `O(V)` 成本。只有引入 indexed output projection 或 sampled head 时，才能进一步声称接近 `O(k)` 的输出层开销。为避免 top-k 近似掩盖教师分布质量，实验必须记录 `teacher_topk_mass`、`teacher_topk_entropy` 与 `student_topk_entropy`。

### 3.5 动态退火发放率正则

在线蒸馏会改变学生中间层的发放率分布，因此 SA-TOPD 显式惩罚过静默或过激活的层。第 `l` 层在 step `k` 的平均发放率为：

```text
FR_l(k) = mean_{tokens, channels, timesteps} S_l(k, t)
```

区间正则定义为：

```text
L_spike(k) =
  sum_l [
    max(0, FR_l(k) - p_max)^2
  + max(0, p_min - FR_l(k))^2
  ]
```

其系数采用退火形式：

```text
lambda_spike(t) =
  lambda_final
  + (lambda_init - lambda_final) * decay(t)
```

早期较大的 `lambda_spike` 用于避免发放率过早进入 silent 或 saturated regime；后期较小的 `lambda_spike` 释放表示拟合自由度。该项不保证每一步发放率都位于区间内，但把 spike-dynamics drift 变成可观测、可惩罚的优化对象。

### 3.6 主方法边界与可选稳定化项

SA-TOPD 的主方法只包含 selected-layer rate bridge、top-k local support 和 annealed spike-rate regularization。全层 OPRD、teacher distribution EMA 和 product-form reliability gate 不进入主方法。全层 OPRD 只作为 cost/over-constraint ablation，因为它会显著增加在线 teacher hidden-state 读取成本，并可能把 ANN 的连续隐空间过强地投射到 SNN 内部表示上。对 teacher distribution 直接做 EMA 也不作为主方法，因为不同 token、不同 prefix 和不同 top-k index set 之间的概率向量没有天然可平均的语义。若需要平滑，应对 loss statistics、teacher top-k mass 或 reliability score 做 EMA，而不是改变教师分布本身。

早期讨论过的 product-form SOD gate 也不作为主贡献。其形式大致为：

```text
w_k = product_u (d_u + eps) / (d_{u+1} + eps)
```

这个连乘项近似退化为 `(d_1 + eps) / (d_k + eps)`，会在后续 token 与教师差异变大时降低梯度权重；但这些 token 往往正是最需要教师纠正的位置。因此 reliability gate 只能作为 appendix heuristic，并且必须带非零下界，例如：

```text
r_k = clip(exp(-alpha * max(0, d_k - EMA(d))), r_min, 1.0)
```

只有当 ablation 证明它在不损害 hard loss 和 teacher agreement 的情况下提升稳定性时，才可以考虑将其纳入扩展版本。

## 4. 理论分析

### 4.1 有界假设下的梯度上界

SA-TOPD 的稳定性分析必须建立在显式有界假设之上。我们假设网络深度与 spiking unroll length 有限，权重 operator norm、adapter 参数、教师 hidden states 和 surrogate derivatives 均在分析区间内有界，且所有 loss weights 有限非负。此时，top-k local-support loss 对局部学生 logits 的导数为：

```text
d L_topk / d z_S^I = (1 / tau) * (p_S^I - p_T^I)
```

由于 `p_S^I` 和 `p_T^I` 都是概率向量，有：

```text
||d L_topk / d z_S^I||_2 <= sqrt(2) / tau
```

feature bridge 的导数为：

```text
d L_bridge,l / d H_S,l
  = (2 / d) * gamma_l * (A_l(H_S,l) - stopgrad(H_T,l))
```

在有界教师状态、有界学生 rate feature 与有界 adapter 参数下，该项也有有限上界。spike-rate interval penalty 的导数在 `[p_min,p_max]` 内为零，在区间外为 piecewise linear；由于 firing rate 位于 `[0,1]`，该项在有限系数下同样有界。因此，在 leaky dynamics、bounded surrogate derivative 和 bounded input 的条件下，任意层参数梯度满足：

```text
||d L_TOPD / d W_l|| <= C_l < infinity
```

其中 `C_l` 依赖权重界、surrogate derivative 界、adapter 界、温度、loss weights、网络深度与序列长度。这个命题只能支持“有界假设下存在有限梯度上界”，不能写成 SA-TOPD 绝对防止梯度爆炸。

### 4.2 局部监督路径与梯度衰减风险

Rate-domain bridge 的主要理论作用是缩短监督路径，而不是保证梯度永不消失。若只使用 output-level OPD，第 `l` 层参数的监督需要经过后续层 Jacobian 的连乘：

```text
d L_output / d W_l
  = d L_output / d H_L
    * product_{i=l}^{L-1} J_i
    * d H_l / d W_l
```

当多个 `||J_i|| < 1` 时，这条路径会随深度快速衰减。若第 `l` 层属于 selected layers，feature bridge 提供额外局部路径：

```text
d L_bridge,l / d W_l
  = d L_bridge,l / d H_l
    * d H_l / d W_l
```

这条路径不包含后续所有层的 Jacobian 乘积，因此减少了训练对长链 surrogate-gradient Jacobian 的依赖。该分析解释了为何多层局部监督可能提升深层 SNN 的训练稳定性，但不构成 strict positive lower bound：局部 feature error 可能接近零，surrogate derivative 可能落入饱和区，不同梯度项也可能抵消。

### 4.3 Top-k 近似的偏差与诊断

Top-k local support 是有偏近似，而不是 full-vocabulary KL 的无损替代。设教师完整分布在 top-k 集合上的概率质量为：

```text
M_k = sum_{i in I_k} p_T_full(i)
```

当 `M_k` 较高时，局部支撑集覆盖教师主要不确定性，局部 KL 更能代表教师分布；当 `M_k` 较低时，top-k 近似会忽略大量长尾概率质量，可能扭曲监督信号。因此，SA-TOPD 必须把 `teacher_topk_mass` 作为主实验诊断指标，而不是只报告训练 loss。

更严格地，设教师完整分布为 `P`，学生完整分布为 `Q`，top-k 集合为 `I`，`M = P(I)`，`q = Q(I)`。将 `P_I/M` 和 `Q_I/q` 表示为归一化后的 top-k 条件分布，将 tail 表示为补集上的条件分布，则 full-vocabulary forward KL 可以分解为：

```text
D(P || Q)
= M * D(P_I / M || Q_I / q)
+ (1 - M) * D(P_tail / (1 - M) || Q_tail / (1 - q))
+ M * log(M / q)
+ (1 - M) * log((1 - M) / (1 - q)).
```

SA-TOPD 的 local-support loss 只优化第一项中的局部条件分布结构，并没有优化 tail 条件分布和 top-k/tail 概率质量分配。因此，论文中应把 top-k local support 写成一种可诊断的 biased approximation，而不是 full-vocabulary KL 的等价替代。这个分解也给出实验指标的必要性：若 `teacher_topk_mass` 高且 `student_topk_mass` 不异常偏离，local-support loss 才更可能代表教师的主要监督信号。

## 5. 实验设计

### 5.1 实验问题

实验围绕五个问题组织。第一，SA-TOPD 是否在相同 checkpoint、token budget 和 hardware budget 下优于 SpAD-only continuation。第二，naive OPD 直接迁移到 spiking causal LM 是否稳定。第三，rate bridge、top-k local support 和 spike-rate regularization 各自贡献多少。第四，质量提升是否以破坏发放率和能效代理指标为代价。第五，趋势是否能在不同 seeds 或 checkpoints 上保持。

### 5.2 Baselines 与 Ablations

主实验应至少包含以下设置：

| ID | 方法 | 作用 |
| --- | --- | --- |
| B0 | SpAD-only continuation | 排除“只是多训练”的解释 |
| B1 | Naive OPD | 检查 ANN-style OPD 是否可直接迁移 |
| B2 | SA-TOPD w/o rate bridge | 验证表征桥接贡献 |
| B3 | SA-TOPD w/o spike-rate reg | 验证动力学约束贡献 |
| B4 | SA-TOPD w/o top-k 或 full-vocab feasible variant | 验证局部支撑蒸馏权衡 |
| B5 | SA-TOPD selected-layer bridge | 主方法 |
| B6 | SA-TOPD full-layer bridge | 分析全层监督的成本与过约束 |

如果 full-vocabulary online KL 在当前硬件预算下不可行，则用 `k in {16,32,64,128}` 的 top-k sweep 替代，并在论文中明确说明 full-vocabulary baseline 的不可行原因。可靠性 gate 只进入 appendix ablation，不作为主表核心模块。

### 5.3 训练协议

所有 candidate 必须先通过小批量 gate，再进入 full-run gate。小批量 gate 使用 `seq_len=512`、`batch_size=2`、`gradient_accumulation=16`、`T_snn=4`，并从同一个 stage-1 checkpoint 出发。candidate 只有在 matched token budget 下同时降低 hard loss 与 soft loss、保持或提升 teacher agreement、且不触发 spike-rate collapse 时，才允许进入 full run。当前历史参考是 loop91b：`hard=6.2209`、`soft=2.6782`、`total=3.0423`、`teacher_top1_agreement=20.74%`、`tokens_seen=36.8M`；但这个数值只能作为历史参考，不能替代 same-checkpoint continuation control。

Full run 使用项目既有 loop14/loop16 geometry：`3 x H200`、`seq_len=1024`、`batch_size_per_rank=4`、`gradient_accumulation=64`、`T_snn=4`、`bf16`。若 hard 与 soft losses 均低于 5，或二者在无 spike collapse 的情况下保持清晰下降趋势，则继续训练；若 hard loss 在 SpAD-only continuation 上方 plateau、soft loss 持续上升、发放率 collapse 或 `teacher_topk_mass` 过低，则停止并诊断。

### 5.4 指标

实验必须同时报告语言质量、教师一致性、脉冲动力学和效率代理指标。语言质量包括 validation PPL、rollout PPL 或 continuation NLL；教师一致性包括 token accuracy、top-5 accuracy、teacher top-1/top-5 agreement、target rank 和 target margin；脉冲动力学包括 layerwise spike-rate mean/std、silent layer count 和 overactive layer count；效率代理包括 SynOps proxy、activation sparsity、peak memory、tokens per second 和 teacher query cost。对于 top-k 蒸馏，还必须报告 `teacher_topk_mass`、`teacher_topk_entropy`、`student_topk_entropy` 与 `KL_on_topk`。

## 6. 限制

SA-TOPD 的第一项限制是 teacher access 成本。第二阶段需要在学生诱导状态上查询教师，并可能读取教师 hidden states 或 top-k logits，因此训练成本会高于纯 SpAD-only continuation。本文将通过 selected-layer bridge、top-k local support 和 teacher-query logging 控制并报告该成本，但这并不消除其存在。

SA-TOPD 的第二项限制是 top-k 近似偏差。局部支撑蒸馏并不等价于 full-vocabulary KL；当教师 top-k mass 较低时，局部监督可能忽略有意义的候选 token。本文通过记录 retained mass 和 top-k sweep 诊断这一问题，但不声称 top-k 在所有状态下都是无损替代。

SA-TOPD 的第三项限制是 reproduction baseline 边界。若无法完全复现官方 BiSpikCLM 的数据规模、教师设置和训练 token budget，则本文只能声称相对于 paper-faithful SpAD reproduction baseline 的提升，而不能声称优于官方 BiSpikCLM 数值。这个边界需要在实验部分和结论中保持一致。

## 7. 结论

本文提出 SA-TOPD，一个面向二值脉冲因果语言模型的两阶段 on-policy 蒸馏框架。SA-TOPD 保留 SpAD 作为离线冷启动阶段，并在第二阶段将教师监督移到学生自生成轨迹上，以缓解 offline-only 蒸馏在自回归推理中的状态分布失配。为了使 OPD 真正适配 SNN，SA-TOPD 同时引入 rate-domain feature bridge、top-k local-support output distillation 和动态退火发放率正则，分别处理表征错配、在线监督开销和脉冲动力学漂移。

本文的核心价值不在于把 ANN OPD 直接迁移到脉冲模型，而在于把 on-policy post-training 重写为一个受 SNN 表示和动力学约束支配的训练问题。后续实验将检验这一框架是否能在 matched checkpoint 和 matched budget 下稳定优于 SpAD-only continuation，并在提升语言建模质量的同时维持发放率与能效代理指标的可控性。

## 参考文献锚点

1. Guo et al. **BiSpikCLM: A Spiking Language Model integrating Softmax-Free Spiking Attention and Spike-Aware Alignment Distillation**. arXiv:2605.13859. https://arxiv.org/abs/2605.13859
2. Agarwal et al. **On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes**. arXiv:2306.13649. https://arxiv.org/abs/2306.13649
3. Song and Zheng. **A Survey of On-Policy Distillation for Large Language Models**. arXiv:2604.00626. https://arxiv.org/abs/2604.00626
4. verl documentation. **On-Policy Distillation (OPD)**. https://verl.readthedocs.io/en/latest/algo/opd.html

## 附录 A：段落反向大纲

摘要：任务、挑战、方法、理论边界和实验协议一次性交代，不声称已有 SA-TOPD 结果。

引言第 1 段：说明 SNN causal LM 的价值和训练难点。

引言第 2 段：说明 BiSpikCLM/SpAD 的意义和 offline-only 的不足。

引言第 3 段：说明 OPD 动机以及为什么 ANN OPD 不能直接套用到 SNN。

引言第 4 段：提出 SA-TOPD 的两阶段框架和三个模块。

引言第 5 段：列出贡献，并避免过度 claim。

相关工作：按脉冲 LM 与 SpAD、OPD、top-k 局部蒸馏三个主题组织，不做 citation dump。

方法：先定义问题，再写两阶段和三个模块，每个模块都包含动机、设计和可验证优势。

理论：只做有界假设下的上界和路径缩短分析，不写绝对稳定。

实验：写成可执行 protocol，明确 baselines、ablations、gate 和 metrics。

限制与结论：保留 teacher cost、top-k bias 和 reproduction boundary。

## 附录 B：五维自审

**贡献**：主贡献被限定为 spike-aware on-policy post-training 的系统设计，而不是 top-k 或 MSE 的单点新颖性。风险可控。

**写作清晰度**：主线从 SNN causal LM 到 SpAD，再到 OPD，再到 SNN-specific adaptation，逻辑连续。术语保持 SA-TOPD、SpAD、rate-domain bridge、top-k local support、spike-rate regularization。

**实验强度**：目前是实验协议，不是已完成结果。投稿前必须补齐 SpAD-only continuation、naive OPD 和模块 ablation。

**评估完整性**：指标覆盖 loss、PPL、teacher agreement、spike dynamics、efficiency proxy 和 top-k diagnostics。仍需实际实现后确认可记录性。

**方法设计合理性**：三个模块分别对应三个 failure modes。SOD 未进入主方法，避免 heuristic 过重。

## 附录 C：Claim-Evidence Map

Claim: BiSpikCLM 证明 SpAD 可以用于二值脉冲因果语言模型冷启动。 | Evidence: BiSpikCLM arXiv 摘要与方法描述。 | Status: supported.

Claim: Offline-only distillation 对自回归模型存在 exposure bias 风险。 | Evidence: GKD/OPD 文献关于 self-generated trajectories 的动机。 | Status: supported at motivation level.

Claim: ANN OPD 不能无改动迁移到 SNN causal LM。 | Evidence: SNN 的 rate-domain representation、temporal unroll 和 firing-rate dynamics 与 ANN 不同；仍需 naive OPD ablation 作为直接实验证据。 | Status: needs evidence.

Claim: SA-TOPD 可以优于 SpAD-only continuation。 | Evidence: 当前仅为实验目标；需要 matched checkpoint 和 matched budget 实验。 | Status: needs evidence.

Claim: Rate-domain bridge 缩短监督路径。 | Evidence: 理论路径分解显示局部 bridge loss 不包含后续层 Jacobian 连乘。 | Status: supported under assumptions.

Claim: Top-k local support 降低在线监督开销但引入近似偏差。 | Evidence: top-k 只优化局部支撑集；teacher_topk_mass 可诊断保留概率质量。 | Status: supported conceptually, needs implementation metrics.

## 附录 D：最终审稿人压力测试与可投写法

本附录记录作者侧模拟审稿人面板的最终共识。它不是投稿正文的一部分，但用于约束主文写法，防止把工程组合写成过强理论贡献。

### D.1 初始拒稿理由

严苛审稿面板对早期 StableOPD/SA-TOPD 草案的初始判断是 Reject。主要问题不是方向错误，而是 claim 过大、模块边界不清、理论证明过强。OPD、top-k sparse supervision、feature MSE 和 firing-rate regularization 都有已有来源或自然对应物；如果论文把这些模块各自写成新发明，创新性会被认为不足。若同时声称“彻底解决显存瓶颈、梯度奇点和物理约束冲突”，理论部分也会被直接打穿。

### D.2 审稿人面板最终共识

| 审稿人角色 | 初始质疑 | 修补后可接受条件 |
| --- | --- | --- |
| R1 创新性 | top-k、feature MSE、rate regularization 都不是单点新颖贡献 | 贡献改写为 binary spiking causal LM 的 system-level spike-aware OPD 适配 |
| R2 理论严谨性 | 不能证明绝对稳定、绝对不爆炸或绝对不消失 | 理论只证明有界假设下的 finite upper bound、短监督路径和 top-k bias decomposition |
| R3 方法设计 | full-layer OPRD 与显存节省冲突，且可能过约束 SNN 表示 | 主方法使用 selected-layer bridge；full-layer 只作为 ablation |
| R4 Devil's Advocate | product-form SOD 可能在最需要纠正的 token 上降低梯度 | SOD 不进入主方法；若测试，必须带非零下界并放 appendix |
| EIC 综合判断 | 方法可能被看作 heuristic soup | 以 SpAD-only continuation、naive OPD 和模块 ablation 为最低实验门槛 |

### D.3 审稿人认可的主方法写法

推荐写法：

> SA-TOPD is a two-stage spike-aware on-policy distillation framework for binary spiking causal language models. Rather than claiming top-k distillation, feature alignment, or firing-rate regularization as isolated inventions, SA-TOPD studies how these components must be jointly constrained when OPD is moved from ANN language models to SNN causal language models. The framework addresses three SNN-specific failure modes: rate-domain mismatch between binary spike features and continuous teacher states, online supervision cost under student-induced trajectories, and firing-rate drift during autoregressive post-training.

中文正文可写为：

> SA-TOPD 的贡献不是提出某个孤立的新损失，而是把 on-policy distillation 重写为一个受 SNN 表征、计算和动力学约束支配的后训练问题。rate-domain bridge、top-k local support 和 spike-rate regularization 分别对应在线 SNN 蒸馏中的三个 failure modes，并通过 matched-budget ablation 验证其必要性。

### D.4 允许和禁止的 Claims

允许写：

1. SA-TOPD 系统研究 binary spiking causal LM 的 spike-aware on-policy post-training。
2. 在 matched checkpoint 和 matched budget 下，SA-TOPD 若实验支持，可以声称优于本文 SpAD-only reproduction baseline。
3. selected-layer rate bridge 提供不经过后续所有层 Jacobian 连乘的局部监督路径。
4. top-k local support 是有偏但可诊断的近似，其偏差通过 teacher top-k mass 和 student mass diagnostics 监控。
5. spike-rate interval penalty 将动力学漂移转化为可观测、可惩罚的优化成本。

禁止写：

1. SA-TOPD 绝对防止梯度爆炸或梯度消失。
2. top-k local support 等价于 full-vocabulary KL。
3. 除非实现 indexed output projection 或 sampled head，否则声称 LM-head compute 从 `O(V)` 降到 `O(k)`。
4. full-layer OPRD 总是优于 selected-layer supervision。
5. 在未完整复现官方设置前，声称优于 official BiSpikCLM。
6. product-form SOD 是主贡献或理论必要项。

### D.5 最终理论证明骨架

**命题一：有限时间梯度上界。** 在有限网络深度、有限 SNN unroll length、有界权重、有界 surrogate derivative、有界 adapter、有界教师 hidden states、有限 loss weights 和 `tau > 0` 的条件下，每个局部 loss source 对其直接输入的导数有界；因此每层参数梯度存在有限上界 `C_l`。该命题只能支持 conditional boundedness，不能支持 unconditional stability。

**命题二：selected-layer bridge 缩短监督路径。** Output-only OPD 的浅层梯度包含后续层 Jacobian 连乘，而 selected-layer bridge 为被监督层提供不经过后续所有层的局部路径。这说明 bridge 降低了训练对长链 surrogate-gradient product 的依赖，但不保证 positive lower bound，因为 feature error、surrogate derivative 和梯度相互抵消都可能使局部梯度变小。

**命题三：top-k 近似偏差可分解。** Full-vocabulary KL 可以分解为 top-k 条件 KL、tail 条件 KL 和 top-k/tail mass allocation 项。SA-TOPD 的 local-support loss 只覆盖 top-k 条件 KL，因此必须报告 `teacher_topk_mass`、`student_topk_mass` 和 entropy diagnostics。这个命题把 top-k 从“无损替代”改写为“可诊断近似”，是审稿人可接受的边界。

### D.6 最低可接受实验门槛

审稿人面板最终要求最小实验包包含：

1. `B0`: SpAD-only continuation，排除“只是多训练”的解释。
2. `B1`: naive OPD，检验 ANN-style OPD 是否能直接迁移到 SNN。
3. `B2-B4`: 分别去掉 rate bridge、top-k local support 和 spike-rate regularization 的 ablation。
4. `B5`: selected-layer SA-TOPD 主方法。
5. `B6`: full-layer bridge 作为 cost/over-constraint ablation。
6. top-k sweep 或 full-vocabulary feasible variant，并报告 retained mass diagnostics。

如果这些对照不完整，论文只能被表述为 proposal 或 technical report，不能写成已验证的 ICLR-level empirical claim。
