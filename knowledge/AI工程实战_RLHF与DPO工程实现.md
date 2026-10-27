# AI 工程实战——RLHF 与 DPO 工程实现

> 偏好对齐的全流程工程实践——奖励模型训练、PPO/DPO/KTO、偏好数据Pipeline、部署监控
> 2026年——RLHF 从"黑魔法"走向"可复现工程"

---

## 一、对齐方法的选择框架

### 1.1 方法矩阵

```
方法         | 需要 reward model | 在线/离线 | 复杂度 | 样本效率 | 2026 适用场景
-------------|-------------------|-----------|--------|----------|---------------
PPO (RLHF)   | 是                | 在线      | 高     | 高       | 需要强对齐的大模型
DPO          | 否                | 离线      | 低     | 中       | 中小模型、快速迭代
KTO          | 否                | 离线      | 低     | 低(但只需要二分类信号) | 只有"好/不好"标签
GRPO         | 否                | 在线      | 中     | 高(组间相对) | DeepSeek 系列、数学推理
Rejection Sampling | 否          | 离线      | 低     | 低       | 初始数据收集
SimPO        | 否                | 离线      | 低     | 中(无需参考模型) | 资源受限场景
```

### 1.2 2026年的主流选择

```
大模型主力训练 (~100B+):
  PPO + GRPO 混合:
    预训练后期: GRPO (组间奖励，不需要 reward model)
    RLHF 阶段: PPO (需要更精细的偏好建模)
    推理强化: GRPO (DeepSeek-R1 风格，self-play)

中小模型 (~7B-70B):
  DPO + Rejection Sampling:
    1. Supervised Fine-tuning (SFT)
    2. 生成多个候选，选择最好的 (Rejection Sampling)
    3. DPO 训练
    整个过程只需要 SFT 数据 + 偏好对

快速实验:
  SimPO (SimPO: Simple Preference Optimization):
    不需要参考模型
    直接在策略模型的 logits 上计算偏好损失
    训练速度: DPO 的 2×
    质量: 接近 DPO
```

---

## 二、RLHF 的完整工程实现

### 2.1 Reward Model 训练

```
Reward Model (RM) 的构建原则:

输入: 文本 + 响应 (AI 生成的回答)
输出: 一个标量分数 (scalar reward)

架构选择:
  - 通常基于 SFT 模型，去掉最后的 LM head
  - 替换为回归 head (一个线性层输出标量)
  - 2026 的改进: 使用 n 个独立的 reward head
    (helpfulness/harmlessness/honesty/hallucination 等维度)

数据准备:

def prepare_preference_data(chosen_responses, rejected_responses):
    # chosen: 人类偏好的回答
    # rejected: 人类不偏好的回答
    pairs = []
    for c, r in zip(chosen_responses, rejected_responses):
        pairs.append({
            "chosen": c,    # [prompt + 偏好回答]
            "rejected": r,  # [prompt + 不偏好回答]
        })
    return pairs

RM 训练损失 (Bradley-Terry 模型):

def reward_model_loss(rm, chosen, rejected):
    reward_chosen = rm(chosen)    # 偏好的奖励
    reward_rejected = rm(rejected)  # 不偏好的
    
    # 偏好回答的奖励应该 > 不偏好的
    loss = -F.logsigmoid(reward_chosen - reward_rejected).mean()
    
    # 可选: 加入 margin (偏好差距越大越好)
    # loss = -F.logsigmoid(reward_chosen - reward_rejected - margin).mean()
    
    return loss

2026 RM 工程关键点:

1. 奖励规范化:
   训练时: 对 batch 内的 reward 做 normalize
   推理时: 使用 running statistics 做标准化
   防止 reward hacking (模型学会利用 RM 的漏洞)

2. 多维度奖励:
   helpfulness_reward + safety_reward + honesty_reward
   加权平均: α·R_h + β·R_s + γ·R_hon
   权重可以动态调整 (安全问题权重高)

3. RM 的评估:
   Accuracy: RM 在验证集上正确选择偏好回答的比例
   Calibration: 奖励分数是否合理分布
   Consistency: 同一输入多次输出是否一致
```

### 2.2 PPO 训练

```
PPO (Proximal Policy Optimization) 是 RLHF 的核心:

4 个模型的配合:

  ① Policy Model (π₀): 被训练的模型，生成响应
  ② Reference Model (ref): 冻结的 π₀, 用于 KL 惩罚
  ③ Reward Model (RM): 给响应打分
  ④ Value Model (V): 估计状态价值 (可以共享 policy 的 body)

PPO 的训练循环:

for step in range(num_steps):
    # 1. 采样
    prompts = sample_batch()  # 从训练集中采样
    responses = policy_model.generate(prompts)  # π₀ 生成
    
    # 2. 打分
    rewards = reward_model(prompts, responses)  # RM 评估
    ref_logprobs = reference_model(prompts, responses)  # KL 参考
    
    # 3. Advantage 计算
    # GAE (Generalized Advantage Estimation)
    advantages = compute_gae(rewards, values)
    
    # 4. PPO 更新
    for _ in range(ppo_epochs):
        # 新策略的 logprobs
        new_logprobs = policy_model(prompts, responses)
        
        # 重要性采样比例
        ratio = exp(new_logprobs - old_logprobs)
        
        # 裁剪的 surrogate 损失
        surr1 = ratio * advantages
        surr2 = clip(ratio, 1-ε, 1+ε) * advantages
        policy_loss = -min(surr1, surr2)
        
        # KL 惩罚 (防止策略偏差太大)
        kl_loss = (new_logprobs - ref_logprobs).mean()
        
        # 总损失
        total_loss = policy_loss + kl_coef * kl_loss

Value Model (批评者) 的更新:

def train_value_model(value_model, states, returns):
    predicted_values = value_model(states)
    value_loss = F.mse_loss(predicted_values, returns)
    return value_loss

价值模型需要估计状态的"真实价值"——
这是 PPO 中最困难的部分，因为"真实价值"不可观测。
```

### 2.3 PPO 的工程坑与解决方案

```
① Reward Hacking: 
   模型学会利用 RM 的漏洞，而不是真正变好
   
   解决:
     - KL 惩罚 (强制模型不偏离参考策略太远)
     - Reward 归一化
     - 使用多个 RM 的平均分

② 训练不稳定:
   PPO 对超参数极其敏感
   
   解决:
     - 学习率: 通常为 SFT 的 1/10 (1e-6 级别)
     - KL 系数: 从 0.01 开始，动态调整
     - 裁剪 epsilon: 0.2

③ 模型坍缩:
   模型变得过于"安全" → 拒绝回答一切问题
   
   解决:
     - 在 reward 中加入 helpfulness 维度
     - 对"拒绝"使用负奖励
     - 退火 KL 系数

④ Generator 和 Reward Model 不匹配:
   生成器产生的内容 RM 无法准确评估
   
   解决:
     - 定期用生成器数据更新 RM
     - 在训练中动态调整 prompt 分布

⑤ PPO 的分布式实现:

   大规模 PPO 训练的 4 阶段流水线:
    
    Stage 1 (数据生成):
      N 个 actor GPU 并行生成响应
     
    Stage 2 (奖励计算):
      RM 对所有生成的响应打分
     
    Stage 3 (价值估计):
      Value Model 计算状态价值
     
    Stage 4 (策略更新):
      所有 actor 同步更新

    流水线重叠:
      Actor GPU 在计算梯度时，另一批已开始生成
      实现 4-stage pipeline parallelism for PPO
```

---

## 三、DPO 的工程实现

### 3.1 DPO 的核心公式

```
DPO (Direct Preference Optimization) 的核心思想:
  不需要显式的 reward model
  直接使用偏好对优化策略

DPO 损失函数:

def dpo_loss(policy_logps, ref_logps, chosen_mask, beta=0.1):
    # policy_logps: 策略模型对偏好对中两个回答的 logprob
    # ref_logps: 参考模型对偏好对中两个回答的 logprob
    
    # 计算 log π(y_w|x) / π_ref(y_w|x) 和 log π(y_l|x) / π_ref(y_l|x)
    policy_log_ratio = policy_logps - ref_logps
    # policy_chosen_log_ratio - policy_rejected_log_ratio
    log_ratio = policy_log_ratio[chosen_mask] - policy_log_ratio[~chosen_mask]
    
    # β 控制 KL 惩罚的强度
    # 小 β → 弱 KL 惩罚 (可能过拟合偏好数据)
    # 大 β → 强 KL 惩罚 (接近 SFT)
    loss = -F.logsigmoid(beta * log_ratio).mean()
    
    return loss
```

### 3.2 DPO 的工程实践

```
数据准备:
  # DPO 需要偏好对
  # (prompt, chosen_response, rejected_response)
  
  数据来源 (2026年):
    1. 人工标注 (高质量，最贵)
    2. AI 标记 (用一个模型标注另一个)
    3. 隐式反馈 (用户点击、停留时间等)
    4. 合成偏好 (用规则生成)

训练配置:
  # DPO 的关键超参数
  beta: 0.1        # KL 惩罚系数
  learning_rate: 5e-7  # 通常极小
  batch_size: 128
  max_length: 2048
  
  # β 的自适应调整
  if kl_divergence > threshold:
      beta *= 1.1  # 增加 KL 惩罚
  if train_accuracy < 0.5:
      beta *= 0.9  # 减少 KL 惩罚(让模型学得更快)

DPO 的变体 (2026):

  Iterative DPO:
    Step 1: 用 SFT 模型 + 初始偏好数据进行 DPO
    Step 2: 用 DPO 后的模型生成新候选
    Step 3: 用 AI 或人工选择偏好 → 扩展偏好数据集
    Step 4: 在新数据集上再做 DPO
    效果: 每次迭代提升 ~5-10%

  Online DPO:
    在训练过程中实时生成偏好对
    用当前的 policy 生成多个候选
    AI 标注 → 偏好数据 → 更新 policy
    优势: 更接近 PPO 的样本效率

  cDPO (Constrained DPO):
    在 DPO 损失中加入对特定维度的约束
    例如: 控制回答长度、控制安全分数
```

### 3.3 PPO vs DPO: 工程决策

```
指标        | PPO                     | DPO
------------|------------------------|------------------------
实现复杂度  | 高 (4 个模型)           | 低 (2 个模型)
训练稳定性  | 低 (需精细调参)         | 高
样本效率    | 高 (在线采样)           | 中 (依赖离线数据)
可扩展性    | 难 (需要 reward model)  | 易
最终性能    | 可能更好                | 接近 PPO
超参数敏感性 | 高                    | 中
内存消耗    | 高 (4 个模型)           | 中 (2 个模型)

工程建议 (2026):
  - 实验室/初期: 用 DPO (快速验证)
  - 生产环境: PPO (更好性能) + GRPO (推理增强)
  - 前沿模型: PPO + GRPO + Iterative DPO 混合
```

---

## 四、偏好数据 Pipeline

### 4.1 数据采集工程

```
偏好数据 Pipeline 的完整流程:

原始数据采集:
  人工标注:
    - 标注员同时对多个维度打分 (1-7 scale)
    - 每个样本至少 3 人标注 (计算一致性)
    - 2026 年: 标注员是自己的 AI 模型?
      (AI 标注成为标准实践)
  
  自动标注 (AI feedback):
    使用一个"评审"模型:
    评审标准: 安全性 + 有用性 + 诚实性
    每个维度生成详细评语和分数
    速度: 比人工快 1000×
    成本: 人工的 1/1000
    质量: 在某些维度上已超过人工 (2026 数据)

数据增强:
  回译: 将偏好对的回答翻译成其他语言再翻译回来
  扰动: 轻微修改偏好回答 (但不改变偏好关系)
  合成: 用规则生成"明显更好的回答"和"明显更差的回答"

质量控制:
  2-pass 验证: 
    Pass 1: 自动检测异常 (response 长度、重复、语言)
    Pass 2: 随机抽样人工检查

  关键指标:
    - 标注者一致性 (Cohen's κ)
    - 偏好强度分布 (不应全是"明显偏好")
    - 领域平衡 (所有领域都有足够的数据)
```

### 4.2 训练管线架构

```
2026年 RLHF 训练管线的标准架构:

┌─────────────────┐
│   数据存储层     │
│ (偏好数据仓库)   │
└────────┬────────┘
         ↓
┌─────────────────┐
│   数据Pipeline  │
│ (采样 → 过滤 →    │
│  格式化 → batching)│
└────────┬────────┘
         ↓
┌─────────────────┐
│   训练引擎       │
│ (Megatron/Deepspeed)│
│  SFT→RM→RL 三阶段 │
└────────┬────────┘
         ↓
┌─────────────────┐
│   评估层         │
│ (Reward Score /   │
│  KL / Helpfulness)│
└────────┬────────┘
         ↓
┌─────────────────┐
│   部署 & 监控    │
│ (在线 Serving    │
│  + 反馈收集)     │
└─────────────────┘

监控 Dashboard (必须指标):

  Training Metrics:
    - Reward mean / std
    - KL divergence
    - Policy entropy
    - Response length
    
  Evaluation Metrics:
    - Helpfulness score (on held-out set)
    - Safety score (on red teaming set)
    - Benchmark scores
    
  Production Metrics:
    - Human rating (crowd sourced)
    - User satisfaction (implicit)
    - Safety incident rate
```

---

## 展望

- **2027: 多模态 RLHF** — 对视觉内容进行偏好对齐，对视频生成进行奖励建模
- **2027: RLHF + Agent** — 对多步推理（Agent 行为）进行整体偏好优化，不仅仅是单轮响应
- **2027: Compressor for Reward Model** — 在不损失精度的情况下压缩 RM（用于加速 PPO）
- **2028: Constitutional AI 2.0** — 完全不需要人类偏好数据，模型通过原则自对齐
