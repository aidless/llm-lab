# AGI 分布式训练（扩展）

---

## 总览：当单卡放不下——并行策略的完整拼图

模型参数从 7B 到万亿，序列长度从 4K 到 1M+——单卡显存早已装不下——分布式训练成为 AGI 研究的"地基工程"。本扩展文档——在 DP/TP/PP 的基础并行之外——聚焦六个实战主题：序列并行、上下文并行、Expert Choice 路由、MoE 全栈并行、超级长序列训练、以及分布式 checkpoint 与容错。

```
并行策略全景（2026）：
数据并行 DP（FSDP/ZERO-3）——每卡一份参数副本（分片），数据不同
张量并行 TP（Megatron）——层内矩阵切分（列/行并行）
流水线并行 PP（1F1B/VPP）——层间切分，微批流水
序列并行 SP（Megatron-SP）——非张量部分的序列维切分
上下文并行 CP（ring attention 风格）——长序列跨设备切分
专家并行 EP（MoE）——专家放在不同设备，token 路由
高维组合：3D/4D 并行（TP × PP × DP × CP/EP）

显存与通信的权衡：
并行维度 × 模型维度 → 切分矩阵（每维度的通信量/显存/负载）
目标是：总吞吐最大（MFU）而非单个维度最优
```

---

## 一、序列并行（Sequence Parallelism）

### 1.1 动机：TP 的显存浪费

**标准张量并行（Megatron TP）**把每层内部的矩阵（Q、K、V、O、MLP 权重）切分到多卡——但**非张量操作**（Dropout、LayerNorm、残差连接、激活函数、词嵌入、损失）仍在每张卡上持有完整副本：

```
标准 TP 的显存分布（每张卡）：
- 矩阵权重：1/TP 份额 ✓ 已切分
- LayerNorm 参数：完整副本 ✗ 冗余
- Dropout mask：完整序列 × 完整隐层 ✗ 冗余
- 激活：完整序列 × 完整隐层 ✗ 冗余
- 词嵌入与输出层（LM head）：完整副本 ✗ 冗余（词表可以很大！）

对于大词表模型（词表 100K-1M）——输出层冗余很可观
对于长序列——激活冗余（batch × 序列 × 隐层）是主要浪费
```

### 1.2 Megatron-SP 的机制

**序列并行（SP）**：把"非张量操作"的序列维也切分——每张卡只处理自己那一段序列：

```
标准 TP 层内数据流：
X（完整序列）→ LN → attn（TP 切分）→ O 投影（TP）→ all-reduce → drop + residual → LN → MLP（TP）→ all-reduce → drop + residual

SP 版数据流：
X 按序列切分（每卡 1/TP 段）→ LN（切分段各自算）→ attn（TP 内 all-to-all 前需 gather QKV）→ 每卡算自己的注意力输出 → O 投影 → all-reduce → 切回序列段 → drop/residual（本段）→ LN（本段）→ MLP → all-reduce → 切回段 → drop/residual

通信变化：
- 原 TP：每层两次全量的 all-reduce（attention 后 + MLP 后）
- SP：同样两次 all-reduce——但融合了序列 gather + 切分
  实际优化：QKV 投影改成列并行（column parallel）+ all-reduce
  一次性通信 = TP 的 gather + 后续的 scatter 合并进同一轮 all-reduce

结论：SP 的通信量 = 原 TP 通信量（不增加）——但显存大幅节省
（Dropout mask、LN 激活、残差激活的冗余全部消除）
```

### 1.3 SP 的工程收益

```
显存收益：
- 激活显存：从 (batch × seq × hidden) × TP 冗余 → (batch × seq × hidden) × 1
- 词表大的模型：LM head 的冗余也消除（若把词嵌入/输出层也做 SP 切分）
- 论文数据（Megatron-LM 原文）：8 卡 TP 下——SP 节省约 30% 激活显存

训练稳定性收益（额外发现，Shoeybi et al., 2022）：
- SP 中的 LN 按段计算——各段统计量略不同——等于隐式正则
- 论文报告：SP 的训练稳定性略优于标准 TP

工程注意：
- SP 要求"序列维可切分"——因果注意力中的序列切分没问题（注意力分数矩阵本身在 TP 内是完整计算的）
- 与 FlashAttention 兼容：每个 TP rank 的注意力核是独立的
- 实现：Megatron-LM（column/row parallel + sequence parallel 标志）——
  主流框架（NVIDIA Megatron、DeepSpeed、vLLM 推理侧）均已内置
```

### 1.4 Ring Attention（序列并行的注意力版本）

**Ring Attention**（Liu et al., 2023）——把序列维直接切到多卡——做"块级注意力"：

```
核心思想：
- 序列切成 P 块（P 台设备）
- 每个设备存一块 Q 块和对应的 KV 块
- 通过 ring 通信（环形传递 KV）——每块 Q 与所有 KV 块轮流计算
- 用"online softmax"（flash attention 的技巧）——分块计算 softmax 统计量
  不需要存储完整注意力矩阵

通信模式：点对点环形传递（P 次传输——每次传 KV 块）
显存：每卡只需 O(seq/P × hidden)——序列可随设备数线性扩展
```

```
Ring Attention 的流水细节：
- 初始化：每卡有 Q_i、K_i、V_i 块
- 循环 P 轮：
  每轮计算 Q_i 对当前 K_j、V_j 的注意力（online softmax 累积）
  然后把自己的 KV 传给下一卡，接收上一卡的 KV
- 总通信量：O(P × seq × hidden)——与序列长度线性——适合超长序列

与 FlashAttention 的关系：Ring = Flash 的在线 softmax + 分布式轮转
2024-2025 发展：Ring Flash Attention（把 flash 融合 + ring 传输——速度翻倍）
```

---

## 二、上下文并行（Context Parallelism）

### 2.1 CP 与 SP 的关系

```
术语澄清：
- 序列并行（SP）：狭义指 Megatron 的"非张量操作序列切分"——通常与 TP 绑定
- 上下文并行（CP）：广义指"把整个注意力计算的序列维切分到多卡"——
  包括 Ring Attention、DeepSpeed-Ulysses（2023）、以及 Megatron 的 CP 扩展

2018 前的 CP 是"后补"维度——因为长序列模型（128K-1M context）才需要它
```

### 2.2 DeepSpeed-Ulysses（all-to-all 风格 CP）

**Ulysses**（Jacobs et al., 2023）：
```
机制：
- 输入序列切 P 块——每卡持有 1/P 的完整序列？——不是：
- Ulysses：每卡持有一块序列的全部隐藏维度 → all-to-all 交换 →
  变成"每卡持有全部序列的 1/P 隐藏维度"
- 然后每卡只算自己那 1/P 个注意力头（头维切分）→ 再 all-to-all 换回

通信：每层两次 all-to-all（序列 → 头、头 → 序列）
显存：注意力分数矩阵 O(seq²/P)——序列越长节省越多

Ring vs Ulysses 对比：
- Ring：通信点对点、均匀——适合 1M+ 超长序列
- Ulysses：all-to-all、通信量 O(seq × hidden × P)——适合中等长序列（128K 级）
- 混合（Hybrid：Ring 内 + Ulysses 间）——2024 后大厂标准方案
```

### 2.3 因果掩码与 CP 的交互

```
因果注意力 + CP 的挑战：
- Ring：每个 Q 块要与"所有更晚序列的 KV"计算——但因果掩码要求只关注前文
- 实际实现：仍然轮转全部 KV——但 attention 核内部跳过未来块（mask 生效）
  代价：未来块的传输是"浪费的"——但换来了无需全量 KV 的显存
- 优化（2024-2025）：因果 ring 只传递"未消费"的 KV——传输减半
- 局部注意力模型（如 DeepSeek-V3 的 MLA 稀疏注意）：KV 传输本身小——
  长序列训练成本大幅下降
```

### 2.4 CP 与 TP/PP 的组合

```
组合实践（4D 并行：TP × PP × CP × DP）：
- 分块优先级：TP 组内先做张量并行（通信密集、需要 NVLink 高速互连）
- CP 通常跨机架（或跨节点）——用 InfiniBand/RoCE
- PP 负责层间——流水线气泡需要微批填充
- DP/FSDP 负责参数分片与梯度同步

通信组划分（NCCL process group 设计）：
- TP group：节点内 8 卡（NVLink）
- CP group：同 TP 扩展组或独立（Ulysses 用 all-to-all——需要全互联）
- PP group：跨节点的层链
- DP group：其余轴

2025 年的行业实践：
- 万亿 MoE 长序列模型：TP×PP×CP×EP×DP 五维并行（DeepSeek-V3 的 Dense 部分 + MLA 长上下文）
- 关键原则：让"通信量最大的维度"在"带宽最高的硬件维度"上——TP 进节点内
```

---

## 三、Expert Choice 路由（MoE 负载均衡的革命）

### 3.1 Token Choice vs Expert Choice

```
经典 MoE 路由（Token Choice）：
- 每个 token 选 top-k 专家（k=1 或 2）
- 问题：热门 token 涌向少数专家 → 负载不均衡 → 丢 token（dropped tokens）
- 经典缓解：辅助负载均衡损失（aux loss，Shazeer 2017）：
  L_aux = α × Σ_experts (fraction_tokens_e × fraction_routed_e)
  （鼓励均匀路由——但牺牲路由质量）

Expert Choice（EP, Zhou et al., 2022）：
- 每个专家选 top-k' 个 token（倒过来！）
- 保证：每个专家恰好处理 k' 个 token——零负载失衡、零丢 token
- 代价：一个 batch 里的所有 token 必须都被某个专家"选走"——
  k' 的设置要保证覆盖（capacity = tokens/experts × k'）
```

```
两种路由的对比：
维度            Token Choice          Expert Choice
负载均衡        需要 aux loss         天然均衡（每个专家固定负载）
丢 token        capacity 不足会丢     不会丢（只要 capacity 覆盖）
路由自由度      每个 token 独立选     专家抢 token——选择权在专家侧
批量大小        不敏感                需要足够大的 batch 才高效
实现复杂度      简单                  需要全对全的 token 交换（all-to-all）
```

### 3.2 Expert Choice 的通信实现

```
Expert Choice 的训练流程（每层）：
1. 每个 token 计算与所有专家的打分（token × E 的 logits——E = 专家数）
   ——打分的计算：token 的 hidden 与每个专家的 router 权重点积
2. 取每列（专家）的 top-k' token → 形成"token → 专家"的分配
3. all-to-all 通信：把 token 送到其被分配的专家所在的设备
4. 每个专家独立前向（矩阵乘）
5. 反向的 all-to-all：把 token 的梯度送回原设备
6. 残差 + 输出

注意：打分需要"所有 token × 所有专家"——即 router logits 矩阵要全局化：
- 小模型：直接在 DP 内 gather 所有 token 的 hidden → 本地算全矩阵（token 数小可接受）
- 大模型：跨设备 gather——通信量 O(batch × hidden)——比 token 交换小一个量级
```

### 3.3 与辅助损失的对比实验

```
实验事实（Zhou et al., 2022 原文）：
- Expert Choice 在 GLUE/SuperGLUE/翻译任务上——无需 aux loss——
  达到或超过 Token Choice + aux loss 的精度
- 负载：Token Choice 的 expert 利用率方差大（0-100%），
  Expert Choice 恒为满负载（专家利用率 100%）
- 训练吞吐：由于不丢 token——梯度噪声更小——收敛更稳

后续（2023-2025）的行业混合方案：
- Switch-GEMM/专家并行的大规模部署——普遍采用：
  "Token Choice + 更好的容量规划 + 高容量系数"或"Expert Choice + 局部 aux 微调"
- 混合路由（Hybrid Routing）：大部分 token 走 token-choice，热点专家用 EC 兜底
- DeepSeek 系列：无 aux loss 的辅助——用"动态偏置"（dynamic bias）替代——
  router 对少数低热度专家加偏置——保持均衡的同时不污染路由分数
```

### 3.4 稀疏注意力中的类似思想

Expert Choice 的核心——"选择权换均衡"——已被推广到注意力/缓存管理：

```
- MoBA（混合块注意力，DeepSeek 2025）：让块"主动招揽"token（类似 EC 反向选择）
- 稀疏 KV 缓存的选择（2025 的 KV 压缩路线）——用"信息量排序选 top-k"——
  本质与 EC 相同：保证每个缓存块负载均衡
- 预测路由（Expert Choice + 预计算）：router 输出提前一轮计算——
  与计算重叠——隐藏通信延迟（2024 的 DeepSpeed 方案）
```

---

## 四、MoE 全栈并行（EP + TP + PP + DP）

### 4.1 专家并行（EP）的基本布局

```
MoE 层的最佳并行 = 专家并行（Expert Parallelism）：
- E 个专家分布在 E/PE 台设备上（PE = 每台设备上的专家数）
- token 通过 all-to-all 路由到目标设备
- 每台设备只计算自己负责的专家的 FFN——专家权重不重复存储
  （对比：TP 会复制专家权重——TP 对 MoE 的扩展性差）

关键数字：
- 专家数 E = 128-256（常见）；每设备专家数 PE = 1-2（大模型）
- 通信：每层两次 all-to-all（去 + 回）——通信量 ≈ token 数 × hidden
- 反直觉：EP 中 TP 只在 attention/dense 部分做（或不做）——
  MoE 层与 dense 层的并行方式不同——需要"分层并行"设计
```

### 4.2 MoE 全栈配置（5 维并行实例）

```
实际配置示例（万亿参数级 MoE，2025 行业做法）：
集群假设：64 节点 × 8 卡 = 512 卡（NVLink 节点内，IB 节点间）
模型：总参数 2T（active 200B，E=256，每 expert 约 7.8B）

并行分组：
- DP = 4（数据并行复制）
- TP = 8（节点内 8 卡——dense 部分张量并行）
- PP = 2（流水线——层间切两段）
- EP = 64（专家分布在 64 组设备上——每组 = 一个 TP 组）
- CP（可选，长序列时）= 4

组拓扑：
- EP 组的成员 = 同一个 TP 组内的卡（专家在同一节点内 → all-to-all 走 NVLink）
- 关键设计：all-to-all 的通信域 < 节点规模 → 专家交换不出节点

每层执行顺序（MoE 层）：
input → router（local）→ all-to-all（token→专家所在设备）→ EP 内专家 FFN → all-to-all（回）→ residual
```

### 4.3 All-to-all 的优化（MoE 的性能命门）

```
all-to-all 是 MoE 训练的最大瓶颈（占 MoE 层时间 20-50%）——优化手段：

1. 通信与计算重叠：
   - token 按目标设备分桶——先发"第一个桶"——计算其余桶时等回包
   - 多流水阶段：发送（bucket 0）→ 计算（bucket 1）→ 接收（bucket 2）循环

2. 减少发送量：
   - 稀疏门控 + 混合精度传输（BF16/FP8 传输 token）
   - token 组合（packing）：多个小 token 打包成一个大的通信块——减少消息数
   - KV/激活压缩传输（2025 的探索：低秩近似传激活）

3. 拓扑感知路由：
   - 先发"同节点"的 token（NVLink 快）——再发跨节点（IB）
   - 分层 all-to-all：节点内 NVLink 交换 + 节点间 IB 交换（DeepSpeed-MoE 的层次化方案）

4. 减少 all-to-all 次数：
   - 把"去程"与"回程"融合：Expert Choice 风格——整层一次交换
   - 两层 MoE 的通信融合（相邻 MoE 层的 all-to-all 合并）

5. FP8 推理/训练（2025 标配）：
   - token 以 FP8 传输——带宽减半——精度损失用"per-token scaling"补偿
```

### 4.4 Dropless MoE 与容量管理

```
dropless（零丢 token）的工程条件：
- capacity factor（容量系数）：每个专家的 token 槽位 = tokens_per_device × c
  c ≥ 1：c=1 时恰好均衡（需 EC）；c=1.25 有 25% 冗余（Token Choice 常见）
- 显存成本：c 越高 → 专家激活缓存越大（c 增大 → 激活显存线性增）
- 工程平衡：c=1.0~1.1 + aux loss 辅助（推荐）或 EC（零冗余但通信复杂度高）

Tutel 的贡献（微软, 2022）：
- 把 capacity 计算 + 分桶 + all-to-all 做成"可微、可重叠"的流水
- 动态容量：按 batch 实际负载自动调节 c（防抖）
- 实验：多框架迁移后提速 1.5-2.6×
```

---

## 五、超级长序列训练（百万 token 上下文）

### 5.1 长序列的成本结构

```
序列长度 S 的关键成本（注意：都是超线性！）：
- 标准注意力：O(S²) 时间、O(S²) 显存（分数矩阵）
- KV 缓存：O(S × hidden) 显存（线性——但 S=1M 时依然很大）
- 显存主导向（推理）：KV 缓存占主导 → MLA（Multi-head Latent Attention）压缩 KV
- 计算主导向（训练）：attention 的 O(S²) 是主要——S=1M 时无法朴素计算

超长序列训练 = 同时解决：
1. 注意力复杂度（FlashAttention/Ring/稀疏化）
2. 序列切分并行（CP——见第二章）
3. KV 显存（MLA/分组 KV）
4. 数据管线（长文本预处理、packing 与填充）
```

### 5.2 FlashAttention 系（长序列的基石）

```
FlashAttention（2022）：
- 用 online softmax（滚动统计）：分块加载 Q/K/V 到 SRAM——逐块计算——
  不落盘完整注意力矩阵——显存 O(S)（原来 O(S²)）
- 精确算法（不是近似！）——数学上与标准 softmax 完全等价

FlashAttention-2（2023）：减少非矩阵乘开销——GPU 利用率 50-70%
FlashAttention-3（2024）：FP8 + 异步流水——Hopper 上再提速 1.5-2×

FlashMLA（DeepSeek, 2025）：
- 面向 MLA（低秩 KV）的闪存高效注意力内核
- 把 MLA 的低秩分解利用到极致：缓存的是低秩投影而非完整 KV
- 长上下文推理：KV 显存减少 5-10 倍

paged attention 系（vLLM 推理侧）：
- KV 缓存按页分配——消除碎片——长序列推理的显存管理革命
```

### 5.3 稀疏注意力与长序列架构

```
稀疏注意力家族（2024-2026 的主流长上下文方案）：
1. 局部 + 全局锚点（Longformer/BigBird）：
   滑动窗口 + 稀疏全局 token——O(S√S) 或 O(S)
2. 分层注意力（LongNet 2023/2024）：
   多尺度——窗口内稠密 + 远程稀疏
3. 聚合注意力（DeepSeek-V3 的 DSA——Dense Sparse Attention）：
   - 前几层：稠密注意力（全 attention——保证信息流）
   - 后几层：稀疏（窗口 + 每 N token 聚合的全局锚）
   - 推理时动态选择稀疏度（DSA 的推理侧按负载调节）
4. MoBA（2025）：块级选择——只让相关块交互（见第三章）
5. 线性注意力（RWKV-7/Mamba2 + 注意力混合）：
   纯线性注意力没有"精准检索"——混合架构（如 DeepSeek-V3 的 MLA+DSA）更实用

工程铁律：稀疏注意力的"训练效率"与"检索能力"是 trade-off——
选择时先看任务（代码/检索需要强注意力；长文档摘要可以稀疏）
```

### 5.4 长序列训练的数据管线

```
长文本预处理（S 从 32K 到 1M+）：
1. 长文档切分策略：
   - 按语义边界切（章节/段落）——避免从句子中间开始
   - 重叠窗口（overlap = 512-1024 token）——跨窗口的信息连续性
   - 超长文档（书/代码库）：层级打包——相关文档相邻
2. Packing（打包训练）：
   - 变长样本打包成固定长度序列——用 <eos> 或 attention mask 隔离
   - 与 CP 的交互：打包序列要按"文档边界"对齐到各设备的序列分块——
     避免一个块内出现两个文档的混切（2024-2025 的细粒度实现）
3. 位置编码的缩放：
   - 长序列训练时——RoPE 的外推（NTK/YaRN 插值）
   - YaRN（2023）：NTK 插值 + 窗口调整——128K 训练可以外推到 1M
4. 序列长度课程：
   - 先短后长（如 4K → 32K → 128K 渐进）——训练稳定性与吞吐的双重收益
   - 长序列阶段的 loss 会先涨后降（long-context 不稳定性）——需要小的 LR 或重新预热
```

### 5.5 推理侧的长上下文（与训练配套）

```
推理的长序列 = 显存工程：
- KV 缓存压缩：MLA（DeepSeek-V3——KV 缩 93%）/ GQA
- 分页 + 预填充/解码分离（2024-2025 的架构标配）
- 投机解码（speculative decoding）：草稿模型先写——大模型验证——长上下文也能加速
- 1M context 的推理成本：依然昂贵——2026 的热点——
  "无限上下文"（检索式 KV 缓存、分层内存外存）
```

---

## 六、分布式 Checkpoint 与容错

### 6.1 Checkpoint 的规模危机

```
万亿参数模型的一个 checkpoint：
- 参数（BF16）：2TB
- 优化器状态（Adam：fp32 主权重 + 一阶 + 二阶矩）：8-12TB
- 保存时间（NVMe 写 5-10GB/s）：20-40 分钟
- 存储成本：TB 级 × 多版本（保留历史 checkpoint）→ 数 TB-数十 TB

问题：
1. 保存期间训练暂停（stall）——浪费 GPU 小时
2. 异步保存（后台写）→ 一致性风险（权重在写过程中被更新）
3. 失败恢复：找"最近的可用 checkpoint"——可能丢失数小时进度
```

### 6.2 分片 Checkpoint（Sharded Checkpoint）

```
核心：优化器状态与模型状态——已经在并行维度分片——
       checkpoint 也应分片保存（每张卡只写自己那部分）——不做 gather

实现要点：
- FSDP/ZERO-3/EP 场景：每 rank 保存自己的分片 + 元数据文件（global metadata）
- 加载：按 rank 加载对应分片——无需全量 gather——恢复时间缩短数倍
- 跨并行配置的迁移：分片是"按逻辑 tensor 位置"存——换 TP/PP 配置也能重组
  （需要分片索引与全局 shape 记录——Pytorch 的 DistributedTensor 索引）

工程细节：
- 元数据文件必须原子写（先写临时文件再 rename）——防半写
- 分片文件按 rank 命名（rank_0000.ckpt）——快速定位损坏分片
- 多副本：双写不同存储路径（容灾）
- 训练/保存并行：保存到本地 NVMe——后台进程再搬到对象存储（两阶段）
```

### 6.3 异步 Checkpoint（Async Checkpoint）

```
异步保存的机制：
1. 训练线程：保持权重更新
2. 保存线程/进程：读取"冻结快照"写入磁盘
   关键：必须在"该步权重完整"的边界取快照——
   方案 A：D2D（device-to-device）拷贝到显存缓冲——再异步写盘
   方案 B：CUDA 事件的同步——确保保存线程看到的权重一致
3. 同步点（sync barrier）：每 N 步等保存完成——防止写盘积压

风险与缓解：
- 保存慢 → 积压 → 训练内存耗尽 → 定期强制同步
- 权重更新与快照竞争 → 使用"双缓冲"（下一份快照的拷贝与当前写盘并行）
- NVMe 带宽抢占 → 保存限速（IO 调度）或保存在计算空闲的窗口（梯度通信时）

实践数据（2024-2025 大厂论文）：
- Async checkpoint 使 checkpoint 开销从"分钟级 stall"→ "近零 stall"
- 万亿参数模型：异步保存 10-20 分钟（写盘）——训练不中断
```

### 6.4 容错与弹性训练（Fault Tolerance）

```
故障统计（2024 年 72B-2T 训练的报告数据）：
- 单卡故障率：数月训练中——总会发生（GPU 故障/掉线）
- 网络抖动/IB 错误：常见
- 结果：不处理故障 → 训练频繁中断 → 实际吞吐损失 20-50%

容错技术栈：
1. 心跳 + 超时检测：通信库（NCCL）断连 → 触发 fault handler
2. 异步故障检测（NCCL fault tolerance，2024-2025）：
   - 检测链路：rank 报告 + 通信库确认
   - 处理：冻结训练 → 拉取最近 checkpoint → 替换故障节点 → 恢复
3. 层级恢复策略：
   - 节点内掉卡：换空卡（同节点热备）
   - 节点掉电：整节点替换 + 从 checkpoint 恢复
4. 弹性训练（elastic training, TorchElastic）：
   - 训练不绑定固定 GPU 数——节点加入/离开自动 re-shard
   - 与 checkpoint 配合：成员变化 → 保存 + 重分配 + 恢复
5. 检查点验证：
   - 加载后校验（checksum/形状检查）——防损坏 checkpoint 导致静默错误
   - "备份 checkpoint"（每 N 步写两个副本）

容错的训练效率数学：
- 目标：MTBF（平均故障间隔）>> 恢复时间 + checkpoint 间隔
- 自动调参：故障率 f、checkpoint 间隔 τ、保存时间 s——
  τ* = sqrt(2s/f)（最小化期望浪费——工程近似公式）
```

### 6.5 CUDA Graphs 与训练稳定性（长训练的可复现）

```
CUDA Graphs（2023-2025 长训练标配）：
- 捕获整个训练 step 的内核序列 → 重放（消除 kernel launch 开销）
- 收益：step 时间抖动降低——对异步 checkpoint/容错更友好
- 与 MoE/动态路由的冲突：graph 要求固定形状——
  变长 all-to-all 需要"多图切换"（按 token 负载选图）或动态形状内核

训练稳定性工程（万亿级训练的隐性要求）：
- BF16/FP8 混合精度：loss 缩放 + 梯度裁剪（clip norm 1.0）
- 损失尖峰检测：NaN/Inf 自动检测 → 回退到上一 checkpoint + 降低 LR
- 周期性的梯度范数/激活统计监控（wandb 级监控）
- 数据顺序固定（shuffle seed）→ 可复现性 → 故障复现友好
```

### 6.6 训练到推理的 Checkpoint 转化

```
推理部署的 checkpoint 需求与训练不同：
- 训练 checkpoint：分片 + 优化器状态（大而全）
- 推理权重：需"合并分片"（TP/EP 合并或重排成目标并行配置）
- 转换管线：
  训练分片 → 按模型结构重组 → 验证加载（跑 20 个代表性样本对比 logits）→ 发布
- 长上下文模型的额外检查：RoPE 配置、KV 缓存格式、MLA 低秩投影的完整性
- 量化版本（FP8/INT4）：需要校准集 + 端到端评估（不能只看困惑度）
```

---

## 七、总结

```
分布式训练的技术脉络（一句话版）：
- 显存不够 → TP/FSDP 分参数；序列太长 → SP/CP 分序列；专家太多 → EP 分专家
- 通信是瓶颈 → all-to-all 优化、重叠、拓扑感知
- 长序列是趋势 → Flash/MLA/Ring/稀疏注意力——训练与推理同构设计
- 训练要连续 → 分片 + 异步 checkpoint + 容错 + 弹性

工程决策速查：
- 模型 < 70B dense → FSDP/ZERO-3 足够；更大 → 加 TP
- 序列 > 32K → 加 CP（Ulysses 到 128K；Ring 到 1M+）
- MoE 模型 → EP 必须；专家交换大 → 优化 all-to-all + 低精度传输
- 训练 > 数天 → checkpoint 必须分片 + 异步 + 容错演练
- 集群异构 → 弹性训练 + 拓扑感知的并行组划分
```

**核心命题**：分布式训练的终点不是"把模型塞进显存"——而是**在固定集群上——以最高 MFU（模型浮点利用率）、最低故障损失——把模型训练到目标规模**。并行维度是手段——吞吐与可靠性才是目的。
