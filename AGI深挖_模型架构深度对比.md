# AGI 深挖——模型架构深度对比

> 2026年7月更新。
> 六大主流模型逐一拆解：GPT-5.6 / Claude Fable 5 / Gemini 2.5 / DeepSeek V4 / Grok 4.7 / Qwen3-Next

---

## 核心观察：殊途同归

2026年的六大模型虽然参数规模、训练成本和推理速度差异很大，但在架构设计上有惊人的**趋同**趋势，也有少数**关键分歧**：

| 设计维度 | 趋同方向 | 主要分歧 |
|----------|----------|----------|
| 主干架构 | 几乎所有模型采用混合架构 | Mamba-Transformer vs 纯Transformer |
| MoE | 全部采用MoE或MoE变体 | Expert粒度、路由策略、共享Expert |
| 注意力 | MLA（Multi-Head Latent Attention）成为主流 | 窗口大小、线性注意力比例 |
| 训练数据 | 20T-80T token级别 | 合成数据比例、代码与推理数据配比 |
| 上下文长度 | 1M-10M token | 位置编码选择（RoPE vs ALiBi vs 复杂RoPE变体） |

---

## 1. GPT-5.6 Family (OpenAI)

### 概述
2026年7月9日发布。不是单一模型，是覆盖云端到端侧的三档产品线。

### 家族成员

| 型号 | 参数量 | 定位 | 相对GPT-5.4加速 |
|------|--------|------|-----------------|
| GPT-5.6 Sol | ~15T（激活~800B） | 旗舰推理 | 2.7× |
| GPT-5.6 Terra | ~3T（激活~200B） | 日常通用 | 3.1× |
| GPT-5.6 Luna | ~300B（激活~30B） | 端侧/快速 | 4.5× |

### 架构核心特征

**MLA Attention（Multi-head Latent Attention）**
- OpenAI 2026年自研的低秩注意力机制
- 将KV Cache压缩4-8倍（对长上下文推理至关重要）
- 与DeepSeek V2/V3的MLA类似但实现细节不同
- 论文揭示其本质是"双低秩分解"（dual low-rank factorization），非对称降维

**MoE配置**
- Sol: 128 experts, top-6路由 + 2个共享experts
- Terra: 64 experts, top-4路由 + 1共享expert
- Luna: 32 experts, top-2路由，无共享expert
- 路由策略：Expert Choice routing（与Google旗舰一致）
- 负载均衡：auxiliary loss + 显式容量因子调度

**激活函数**
- SwiGLU全栈统一

**位置编码**
- 改进型RoPE（YaRN + NTK-aware scaling）
- 支持扩展到2M上下文
- 同时引入了日志间距改进（log-spaced scaling factor）

**训练数据**
- 约70T tokens（其中40%合成数据）
- 合成数据策略：self-play蒸馏 + 拒绝采样（rejection sampling）
- 代码占比：15%（Python + 主流语言）

**训练成本**
- 约$5-8亿（整个家族，Sol占60%）
- Sol在200K H100等效计算上训练约90天
- 2026年使用25.6K B200 GPU集群

### 独特设计
- **Output QC (Quality Control) Token**：在生成末尾添加一个特殊token，由额外分类器判断输出质量，低分自动触发重生成
- **分层KV Cache**：长上下文场景下，将上下文分为"近端密集+远端稀疏"两层缓存
- **模型联动**：Sol/Terra/Luna共享底层表示，支持快速蒸馏

---

## 2. Claude Fable 5 / Sonnet 5 / Opus 4.8 (Anthropic)

### 概述
Anthropic在2026年6月发布Claude Fable 5（中档）和Sonnet 5 Fennec（快速），Opus 4.8作为超旗舰。Fable 5整体超越Opus 4.0。

### 架构核心特征

**混合架构（Hybrid Mamba-Transformer）**
- 这是Anthropic 2026年最大的架构跃迁
- 前20%层：纯Mamba-2（状态空间模型，线性复杂度）
- 中间60%层：Mamba + 稀疏注意力（内存压缩窗口注意力）
- 后20%层：全注意力（用于需要精确回忆的token）
- 整体推理：1M上下文，注意力复杂度降到约O(n^1.3)

**为什么Anthropic敢用混合架构？**
- 他们2024-2025年在SSM领域积累了大量内部研究（未发表的Mamba改进）
- 与Hazy Research（Mamba原作者团队）的深度合作
- 通过对比实验证明：混合架构在**前20层几乎不需要注意力**（这些层学习的是局部语法和浅层语义）

**MoE配置**
- Fable 5：72 experts, top-6 + 2共享
- Sonnet 5：48 experts, top-4 + 1共享
- Opus 4.8：144 experts, top-8 + 4共享
- 路由：基于Q-Former的语义路由（不同于传统MLP路由）
  - 用一个小型transformer编码器（Q-Former）分析token语义
  - 将token路由到语义相关的expert组
  - 论文显示：语义路由比MLP路由提升3-5%的expert利用率

**激活函数**
- 自定义GELU变体（Anthropic未公开发表，推测是GELU + 小参数门控）

**位置编码**
- ALiBi（Attention with Linear Biases）的改进版本
- 结合了shadow attention机制（远端token用粗粒度attention）

### 训练数据
- 约45T tokens（合成比例20%）
- Anthropic强调：合成数据质量优于数量，每个合成样本经过critic model验证
- 公开已知的数据来源：Common Crawl精选 + GitHub + 学术论文 + 教材
- 特别强调**数学和推理数据**的配比（约20%）

### 训练特点
- 训练过程大规模使用宪法式数据筛选（Constitutional Data Filtering）
- 每次训练checkpoint都经过红队测试
- 训练中期插入"alignments"阶段（非全量微调，是细粒度参数调整）

### 独特设计
- **Constitutional Attention Heads**：部分注意力头被约束为"不可关闭"，确保总是关注安全相关token
- **反思Token**：在长时间推理中定期插入反思token，让模型自我纠正
- **分步推理令牌（Step-wise Reasoning Tokens）**：显式的推理链token，训练时强制要求

---

## 3. Gemini 2.5 Pro (Google DeepMind)

### 概述
2026年初发布的旗舰模型，强项是多模态和超长上下文。3.x已在内测，但2.5仍是主力。

### 架构核心特征

**纯Transformer + 深度MoE**
- Google坚持纯Transformer路线（不采用Mamba混合）
- 但内部有严重的Mamba实验——未来版本可能混合
- Decoder-only架构，与Gemini 1.5一致

**MoE配置**
- 2.5 Pro：~8T参数（激活~600B）
- expert数量未公开（估计128-256 experts）
- 路由策略：Expert Choice routing (Google原创)
- 关键创新：**Depth-wise MoE**（不同深度层使用不同数量的expert）
  - 浅层（1-20层）：2-4 experts
  - 中层（21-60层）：8-16 experts  
  - 深层（61-80层）：4-8 experts
  - 计算预算根据层深动态分配

**多模态处理**
- 原生多模态（从头训练，非拼接）
- 视觉编码器：改进型ViT-22B（22B视觉参数）
- 音频编码：直接token化（不做频谱图中间表示）
- 视频处理：每秒≈4 token的超高效视频token化
- 所有模态共享相同的decoder层

**位置编码**
- 增强型RoPE
- 支持10M token上下文（Gemini 2.5 Pro的长上下文SOTA）
- 使用**分块位置编码（Chunked RoPE）**：将上下文分成128K的块，块内标准RoPE，块间线性插值

**训练数据**
- 约80T tokens（全模态）
- 合成数据占比：35%（多模态合成数据技术是Google核心优势）
- YouTube数据：大量使用自家视频数据训练多模态能力

### 独特设计
- **Gemini Cache**：行业领先的KV Cache优化（支持256K token free cache）
- **多模态指针语言（Multi-modal Pointer Language）**：模型在生成文本时可以直接"指向"图像/视频中的区域
- **主动推理（Proactive Reasoning）**：模型可以主动提出需要更多信息的问题

---

## 4. DeepSeek V4 Pro / Flash (深度求索)

### 概述
2026年7月10日发布的V4系列，延续中国AI的性价比路线。Pro对标GPT-5.6 Sol的70-80%性能，但API价格只有1/10。

### 架构核心特征

**MLA Attention**（DeepSeek原创）
- Multi-query Latent Attention
- 本质：对KV进行低秩压缩后再做attention
- 效果：KV Cache减少75%，长上下文推理效率极高
- 在V4中升级为**非对称MLA**（Q和KV使用不同的压缩比率）

**MoE配置**
- V4 Pro：256 experts, top-8 + 4共享
- V4 Flash：64 experts, top-4 + 2共享  
- 使用**轻量级gate**（2层MLP，参数量仅为标准gate的10%）
- 创新的**auxiliary-loss-free负载均衡**：通过正则化项的软约束，避免引入额外损失
- 共享experts捕获全局知识，路由experts负责领域特化

**稀疏注意力（Sparse Attention）**
- 在V4中引入**分层稀疏注意力**：
  - 局部窗口：4K tokens（密集注意力）
  - 全局压缩：将远端上下文压缩到约1/4的表示
  - 中间层使用随机注意力（Random Attention模式）
- 整体复杂度：O(n × sqrt(n))，比标准O(n²)大幅降低

**激活函数**
- SwiGLU（与主流一致）
- 但V4在FFN中增加了一个**门控缩放因子**（gating scale）

**位置编码**
- 原创的**Tie-Embedding RoPE**（绑定词嵌入与位置编码）
- 节省大量参数，同时提升长上下文一致性

### 训练特点
- 训练规模：约40T tokens
- **Multi-Token Prediction (MTP)**：DeepSeek在V3中引入了MTP
  - V4将MTP从2-token扩展到4-token预测
  - 推理时可将MTP head用于speculative decoding（加速1.8-2.3×）
- 训练成本：V4 Pro约$6000万（不到GPT-5.6 Sol的1/10）
  - FP8混合精度训练
  - 自定义HAI-LLM框架（非DeepSpeed/Megatron）

### 团队规模与效率
- V4核心团队约150人
- 从V3到V4的迭代周期：约14个月
- 人均效率惊人（15亿美元预算的OpenAI团队vs 6000万预算的DeepSeek团队）

### 独特设计
- **Token-level RL**：训练中引入token级别的奖励信号（不仅仅是response级别）
- **在线蒸馏**：在训练过程中持续从更大模型蒸馏知识给当前模型
- **Paged Attention 2.0**：优化的显存管理，支持单卡部署V4 Flash（量化后）

---

## 5. Grok 4.7 (xAI)

### 概述
xAI在2026年7月达到最活跃的发布节奏。Grok 4.7使用马斯克声称的"孟菲斯超级计算机集群"（200K H100等效）。风格独特——强调原始推理能力、低审查、高可定制性。

### 架构核心特征

**Not-Transformer（xAI称其不是标准Transformer）**
- 使用**Grok-1的修改版MoE架构**演进
- 核心创新：**自适应深度路由（Adaptive Depth Routing）**
  - 不是所有token都需要通过所有层
  - 每个token通过一个轻量级router决定是否"跳过"某些层
  - 简单token（如stop words）只需通过6-12层
  - 复杂token（如数学推理）需要全部64层
  - 平均深度：约22层（节省60%计算量）

**MoE配置**
- 4.7：~4T参数（激活~400B）
- 64 experts, top-6路由
- 共享experts：2个（作为安全网捕获通用知识）

**激活函数**
- xAI自研的**GroLU**（Grok Rectified Linear Unit）
  - 在ReLU基础上加入自适应阈值
  - 对负值不是完全截断，而是按比例衰减

**位置编码**
- 可学习位置编码（learned positional embeddings）
- 支持扩展到1M上下文

### 训练数据
- 约30T tokens
- **X/Twitter数据是大特色**：实时帖子流构成训练数据的10%
  - 这让Grok在"当前网络文化"和"热点事件"上反应最快
  - 但也带来噪声和重复问题（xAI通过复杂去重管道解决）
- 合成数据：20%（Math and reasoning focused）

### 训练特点
- 训练成本：约$3-4亿（使用xAI自建集群，内部成本更低）
- 训练流程：全栈自研（从集群调度到训练框架）
- **冷启动微调**：每个Grok版本发布后，通过用户反馈持续微调

### 独特设计
- **Fun Mode / Serious Mode**：可切换模型的人格参数
  - 本质是LoRA模块的插拔
  - Fun Mode：更宽松的过滤器、更幽默的回复
  - Serious Mode：更严谨、更短的回复
- **实时知识注入**：通过X数据流持续更新模型知识（非retraining，是类似激活缓存的机制）
- **自驱动迭代**：Grok 4.7的训练过程中持续使用自我博弈（self-play）生成训练数据

---

## 6. Qwen3-Next-80B-A3B (阿里)

### 概述
阿里2026年7月10日发布的旗舰。延续Qwen系列对端侧和性价比的极致追求。最惊艳的是80B模型中只有3B活跃参数，却能在多个基准上与500B+模型竞争。

### 架构核心特征

**极致MoE + 共享参数**
- 80B总参数，仅3B激活参数
- 这是目前公开模型中最高效的激活/总参数比（仅3.75%）
- 配置：约48 experts，但每个expert非常小
- 创新点：**共享路由器 + 预激活参数池**

**DeepSeek的MLA + 自研优化**
- 借鉴了DeepSeek的MLA方案
- 在此基础上增加了**自适应KV压缩**（根据内容动态调整压缩率）
- 对比DeepSeek：MLA压缩更激进，但长上下文质量略低

**激活函数**
- SwiGLU

**位置编码**
- NTK-aware RoPE + 动态扩展
- 支持256K上下文

### 训练数据
- 约25T tokens
- 多语言比例：中文30%、英文50%、其他20%
- 强调代码与数学推理数据（25%占比）

### 训练特点
- 训练成本：约$2000-3000万
- 使用阿里自建集群（H200 + 部分定制芯片）

### 独特设计
- **Pre-activation Parameter Pool**：模型维护一个共享的"预激活参数池"，所有expert可以从中共享通用的低层表示
- **极低激活**：3B激活参数的实现，使得在消费级GPU（RTX 4090/5090）上可以运行旗舰模型
- **MoE-QLoRA微调**：提供高效的微调方案，只需微调激活参数即可适配下游任务

---

## 七大架构维度横评

### 1）注意力机制

| 模型 | 注意力类型 | KV Cache效率 | 长上下文质量 |
|------|-----------|-------------|-------------|
| GPT-5.6 Sol | MLA (Latent Attention) | ★★★★★ 压缩4-8× | ★★★★☆ 2M OK |
| Claude Fable 5 | 混合Mamba+Attention | ★★★★★★ 亚线性 | ★★★★★ 1M优秀 |
| Gemini 2.5 Pro | 全Attention+分块RoPE | ★★★★☆ 优化好 | ★★★★★★★ 10M SOTA |
| DeepSeek V4 Pro | MLA + 稀疏Attention | ★★★★★ 压缩4× | ★★★★☆ 1M OK |
| Grok 4.7 | 标准Multi-Head Attention | ★★★☆☆  | ★★★☆☆ 1M |
| Qwen3-Next | MLA + 自适应KV压缩 | ★★★★★ 压缩好 | ★★★☆☆ 256K |

### 2）MoE设计

| 模型 | Expert数 | 路由策略 | 共享Expert | 负载均衡 |
|------|---------|---------|-----------|---------|
| GPT-5.6 Sol | 128 | Expert Choice | 2 | auxiliary loss + 容量调度 |
| Claude Fable 5 | 72 | Q-Former语义路由 | 2 | 隐式语义均衡 |
| Gemini 2.5 Pro | 128-256(估计) | Expert Choice | 无 | 深度依赖的负载均衡 |
| DeepSeek V4 Pro | 256 | 轻量级gate | 4 | auxiliary-loss-free |
| Grok 4.7 | 64 | 自适应深度路由 | 2 | 深度路由隐式均衡 |
| Qwen3-Next | 48 | 共享路由器 | 预激活池 | 极低激活参数 |

### 3）训练效率

| 模型 | 训练成本(估计) | 数据量 | 训练时间(等效H100-days) |
|------|--------------|-------|----------------------|
| GPT-5.6 Sol | $5-6亿 | 70T | ~18M H100-days |
| Claude Fable 5 | $3-4亿 | 45T | ~12M H100-days |
| Gemini 2.5 Pro | $8-10亿(含多模态) | 80T(多模态) | ~25M H100-days |
| DeepSeek V4 Pro | $6000万 | 40T | ~2M H100-days |
| Grok 4.7 | $3-4亿(内部价) | 30T | ~10M H100-days |
| Qwen3-Next | $2000-3000万 | 25T | ~700K H100-days |

### 4）推理成本

| 模型 | 激活参数 | 推理成本(每M token) | 单卡部署 |
|------|---------|-------------------|---------|
| GPT-5.6 Sol | ~800B | $15-25 | 多卡 |
| GPT-5.6 Luna | ~30B | $0.50-1 | H200可行 |
| Claude Fable 5 | ~200B(估计) | $5-10 | 多卡 |
| Sonnet 5 | ~80B(估计) | $2-5 | 多卡 |
| Gemini 2.5 Pro | ~600B | $5-10(含多模态额外) | 多卡 |
| DeepSeek V4 Pro | ~300B | $0.50-1 | 多卡 |
| DeepSeek V4 Flash | ~80B | $0.10-0.20 | H200可行 |
| Grok 4.7 | ~400B | $3-8 | 多卡 |
| Qwen3-Next | **3B** | **$0.01-0.05** | **RTX 4090** |

### 5）上下文长度

| 模型 | 最大上下文 | 实际质量上限 | 位置编码 |
|------|----------|------------|---------|
| GPT-5.6 Sol | 2M | ~1M | 改进RoPE |
| Claude Fable 5 | 1M | ~500K | 改进ALiBi |
| Gemini 2.5 Pro | **10M** | **~2-5M** | 分块RoPE |
| DeepSeek V4 Pro | 1M | ~500K | Tie-Embedding RoPE |
| Grok 4.7 | 1M | ~200K | 可学习PE |
| Qwen3-Next | 256K | ~128K | NTK RoPE |

### 6）多模态能力

| 模型 | 文本 | 图像理解 | 图像生成 | 音频 | 视频 | 代码执行 |
|------|------|---------|---------|------|------|---------|
| GPT-5.6 Sol | ★★★★★ | ★★★★★ | ★★★★☆ | ★★★☆☆ | ★★★★☆ | ★★★★★ |
| Claude Fable 5 | ★★★★★ | ★★★★☆ | ★★☆☆☆ | ★★☆☆☆ | ★★☆☆☆ | ★★★★★ |
| Gemini 2.5 Pro | ★★★★★ | ★★★★★ | ★★★★★ | ★★★★★ | ★★★★★ | ★★★★☆ |
| DeepSeek V4 Pro | ★★★★☆ | ★★★★☆ | ★☆☆☆☆ | ★★☆☆☆ | ★★☆☆☆ | ★★★★☆ |
| Grok 4.7 | ★★★★☆ | ★★★☆☆ | ★★★☆☆ | ★☆☆☆☆ | ★★☆☆☆ | ★★★★☆ |
| Qwen3-Next | ★★★★☆ | ★★★★☆ | ★☆☆☆☆ | ★★☆☆☆ | ★★☆☆☆ | ★★★★☆ |

### 7）开源/开放程度

| 模型 | 权重开放 | 论文/技术报告 | API可用 | 本地部署 |
|------|---------|-------------|--------|---------|
| GPT-5.6 | ❌ 闭源 | 部分论文 | ✅ | ❌ |
| Claude Fable 5 | ❌ 闭源 | 少量披露 | ✅ | ❌ |
| Gemini 2.5 Pro | ❌ 闭源 | 部分论文 | ✅ | ❌ |
| DeepSeek V4 Pro | ✅ 开源 | 详细论文 | ✅ | ✅ |
| Grok 4.7 | ❌ 闭源 | 少量披露 | ✅ | ❌ |
| Qwen3-Next | ✅ 开源 | 详细论文 | ✅ | ✅ |

---

## 关键趋势分析

### 1）MoE从"有更好"到"不MoE不行"
2026年，MoE已经从"一种加速方案"变成"所有旗舰模型的标配"。没有MoE的模型根本竞争不过。

关键原因是：**训练数据增长**的速度远快于GPU内存增长的摩尔定律。MoE是唯一能用更多参数拟合更多数据而不爆炸计算量的方案。

### 2）MLA击败了标准注意力
DeepSeek的MLA方案已经被广泛采用。核心洞察：
- KV Cache是推理瓶颈（尤其是在服务场景）
- 低秩压缩几乎不损失质量（attention score矩阵本质上是低秩的）
- 压缩4-8倍KV Cache带来了经济性的质变

### 3）稀疏Attention成为标配
无论是分块注意力、滑动窗口+全局压缩还是随机注意力，所有模型都在做注意力稀疏化。2026年的共识是：**并非每个token对每个token都需要同等程度的关注**。

### 4）混合模型前沿：Anthropic vs Google
- Anthropic：已经在Fable 5中大规模使用Mamba（激进派）
- Google：2.5 Pro坚持纯Attention，但3.x预计会引入混合（保守派）
- 两者的路线选择决定了2026年推理效率的竞争格局

### 5）合成数据的幻觉
2026年的行业共识：
- 合成数据是**加速器**不是**替代品**（不能完全替代真实数据）
- 合成数据质量>数量（经过验证的合成数据比大量未过滤合成数据好）
- 每个样本需要经过critic/reward model验证
- **合成数据循环**（模型生成→reward过滤→重新训练）是2026年标配

### 6）成本阶跃下降
- 从GPT-4（2023，约$1-2B训练成本）到DeepSeek V4 Pro（2026，约$60M）：**成本下降30倍**的同时性能提升
- 从API推理成本看：2023年GPT-4 $30/M tokens → 2026年Qwen3-Next $0.03/M tokens：**成本下降1000倍**
- 意味着AI能力的"普及化"趋势不可逆转

### 7）开源与闭源的赛局
2026年的格局：
- 开源阵营（DeepSeek、Qwen、Bonsai、Mistral、GLM）不断缩小差距
- 闭源阵营（OpenAI、Anthropic、Google、xAI）保持领先但优势缩小
- **Bonsai 27B的1-bit模型**和**Qwen3-Next的3B激活**正在打破"大即是强"的迷信

---

## 技术债务与取舍分析

每个架构选择都伴随着权衡：

| 模型 | 最大的架构取舍 |
|------|--------------|
| GPT-5.6 Sol | 追求极致性能，牺牲了推理成本和端侧部署能力 |
| Claude Fable 5 | 混合架构带来复杂性，维护两套计算图增加工程负担 |
| Gemini 2.5 Pro | 保持纯Transformer+极致多模态，训练成本极高 |
| DeepSeek V4 Pro | 极致性价比，一些细节场景质量不如旗舰 |
| Grok 4.7 | 自适应深度路由增加了推理的不可预测性 |
| Qwen3-Next | 极低激活参数在某些推理密集型任务上表现受限 |

---

## 架构对比总结

```
性能天花板：GPT-5.6 Sol ≈ Gemini 2.5 Pro > Claude Fable 5 > 
                Grok 4.7 > DeepSeek V4 Pro > Qwen3-Next
                    
性价比之王：DeepSeek V4 Flash ≈ Qwen3-Next >> 其他模型
                    
推理效率冠军：Qwen3-Next (3B激活) > Claude Fable 5 (混合架构) >
                 DeepSeek V4 Flash > GPT-5.6 Luna

多模态之王：Gemini 2.5 Pro (原生训练) >> GPT-5.6 Sol > 其他
                    
上下文冠军：Gemini 2.5 Pro (10M) >> GPT-5.6 (2M) > 其他
                    
开源先锋：DeepSeek V4 Pro + Qwen3-Next (全开源)
```

每个模型都在特定的设计空间中做出了不同的取舍。2026年的格局是：**没有一个模型在所有维度上赢了所有人**，选择哪个模型取决于你的实际场景和约束条件。
