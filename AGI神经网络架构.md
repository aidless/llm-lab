# AGI 神经网络架构

---

## 总览：从神经元到千亿参数

神经网络架构——是 AGI 的"身体结构"。2017 年之前——主流是 CNN（视觉）与 RNN（序列）；2017 年 Transformer 问世后——架构进入"注意力统治"时代：语言、视觉、音频、多模态、强化学习策略——全部统一到 Transformer 之上。2026 年的格局是：**稠密 Transformer 是基线，MoE 是规模化主流，混合线性注意力（Mamba、RWKV、线性注意力）是效率前线**——而 CNN 以 tokenizer/视觉编码器的形式活在多模态模型内部，GNN 则在知识图谱与 GraphRAG 中扮演特殊角色。

```
              ┌────────────────────────────────────────────┐
              │            Transformer Block               │
              │                                            │
输入 ──→ Embedding ──→ [多头注意力 ─→ FFN] × L ──→ 输出层    │
              │           ↑  │        ↑  │                 │
              │       残差连接  │  残差连接  │                 │
              │        LayerNorm  │  LayerNorm  │            │
              └────────────────────────────────────────────┘
                                    │
     ┌──────────────┬───────────────┼───────────────┐
     ↓              ↓               ↓               ↓
  GPT/Llama      Mistral         DeepSeek         Mamba
  (密集+RoPE)   (滑动窗口+GQA)   (MLA+MoE)       (线性注意力)
```

---

## 一、Transformer 架构深度

### 1.1 总体结构

**Transformer（Vaswani et al., 2017）**——"Attention Is All You Need"。核心设计决策：**抛弃循环**——用自注意力对序列建立任意距离的直接依赖。

```
Encoder-Decoder（原始版）:  机器翻译
  Encoder: 双向注意力（每 token 可见全序列）
  Decoder: 带因果掩码的自注意力 + 交叉注意力

Decoder-only（现代 LLM 标准）:
  x₀ = Embedding(tokens) + PositionEncoding
  for l in 1..L:
      xₗ = Block_l(xₗ₋₁)
  logits = x_L · W_outᵀ       # 权重共享（tied embedding）可选
  loss = CE(logits, next_token)
```

**每个 Block 内部（以 Llama 3 为例）**：
```
x ← x + Attention(Norm(x))       # Pre-Norm + 残差
x ← x + FFN(Norm(x))             # SwiGLU
```

### 1.2 自注意力（Self-Attention）详解

**Scaled Dot-Product Attention**：
```
Q = XW_Q,  K = XW_K,  V = XW_V        # X ∈ ℝ^{T×d}
Attention(Q,K,V) = softmax(QKᵀ/√d_k)·V
```

**逐元素推导**——第 i 个 token 的输出：
```
outᵢ = Σⱼ aᵢⱼ·vⱼ,   aᵢⱼ = softmax(qᵢ·kⱼ/√d_k)

aᵢⱼ 的含义: token i 对 token j 的"注意力权重"——由两者语义相似度（点积）决定
```

**为什么缩放 √d_k**：
- 若 q, k 各分量 i.i.d. N(0,1)——点积 q·k 的方差 = d_k
- 不缩放时——d_k 越大分布越尖锐——softmax 输入大 → 梯度趋零（饱和）
- 除以 √d_k 使方差回到 1——softmax 保持"温和"——梯度流畅

**因果掩码（Causal Mask）**——Decoder-only 的关键：
```
未掩码:  A = softmax(QKᵀ/√d)                  # 全部位置可见
掩码后:  A = softmax((QKᵀ/√d) + M),  Mᵢⱼ = -∞ (j > i)
        等价于把未来位置的概率质量清零（softmax(-∞) = 0）
```

**注意力的三个角色**：从信息论看——注意力是"可微的信息检索"：
```
Query:  "我现在要找什么"（当前 token 的需求）
Key:    "我有什么"（每个 token 的内容标签）
Value:  "找到后给我什么"（实际内容）
```

### 1.3 多头注意力（Multi-Head Attention）

**动机**：单头只能捕捉一种相似度模式；多头让不同头学习不同的关系——语法、共指、主题、位置。

```
h = 8/16/32/64 个头
每个头:  Qₕ = XW_Qₕ, Kₕ = XW_Kₕ, Vₕ = XW_Vₕ    # d/h 维
输出:    Concat(O₁,...,Oₕ)·W_O
```

**参数与计算**：
```
总投影参数: 4·d²（W_Q, W_K, W_V, W_O）
与单头相同参数量——只是分块 + 输出拼接——FLOPs 不变
注意: 头间不共享参数——每个头在子空间 d/h 内独立学习
```

### 1.4 位置编码（Position Encoding）

**问题**：注意力对 token 顺序**置换不变**（没有位置信息，模型看到的是词袋）。必须显式注入位置。

**绝对位置编码（sinusoidal，原始版）**：
```
PE(pos, 2i)   = sin(pos / 10000^(2i/d))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
可学习版本:    GPT-2/3 用可学习参数表（上限 2048 位置——训练后不能超长）
```

**相对位置编码**——不编码绝对位置而是位置差：
```
经典相对（T5 用相对偏置表: 每个相对距离一个标量）
ALiBi（Mistral 70B 用过）:
  attention_logits += -m·|i-j|     # m 随头数递减——越远的 token 越不被注意
  → 零参数、天然外推、无需位置表
```

**旋转位置编码（RoPE, Rotary Position Embedding）——2026 年事实标准**：
```
核心思想: 把 q,k 按 2 维子空间旋转——旋转角度 = 位置差

RoPE(q, pos) = R(pos)·q,  其中 R 是块对角旋转矩阵:
  R(pos) = diag( R(pos,1), R(pos,2), ..., R(pos,d/2) )
  每个 2×2 块: [cos(pos·θᵢ)  -sin(pos·θᵢ)]
               [sin(pos·θᵢ)   cos(pos·θᵢ)]
  θᵢ = 10000^(-2i/d)（与 sinusoidal 相同的基频）

关键性质:  ⟨R(p₁)q, R(p₂)k⟩ = ⟨q, k⟩ 旋转 (p₂-p₁) ——点积只依赖相对位置！
```

**RoPE 的实用工程**：
```
1. 高频外推衰减（NTK-aware / YaRN）: 调整基频使位置超训时仍可用
   θᵢ' = θᵢ · (base_ratio)^(2i/d)——把"旋钮"拧到更长上下文
2. 上下文扩展: Llama-3 8k → 128k 靠 RoPE 基频调整 + 少量继续训练
3. 实现优化: 不构造旋转矩阵（O(T·d²) 太贵）——
   而是 q 复数形式 q = q_even + i·q_odd，逐对乘 e^{iθ·pos}（O(T·d)）
```

### 1.5 归一化：LayerNorm → RMSNorm

**LayerNorm（LN）**——对每个 token 的特征维度归一化：
```
μₜ = (1/d)Σᵢ xₜᵢ,  σₜ² = (1/d)Σᵢ (xₜᵢ-μₜ)²
yₜ = γ·(xₜ-μₜ)/√(σₜ²+ε) + β       # γ, β 可学习

与 BatchNorm 区别: 对"序列维度"归一化而非"batch 维度"
→ 与 batch size 无关；长度可变；训练/推理一致（BN 用跑动统计量）
```

**RMSNorm（Root Mean Square Norm, Llama 家族标配）**——去掉均值与偏置：
```
yₜ = γ·xₜ / RMS(xₜ),   RMS(xₜ) = √((1/d)Σᵢ xₜᵢ² + ε)

优势: 少算均值——少一组偏置参数——速度 +10% 左右
数学: 缩放不变性保留（layer 输出的尺度稳定）——均值信息对 Transformer 冗余
```

### 1.6 FFN 变体谱系

| 变体 | 公式 | 代表模型 |
|------|------|---------|
| 标准 ReLU FFN | W₂·ReLU(W₁x) | GPT-2 |
| GELU FFN | W₂·GELU(W₁x) | GPT-3 |
| SwiGLU | W₂·(SiLU(W₁ᵍx) ⊙ W₁ᵘx) | Llama、Mistral、Gemma、Qwen |
| GeGLU | W₂·(GELU(W₁ᵍx) ⊙ W₁ᵘx) | PaLM、Falcon |
| 无偏置 | 去掉 b（节省参数、便于张量并行） | Llama 1/2/3 |

**SwiGLU 为何流行**：门控机制（σ(x)⊙x）提供自适应非线性——实测语言建模 PPL 稳定优于 GELU（约 0.1~0.3）；代价是参数 ×1.5——用 `(8/3)d` 中间维抵消（效果：参数量接近 4d 但 PPL 更优）。

### 1.7 高效注意力变体：GQA / MQA / Sliding Window

**背景**：推理时 KV cache 显存 = 2·T·d·L·字节数（每 token 每层）。长上下文下——KV cache 是显存天花板（128k 上下文 × 70B 模型可达上百 GB）。

```
MHA（Multi-Head）:     每头独立 K、V     —— 全质量、高缓存
MQA（Multi-Query）:    所有头共享 K、V    —— 缓存 1/h，质量略降（PaLM 用过）
GQA（Grouped-Query）:  头分 G 组，组内共享 K、V —— 质量/缓存的甜点
                       Llama 2 70B: 8 组; Llama 3: 8 组; Gemma: 8 组
```

**GQA 的量化效果**：d=4096, h=32, G=8——KV cache 降为 MHA 的 8/32 = 1/4。

**滑动窗口注意力（Mistral 7B 初版）**：
```
每层只看窗口 W=4096 内的 token:
  Aᵢⱼ = 0 (|i-j| > W)
信息仍可远距离传播: 层间堆叠——每层传播 W——L 层可达 L·W
（Mistral 后续版本放弃——因为 Rope 长上下文训练直接解决）
```

### 1.8 KV Cache 与推理效率

```
推理逐 token 生成时——前面的 K、V 可复用:
KV cache: 存储所有层、所有头的历史 K、V 张量
内存:  2 · L · h · d_k · T · 字节数（bf16: ×2 bytes）

PagedAttention（vLLM）: 把 KV cache 分页——类似操作系统虚拟内存
→ 显存利用率从 ~40% 提升到 ~90%+，支持并发批处理
```

---

## 二、LLM 家族架构演进（GPT / Llama / Mistral / DeepSeek）

### 2.1 GPT 系列：架构事实标准的确立

```
GPT-1 (2018, 117M):  Decoder-only 12层, 768d, 12头, 可学习位置编码
                     = 把"预训练语言模型+微调"范式确立
GPT-2 (2019, 1.5B):  48层, 1600d, LN 移到每个子层前（Pre-Norm!）
                     上下文 1024；LayerNorm 前置是深模型能训的功臣
GPT-3 (2020, 175B):  96层, 12288d, 96头, 稀疏注意力（局部+全局混合）
                     bpe 词表 50257；初始化 σ=0.02；上下文 2048
GPT-4 (2023, ~1.8T MoE 传闻): 8×220B 专家——每个 token 激活 ~2 个
                     （OpenAI 未官方确认——但多份泄漏与论文分析支持）
```

**GPT 的 Pre-Norm 决策**：Post-Norm（原始 Transformer）在 100+ 层时会退化——因为每层输出方差累积。Pre-Norm 让残差主干保持单位尺度——深模型可训。这是所有后续 LLM 遵循的基线。

### 2.2 Llama 家族：开源标准的"配方"

**Llama 1（2023, 7B/13B/33B/65B）**——把开源预训练配方固化：
```
RMSNorm（去均值）
RoPE（旋转位置编码）
SwiGLU 激活（中间维 8/3 d）
无偏置
AdamW + cosine schedule + warmup 2000 步
权重衰减 0.1；梯度裁剪 1.0；bf16
```

**Llama 2（2023）**——数据集 2T token、上下文翻倍 4096、GQA（70B 用）：
```
SFT + RLHF 全链路公开——后训练范式开源化
```

**Llama 3（2024, 8B/70B/405B）**：
```
Tokenizer: 128K 词表（tiktoken 式 BPE）；上下文 8k → RoPE 扩展 128k
405B: 126 层, 16384d, 128 头（GQA 8组）
数据: 15T+ token 多语种；后训练: SFT → DPO（而非 PPO！）
训练: 使用标准 dense——MoE 留到研究版
```

**Llama 4（2025 传闻/部分发布）**：MoE 路线（Scout/Maverick 等——数百万上下文 + MoE 专家路由）——标志 Meta 从稠密转向稀疏。

### 2.3 Mistral 家族：效率路线

```
Mistral 7B (2023): 滑动窗口注意力 + GQA——8B 级性能标杆
Mixtral 8x7B (2023): 第一个开源 MoE——8 专家 Top-2 路由——46.7B 总参 12.9B 激活
Mixtral 8x22B (2024): 141B 总参 39B 激活
Mistral Large (2024+): 闭源稠密/MoE——与 Microsoft 合作
```

**MoE 路由机制（Mixtral 版）**：
```
每层 FFN 换成 8 个专家 + 路由器:
  logits = x·W_r               # 路由打分 (d → 8)
  top-2 专家被激活:
  out = Σ_{k∈top2} softmax(logits)[k] · Expert_k(x)
负载均衡: 辅助 loss 惩罚路由不均（防止专家坍塌/退化）
```

### 2.4 DeepSeek 家族：创新前沿

**DeepSeek-V2（2024）**——两项关键创新：
```
1. MLA（Multi-head Latent Attention）:
   把 KV 压缩到低维隐向量 c:
     cₜ = W_DKV·hₜ            # d → d_c（如 512，而原始 KV 是 4×d）
     Kₜ, Vₜ = W_UK·cₜ, W_UV·cₜ
   → KV cache 减少 ~93%——长上下文成本大幅下降
   （解耦 RoPE 的 key 加回——因为压缩向量不能直接旋转）

2. DeepSeekMoE: 细粒度专家 + 共享专家 + 无辅助损失的负载均衡
   (通过偏置项自适应调节路由 logits——取代辅助 loss)
```

**DeepSeek-V3（2024-12）**——671B 总参、37B 激活：
```
结构: 61 层稠密 + MoE 层（256 专家, 1 共享 + 8 路由, top-8）
MLA + 多 token 预测（MTP, Multi-Token Prediction 辅助头）
训练: 14.8T tokens——2048 GPU——2.788M H800 GPU 小时
FP8 混合精度训练——低成本超大规模训练的标志性工程
```

**DeepSeek-R1（2025）**——纯 RL 推理强化：
```
GRPO（组相对策略优化）替代 PPO（免 critic 模型）:
  Aᵢ = (rᵢ - mean(r)) / std(r)      # 组内归一化优势
冷启动 SFT → 大规模 RL（推理奖励：格式+正确性）→ 蒸馏
```

**通用趋势总结（2026）**：
```
稠密 → MoE（DeepSeek、Llama 4、Qwen 3-Max 等——稀疏化是规模化必经）
MHA → GQA → MLA（KV 压缩是长上下文钥匙）
Post-Norm → Pre-Norm → QK-Norm（查 1.6: QK-LayerNorm 稳定大模型）
激活: GELU → SwiGLU（稳定赢家）
位置: 绝对 → RoPE（+YaRN 等扩展）
训练: bf16/FP8；梯度裁剪；WSD 或 cosine 调度（见训练方法论）
```

---

## 三、CNN（卷积神经网络）

### 3.1 卷积与池化

**卷积**——局部连接 + 权重共享：
```
y[i,j] = Σ_{u,v} x[i+u, j+v] · W[u,v] + b      # W: k×k 卷积核

关键性质:
  局部性: 只看感受野 k×k
  平移等变: 同一特征在图像任何位置由同一核检测
  参数量: k²·c_in·c_out（vs 全连接 d_in·d_out——指数级压缩）
```

**池化（Pooling）**——下采样 + 平移不变：
```
MaxPool 2×2:  取窗口最大值（最显著特征）
AvgPool:      取平均（平滑）
GlobalAvgPool: 全图平均 → 分类头（ResNet 末层）
```

### 3.2 残差网络（ResNet, 2015）

**退化问题**：更深网络训练误差反而更高（不是过拟合——是优化失败——梯度流断裂）。

**解决方案——恒等捷径**：
```
y = F(x, W) + x          # F 是 2~3 层卷积
梯度: ∂y/∂x = 1 + ∂F/∂x  # 恒等项 1 保证梯度通路
```

**ResNet 设计规范（后续所有 CV 骨干的基线）**：
```
BasicBlock (18/34 层):  3×3 Conv → BN → ReLU → 3×3 Conv → BN → +x
Bottleneck (50/101/152): 1×1 → 3×3 → 1×1（压缩-扩张——3.9 倍参数效率）
下采样: 空间降维时用 1×1 卷积 + stride 2 对齐维度
```

### 3.3 CNN 在 2026 年 AGI 生态中的位置

**CNN 没有被淘汰——而是退居"感知前端"**：
```
ViT（Vision Transformer, 2021）: 把图像切成 16×16 patch 当 token
  → 纯 Transformer 做视觉——大模型时代主流
但混合架构仍是效率王:
  - ConvNeXt、EfficientViT（卷积 token 混合）
  - Gemini、GPT-4V 的内部视觉编码器（未公开——推测含卷积下采样）
  - SD/SDXL 的 VAE 编码器（下采样卷积）
  - Whisper 的 Mel 特征 CNN 前端
```

**Patch Embedding 数学**：
```
图像 224×224×3 → 16×16 patch → 14×14 = 196 个 token
每个 patch 展平 768 维 → Linear 投影到 d 维 → 加位置编码
→ 视觉问题变成序列问题——与 LLM 共享同一套 Transformer 骨架
```

---

## 四、RNN 及其变体

### 4.1 RNN 的结构与困境

```
标准 RNN:
hₜ = tanh(W_hh·hₜ₋₁ + W_xh·xₜ + b)
yₜ = W_hy·hₜ

困境: BPTT（沿时间反向传播）——梯度沿时间链式相乘:
∂hₜ/∂h₀ = Π W_hhᵀ·diag(tanh')
→ 时间步 > 20~30 时梯度消失/爆炸——长期依赖学不到
```

### 4.2 LSTM（Long Short-Term Memory, 1997）

**核心思想**：引入细胞状态 cₜ 作为"传送带"——门控控制读写：
```
遗忘门:  fₜ = σ(W_f·[hₜ₋₁, xₜ] + b_f)
输入门:  iₜ = σ(W_i·[hₜ₋₁, xₜ] + b_i)
候选:    c̃ₜ = tanh(W_c·[hₜ₋₁, xₜ] + b_c)
细胞更新: cₜ = fₜ ⊙ cₜ₋₁ + iₜ ⊙ c̃ₜ
输出门:  oₜ = σ(W_o·[hₜ₋₁, xₜ] + b_o)
隐状态:  hₜ = oₜ ⊙ tanh(cₜ)
```

**为什么有效**：梯度可以通过 `cₜ = fₜ⊙cₜ₋₁ + ...` 的恒等通路传播（fₜ 接近 1 时梯度近似不减）——长期记忆由遗忘门调节。

### 4.3 GRU（2014）——LSTM 的简化版

```
更新门:  zₜ = σ(W_z·[hₜ₋₁, xₜ])
重置门:  rₜ = σ(W_r·[hₜ₋₁, xₜ])
候选:    h̃ₜ = tanh(W·[rₜ ⊙ hₜ₋₁, xₜ])
隐状态:  hₜ = (1-zₜ) ⊙ hₜ₋₁ + zₜ ⊙ h̃ₜ        # 无独立细胞状态
参数:    LSTM 的 3/4——性能几乎持平——现代 RNN 首选
```

### 4.4 RNN 的遗产与"线性注意力"复兴

**2023-2026 的范式转移**：Transformer 的 O(T²) 注意力在超长序列上不可扩展——"subquadratic"架构复兴了 RNN 思想：

```
Mamba（选择性状态空间模型, S4/SSM 路线）:
  hₜ = Ā(xₜ)·hₜ₋₁ + B̄(xₜ)·xₜ        # A, B 依赖输入——选择性记忆
  yₜ = C(xₜ)·hₜ
  并行化: 借助分段卷积 + 扫描（scan）算法——训练并行、推理 O(1) 每步

RWKV:  "线性 Transformer"——用 token shift + 指数衰减替代 softmax:
  等价于带固定衰减率的 RNN——可训练的"无限上下文"特性

混合路线（2025+ 主流）:
  Jamba（AI21）、Zamba、Hymba、Llama 4 部分层——Transformer 层 + SSM 层交错
  → 兼顾 Transformer 的质量与线性注意力的效率
```

**为什么混合架构在 2026 年胜出**：纯线性注意力在长程检索（记忆精确召回）上仍弱于 softmax 注意力；交错混合——让部分层保留 softmax 检索能力——是工程上的帕累托最优。

---

## 五、GNN（图神经网络）

### 5.1 消息传递框架（Message Passing）

**GNN 的统一数学框架**——节点通过邻居交换信息：
```
聚合（Aggregate）:  aᵥ⁽ˡ⁾ = AGG({ hᵤ⁽ˡ⁾ : u ∈ N(v) })
更新（Update）:     hᵥ⁽ˡ⁺¹⁾ = UPDATE(hᵥ⁽ˡ⁾, aᵥ⁽ˡ⁾)

L 层后——节点的表示包含 L 跳邻居信息
```

### 5.2 典型 GNN 模型

**GCN（图卷积网络, Kipf & Welling, 2017）**：
```
H⁽ˡ⁺¹⁾ = σ( D̂^(-1/2) · Â · D̂^(-1/2) · H⁽ˡ⁾ · W⁽ˡ⁾ )
  Â = A + I（加自环）, D̂ = 对角矩阵（度）
归一化 D̂^(-1/2)·Â·D̂^(-1/2): 防止高连接节点主导——同时做度归一化
```

**GAT（图注意力网络）**——注意力权重替代固定邻接：
```
eᵥᵤ = LeakyReLU(aᵀ·[Whᵥ || Whᵤ])       # 注意力打分
αᵥᵤ = softmax(eᵥᵤ)                     # 邻居间归一化
hᵥ' = σ(Σᵤ αᵥᵤ·Whᵤ)                    # 多头可叠加（同 Transformer）
```

**GraphSAGE**：采样邻居（可扩展到大图）+ 多种聚合器（mean/LSTM/pool）。

### 5.3 GNN 在 AGI/LLM 生态中的角色

```
知识图谱与 GraphRAG:
  - 实体 → 节点, 关系 → 边——知识图谱作为 LLM 的外部结构化记忆
  - GraphRAG（微软, 2024）: 文本 → 实体/关系图 → 社区摘要 → 检索增强
  - 2026: 图索引 + 向量检索混合——查询先到图、再到文本块

结构化推理:
  - 代码依赖图、分子图、蛋白质结构（AlphaFold 的 Evoformer 是图注意力变体）
  - 多跳推理显式路径（与 LLM 隐式推理互补）

与 Transformer 的关系:
  - Transformer 本质是"完全图上的注意力"（每 token 与所有 token 相连）
  - 稀疏注意力（局部窗口）≈ 图结构的特例
  - 长上下文 + 图结构先验（如位置编码编码树结构）——研究前沿
```

---

## 六、架构选择的决策框架

### 6.1 稠密 vs MoE vs 线性注意力

| 维度 | 稠密 Transformer | MoE（稀疏） | 线性注意力/SSM |
|------|----------------|-------------|---------------|
| 训练 FLOPs/token | 6N | ~2-3N（激活部分） | ~6N（但长序列省） |
| 参数质量比 | 基线 | 同 FLOPs 更高性能 | 略低（检索弱） |
| 推理显存 | 全部参数 | 全部参数（专家也在） | 全部参数 |
| KV cache | 大（GQA 可减） | 同左（MLA 可减） | O(1)（状态） |
| 长上下文 | O(T²) 注意力 | 同左 | O(T) 线性 |
| 实现复杂度 | 低 | 中（路由+负载均衡） | 中（扫描算子） |

**2026 年的现实结论**：**预训练大规模用 MoE**（同样的 FLOPs 预算下性能上限更高）；**推理/边缘部署用线性注意力或混合**；**稠密模型仍是研究基线和中小规模主力**。

### 6.2 位置编码决策树

```
需要超长上下文外推? → 是 → RoPE（+YaRN/NTK 扩展）或 ALiBi
                      → 否 → 可学习绝对位置（简单）
需要零参数?        → ALiBi
兼容并行扫描（SSM 混合）? → 层内注意 RoPE、SSM 层不用
```

### 6.3 架构超参数的典型配置（7-8B 级稠密模型，2026 标准）

```
层数 L:          32
隐藏维 d:        4096
头数 h:          32（GQA 组 8）
FFN 中间维:      14336（SwiGLU, 8/3 规则）
词表:            128K（tiktoken BPE / 自定义 BPE）
上下文:          8K-128K（RoPE）
初始 LR:         3e-4（AdamW, β=(0.9, 0.95), eps=1e-8）
权重衰减:        0.1
梯度裁剪:        1.0
训练 token:      2T-15T（Chinchilla 20/参数 的 10-30 倍——超训普遍）
```

---

## 七、总结

```
2017  Transformer 诞生:  attention 统一序列建模
2018-20  GPT 确立 decoder-only + 预训练范式
2021-23  Llama 固化开源配方（RoPE+RMSNorm+SwiGLU+GQA）
2023-24  MoE 复兴（Mixtral→DeepSeek）+ KV 压缩（MLA）+ 长上下文
2024-26  混合架构（Transformer+SSM）、RL 推理、Test-time compute
```

架构演进的底层逻辑——三条主线：
1. **质量**：更强的依赖建模（绝对→相对→RoPE 旋转）、更好的门控（GELU→SwiGLU）
2. **效率**：KV 压缩（GQA→MLA）、稀疏化（MoE）、亚二次注意力（SSM/线性）
3. **稳定性**：归一化（Post→Pre→RMS→QK-Norm）、初始化（残差缩放、DeepNorm）

对 AGI 而言——架构是"表达能力"的载体：Attention 提供任意长度的软检索，MoE 提供超大容量，SSM 提供无限状态流。未来 AGI 架构很可能不是单一范式的胜利——而是**多种归纳偏置的组合系统**：感知用卷积/层次结构、推理用注意力、长期记忆用图或状态空间、行动用世界模型——正如大脑的模块化组织。

