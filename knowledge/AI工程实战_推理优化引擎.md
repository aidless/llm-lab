# AI 工程实战——推理优化引擎

> 从 KV Cache 到 Speculative Decoding，从量化到推理框架选型
> 2026年——推理成本下降 1000×，但需求增长 10,000×

---

## 一、推理优化的全景图

### 1.1 优化层次

```
┌───────────────────────────────────────────────┐
│              算法层                             │
│  Speculative Decoding | Medusa | Lookahead     │
│  并行生成 | 投机推理 | Cascade                  │
├───────────────────────────────────────────────┤
│              系统层                             │
│  KV Cache 管理 | PagedAttention | 前缀缓存      │
│  动态批处理 | Continuous Batching              │
├───────────────────────────────────────────────┤
│              模型层                             │
│  量化 (FP8/INT4/INT2) | 蒸馏 | 剪枝            │
│  MoE 稀疏激活 | 层折叠                          │
├───────────────────────────────────────────────┤
│              硬件层                             │
│  Flash Attention | 算子融合 | CUDA Graph       │
│  内存布局优化 | TensorRT 编译                    │
└───────────────────────────────────────────────┘

优化空间:
  硬件层: ~5× (算子优化)
  模型层: ~10× (量化 + 稀疏化)
  系统层: ~10× (批处理 + 缓存)
  算法层: ~3× (投机解码)
  总体: ~1500×
```

### 1.2 吞吐 vs 延迟：不同的优化方向

```
在线服务 (延迟敏感):
  目标: 首 token < 200ms, 每个 token < 30ms
  策略:
    - 小批次 (batch=1-8)
    - CUDA Graph 消除 kernel launch 开销
    - KV Cache 预分配
    - 投机解码
    
  典型: ChatGPT, Claude, 产品级助手

离线批处理 (吞吐优先):
  目标: 最大化 tokens/sec, 延迟不敏感
  策略:
    - 大批次 (batch=64-512)
    - Continuous Batching
    - Flash Attention
    - FP8/INT4 量化
    
  典型: 数据标注、知识库批量生成、合成数据生产
```

---

## 二、KV Cache 深度优化

### 2.1 KV Cache 的内存消耗

```
对于 70B 模型, batch=1, seq_len=4096:

  每个 token 需要:
    key 矩阵: 80L × 128d × 2B = 20,480 bytes
    value 矩阵: 80L × 128d × 2B = 20,480 bytes
    ≈ 40KB per token ※
  
  4096 tokens: 40KB × 4096 = 160MB per request
  
  batch=64: 10GB+ 用于 KV Cache
  
  ※ 不同模型架构的实际配置:
    LLaMA-70B: 80 layers, 8 KV heads, 128 dim per KV head = FP16

长序列的 KV Cache (2026年典型场景):

  上下文 128K, batch=32:
    KV Cache ≈ 40KB × 128K × 32 = 160GB
    需要 5 张 A100 80G 只存 KV Cache!
```

### 2.2 PagedAttention (vLLM 核心)

```
问题: KV Cache 是碎片化的
  传统: 预分配最大序列长度的连续内存
  浪费: 短序列浪费大量预分配内存

PagedAttention 的思路:
  将 KV Cache 分成固定大小的 "pages" (blocks)
  像虚拟内存一样管理:

  Physical KV blocks (GPU memory):
    ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐
    │ B0   │ │ B1   │ │ B2   │ │ B3   │ ...
    └──────┘ └──────┘ └──────┘ └──────┘

  Logical pages (per request):
    Request A: [B0, B2, B5, B8]  ← 非连续!
    Request B: [B1, B3, B4, B6]  ← 灵活的物理分配

优势:
  - 几乎零内存浪费 (仅最后一页可能部分填满)
  - 内存共享: 多个请求可以共享前缀的 KV pages
  - Copy-on-Write: Prefix Caching 的基础

实现关键:

  class PagedAttention:
      def __init__(self, block_size=16, num_blocks=total_gpu_mem//block):
          self.block_size = block_size  # 每块 16 个 token
          self.num_blocks = num_blocks
          self.free_blocks = set(range(num_blocks))
          self.block_table = {}  # request_id → [block_ids]
      
      def allocate(self, request_id, num_tokens):
          """新序列的内存分配"""
          num_blocks_needed = ceil(num_tokens / self.block_size)
          allocated = []
          for _ in range(num_blocks_needed):
              block = self.free_blocks.pop()
              allocated.append(block)
          self.block_table[request_id] = allocated
      
      def append_token(self, request_id, key, value):
          """为已有序列追加一个 token"""
          blocks = self.block_table[request_id]
          last_block = blocks[-1]
          
          if last_block is not full:
              # 写入已有块
              offset = (current_seqlen - 1) % self.block_size
              self.kv_blocks[last_block][offset] = (key, value)
          else:
              # 分配新块
              new_block = self.free_blocks.pop()
              blocks.append(new_block)
              self.kv_blocks[new_block][0] = (key, value)
```

### 2.3 前缀缓存 (Prefix Caching)

```
核心观察: 很多请求共享相同的前缀
  例如: 系统提示词、上下文文档、对话历史

实现:

  class PrefixCache:
      def __init__(self):
          self.cache = {}  # hash → KV blocks
      
      def compute_prefix_hash(self, tokens: List[int]) -> str:
          """用 rolling hash 计算前缀的哈希"""
          # 使用 RRPR 或 MinHash
          # 粒度: 每 block_size 个 token 一个 hash
          return hash(tuple(tokens[:self.block_size]))
      
      def lookup(self, prompt_tokens):
          """尝试缓存命中"""
          prefix_blocks = []
          
          for i in range(0, len(prompt_tokens), self.block_size):
              chunk = prompt_tokens[i:i+self.block_size]
              chunk_hash = self.compute_prefix_hash(chunk)
              
              if chunk_hash in self.cache:
                  # 命中: 复用已有 block
                  prefix_blocks.append(self.cache[chunk_hash])
              else:
                  # 未命中: 停止前缀匹配
                  break
          
          return prefix_blocks  # 已缓存的 blocks
      
      def store(self, request):
          """处理完成后将 KV 存入缓存"""
          for block_id, tokens in request.cached_blocks:
              h = self.compute_prefix_hash(tokens)
              self.cache[h] = block_id

Auto-Prefix Detection (2026年):

  不再需要手动标记系统提示
  自动检测重复前缀模式
  跨会话共享缓存
  缓存淘汰: LRU + 引用计数
```

---

## 三、投机解码 (Speculative Decoding)

### 3.1 为什么需要投机解码

```
大模型的推理瓶颈在于:

  - Prefill 阶段 (处理输入): 计算密集
    可以充分利用 GPU 并行能力
  
  - Decode 阶段 (逐 token 生成): 带宽密集
    GPU 算力利用不充分
    一次 forward 只生成 1 个 token
    
  自动回归的制约:
    因为每个 token 依赖前一个 token
    无法并行
    
  投机解码的思路:
    用一个小的 "draft model" 先快速猜多个 token
    然后让大模型并行验证
```

### 3.2 Draft Model 方案

```
架构:

  Draft Model (小模型, ~1B 参数):
    - 快速生成 k 个投机 token
    - 生成速度: ~100 tokens/ms
  
  Target Model (大模型, ~70B):
    - 并行验证投机 token
    - 一次前向传播验证所有 k 个 token
    - 保留第一个不接受的 token 之前的所有 token

流程:

  1. Draft: 
     draft_model 快速生成 [t₁, t₂, t₃, t₄] (k=4)
  
  2. Verify:
     target_model 一次 forward:
       输入: [prompt, t₁, t₂, t₃, t₄]
       输出: [logits_1, logits_2, logits_3, logits_4]
       
  3. Accept/Reject:
     比较 draft 与 target 的分布
     对于每个位置 j, 如果 draft_token 被接受:
       继续检查 j+1
     如果拒绝:
       从 target 分布采样
       返回已接受部分

接受率:

  典型接受率: 60-80% (单 token)
  一次验证 4 个, 期望接受 ~2.5 个
  加速比: ~2.5×
```

### 3.3 Self-Speculative (2026年最优方案)

```
使用大模型自身的浅层作为 draft model:

  LLaMA-70B:
    完整: 80 layers → target model
    浅层: 前 20 layers → draft model
    
  Draft 阶段:
    前 20 layers + LM head → 快速生成 4 个投机 token
  
  Verify 阶段:
    从第 21 层开始, 继续用完整的 80 层验证

优势:
  - 不需要额外训练 draft model
  - draft 和 target 的分布天然一致 → 高接受率
  - 权重共享 → 零额外内存

工程实现:

  class SelfSpeculativeDecoder:
      def __init__(self, model, draft_layers=20):
          self.model = model
          self.draft_layers = draft_layers
          # draft: 前 draft_layers 层
          # target: 完整层
      
      def generate(self, prompt, max_tokens):
          output = prompt.copy()
          
          while len(output) < max_tokens:
              k = min(5, max_tokens - len(output))
              
              # 1. Draft
              draft_tokens = self._draft(output, k)
              
              # 2. Verify
              verified = self._verify(output, draft_tokens)
              
              # 3. Accept
              output.extend(verified)
          
          return output
```

### 3.4 Medusa (多投头方案)

```
核心创新:
  不再使用单独的 draft model
  在模型最后添加多个 "medusa head"

  每个 head 预测未来第 i 个 token:
    Head 1: 预测下一个 token
    Head 2: 跳过 1 步, 预测下下个
    Head 3: 跳过 2 步, 预测下下下个

  树状验证:
    所有 head 的预测组合成树
    一次 forward 验证所有路径

  加速: 2-3× (典型值)
  优势: 不需要额外模型
```

---

## 四、量化推理

### 4.1 量化位宽选择

```
精度     | 内存节省 | 速度提升 | 模型质量
---------|---------|---------|---------
FP16     | 1×      | 1×      | 基线
FP8      | 2×      | 1.5-2×  | 几乎无损
INT4     | 4×      | 2-3×    | 轻微损失
INT2     | 8×      | 3-4×    | 显著损失
NF4      | 4×      | 不可加速 | 近无损 (QLoRA)
MXFP4    | 4×      | 2-3×    | 近无损 (2026 新格式)

工程建议:
  70B 以下: FP8 + FP16 KV Cache
  70B-200B: INT4 权重 + FP16/FP8 KV Cache
  200B+: NF4 + 稀疏 KV Cache
  端侧: INT2 + 结构化剪枝
```

### 4.2 2026年主流量化技术

```
GPTQ (One-shot Weight Quantization):

  用少量校准数据
  逐层最小化量化误差
  支持 INT3/INT4
  
  原理:
    min_W_hat ||WX - W_hat X||²_F
    逐列优化量化权重
    
  优势: 量化后不需要微调
  适用: 70B 以下模型

AWQ (Activation-aware Weight Quantization):

  观察: 权重中只有 1% 的重要通道
  保护重要通道不被量化
  
  方法:
    1. 用校准数据观察激活
    2. 识别重要通道 (高激活幅度)
    3. 保护重要通道 (更高精度或缩放)
    
  优势: 比 GPTQ 质量更好
  劣势: 实现复杂

Bitsandbytes NF4 (QLoRA 量化):

  Normal Float 4: 假设权重重正态分布优化的量化级别
  用于微调时的量化 (QLoRA)
  不可用于推理加速 (没有优化的 kernel)

SmoothQuant (INT8 量化):

  观察: 激活值比权重更难量化
  方案: 将激活的量化难度 "平滑" 到权重
  
    X' = X / s
    W' = W × s  # 平滑因子
    
  使得 X' 和 W' 都容易被 INT8 量化
  
  2026: 标准 INT8 推理方案
```

### 4.3 KV Cache 量化

```
观察: KV Cache 是推理的内存瓶颈

解决方案:
  将 KV Cache 量化为 FP8 或 INT4

FP8 KV Cache:
  每 token KV 从 40KB → 20KB
  质量损失: 几乎为零
  主流方案: 2025+ 所有推理框架支持

INT4 KV Cache:
  每 token KV 从 40KB → 10KB
  质量损失: 长上下文略有影响
  需要: KV Cache 的通道量化 (不同通道不同缩放)

量化 KV Cache 的关键:

  K 矩阵可以轻度量化 (注意力对 K 精度不敏感)
  V 矩阵需要较高精度 (V 的误差会累积到输出)
  建议: K→INT4, V→FP8
```

---

## 五、推理框架

### 5.1 2026 主流推理引擎

```
框架       | 硬件       | 优势           | 劣势           | 最佳场景
-----------|-----------|---------------|---------------|----------
vLLM       | NVIDIA    | PagedAttention | 对 AMD 支持差   | LLM 服务
TensorRT-LLM | NVIDIA | 极致性能       | 编译时间长      | 生产部署
SGLang     | NVIDIA    | RadixAttention | 社区较小       | 结构化推理
llama.cpp  | CPU/MPS   | 广泛兼容       | GPU 性能不如专用  | 端侧/本地
MLC-LLM    | 全平台      | 统一部署       | 成熟度中等      | 移动端
OpenPPL    | 全平台      | AI 推理统一     | 中文生态为主    | 国产 GPU
TPU 推理    | Google TPU | 自家硬件       | 需要 MXU 优化   | Google 生态
```

### 5.2 推理服务架构

```
生产级架构:

  ┌──────┐  ┌──────┐  ┌──────┐
  │ Load │→ │ API  │→ │ Model│
  │      │  │ Gate │  │      │
  └──────┘  └──────┘  └──────┘
  Balancer  Auth/Cache  Server
  
  ┌──────────────────────────────────┐
  │       分布式推理集群              │
  │  ┌────────┐ ┌────────┐ ┌────────┐│
  │  │ Server │ │ Server │ │ Server ││
  │  │  1     │ │  2     │ │  N     ││
  │  └────────┘ └────────┘ └────────┘│
  └──────────────────────────────────┘

关键组件:

  1. Request Router:
     - 按需求路由到不同模型
     - 缓存命中返回缓存结果
     - 负载均衡
     
  2. Data Plane (推理节点):
     - vLLM / TRT-LLM Serving
     - 支持 dynamic batching
     - 支持 KVCache 复用
     - 自动扩缩容
     
  3. Control Plane:
     - 模型版本管理
     - 灰度发布
     - 模型 warmup
     - 健康检查
```
