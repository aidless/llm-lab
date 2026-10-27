# AI 工程实战——大规模分布式训练

> 3D/4D 并行策略全解、集群网络拓扑、训练稳定性、故障恢复、2026年十万卡训练
> 训练一个前沿大模型需要 10,000+ GPU、3 个月、$100M+

---

## 一、并行策略全景

### 1.1 四大并行维度的工程选择

```
维度         | 切分方式         | 通信模式      | 主要开销      | 适用场景
-------------|------------------|--------------|--------------|----------
数据并行(DP) | 数据切分          | All-Reduce   | 梯度同步       | 通用
张量并行(TP) | FFN/Attention 切分 | All-Reduce   | 通信频繁       | 单节点内
流水线并行(PP) | 层切分          | P2P Send/Recv | 气泡 (Bubble) | 跨节点
序列并行(SP) | 序列维度切分      | All-Reduce   | 通信 + 重计算   | 长序列

通信开销对比:
  DP: 每步一次 all-reduce (梯度)
  TP: 每层多次 all-reduce (forward + backward)
  PP: 每步 N 次 P2P (微批次数量)
  SP: 每层一次 all-reduce (序列维度)

并行度决策树:

  模型能否放进单 GPU?
    ├── 是 → 数据并行足矣 (小模型)
    └── 否 → 参数量多大?
         ├── < 10B → TP + DP
         ├── 10B-100B → TP + PP + DP (3D 并行)
         └── > 100B → 4D 并行 (TP + PP + DP + SP)
```

### 1.2 数据并行 (Data Parallelism)

```
传统数据并行 (DDP):

  GPU 0: [data batch 0] → [完整模型] → loss → 梯度 all-reduce
  GPU 1: [data batch 1] → [完整模型] → loss → 梯度 all-reduce
  ...

  问题: 每个 GPU 保存完整模型 → 大模型放不下

ZeRO (Zero Redundancy Optimizer):

  ZeRO Stage 1: 切分优化器状态
    每个 GPU 只保存一部分优化器状态
    通信量: 不变 (还是 all-reduce 梯度)

  ZeRO Stage 2: 切分梯度
    每个 GPU 只保存一部分梯度
    通信量: 增加 (reduce-scatter + all-gather)

  ZeRO Stage 3: 切分参数
    每个 GPU 只保存一部分参数
    通信量: 显著增加 (每个 forward/backward 都需要 all-gather/scatter)
    
  工程选择:
    - 7B 以下: DDP 即可
    - 7B-70B: ZeRO-2
    - 70B-500B: ZeRO-3
    - 500B+: ZeRO-3 + TP + PP

FSDP (Fully Sharded Data Parallelism):

  PyTorch 官方实现
  与 ZeRO-3 等价但更灵活
  2026年: FSDP2 成为主流

  FSDP2 的关键改进:
    1. 自动参数预取 (prefetching next layer's parameters)
    2. 混合分片 (部分参数分片，部分复制)
    3. 通信重叠 (在计算当前层时预取下一层的参数)
```

### 1.3 张量并行 (Tensor Parallelism)

```
核心思想: 将一个矩阵乘法切分到多个 GPU 上

行切分 (Row-wise):
  Y = X @ W → [X @ W1 | X @ W2]  # W 按行切分
  GPU 0: X @ W1
  GPU 1: X @ W2
  需要: 一次 all-reduce (合并结果)

列切分 (Column-wise):
  Y = X @ W → Y1 + Y2  # W 按列切分
  GPU 0: X1 @ W
  GPU 1: X2 @ W  
  需要: 一次 all-reduce (求和)

Transformer 中 TP 的实现:

  Attention:
    QKV 的投影可以列切分
    输出投影可以行切分
    每层需要 2 次 all-reduce

  FFN:
    第一个线性层: 列切分
    第二个线性层: 行切分
    每层需要 2 次 all-reduce

工程要点:
  - TP 通信非常频繁 (每层 2-4 次 all-reduce)
  - 要求 TP 域内的 GPU 有高速互联 (NVLink/NVSwitch)
  - 通常 TP=8 (一个节点内)
```

### 1.4 流水线并行 (Pipeline Parallelism)

```
核心思想: 将模型按层分到不同 GPU

朴素流水线 (Naive):

  GPU 0: Layer 1-4 → forward → 传给 GPU 1
  GPU 1: Layer 5-8 → forward → 传给 GPU 2
  GPU 2: Layer 9-12 → forward

  问题: GPU 利用率极低 (一次只处理一个 micro-batch)

Gpipe (微批次流水线):

  将一个 batch 分成多个 micro-batch
  隐式流水线: 利用率提高但仍存在气泡

  气泡 (Bubble):
    流水线启动和排空时的空闲时间
    气泡率 = (P-1) / (M+P-1) (P=流水线深度, M=微批次数)
    M >> P 时气泡≈0

  1F1B (One-Forward-One-Backward):

    GPU 0: F1 → F2 → ... → F4 → B4 → B3 → ... → B1
    GPU 1:       F1 → F2 → ... → F4 → B4 → ... → B1
    
    优势: 减少内存使用 (不需要保存所有中间激活)
    2026年主流方案

VPP (Virtual Pipeline Parallelism):

  将流水线分成更多的虚拟阶段
  增加微批次数量进一步减少气泡
  但: 增加通信量
```

### 1.5 4D 并行 (2026年标准配置)

```
前沿模型的标准配置 (例如: 训练一个 ~1T 参数的 MoE):

  物理集群: 4096 GPU (512 节点 × 8 GPU)
  
  并行配置:
    张量并行 (TP): 8
    流水线并行 (PP): 16
    数据并行 (DP): 32
    序列并行 (SP): 启用
  
  并行拓扑:
    每 8 GPU (1 节点) 内: TP (NVLink)
    每 16 节点: PP 阶段 (跨节点 RDMA)
    32 组 PP: DP (数据并行复制)
  
  通信矩阵:
    TP 域内: NVLink (900 GB/s)
    PP 之间: RDMA (400 Gbps per link)
    DP 之间: RDMA (梯度 All-Reduce)
```

---

## 二、集群网络拓扑

### 2.1 GPU 互联的比较

```
互联技术      | 带宽       | 延迟 | 范围     | 2026 年状态
--------------|-----------|------|----------|-----------
NVLink 6      | 1800 GB/s | 低   | 节点内   | H100/B200
NVSwitch      | 900 GB/s  | 低   | 节点内   | 8/16 GPU 互联
InfiniBand NDR800 | 800 Gbps | 极低 | 跨节点 | 主流 AI 集群
RoCEv3        | 400 Gbps  | 中   | 跨节点   | 成本敏感方案
Ethernet (UEC) | 800 Gbps | 中   | 跨节点   | 2026 新兴方案
CXL           | 128 GB/s  | 中   | 跨节点   | 内存池化

关键约束:
  TP 通信: 每层 2-4 次 all-reduce, 要求 NVLink 级带宽
  PP 通信: 每 micro-batch 次传输激活值, InfiniBand 即可
  DP 通信: 每步一次梯度同步, 最不敏感
```

### 2.2 网络拓扑设计

```
最佳拓扑: 3D Torus (Google TPU 方案)

  TPU v5p 拓扑 (8192 TPU):
    4×4×4 Torus → 64 芯片超级块
    超级块之间通过光互连连接
    
  优势:
    - 所有节点距离短 (最大 3 跳)
    - 无瓶颈点
    - 容错: 单个链路故障不影响全局

  替代方案: Fat-Tree (多数 GPU 集群)

    2 层 Fat-Tree:
      Leaf switch: 连接 GPU 节点
      Spine switch: 连接 Leaf
    
    3 层 Fat-Tree:
      Pod → Leaf → Spine → Super Spine
    
    优势:
      - 容易扩展 (增加 pod)
      - 对称性 (任意节点间等距)
    劣势:
      - 成本高 (大量交换机)
      - GPU 集群体积大

Dragonfly+ (2026 新兴方案):

    组内: 全连接
    组间: 少量全局链路
    
    优势:
      - 极低延迟 (任意节点最多 3 跳)
      - 适合 All-to-All (MoE)
      - 成本低于 Fat-Tree
```

---

## 三、训练故障恢复

### 3.1 十万卡训练的故障率

```
10,000+ GPU 训练的现实:

  单 GPU MTBF (Mean Time Between Failures): ~10,000 小时
  10,000 GPU 集群: 平均每 1 小时有 1 个 GPU 故障
  
  加上其他故障:
    - 网卡: 每 2-3 小时一次
    - 光模块: 每 4-6 小时一次
    - 交换机: 每天 1-2 次
    - 电源/冷却: 每周 1-2 次

  结论: 十万卡训练大约每小时发生一次故障
  训练 3 个月: ~2000 次故障
```

### 3.2 Checkpoint 策略

```
同步 Checkpoint (性能差但简单):

  每 N 步所有 GPU 同时保存:
    GPU 0: optimizer state shard 0
    GPU 1: optimizer state shard 1
    ...
    全部完成后继续训练
  
  时间: 大模型 ~10-30 分钟 (写入 KV cache + 优化器状态)
  问题: 每次 checkpoint 训练暂停 10-30 分钟

异步 Checkpoint (2026 主流):

  训练不暂停，在后台保存 checkpoint:
    
    1. 冻结当前参数 → 复制到 CPU 内存
    2. CPU 线程异步写入持久化存储
    3. 训练继续 (不等待写入完成)
    4. 写入完成后标记 checkpoint 有效

  优点: 零停机时间
  缺点: checkpoint 可能对应未来几秒后的状态

选择性 Checkpoint:

  不全量保存，只保存:
    - 模型权重 (每 1000 步)
    - 优化器状态 (每 5000 步)
    - MoE routing 统计 (每 100 步)
  
  恢复方式:
    最近权重 + 最近优化器 + routing 统计
    损失: 后 100 步需要重新计算
```

### 3.3 故障恢复流程

```
自动恢复工作流:

检测到故障:
  1. 心跳检测 → 发现 GPU 5 无响应
  2. 自动排除故障节点
  3. 从最近 checkpoint 恢复
  4. 调整并行配置 (去掉一个节点)
  5. 继续训练

热替换 (Hot Swap):

  2026 年顶级 AI 实验室的能力:
    故障 GPU 被自动替换 (备用 GPU 池)
    不需要重启训练
    替换过程: ~30 秒
    训练影响: 数据并行度略降，但训练不中断
    
  实现: 
    - 集群有 5-10% 的备用 GPU
    - 训练框架支持动态调整并行组大小
    - NCCL 支持动态成员变更
```

---

## 四、训练效率优化

### 4.1 内存优化

```
激活重计算 (Activation Recomputation / Checkpointing):

  不保存所有中间激活
  在 backward 时重新计算
  
  选择性策略:
    Attention 输出: 保存 (计算量小)
    FFN 中间: 不保存 (计算量大但内存省)
    特定层: 保存
    
  2026 最佳实践:
    保存约 30% 的激活
    重新计算约 70% 的激活
    内存节省: ~40%
    计算开销: ~15%

CPU Offloading:

  将优化器状态 offload 到 CPU 内存
  当 GPU 显存不足时
  
  问题: CPU-GPU 传输带宽有限 (PCIe 4.0: 32 GB/s)
  方案: 只在 optimizer step 时传输 (每步 ~1ms)
```

### 4.2 通信优化

```
通信重叠 (Communication Overlap):

  在 backward 计算梯度的同时通信:
    
    正常: 计算 backward → all-reduce 梯度 → 更新参数
    重叠: 计算 backward → |all-reduce 第一层梯度 ← 同时计算第二层 backward|
    
  收益: 通信时间从串行变为与计算重叠
  实现: 使用异步 NCCL (ncclGroupStart/End)
  效果: 总时间减少 ~20%

梯度压缩:

  将梯度量化为 FP8 再进行 all-reduce
  接收后恢复为 FP16/32
  
  误差补偿:
   发送前: gradient - error_buffer
   计算误差: quantize(gradient) - gradient → error_buffer
  
  效果: 2× 通信带宽 (压缩比) + 训练质量几乎无损
```

### 4.3 训练监控

```
关键监控指标 (2026 训练 Dashboard):

  硬件指标:
    GPU 利用率: 95%+ 为健康
    HBM 带宽利用率: 80%+
    NVLink 利用率: 单 TP 域内 60%+
    InfiniBand 利用率: 跨节点 40%+

  训练指标:
    TFLOP/s per GPU: 预期值的 60-70% 为正常 (考虑通信开销)
    MFU (Model FLOPS Utilization): ~50% 为优秀
    Step time 稳定性: 标准差 < 平均值的 5%
    
  通信指标:
    All-reduce 延迟: 大模型 < 200ms
    P2P 延迟 (PP): < 1ms per micro-batch
    通信计算比: 通信时间 / (通信+计算) < 20%

  异常告警:
    GPU 温度 > 85°C
    HBM ECC 错误率 > 阈值
    网络丢包率 > 0.01%
    Step time 突增 > 20%
```
