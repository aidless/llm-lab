# 产业深度——AI 芯片战争

> NVIDIA 的护城河、挑战者的突围、地缘政治与算力独立
> 2026年——AI 算力成为国家战略资源

---

## 一、2026 年 AI 芯片格局

### 1.1 市场全景速览

```
AI 芯片市场 (2026, 估算):

  总市场规模: ~$1500 亿 (2023: ~$500 亿)
  训练芯片: ~$600 亿
  推理芯片: ~$800 亿 (2024 年超过训练)
  边缘/AIoT: ~$100 亿

主要玩家:

  级别     | 公司           | 核心产品           | 制程 | 优势
  ---------|---------------|-------------------|------|------
  领导者   | NVIDIA        | B300, Rubin       | 4nm  | 生态无敌
  挑战者   | AMD           | MI400X            | 3nm  | 性价比
  云厂自研 | Google        | TPU v7            | 3nm  | 垂直整合
           | Amazon        | Trainium 3        | 3nm  | AWS 绑定
           | Microsoft     | Maia 200          | 3nm  | Azure 生态
           | Meta          | MTIA v3           | 4nm  | 内部推理
  新势力   | Cerebras      | Wafer-Scale CS-4  | 5nm  | 超大晶圆
           | Groq          | LPU Gen 3         | 4nm  | 低延迟推理
           | SambaNova     | SN40L             | 5nm  | 可重构
           | Tenstorrent   | Wormhole          | 5nm  | 开源生态
  国产     | 华为          | Ascend 910C       | 7nm  | 中国最大
           | 寒武纪        | MLU590            | 7nm  | 推理专用
           | 海光          | DCU K100          | 7nm  | x86 兼容

NVIDIA 的份额: ~80% (训练), ~60% (推理)
    2022: ~90% 训练, 但推理竞争加剧
```

### 1.2 NVIDIA 的护城河有多深？

```
硬件层优势:

  NVLink 6: 1800 GB/s GPU-GPU 互联
    竞争对手没有对等的互联方案
    AMD Infinity Fabric: ~800 GB/s
    自研芯片: 节点内互联是弱点
  
  NVSwitch: 576 GPU 直接互联
    跨节点: InfiniBand (标准）
    
  CUDA + Tensor Core:
    每个新架构新增: Transformer Engine, FP8, FP4
    B300: 新的 Sparsity Engine

真正的护城河是 CUDA 生态:

  1. 兼容性:
    你写的 CUDA 代码在 15 年前的 GPU 上也能跑
    ROCm 每次大版本破坏兼容性
    
  2. 工具链:
    PyTorch 的绝大多数 CUDA 实现只有 NVIDIA 上优化过
    cuDNN, cuBLAS, NCCL → 竞争对手需要重新实现
    
  3. 社区:
    Hugging Face + PyTorch + TensorFlow 都默认 NVIDIA
    切换到 AMD 需要 "just works" 而不是 "需要改代码"

  4. 人才:
    全世界的 AI 工程师都会 CUDA
    会 ROCm 或者 Triton 的少一个数量级

  5. 云:
    AWS/Azure/GCP 上最好的 GPU 全是 NVIDIA
    自研芯片在自家云上，跨云不可用

NVIDIA 的真正风险:
  不是竞争对手的芯片更好
  而是大客户 (Google, Microsoft) 自研芯片
  内部工作负载迁移到自研芯片
  长期: 失去云的训练市场
  短期 (2026): NVIDIA 仍然 >80% 训练芯片份额
```

---

## 二、AMD 能否挑战 NVIDIA？

### 2.1 ROCm 生态的现状

```
ROCm 6.x 的进展 (2026):

  好处:
    主要的训练框架已支持 ROCm
    PyTorch ROCm 版本: 功能基本持平 CUDA
    MI400X 的 FP8 训练性能接近 H200
    
  仍然存在的问题:
    第三方库支持不完整 (Flash Attention 等)
    大集群训练稳定性不如 CUDA (经过 10 万卡验证 vs ~1 万卡)
    部署工具链不够成熟 (模型需要额外测试)
    文档和 Debug 工具不如 NVIDIA
    
  社区工作流:
    Docker 运行 ROCm: 常用但有限制
    本地 AMD 卡测试: 开发者稀缺
    "It runs on CUDA, should also work on ROCm" → 经常不

MI400X vs B300 (性能对比, 2026):

  性能指标:    MI400X     B300
  FP8 TFLOPS:  ~2,000    ~3,000
  HBM 容量:    256 GB     288 GB
  HBM 带宽:    4.0 TB/s   8.0 TB/s
  TDP:         700W       1000W
  NVLink 等效: Infinity  ~NVLink 5 水平
  价格:        ~$25,000   ~$35,000
  性价比:      获胜        稍弱
  生态成熟度:   6/10       10/10
```

### 2.2 AMD 的战略困境

```
AMD 需要解决的问题:

  1. 生态鸡和蛋:
    开发者不愿意为 AMD 开发因为用户少
    用户不愿意用 AMD 因为开发者生态少
    
  2. 切换成本:
    训练基础设施围绕 NVIDIA 搭建
    换 AMD 需要重做: NCCL→RCCL, cuDNN→MIOpen, 自定义 kernel
  
  3. 大客户策略:
    AMD 只能卖芯片
    NVIDIA 提供完整解决方案 (DGX + InfiniBand + Base Command)
    大客户 (OpenAI, Meta) 需要的是方案，不是芯片

  4. 微软的摇摆:
    AMD 最大的潜在客户 → 但微软在推自研 Maia 芯片
    如果微软推自研，AMD 失去最大的推理市场
```

---

## 三、自研芯片

### 3.1 Google TPU 的独特之路

```
TPU 架构 vs GPU 架构:

  GPU: 通用并行计算 → 适合大部分 AI 任务
  TPU: 矩阵乘法专用 → 比 GPU 效率更高 (特定场景)
  
  TPU v7 (2025-2026):
    3nm 制程
    MXU (Matrix Multiply Unit): 每个 chip 4 个
    每个 MXU 128×128 systolic array
    SparseCore: 专门做稀疏计算的加速器
    
  Pod 拓扑:
    一个 Pod 8960 TPU v7
    ICI 互联: 1600 GB/s per link
    3D Torus 拓扑

TPU 的软肋:
  只对特定工作负载优
  Custom kernel 难写 (Pallas/Triton 兼容性不如 CUDA)
  训练 PyTorch 需要特殊适配
  Google 自己的模型用 TPU 没问题，但外部用户少
```

### 3.2 AWS Trainium 的渐进之路

```
Trainium 3 (2025-2026):

  制程: 3nm
  性能: ~1.5× H200 per chip
  互联: EFA (Elastic Fabric Adapter)
  关键: NeuronLink 节点内互联
  
Neuron 生态:
  PyTorch Neuron: AWS 维护的 PyTorch fork
  问题: 新功能滞后 (Flash Attention 等)
  解决方案: AWS 出钱大模型团队适配
  
AWS 的策略:
  内部推理 (Alexa, 推荐系统): Trailium, 成本低
  外部客户: 主要提供 NVIDIA
  不能只用自己的芯片 (客户需要用 CUDA)

Trainium 的挑战:
  Neuron 与 CUDA API 差异较大
  缺乏 Flash Attention 之类的高级优化
  大客户只信任 NVIDIA
```

### 3.3 自研芯片的经济学

```
为什么超大规模云厂要自研?

  1. 成本: NVIDIA 的利润 >80%
     自研芯片成本降低 40-60%
     Google 自研 TPU 节省数十亿美元

  2. 控制:
     NVIDIA 的 roadmap 不透明
     自己控制硬件与软件的协同设计
     TPU 与 TensorFlow/JAX 深度整合

  3. 差异化:
     自己的模型运行在自己的芯片上
     Google: Gemini 在 TPU 上训练
     Meta: 不想被 NVIDIA 锁定 (但训练还在用 NVIDIA)

  自研并不容易:
    开发成本: 5 亿美元 + 3-5 年
    晶圆产能: 台积电 3nm 产能争夺
    生态: 需要独立的软件栈
    规模效应: 只有超大规模云厂能摊平成本
```

---

## 四、国产芯片

### 4.1 美国出口管制的影响

```
2019: 华为被加入实体清单
2022: 限制 A100/H100 对华出口
2023: 进一步限制 A800/H800, 缩小性能阈值
2024-2026: 持续收紧, 性能阈值几乎为零

管制带来的后果:

  1. 中国 AI 算力受限:
    训练成本是美国的 2-3×
    推理效率低 30-50%
    模型规模受限 ~300B (vs 美国的 1T+)

  2. 国产芯片快速发展:
    华为 Ascend 系列从 7nm 到未来的 5nm (受限)
    芯片设计追上来了，但制造被卡在 7nm
    
  3. 替代生态:
    CANN (华为 AI 计算框架): 兼容 CUDA API
    昇思 MindSpore: 国内框架
    PyTorch 华为适配: 社区维护

  4. 突破点:
    Chiplet: 用成熟制程拼出高性能
    CXL: 内存池化弥补 HBM 限制
    先进封装: 3D 堆叠提升性能
```

### 4.2 华为 Ascend 的崛起

```
昇腾 910C (2026 年):

  制程: 7nm (受美国限制)
  性能: 接近 A100 (某些场景), 远低于 H200 (泛化)
  HBM: 受限 (无法采购最先进的 HBM3E)
  Die: 用 Chiplet 堆叠弥补单体性能

CANN (华为 AI 计算框架):

  目标: CUDA 兼容
  OKAPI: 自动将 CUDA kernel 翻译到华为硬件
  实际: 简单 kernel 兼容好, 复杂 kernel 需重写
  开发者: 中国 AI 工程师的标配

MindSpore (昇思框架):

  华为自家的 PyTorch 替代品
  模型：在昇腾上适配的模型数量快速增长
  社区: 国内活跃，国外几乎没有
  问题: 生态迁移的成本让大团队犹豫
```

### 4.3 其他国产厂商

```
寒武纪 MLU590:
  定位: AI 推理专用
  性能: 对标 A100 推理 (2× 性价比)
  问题: 软件生态弱，主要在国内政企市场

海光 DCU K100:
  架构: x86 兼容 (海光是 AMD Zen 架构授权)
  性能: 每芯片 ~A100 的 60%
  优势: x86 兼容性最好 (无需改 CUDA, 直接跑)
  劣势: 总带宽不如 NVIDIA

国产芯片的瓶颈:
  制造: 只有 7nm, 无法获得 3nm
  HBM: 被韩国/美国控制, 拿不到最新 HBM3E
  EDA: 芯片设计工具受管制
  产能: 台积电不给做, 中芯国际产能不足

市场现状:
  华为: 国内大厂首选 (需要的适配工作最多但性能最好)
  寒武纪: 政企市场推理 (人脸识别, 安防)
  海光: 边缘/信创市场
  GPU 互换方案: 部分公司用通用 GPU 方案 (景嘉微)
  
  2026 年: 
    国产 AI 芯片总份额约占中国市场的 ~40%
    但性能和生态仍无法与 NVIDIA 竞争
```

---

## 五、AI 计算的未来

### 5.1 算力的阶层分化

```
AI 算力正在成为新的权力象征:

  Tier 1: 国家级算力 (US, CN, EU)
    拥有 100K+ GPU 集群
    训练万亿级以上模型
    核聚变级别的电力需求

  Tier 2: 公司级算力 (OpenAI, Meta, Google, 微软)
    拥有 10K-100K GPU
    训练千亿级模型
    被 Tier 1 的出口管制影响

  Tier 3: 从业者
    使用云服务 (HGX, DGX Cloud)
    API 调用, 微调
    完全依赖 Tier 1 和 2 的算力

算力的未来趋势:

  1. 算力集中化: 少数超大规模集群
  2. 电力成为瓶颈: 一个十万卡集群需要 150MW
  3. 芯片供给: TSMC/A 的 3nm 产能决定一切
  4. 推理 > 训练: 推理占 60%+ 的算力消耗
  5. 边缘算力: 手机/PCLaptops 端侧模型兴起
```

### 5.2 未来的技术路线

```
2026-2030 年 AI 芯片技术趋势:

1. 存算一体:
   当前: 计算和存储分离 (冯诺依曼瓶颈)
   未来: 在存储器内完成矩阵计算
   代表: Mythic, Syntiant

2. 光学互联:
   当前: 电互联 (功耗高, 带宽受限)
   未来: 光互联 (低功耗, 极低延迟)
   NVIDIA 正在探索, 预计 2028+

3. 3D 堆叠:
   当前: HBM 3D 堆叠 (12-Hi)
   未来: 逻辑芯片 3D 堆叠
   三星/Micron 正在开发

4. 模拟计算:
   利用物理定律 (电流/电压) 做矩阵乘法
   能效: 比数字计算高 100-1000×
   挑战: 精度、噪声、规模化

5. 量子 AI:
   Google Willow: 量子纠错里程碑
   但: 距离实用的量子 AI 还需 10-20 年
   短期 (2026): 不影响 AI 芯片格局
```
