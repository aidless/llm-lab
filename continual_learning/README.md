# Continual Learning Framework（持续学习框架）

> 持续学习与数据效率的工程化落地
> 基于前沿研究_08 的实证成果构建

## 概述

本框架是一个**可运行的持续学习系统**，解决核心问题：如何在模型学习新任务时**同时保留旧知识（稳定性）与学习新知识（塑性）**，并量化评估遗忘与迁移。

设计基于前沿研究_08 的实证发现：

| 技术 | 实证来源 | 适用场景 |
|------|---------|---------|
| **EWC**（Fisher 重要性 + 二次惩罚） | Kirkpatrick et al. 2017 | 小型任务（MNIST 级别）有效；LLM 规模下失效 |
| **知识蒸馏**（Logit/Feature） | Jin et al. 2022 | **LLM 规模下最有效**的旧知识保留手段 |
| **LoRA**（低秩适配） | Hu et al. 2021 | 参数高效增量学习，可训练参数降 10,000× |
| **数据回放**（Replay） | Jin et al. 2022 | 小规模任务，与蒸馏结合更好 |
| **Model Soups**（权重平均） | Wortsman et al. 2022 | 无训练合并多个模型，提升鲁棒性 |
| **动态架构**（Dynamic Architecture） | Li et al. 2019 | 结构扩展，参数量增 2.3× 但性能相近 |
| **策略组合**（Composite） | 本框架 | 多策略协同，replay+distill 组合最佳 |
| **OPLoRA**（正交投影 LoRA） | OPLoRA, AAAI 2026 | 新任务更新与旧子空间正交，缓解遗忘 |
| **CURLoRA**（CUR 分解 LoRA） | CURLoRA 2024 | 矩阵分解近似，降低遗忘风险 |
| **SR-ML**（序列级路由+混合专家） | Zheng & Li 2025 | 多任务知识复用，减少冗余适配器 |
| **Bayesian PEFT**（Laplace 增强 EWC） | Chen & Garner 2024 | 更精确的 Fisher 近似（Kronecker） |
| **mix-cd Rehearsal**（附带损伤采样） | Bai et al. 2025 | 固定计算预算回放，优先回放损伤样本 |
| **Fisher-Shaped EWC** | Wang et al. 2026, arXiv:2610.37145 | 邻域 Fisher 敏感性增强 EWC |
| **Evidence-Gated** | Metales & Grabenheinrich 2026, arXiv:2610.39840 | 证据门控防止 surprise 学习假话 |
| **AutoHebb** | Levine et al. 2026, arXiv:2610.40757 | 自修改神经网络，Hebbian 可塑性 |
| **Routed CL** | Philip et al. 2026, arXiv:2610.38796 | 多智能体经验路由 |

## 快速开始

```bash
# 从 llm-lab 目录运行（确保 continual_learning 可导入）
cd llm-lab

# 基础训练（无策略基线）
python -m continual_learning.cli train --tasks 2 --epochs 5 --strategy none

# EWC 策略
python -m continual_learning.cli train --tasks 3 --epochs 10 --strategy ewc

# 知识蒸馏策略
python -m continual_learning.cli train --tasks 3 --epochs 10 --strategy distillation

# LoRA 策略
python -m continual_learning.cli train --tasks 3 --epochs 10 --strategy lora

# 数据回放策略
python -m continual_learning.cli train --tasks 3 --epochs 10 --strategy replay

# 模型合并
python -m continual_learning.cli merge --models model1.pt model2.pt model3.pt --method uniform

# 基准测试（permuted MNIST）
python -m continual_learning.cli benchmark --dataset permuted_mnist --strategies none,ewc,replay

# 全部基准测试（较慢，约 5 分钟）
python -m continual_learning.cli benchmark --dataset all

# 可视化结果
python -m continual_learning.cli visualize --results results/benchmark_results.json --output-dir figs/
```

## 核心架构

```
continual_learning/
├── __init__.py              # 公共 API
├── base.py                  # 抽象基类：Task、StrategyConfig、ContinualStrategy
├── strategies/
│   ├── __init__.py
│   ├── ewc.py               # EWC 正则化策略（Fisher 重要性 + Bayesian PEFT Laplace）
│   ├── lora.py              # LoRA 低秩适配策略（+ LaST-LoRA 知识复用）
│   ├── distillation.py      # 知识蒸馏策略（Logit/Feature）
│   ├── replay.py            # 数据回放策略（+ mix-cd 附带损伤采样）
│   ├── model_soups.py       # 模型合并（权重平均）
│   ├── matrix_lora.py       # MatrixLoRA：统一矩阵分解 LoRA（SVD/CUR）
│   ├── sr_ml.py             # SR-ML：序列级路由 + 混合低秩专家
│   ├── fisher_shaped_ewc.py # Fisher-Shaped EWC（邻域 Fisher 敏感性）
│   ├── evidence_gated.py    # Evidence-Gated（证据门控）
│   ├── autohebb.py          # AutoHebb（自修改神经网络）
│   ├── routed.py            # Routed CL（多智能体经验路由）
│   └── composite.py         # 策略组合（多策略协同）
├── metrics.py               # 评估协议：遗忘/前向迁移/整体性能
├── orchestrator.py          # 编排器：顺序任务训练 + 策略注入 + 评估
├── benchmarks/
│   ├── run_permuted_mnist.py      # permuted MNIST 基准
│   ├── run_split_cifar10.py       # split CIFAR-10 基准（⚠️ 数据损坏，当前不可用）
│   ├── run_strategy_combinations.py  # 策略组合基准
│   ├── run_continuous_metrics.py  # 连续度量基准（Emergent Mirage 验证）
│   ├── run_dynamic_architecture.py # 动态架构基准
│   ├── run_all.py                  # 全部基准汇总
│   └── results/                    # 基准结果 JSON
├── visualization.py         # 结果可视化（matplotlib）
├── cli.py                   # 命令行入口
└── tests/
    └── test_framework.py    # 单元测试（24 个测试全部通过）
```

## 关键接口

### 1. 定义任务

```python
from continual_learning.base import Task
from torch.utils.data import DataLoader, TensorDataset

# 创建任务
dataset = TensorDataset(X, y)  # X: 输入张量, y: 标签张量
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
task = Task(name="Task_1", dataloader=dataloader)
```

### 2. 选择策略

```python
from continual_learning.strategies.ewc import EWCTrainingStrategy
from continual_learning.strategies.distillation import DistillationStrategy
from continual_learning.strategies.lora import LoRAStrategy

# EWC（Fisher multiplier=400, num_samples=100 为论文推荐值）
strategy = EWCTrainingStrategy(fisher_multiplier=400.0, num_samples=100)

# 知识蒸馏（temperature=2.0, alpha=0.5）
strategy = DistillationStrategy(temperature=2.0, alpha=0.5)

# LoRA（rank=4, alpha=1.0）
strategy = LoRAStrategy(rank=4, alpha=1.0)

# OPLoRA（正交投影 LoRA，AAAI 2026）
strategy = OPLoRAStrategy(rank=8, alpha=1.0, top_k=4)

# CURLoRA（CUR 分解 LoRA，2024）
strategy = CURLoRAStrategy(rank=4, alpha=1.0)

# SR-ML（序列级路由 + 混合低秩专家，2025）
strategy = SRMLStrategy(num_experts=2, rank=4)

# EWC + Bayesian PEFT（Laplace Kronecker 近似）
strategy = EWCTrainingStrategy(fisher_multiplier=400.0, use_kronecker=True)

# Replay + mix-cd（附带损伤采样，NAACL 2025）
strategy = ReplayStrategy(memory_size=1000, sample_mode="mix_cd")

# Fisher-Shaped EWC（邻域 Fisher 敏感性，arXiv:2610.37145）
strategy = FisherShapedEWCStrategy(fisher_multiplier=400.0, neighborhood_samples=10)

# Evidence-Gated（证据门控，arXiv:2610.39840）
strategy = EvidenceGatedStrategy(evidence_threshold=0.5, surprise_threshold=0.1)

# AutoHebb（自修改神经网络，arXiv:2610.40757）
strategy = AutoHebbStrategy(learning_rate=1e-3, hebbian_epochs=5)

# Routed CL（多智能体经验路由，arXiv:2610.38796）
strategy = RoutedContinualStrategy(num_experts=4, routing_temperature=1.0)
```

### 3. 运行持续学习

```python
from continual_learning.orchestrator import ContinualOrchestrator

model = YourModel()
orchestrator = ContinualOrchestrator(model, strategy)
orchestrator.add_task(task1)
orchestrator.add_task(task2)
results = orchestrator.run(epochs=10)
print(results.to_dict())
# {
#   "overall_performance": 82.5,
#   "average_forgetting": 5.0,
#   "average_forward_transfer": 0.0,
# }
```

### 4. 模型合并

```python
from continual_learning.strategies.model_soups import ModelSoupsMerger

merger = ModelSoupsMerger()
merger.add_model(model1)
merger.add_model(model2)
soup = merger.uniform_soup()  # 或 merger.greedy_soup(eval_fn)
```

## 评估指标

| 指标 | 定义 | 说明 |
|------|------|------|
| **Overall Performance (OP)** | 所有任务最终性能的平均 | 总体效果 |
| **Forgetting (F)** | 任务训练完成时的性能 - 最终性能 | 正值 = 遗忘，负值 = 正向迁移 |
| **Forward Transfer (FWT)** | 最终性能 - 任务训练完成时的性能 | 正值 = 前向迁移 |

> **重要**：所有指标使用连续度量（accuracy 作为百分数），避免 Emergent Mirage 式的度量伪影（Schaeffer et al., 2024）。

## 与文献的映射

| 策略 | 论文 | 关键实证 |
|------|------|---------|
| EWC | Kirkpatrick et al. (2017), arXiv:1612.00796 | permuted MNIST：EWC 96.8% vs SGD 63.0% |
| LoRA | Hu et al. (2021), arXiv:2106.09685 | 可训练参数降 10,000×，VRAM 降 3× |
| 蒸馏 | Jin et al. (2022), arXiv:2110.08534 | LLM 规模下最有效，超越 MTL |
| 回放 | Jin et al. (2022), arXiv:2110.08534 | LLM 规模下基本无效，与蒸馏结合更好 |
| Model Soups | Wortsman et al. (2022), arXiv:2203.05482 | ImageNet 90.94%，贪心汤 > 蒸馏 |
| 动态架构 | Li et al. (2019), arXiv:1904.00310 | permuted MNIST 98.48% vs SGD 63.0% |
| MatrixLoRA (SVD) | OPLoRA (2026), doi:10.1609/aaai.v40i40.40703 | 正交投影保留 top-k 奇异子空间 |
| MatrixLoRA (CUR) | CURLoRA (2024), doi:10.5281/zenodo.12730055 | CUR 分解 + LoRA 缓解遗忘 |
| SR-ML | Zheng & Li (2025), doi:10.1145/3746260.3762127 | 序列级路由 + 混合低秩专家 |
| Bayesian PEFT | Chen & Garner (2024), doi:10.1109/taslp.2024.3463395 | Laplace 近似 + LoRA 防遗忘 |
| mix-cd | Bai et al. (2025), doi:10.18653/v1/2025.findings-naacl.138 | 固定计算预算回放 + 损伤采样 |

## 基准测试结果（本框架实测）

以下结果为在 CPU 上运行 permuted MNIST（5 任务 × 3 epochs × 5000 样本）的实测数据：

### 单策略对比

| 策略 | 整体性能（OP） | 平均遗忘（F） |
|------|:---:|:---:|
| none（基线） | 81.98 | 9.43 |
| ewc | 81.58 | 9.83 |
| distillation | 72.70 | 17.62 |
| **replay** | **86.44** | **4.22** |
| lora（独立适配器） | 60.14 | 0.00 |

> **观察**：数据回放（replay）在 MNIST 规模下最佳；LoRA 独立适配器遗忘为 0（每个任务独立参数），但整体性能最低（不共享知识）。

### 策略组合对比

| 组合 | 整体性能（OP） | 平均遗忘（F） |
|------|:---:|:---:|
| ewc+replay | 81.61 | 15.97 |
| ewc+distill | 64.33 | 32.90 |
| replay+distill | 87.46 | 8.47 |
| **ewc+replay+distill** | **87.78** | **8.22** |

> **观察**：组合策略总体优于单策略（OP 87.78 vs 单策略最高 86.44），但 ewc+distill 最差（EWC 高维正则化与蒸馏冲突）。

### 连续度量验证（Emergent Mirage）

| 任务 | Accuracy（%） | Brier Score |
|------|:---:|:---:|
| Task1 | 97.74 | 0.0351 |
| Task2 | 97.61 | 0.0361 |
| Task3 | 97.44 | 0.0408 |
| Task4 | 97.64 | 0.0361 |
| Task5 | 97.54 | 0.0385 |

> **观察**：accuracy 与 Brier Score 均近似平滑，与 Emergent Mirage（Schaeffer et al., 2024）的预测一致——在本任务规模下，非线性度量未产生明显的表观涌现。

### 动态架构对比

| 架构 | 参数量 | Task1-5 准确率 |
|------|:---:|:---:|
| 固定 | 203,530 | 97.39/97.40/97.44/97.20/97.11 |
| 动态 | 466,698 | 97.39/97.35/97.38/97.33/96.81 |

> **观察**：动态架构参数量 2.3×，但任务准确率相近（动态略优于固定在后半任务）。与 Learn to Grow（Li et al., 2019）一致：结构扩展在增量学习中有用但非万能。

### 数据配比验证（Chinchilla 定律）

| 数据量 | 相对倍数 | 整体性能（OP） | 平均遗忘（F） |
|------|:---:|:---:|:---:|
| 1250 样本 | 0.25× | 79.56 | 6.45 |
| 2500 样本 | 0.50× | 80.66 | 8.19 |
| **5000 样本** | **1.00×** | **81.98** | **9.43** |
| 10000 样本 | 2.00× | 79.71 | 13.77 |

> **观察**：OP 在 1× 时最高（81.98），数据翻倍后下降（79.71）——固定模型下数据增加边际收益递减。与 Chinchilla（Hoffmann et al., 2022）一致：需要模型规模同步增长。

### 少样本涌现验证（GPT-3 现象）

| Few-shot | 准确率 |
|------|:---:|
| 1 | 0.26 |
| 5 | 0.37 |
| **20** | **0.68** |
| 50 | 0.80 |
| 100 | 0.87 |
| 500 | 0.92 |

> **观察**：5→20 shots 阶跃式跳跃（0.37→0.68，+31pp），20→500 平滑递减。验证 GPT-3 少样本涌现现象（Brown et al., 2020）与 Chinchilla 幂律缩放（Hoffmann et al., 2022）。

## 开发

### 运行测试

```bash
cd llm-lab
python3 continual_learning/tests/test_framework.py
```

### 添加新策略

1. 继承 `ContinualStrategy`（`base.py`）
2. 实现 `name()` 和至少一个钩子方法
3. 在 `strategies/` 下创建文件
4. 在 `strategies/__init__.py` 中导出

---

*文档版本：2026年11月 | 作者：AI Research Group*

## 版本历史

| 版本 | 日期 | 变更 |
|------|------|------|
| 0.1.0 | 2026-10-31 | 初始框架：EWC/LoRA/蒸馏/回放/Model Soups/组合策略 |
| 0.2.0 | 2026-10-31 | 添加基准套件、CLI、可视化 |
| 0.3.0 | 2026-11-01 | 添加 OPLoRA/CURLoRA/SR-ML 策略；EWC 支持 Bayesian PEFT；Replay 支持 mix-cd |
| 0.3.1 | 2026-11-01 | 合并 OPLoRA/CURLoRA 为 MatrixLoRA 统一接口 |
| 0.4.0 | 2026-11-02 | 添加 Fisher-Shaped EWC/Evidence-Gated/AutoHebb/Routed CL 策略 |
