# Continual Learning Framework — 架构设计

> 持续学习与数据效率的工程化落地
> 基于前沿研究_08 的实证成果，构建可运行的持续学习框架

## 1. 设计目标

本框架解决的核心问题：**如何在 LLM 持续学习过程中同时保留旧知识（稳定性）与学习新知识（塑性）**，并量化评估遗忘与迁移。

基于前沿研究_08 的关键发现：
- EWC（重要性加权正则化）在小型任务上有效，LLM 规模下失效
- 知识蒸馏是 LLM 规模下最有效的知识保留手段
- LoRA（低秩适配）提供参数高效的增量学习路径
- Model Soups（权重平均）提供无训练的模型能力组合
- 评估必须使用连续度量（Emergent Mirage 的教训）

## 2. 核心架构

```
continual_learning/
├── __init__.py              # 公共 API
├── base.py                  # 抽象基类：Task、Strategy、Evaluator
├── strategies/
│   ├── __init__.py
│   ├── ewc.py               # EWC 正则化策略（Fisher 重要性）
│   ├── lora.py              # LoRA 低秩适配策略
│   ├── distillation.py      # 知识蒸馏策略（Logit/Feature）
│   ├── replay.py            # 数据回放策略
│   └── model_soups.py       # 模型合并（权重平均）
├── metrics.py               # 评估协议：遗忘/前向迁移/整体性能
├── orchestrator.py          # 编排器：顺序任务训练
├── cli.py                   # 命令行入口
└── README.md                # 使用说明
```

## 3. 数据流

```
Task 1 ──► Strategy ──► Model ──► Eval ──► Metrics
Task 2 ──► Strategy ──► Model ──► Eval ──► Metrics
...
Task N ──► Strategy ──► Model ──► Eval ──► Metrics
                                  │
                                  └──► Model Soup（合并）
```

## 4. 关键接口

```python
class ContinualStrategy(ABC):
    """持续学习策略基类"""
    def train(self, model, task, optimizer): ...
    def compute_loss(self, model, batch, task): ...
    def after_task(self, model, task): ...  # 任务后处理（如更新 Fisher）

class Task:
    """任务定义：数据加载器 + 评估器"""
    dataloader: DataLoader
    eval_dataloader: DataLoader
    name: str

class Metrics:
    """评估协议：遗忘 / 前向迁移 / 整体性能"""
    def forgetting(self, task_id): ...
    def forward_transfer(self, task_id): ...
    def overall_performance(self): ...
```

## 5. 技术选型

- **PyTorch 2.14** — 深度学习核心（已验证可用）
- **NumPy 2.5** — 数值计算（已验证可用）
- **纯 Python + PyTorch** — 无额外重型依赖，开箱即用

## 6. 实现优先级

1. **核心策略**：EWC、蒸馏、LoRA、回放、Model Soups（按前沿研究_08 的实证有效性排序）
2. **评估协议**：遗忘/前向迁移/整体性能（连续度量）
3. **编排器**：顺序任务训练 + 策略注入
4. **CLI**：命令行入口，支持快速实验
