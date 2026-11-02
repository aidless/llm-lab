"""持续学习框架公共 API。

基于前沿研究_08 的实证成果，实现可运行的持续学习框架。
核心能力：
- 多任务增量训练（顺序任务学习）
- 知识保留策略（EWC / 蒸馏 / 回放 / LoRA）
- 模型合并（Model Soups）
- 评估协议（遗忘 / 前向迁移 / 整体性能）

扩展策略（2026-11-01，来自前沿论文搜索）：
- MatrixLoRA（统一矩阵分解 LoRA，合并 OPLoRA + CURLoRA）
- SR-ML（2025）：序列级路由 + 混合低秩专家
"""

from .base import Task, ContinualStrategy, StrategyConfig
from .metrics import CLMetrics
from .orchestrator import ContinualOrchestrator
from .strategies import (
    EWCTrainingStrategy,
    DistillationStrategy,
    ReplayStrategy,
    LoRAStrategy,
    ModelSoupsMerger,
    MatrixLoRAStrategy,
    SRMLStrategy,
)

try:
    from .strategies.composite import CompositeStrategy
except ImportError:
    CompositeStrategy = None  # 兼容：组合策略为可选

# 向后兼容：OPLoRA/CURLoRA 已合并为 MatrixLoRA，重定向到统一接口
OPLoRAStrategy = MatrixLoRAStrategy
CURLoRAStrategy = MatrixLoRAStrategy

__all__ = [
    "Task",
    "ContinualStrategy",
    "StrategyConfig",
    "CLMetrics",
    "ContinualOrchestrator",
    "EWCTrainingStrategy",
    "DistillationStrategy",
    "ReplayStrategy",
    "LoRAStrategy",
    "ModelSoupsMerger",
    "CompositeStrategy",
    "MatrixLoRAStrategy",
    "SRMLStrategy",
    "OPLoRAStrategy",
    "CURLoRAStrategy",
]

__version__ = "0.3.1"
