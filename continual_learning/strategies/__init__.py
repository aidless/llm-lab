"""持续学习策略集合。

按前沿研究_08 的实证有效性排序：
1. 蒸馏（最有效）
2. LoRA（参数高效替代路径）
3. EWC（小型任务有效）
4. 回放（需结合蒸馏）
5. Model Soups（无训练合并）

扩展策略（2026-11-01 新增，来自前沿论文搜索）：
- MatrixLoRA：统一矩阵分解 LoRA（合并 OPLoRA + CURLoRA）
- SR-ML：序列级路由 + 混合低秩专家（2025）
- Bayesian PEFT：Laplace 近似增强 EWC（2024）
- mix-cd：附带损伤采样增强 Replay（NAACL 2025）
"""

from .ewc import EWCTrainingStrategy
from .lora import LoRAStrategy
from .distillation import DistillationStrategy
from .replay import ReplayStrategy
from .model_soups import ModelSoupsMerger
from .composite import CompositeStrategy
from .matrix_lora import (
    MatrixLoRAStrategy,
    MatrixLoRAModel,
    MatrixLoRALayer,
    svd_projection,
    cur_decomposition,
)
from .sr_ml import SRMLStrategy, SRMLModel, LowRankExpert, SequenceRouter

# 向后兼容：OPLoRA/CURLoRA 已合并为 MatrixLoRA，旧导入重定向到统一接口
OPLoRAStrategy = MatrixLoRAStrategy
OPLoRALayer = MatrixLoRALayer
OPLoRAModel = MatrixLoRAModel
CURLoRAStrategy = MatrixLoRAStrategy
CURLoRALayer = MatrixLoRALayer
CURLoRAModel = MatrixLoRAModel

__all__ = [
    "EWCTrainingStrategy",
    "LoRAStrategy",
    "DistillationStrategy",
    "ReplayStrategy",
    "ModelSoupsMerger",
    "CompositeStrategy",
    "MatrixLoRAStrategy",
    "MatrixLoRAModel",
    "MatrixLoRALayer",
    "svd_projection",
    "cur_decomposition",
    "SRMLStrategy",
    "SRMLModel",
    "LowRankExpert",
    "SequenceRouter",
    # 向后兼容导出
    "OPLoRAStrategy",
    "OPLoRALayer",
    "OPLoRAModel",
    "CURLoRAStrategy",
    "CURLoRALayer",
    "CURLoRAModel",
]
