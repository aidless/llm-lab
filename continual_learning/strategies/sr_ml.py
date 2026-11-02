"""SR-ML（Sequence-level Routing with Mixed Low-rank Experts）策略。

基于 Zheng & Li (2025)（doi:10.1145/3746260.3762127）：
序列级路由 + 混合低秩专家框架，用于持续学习。

SR-ML 的核心思想：
1. 将多个任务的知识组织为多个低秩专家（LoRA 专家）
2. 使用序列级路由（sequence-level routing）在专家间分配输入序列
3. 混合不同低秩专家，实现知识复用与遗忘缓解

关键实证数据：
- 在持续学习基准上优于单一 LoRA
- 序列级路由比 token 级路由更高效（计算成本更低）
- 混合低秩专家可复用旧任务知识，减少冗余适配器
"""

from __future__ import annotations

import math
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base import ContinualStrategy, StrategyConfig, Task


class LowRankExpert(nn.Module):
    """单个低秩专家（LoRA 模块）。"""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 4,
        alpha: float = 1.0,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha

        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.scaling = alpha / rank

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """计算专家输出。"""
        return (x @ self.lora_A.t() @ self.lora_B.t()) * self.scaling

    def merge(self) -> torch.Tensor:
        """合并为权重矩阵。"""
        return self.lora_B @ self.lora_A


class SequenceRouter(nn.Module):
    """序列级路由器。

    输入序列特征，输出各专家的注意力权重（softmax）。
    对每个输入序列生成一个路由向量（不是每个 token）。
    """

    def __init__(
        self,
        input_dim: int,
        num_experts: int,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_experts)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """计算路由权重。

        Args:
            x: 序列特征 (batch, input_dim)。

        Returns:
            路由权重 (batch, num_experts)，每行 softmax。
        """
        h = F.relu(self.fc1(x))
        logits = self.fc2(h)
        return F.softmax(logits, dim=-1)


class SRMLModel(nn.Module):
    """SR-ML 适配模型包装器。

    冻结原始模型，为每个任务分配低秩专家，使用序列级路由混合。
    """

    def __init__(
        self,
        base_model: nn.Module,
        target_modules: Optional[list[str]] = None,
        rank: int = 4,
        alpha: float = 1.0,
        num_experts: int = 4,
        max_experts: int = 8,
    ) -> None:
        super().__init__()
        self.base_model = base_model
        self.rank = rank
        self.alpha = alpha
        self.num_experts = num_experts
        self.max_experts = max_experts

        for param in base_model.parameters():
            param.requires_grad = False

        self.experts: dict[str, dict[str, LowRankExpert]] = {}
        self.routers: dict[str, SequenceRouter] = {}
        self._inject_srml(target_modules)

    def _inject_srml(self, target_modules: Optional[list[str]]) -> None:
        """初始化 SR-ML 结构。"""
        if target_modules is None:
            target_modules = []
            for name, module in self.base_model.named_modules():
                if isinstance(module, nn.Linear):
                    target_modules.append(name)

        self.target_modules = target_modules
        for name, module in self.base_model.named_modules():
            if name in target_modules and isinstance(module, nn.Linear):
                # 为每个目标层创建专家池
                experts = {
                    f"expert_{i}": LowRankExpert(
                        module.in_features, module.out_features, self.rank, self.alpha
                    )
                    for i in range(self.num_experts)
                }
                self.experts[name] = experts
                # 路由器
                router = SequenceRouter(module.in_features, self.num_experts)
                self.routers[name] = router
                setattr(self, f"srml_experts_{name.replace('.', '_')}", nn.ModuleDict(experts))
                setattr(self, f"srml_router_{name.replace('.', '_')}", router)

    def add_expert(self, layer_name: str) -> Optional[LowRankExpert]:
        """动态添加新专家（用于新任务）。"""
        if layer_name not in self.experts:
            return None
        if len(self.experts[layer_name]) >= self.max_experts:
            return None

        # 获取该层信息
        for name, module in self.base_model.named_modules():
            if name == layer_name and isinstance(module, nn.Linear):
                expert = LowRankExpert(
                    module.in_features, module.out_features, self.rank, self.alpha
                )
                expert_id = f"expert_{len(self.experts[layer_name])}"
                self.experts[layer_name][expert_id] = expert
                setattr(self, f"srml_experts_{layer_name.replace('.', '_')}", nn.ModuleDict(self.experts[layer_name]))
                return expert
        return None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播：原始模型 + SR-ML 混合专家。"""
        hooks = []

        def make_hook(layer_name: str):
            def hook_fn(module, input, output):
                # 序列级路由：对每个样本的输入特征计算路由权重
                inputs = input[0]  # (batch, seq_len, features) 或 (batch, features)
                if inputs.dim() == 3:
                    # 序列输入：取均值作为序列特征
                    seq_features = inputs.mean(dim=1)
                else:
                    seq_features = inputs

                # 路由权重
                routing = self.routers[layer_name](seq_features)  # (batch, num_experts)

                # 混合专家输出
                expert_outputs = 0
                for expert_id, expert in self.experts[layer_name].items():
                    expert_idx = int(expert_id.split("_")[1])
                    if expert_idx < routing.shape[1]:
                        contrib = expert(inputs)
                        expert_outputs = expert_outputs + routing[:, expert_idx].unsqueeze(-1) * contrib

                return output + expert_outputs
            return hook_fn

        for name, module in self.base_model.named_modules():
            if name in self.experts:
                hook = module.register_forward_hook(make_hook(name))
                hooks.append(hook)

        try:
            output = self.base_model(x)
        finally:
            for hook in hooks:
                hook.remove()

        return output

    def trainable_parameters(self) -> int:
        """计算可训练参数量。"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class SRMLStrategy(ContinualStrategy):
    """SR-ML 持续学习策略。

    每个任务分配新的低秩专家，通过序列级路由混合。
    支持动态专家添加（add_expert），实现知识复用。
    """

    def __init__(
        self,
        rank: int = 4,
        alpha: float = 1.0,
        num_experts: int = 4,
        max_experts: int = 8,
        config: Optional[StrategyConfig] = None,
    ) -> None:
        super().__init__(config)
        self.rank = rank
        self.alpha = alpha
        self.num_experts = num_experts
        self.max_experts = max_experts
        self.adapters: dict[str, SRMLModel] = {}

    def name(self) -> str:
        return "SR-ML"

    def create_adapter(self, base_model: nn.Module, task_name: str) -> SRMLModel:
        """为指定任务创建 SR-ML 适配器。"""
        adapter = SRMLModel(
            base_model,
            rank=self.rank,
            alpha=self.alpha,
            num_experts=self.num_experts,
            max_experts=self.max_experts,
        )
        self.adapters[task_name] = adapter
        return adapter

    def get_adapter(self, task_name: str) -> Optional[SRMLModel]:
        """获取指定任务的适配器。"""
        return self.adapters.get(task_name)

    def add_expert(self, task_name: str, layer_name: str) -> Optional[LowRankExpert]:
        """为指定任务添加新专家。"""
        adapter = self.adapters.get(task_name)
        if adapter is None:
            return None
        return adapter.add_expert(layer_name)

    def state_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "alpha": self.alpha,
            "num_experts": self.num_experts,
            "max_experts": self.max_experts,
            "adapters": {
                name: adapter.state_dict() for name, adapter in self.adapters.items()
            },
        }
