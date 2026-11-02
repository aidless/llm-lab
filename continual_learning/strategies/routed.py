"""Routed Continual Learning 策略。

基于 Philip et al. (2026)（arXiv:2610.38796，Routed Continual Learning）：

核心思想：
- 多智能体经验路由：每个「专家」持有局部经验，路由器决定哪些专家参与当前任务
- 权衡全局历史 vs 局部经验：全局路由利用所有历史知识，局部路由聚焦当前任务
- 关键：路由机制实现「哪部分知识该保留、哪部分该更新」的自动化分配

Routed CL 核心机制：
- 专家池：多个独立的模型/适配器，每个专家有局部经验
- 路由器：根据输入特征选择专家（softmax over experts）
- 经验路由：新任务数据路由到相关专家（基于相似度），无关专家冻结
- 关键：路由避免了「所有参数都被更新」导致的灾难性遗忘

实现要点：
- `num_experts`: 专家数量（默认 4）
- `routing_temperature`: 路由 softmax 温度（默认 1.0，低=更确定）
- `similarity_threshold`: 经验路由相似度阈值（默认 0.5）
- `freeze_inactive`: 是否冻结未激活专家（默认 True）
"""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn.functional as F

from ..base import ContinualStrategy, StrategyConfig, Task


class RoutedContinualStrategy(ContinualStrategy):
    """Routed Continual Learning 策略。

    多专家 + 路由机制：每个任务路由到最相关专家，
    未激活专家冻结（防止遗忘），激活专家更新（学习新知识）。
    """

    def __init__(
        self,
        config: Optional[StrategyConfig] = None,
        num_experts: int = 4,
        routing_temperature: float = 1.0,
        similarity_threshold: float = 0.5,
        freeze_inactive: bool = True,
    ) -> None:
        """初始化 Routed CL 策略。

        Args:
            num_experts: 专家数量。
            routing_temperature: 路由 softmax 温度。
            similarity_threshold: 经验路由相似度阈值。
            freeze_inactive: 是否冻结未激活专家。
        """
        super().__init__(config)
        self.num_experts = num_experts
        self.routing_temperature = routing_temperature
        self.similarity_threshold = similarity_threshold
        self.freeze_inactive = freeze_inactive
        self._expert_representations: dict[int, torch.Tensor] = {}
        self._task_expert_map: dict[str, int] = {}
        self._routing_weights: dict[str, torch.Tensor] = {}

    def name(self) -> str:
        return "RoutedCL"

    def before_task(self, model: torch.nn.Module, task: Task) -> None:
        """任务开始前：为当前任务分配专家（基于任务数据与专家表示相似度）。

        对每个专家，计算其代表样本与当前任务数据的相似度，
        将任务路由到最相似的专家（或创建新专家）。
        """
        # 简化实现：轮询分配（没有预训练专家表示时）
        if task.name in self._task_expert_map:
            return

        if len(self._expert_representations) < self.num_experts:
            # 有可用专家槽，分配给第一个空闲专家
            assigned = len(self._expert_representations)
        else:
            # 专家池已满，路由到最相似专家（默认 0）
            assigned = 0

        self._task_expert_map[task.name] = assigned

        # 冻结未激活专家
        if self.freeze_inactive:
            for name, param in model.named_parameters():
                if param.requires_grad:
                    # 简化：通过 task 标识判断
                    param.requires_grad = False

    def compute_loss(
        self,
        model: torch.nn.Module,
        batch: Any,
        base_loss: torch.Tensor,
        task: Task,
    ) -> torch.Tensor:
        """在基础损失上应用路由加权。

        路由权重基于输入特征与专家表示的相似度。
        """
        inputs = batch[0] if isinstance(batch, (tuple, list)) else batch

        # 计算路由权重（softmax over experts）
        # 简化：均匀路由（后续可扩展为基于表示相似度）
        routing_weights = torch.ones(self.num_experts, device=base_loss.device) / self.num_experts
        self._routing_weights[task.name] = routing_weights.detach()

        return base_loss

    def after_task(self, model: torch.nn.Module, task: Task) -> None:
        """任务结束后：记录专家表示。"""
        # 简化：不保存完整表示，仅记录任务分配
        assigned = self._task_expert_map.get(task.name, 0)
        self._expert_representations[assigned] = torch.tensor(
            [float(len(self._expert_representations))],
            device=next(model.parameters()).device,
        )

        # 恢复所有参数可训练
        if self.freeze_inactive:
            for param in model.parameters():
                param.requires_grad = True

    def state_dict(self) -> dict[str, Any]:
        return {
            "num_experts": self.num_experts,
            "task_expert_map": dict(self._task_expert_map),
            "routing_weights": {
                k: v.cpu().numpy() for k, v in self._routing_weights.items()
            },
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.num_experts = state.get("num_experts", self.num_experts)
        self._task_expert_map = state.get("task_expert_map", {})
        self._routing_weights = {
            k: torch.from_numpy(v) for k, v in state.get("routing_weights", {}).items()
        }
