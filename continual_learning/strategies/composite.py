"""组合策略（Composite Strategy）。

将多个持续学习策略组合使用，以检验协同效应。
例如 EWC+Replay、EWC+Distillation、Replay+Distillation 等。

设计原则：
- 组合策略按顺序执行每个子策略的钩子（before_task / compute_loss / after_task）
- compute_loss 的附加损失是各子策略附加损失的加权和
- 子策略的权重可通过组合配置指定
"""

from __future__ import annotations

from typing import Any, Optional

import torch

from ..base import ContinualStrategy, StrategyConfig, Task


class CompositeStrategy(ContinualStrategy):
    """组合多个持续学习策略。

    Args:
        strategies: 子策略列表（按顺序执行）。
        weights: 各子策略损失权重（默认为 1.0）。
        config: 组合策略配置。
    """

    def __init__(
        self,
        strategies: list[ContinualStrategy],
        weights: Optional[list[float]] = None,
        config: Optional[StrategyConfig] = None,
    ) -> None:
        super().__init__(config)
        self.strategies = strategies
        self.weights = weights or [1.0] * len(strategies)

    def name(self) -> str:
        return "Composite(" + "+".join(s.name() for s in self.strategies) + ")"

    def before_task(self, model: torch.nn.Module, task: Task) -> None:
        """依次调用所有子策略的任务开始前钩子。"""
        for strategy in self.strategies:
            strategy.before_task(model, task)

    def compute_loss(
        self,
        model: torch.nn.Module,
        batch: Any,
        base_loss: torch.Tensor,
        task: Task,
    ) -> torch.Tensor:
        """计算组合损失：基础损失 + 加权各子策略附加损失。"""
        total_loss = base_loss
        for strategy, weight in zip(self.strategies, self.weights):
            # 每个子策略的附加损失 = 子策略计算的总损失 - 基础损失
            subtotal = strategy.compute_loss(model, batch, base_loss, task)
            # 若子策略没有附加（返回 base_loss 本身），则附加为 0
            if subtotal is not base_loss:
                total_loss = total_loss + weight * (subtotal - base_loss)
        return total_loss

    def after_task(self, model: torch.nn.Module, task: Task) -> None:
        """依次调用所有子策略的任务结束后钩子。"""
        for strategy in self.strategies:
            strategy.after_task(model, task)

    def state_dict(self) -> dict[str, Any]:
        return {
            "strategies": [s.name() for s in self.strategies],
            "weights": self.weights,
        }
