"""抽象基类与数据结构定义。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import torch
from torch.utils.data import DataLoader


@dataclass
class Task:
    """持续学习中的单个任务。

    Attributes:
        name: 任务名称（唯一标识）。
        dataloader: 训练数据加载器。
        eval_dataloader: 评估数据加载器（可选，默认为训练数据）。
        eval_fn: 评估函数，输入模型和评估数据，返回指标字典（如 accuracy）。
    """

    name: str
    dataloader: DataLoader
    eval_dataloader: Optional[DataLoader] = None
    eval_fn: Optional[Callable[[torch.nn.Module, DataLoader], dict[str, float]]] = None

    def __post_init__(self) -> None:
        if self.eval_dataloader is None:
            self.eval_dataloader = self.dataloader


@dataclass
class StrategyConfig:
    """策略配置基类。"""

    name: str = "base"
    params: dict[str, Any] = field(default_factory=dict)


class ContinualStrategy(ABC):
    """持续学习策略基类。

    所有具体策略（EWC、蒸馏、回放、LoRA）继承此基类。
    策略通过钩子函数接入训练流程：
    - before_task: 任务开始前（如重置优化器、加载记忆）
    - compute_loss: 计算总损失（基础损失 + 正则化/蒸馏损失）
    - after_task: 任务结束后（如更新 Fisher 信息）
    """

    def __init__(self, config: Optional[StrategyConfig] = None) -> None:
        self.config = config or StrategyConfig()

    @abstractmethod
    def name(self) -> str:
        """策略名称。"""

    def before_task(self, model: torch.nn.Module, task: Task) -> None:
        """任务开始前的钩子。默认不执行任何操作。"""

    def compute_loss(
        self,
        model: torch.nn.Module,
        batch: Any,
        base_loss: torch.Tensor,
        task: Task,
    ) -> torch.Tensor:
        """计算总损失。

        Args:
            model: 当前模型。
            batch: 当前批次数据。
            base_loss: 基础损失（如交叉熵）。
            task: 当前任务。

        Returns:
            总损失（基础损失 + 策略附加项）。
        """
        return base_loss

    def after_task(self, model: torch.nn.Module, task: Task) -> None:
        """任务结束后的钩子。默认不执行任何操作。"""

    def state_dict(self) -> dict[str, Any]:
        """导出策略状态（用于保存/加载）。"""
        return {}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """加载策略状态。"""
