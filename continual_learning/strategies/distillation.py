"""知识蒸馏策略（Distillation Strategy）。

基于 Jin et al. (2022)（arXiv:2110.08534，NAACL 2022）：
蒸馏是 LLM 规模下最有效的知识保留手段。

关键实证数据：
- Logit 蒸馏 / SEED-Logit 蒸馏在 D1/D2 域下游 F1 至少提升 1.0 个百分点
- 蒸馏甚至能超越多任务学习（MTL）——课程效应
- 过度蒸馏降低塑性（D3/D4 域无效）

本实现支持三种蒸馏变体：
- Logit：匹配教师和学生输出 logits（KL 散度）
- Feature：匹配教师和学生中间层特征（MSE）
"""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn

from ..base import ContinualStrategy, StrategyConfig, Task


class DistillationStrategy(ContinualStrategy):
    """知识蒸馏策略。

    使用旧模型作为教师，约束新模型的输出分布，从而在训练新任务时保留旧任务行为。
    """

    def __init__(
        self,
        teacher: Optional[nn.Module] = None,
        temperature: float = 2.0,
        alpha: float = 0.5,
        distillation_type: str = "logit",
        config: Optional[StrategyConfig] = None,
    ) -> None:
        super().__init__(config)
        self.teacher = teacher
        self.temperature = temperature
        self.alpha = alpha  # 蒸馏损失权重
        self.distillation_type = distillation_type  # "logit" | "feature"

    def name(self) -> str:
        return f"Distillation({self.distillation_type})"

    def set_teacher(self, teacher: nn.Module) -> None:
        """设置教师模型（通常是上一任务训练完的模型）。"""
        self.teacher = teacher
        self.teacher.eval()

    def compute_loss(
        self,
        model: nn.Module,
        batch: Any,
        base_loss: torch.Tensor,
        task: Task,
    ) -> torch.Tensor:
        """在基础损失上加入蒸馏损失。"""
        if self.teacher is None:
            return base_loss

        # 提取输入
        inputs = batch[0] if isinstance(batch, (tuple, list)) else batch

        # 教师输出
        with torch.no_grad():
            teacher_outputs = self.teacher(inputs)

        # 学生输出
        student_outputs = model(inputs)

        if self.distillation_type == "logit":
            dist_loss = self._logit_distillation(teacher_outputs, student_outputs)
        elif self.distillation_type == "feature":
            dist_loss = self._feature_distillation(teacher_outputs, student_outputs)
        else:
            raise ValueError(f"Unknown distillation type: {self.distillation_type}")

        return base_loss + self.alpha * dist_loss

    def _logit_distillation(
        self, teacher_outputs: torch.Tensor, student_outputs: torch.Tensor
    ) -> torch.Tensor:
        """Logit 蒸馏：使用 KL 散度匹配教师和学生输出分布。

        使用温度缩放（Hinton et al., 2015）：
            L_KD = KL(softmax(teacher_logits / T) || softmax(student_logits / T)) * T^2
        """
        teacher_soft = torch.nn.functional.softmax(
            teacher_outputs / self.temperature, dim=-1
        )
        student_log_soft = torch.nn.functional.log_softmax(
            student_outputs / self.temperature, dim=-1
        )
        kl_div = torch.nn.functional.kl_div(
            student_log_soft, teacher_soft, reduction="batchmean"
        )
        return kl_div * (self.temperature**2)

    def _feature_distillation(
        self, teacher_features: torch.Tensor, student_features: torch.Tensor
    ) -> torch.Tensor:
        """特征蒸馏：使用 MSE 匹配教师和学生中间层特征。"""
        return torch.nn.functional.mse_loss(student_features, teacher_features)

    def after_task(self, model: nn.Module, task: Task) -> None:
        """任务结束后，将当前模型保存为教师（用于下一个任务的蒸馏）。"""
        # 深拷贝当前模型作为教师
        import copy

        self.teacher = copy.deepcopy(model)
        self.teacher.eval()

    def state_dict(self) -> dict[str, Any]:
        return {
            "temperature": self.temperature,
            "alpha": self.alpha,
            "distillation_type": self.distillation_type,
        }
