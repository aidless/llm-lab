"""评估协议：遗忘 / 前向迁移 / 整体性能。

基于前沿研究_08 的关键发现，使用连续度量（而非 accuracy 这类不连续度量）
来避免 Emergent Mirage 式的度量伪影。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class CLMetrics:
    """持续学习评估指标。

    Attributes:
        accuracies: 二维数组 accuracies[i][j] = 模型在任务 j 上训练完任务 i 后的性能。
        task_names: 任务名称列表。
    """

    accuracies: list[list[float]] = field(default_factory=list)
    task_names: list[str] = field(default_factory=list)

    def record(self, current_task_idx: int, eval_task_idx: int, task_name: str, accuracy: float) -> None:
        """记录模型在指定任务上的性能。

        Args:
            current_task_idx: 刚完成训练的任务索引（决定行）。
            eval_task_idx: 被评估的任务索引（决定列）。
            task_name: 被评估任务名称。
            accuracy: 性能指标。
        """
        # 确保二维数组尺寸足够
        while len(self.accuracies) <= current_task_idx:
            self.accuracies.append([])
        # 确保该行包含之前所有任务的性能记录
        while len(self.accuracies[current_task_idx]) < eval_task_idx:
            self.accuracies[current_task_idx].append(float("nan"))
        self.accuracies[current_task_idx].append(accuracy)
        if len(self.task_names) <= eval_task_idx:
            self.task_names.append(task_name)

    def forgetting(self, task_idx: int) -> float:
        """遗忘度（Forgetting）：任务 j 训练完成时的性能 - 最终性能。

        正值表示遗忘（性能下降），负值表示正向迁移（性能提升）。

        Args:
            task_idx: 任务索引。

        Returns:
            遗忘度。
        """
        if task_idx >= len(self.accuracies) or len(self.accuracies[task_idx]) <= task_idx:
            return float("nan")
        # 任务训练完成时的性能（对角线元素）
        best = self.accuracies[task_idx][task_idx]
        # 最终性能（最后一行）
        final = self.accuracies[-1][task_idx] if len(self.accuracies) > 0 else float("nan")
        return best - final

    def average_forgetting(self) -> float:
        """平均遗忘度（所有任务）。"""
        if not self.task_names:
            return float("nan")
        forgets = [self.forgetting(i) for i in range(len(self.task_names))]
        valid = [f for f in forgets if not np.isnan(f)]
        return float(np.mean(valid)) if valid else float("nan")

    def forward_transfer(self, task_idx: int) -> float:
        """前向迁移（Forward Transfer）：模型在任务 j 上的最终性能 - 单独训练基线。

        注意：这里简化为「最终性能 - 任务完成时的性能」，近似前向迁移。
        正值表示前向迁移（新任务学习受益于旧任务），负值表示干扰。

        Args:
            task_idx: 任务索引。

        Returns:
            前向迁移量。
        """
        if task_idx >= len(self.accuracies) or len(self.accuracies[task_idx]) <= task_idx:
            return float("nan")
        # 最终性能
        final = self.accuracies[-1][task_idx] if len(self.accuracies) > 0 else float("nan")
        # 任务完成时的性能（对角线）
        diag = self.accuracies[task_idx][task_idx]
        return final - diag

    def average_forward_transfer(self) -> float:
        """平均前向迁移。"""
        if not self.task_names:
            return float("nan")
        fts = [self.forward_transfer(i) for i in range(len(self.task_names))]
        valid = [f for f in fts if not np.isnan(f)]
        return float(np.mean(valid)) if valid else float("nan")

    def overall_performance(self) -> float:
        """整体性能（Overall Performance）：所有任务最终性能的平均。"""
        if not self.task_names or not self.accuracies:
            return float("nan")
        final_row = self.accuracies[-1]
        valid = [v for v in final_row if not np.isnan(v)]
        return float(np.mean(valid)) if valid else float("nan")

    def to_dict(self) -> dict[str, float]:
        """导出全部指标为字典。"""
        return {
            "overall_performance": self.overall_performance(),
            "average_forgetting": self.average_forgetting(),
            "average_forward_transfer": self.average_forward_transfer(),
        }
