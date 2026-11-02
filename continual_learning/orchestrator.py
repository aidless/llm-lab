"""持续学习编排器。

按顺序执行多个任务，并记录评估指标。
支持策略注入（EWC / 蒸馏 / 回放 / LoRA）。
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .base import ContinualStrategy, Task
from .metrics import CLMetrics


class ContinualOrchestrator:
    """持续学习编排器。

    用法：
        orchestrator = ContinualOrchestrator(model, strategy)
        orchestrator.add_task(task1)
        orchestrator.add_task(task2)
        results = orchestrator.run(epochs=10)
    """

    def __init__(
        self,
        model: nn.Module,
        strategy: Optional[ContinualStrategy] = None,
        device: str = "cpu",
    ) -> None:
        self.model = model
        self.strategy = strategy
        self.device = torch.device(device)
        self.tasks: list[Task] = []
        self.metrics = CLMetrics()
        self.model.to(self.device)

    def add_task(self, task: Task) -> None:
        """添加一个任务。"""
        self.tasks.append(task)

    def run(
        self,
        epochs: int = 10,
        learning_rate: float = 0.001,
        log_interval: int = 10,
    ) -> CLMetrics:
        """按顺序训练所有任务。

        Args:
            epochs: 每个任务的训练轮数。
            learning_rate: 学习率。
            log_interval: 日志打印间隔（batch 数）。

        Returns:
            评估指标。
        """
        for task_idx, task in enumerate(self.tasks):
            print(f"\n=== 训练任务 {task_idx}: {task.name} ===")

            # 策略钩子：任务开始前
            if self.strategy:
                self.strategy.before_task(self.model, task)

            # 训练当前任务
            self._train_task(task, epochs, learning_rate, log_interval)

            # 策略钩子：任务结束后（如更新 Fisher）
            if self.strategy:
                self.strategy.after_task(self.model, task)

            # 评估当前任务在所有任务上的性能
            self._evaluate_all(task_idx)

        return self.metrics

    def _train_task(
        self,
        task: Task,
        epochs: int,
        learning_rate: float,
        log_interval: int,
    ) -> None:
        """训练单个任务。"""
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = torch.nn.CrossEntropyLoss()

        for epoch in range(epochs):
            total_loss = 0.0
            num_batches = 0

            for batch_idx, batch in enumerate(task.dataloader):
                # 移动到设备
                batch = self._to_device(batch)
                inputs, labels = batch

                optimizer.zero_grad()
                outputs = self.model(inputs)
                base_loss = criterion(outputs, labels)

                # 策略注入：计算总损失
                if self.strategy:
                    total = self.strategy.compute_loss(
                        self.model, batch, base_loss, task
                    )
                else:
                    total = base_loss

                total.backward()
                optimizer.step()

                total_loss += total.item()
                num_batches += 1

                if (batch_idx + 1) % log_interval == 0:
                    avg_loss = total_loss / num_batches
                    print(
                        f"  Epoch {epoch+1}/{epochs}, "
                        f"Batch {batch_idx+1}/{len(task.dataloader)}, "
                        f"Loss: {avg_loss:.4f}"
                    )

            avg_epoch_loss = total_loss / max(1, num_batches)
            print(f"  Epoch {epoch+1}/{epochs} 完成，平均 Loss: {avg_epoch_loss:.4f}")

    def _evaluate_all(self, current_task_idx: int) -> None:
        """评估当前模型在所有任务上的性能。"""
        self.model.eval()
        for task_idx, task in enumerate(self.tasks):
            if task_idx > current_task_idx:
                break  # 只评估已训练的任务

            # 使用任务的评估函数（默认 accuracy）
            if task.eval_fn:
                accuracy = task.eval_fn(self.model, task.eval_dataloader)
            else:
                accuracy = self._default_eval(self.model, task.eval_dataloader)

            self.metrics.record(current_task_idx, task_idx, task.name, accuracy)
            print(f"  Task {task_idx} ({task.name}) accuracy: {accuracy:.4f}")

    def _default_eval(
        self, model: nn.Module, dataloader: torch.utils.data.DataLoader
    ) -> float:
        """默认评估函数：accuracy。"""
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in dataloader:
                batch = self._to_device(batch)
                inputs, labels = batch
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        return correct / max(1, total) * 100.0

    def _to_device(self, batch):
        """将批次数据移动到设备。"""
        if isinstance(batch, (tuple, list)):
            return tuple(
                item.to(self.device) if isinstance(item, torch.Tensor) else item
                for item in batch
            )
        return batch
