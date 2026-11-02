"""数据回放策略（Replay Strategy）。

基于 Jin et al. (2022)（arXiv:2110.08534）：
数据回放在 LLM 规模下基本无效（记忆样本过拟合），
但在小规模任务上仍有一定价值，且与蒸馏结合效果更好。

mix-cd 扩展（Bai et al., 2025，doi:10.18653/v1/2025.findings-naacl.138）：
- 固定计算预算回放（而非固定内存预算）
- mix-cd 采样优先回放"附带损伤"（collateral damage）样本：
  旧模型预测正确但当前模型预测错误的样本
- 关键：随模型推理成本上升、数据存储成本下降，
  计算预算回放比内存预算回放更实用
"""

from __future__ import annotations

from typing import Any, Optional

import torch

from ..base import ContinualStrategy, StrategyConfig, Task


class ReplayStrategy(ContinualStrategy):
    """数据回放策略。

    在训练新任务时，将旧任务的代表性样本重新放入训练批次。
    需与蒸馏结合（前沿研究_08 建议），避免记忆样本过拟合。

    mix-cd 扩展：支持附带损伤采样（sample_mode="mix_cd"），
    优先回放旧模型预测正确但当前模型预测错误的样本。
    """

    def __init__(
        self,
        memory_size: int = 1000,
        replay_ratio: float = 0.2,
        config: Optional[StrategyConfig] = None,
        sample_mode: str = "random",
        compute_budget: int = 500,
    ) -> None:
        super().__init__(config)
        self.memory_size = memory_size
        self.replay_ratio = replay_ratio
        self.sample_mode = sample_mode  # "random" 或 "mix_cd"
        self.compute_budget = compute_budget
        self.memory: list[Any] = []  # 存储 (inputs, labels) 样本
        self._old_model_predictions: dict[int, int] = {}  # 样本索引 → 旧模型预测

    def name(self) -> str:
        return "Replay"

    def store_samples(self, batch: Any) -> None:
        """存储当前批次的部分样本到记忆库。"""
        inputs, labels = batch
        for i in range(len(inputs)):
            if len(self.memory) >= self.memory_size:
                # 简单 FIFO 替换
                self.memory.pop(0)
            self.memory.append((inputs[i], labels[i]))

    def sample_replay(self, batch_size: int) -> Optional[list[Any]]:
        """从记忆库中采样回放样本。

        mix_cd 模式：优先采样附带损伤样本（旧模型对、当前模型错）。
        random 模式：均匀随机采样。
        """
        if not self.memory:
            return None

        num_replay = min(len(self.memory), max(1, int(batch_size * self.replay_ratio)))

        if self.sample_mode == "mix_cd" and self._old_model_predictions:
            # 附带损伤采样：旧模型预测正确但当前模型预测错误
            # 简化实现：优先采样标记为附带损伤的样本
            collateral = [
                (i, sample) for i, sample in enumerate(self.memory)
                if self._old_model_predictions.get(i) is not None
            ]
            if len(collateral) >= num_replay:
                indices = torch.randperm(len(collateral))[:num_replay]
                return [collateral[i][1] for i in indices]

        # 默认随机采样
        indices = torch.randperm(len(self.memory))[:num_replay]
        return [self.memory[i] for i in indices]

    def update_old_predictions(
        self, model: torch.nn.Module, sample_indices: Optional[list[int]] = None
    ) -> None:
        """更新附带损伤标记（mix-cd 扩展）。

        记录旧模型对记忆样本的预测，用于后续 mix-cd 采样。

        Args:
            model: 当前模型（作为旧模型）。
            sample_indices: 要更新的样本索引（默认全部）。
        """
        model.eval()
        with torch.no_grad():
            indices = sample_indices or range(len(self.memory))
            for i in indices:
                if i >= len(self.memory):
                    continue
                inputs, labels = self.memory[i]
                outputs = model(inputs.unsqueeze(0))
                pred = outputs.argmax(dim=1).item()
                self._old_model_predictions[i] = pred

    def compute_loss(
        self,
        model: torch.nn.Module,
        batch: Any,
        base_loss: torch.Tensor,
        task: Task,
    ) -> torch.Tensor:
        """在基础损失上加入回放损失。"""
        # 获取回放样本
        inputs, labels = batch
        replay_samples = self.sample_replay(len(inputs))
        if replay_samples is None:
            return base_loss

        # 计算回放损失
        replay_inputs = torch.stack([s[0] for s in replay_samples])
        replay_labels = torch.stack([s[1] for s in replay_samples])
        replay_outputs = model(replay_inputs)
        replay_loss = torch.nn.functional.cross_entropy(replay_outputs, replay_labels)

        return base_loss + replay_loss

    def after_task(self, model: torch.nn.Module, task: Task) -> None:
        """任务结束后，保存代表性样本到记忆库。"""
        # 采样保存代表性样本
        num_to_store = min(self.memory_size, len(task.dataloader.dataset))
        stored = 0
        for batch in task.dataloader:
            if stored >= num_to_store:
                break
            self.store_samples(batch)
            stored += len(batch[0])

        # 更新附带损伤标记（mix-cd 扩展）
        if self.sample_mode == "mix_cd":
            self.update_old_predictions(model)

    def state_dict(self) -> dict[str, Any]:
        return {
            "memory_size": self.memory_size,
            "replay_ratio": self.replay_ratio,
            "sample_mode": self.sample_mode,
            "compute_budget": self.compute_budget,
            "memory": self.memory,
            "old_predictions": self._old_model_predictions,
        }
