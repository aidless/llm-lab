"""Fisher-Shaped EWC 策略。

基于 Wang et al. (2026)（arXiv:2610.37145，Beyond Parameter Protection: Fisher-Shaped EWC for Replay-Free Continual Learning）：

标准 EWC 的局限：
- FIM 局部估计（对当前参数 theta 的 Fisher）不足以描述有限位移后的参数敏感性
- 在高维 LLM 上，对角线 Fisher 近似失效（Lifelong Pretraining 实证）

Fisher-Shaped EWC 的核心改进：
- 不只用当前参数的 Fisher，而是控制参数邻域的 Fisher 敏感性
- 使用「Fisher 形状」加权：对每个参数施加一个形状函数，惩罚幅度随位移量非线性增长
- 关键：Fisher 形状通过采样邻域参数估计（N 个扰动样本），对每个扰动点计算 Fisher 再平均

实现要点：
- `neighborhood_samples`: 邻域采样数（默认 10，论文建议 5-20）
- `perturbation_scale`: 扰动尺度 epsilon（默认 1e-3，控制邻域半径）
- `shape_exponent`: Fisher 形状指数 p（默认 1.0，标准二次；>1 时更强惩罚，<1 时更宽容）
- 在标准 EWC 惩罚基础上，引入邻域 Fisher 形状加权
"""

from __future__ import annotations

from typing import Any, Optional

import torch

from ..base import ContinualStrategy, StrategyConfig, Task
from .ewc import EWCTrainingStrategy


class FisherShapedEWCStrategy(ContinualStrategy):
    """Fisher-Shaped EWC 策略。

    在标准 EWC 基础上，通过邻域 Fisher 敏感性加权增强惩罚：
        L_ewc = lambda * sum_i F_shape_i * |theta_i - theta_old_i|^p

    其中 F_shape_i 是邻域平均 Fisher（对参数加扰动后重新估计），
    p 是形状指数（默认 2.0 为二次惩罚，标准 EWC 形式）。
    """

    def __init__(
        self,
        fisher_multiplier: float = 400.0,
        num_samples: int = 100,
        config: Optional[StrategyConfig] = None,
        neighborhood_samples: int = 10,
        perturbation_scale: float = 1e-3,
        shape_exponent: float = 2.0,
    ) -> None:
        super().__init__(config)
        self.fisher_multiplier = fisher_multiplier
        self.num_samples = num_samples
        self.neighborhood_samples = neighborhood_samples
        self.perturbation_scale = perturbation_scale
        self.shape_exponent = shape_exponent
        self._fisher_shape: dict[str, torch.Tensor] = {}
        self._old_params: dict[str, torch.Tensor] = {}

    def name(self) -> str:
        return "FisherShapedEWC"

    def compute_loss(
        self,
        model: torch.nn.Module,
        batch: Any,
        base_loss: torch.Tensor,
        task: Task,
    ) -> torch.Tensor:
        """在基础损失上加入 Fisher-Shaped EWC 惩罚项。"""
        if not self._fisher_shape or not self._old_params:
            return base_loss

        ewc_loss = torch.tensor(0.0, device=base_loss.device)
        for name, param in model.named_parameters():
            if name in self._fisher_shape and param.grad is not None:
                fisher = self._fisher_shape[name]
                old = self._old_params[name]
                delta = (param - old).abs()
                # Fisher 形状惩罚：F_shape * |delta|^p
                ewc_loss += (fisher * delta.pow(self.shape_exponent)).sum()

        return base_loss + self.fisher_multiplier * ewc_loss

    def after_task(self, model: torch.nn.Module, task: Task) -> None:
        """任务结束后计算邻域 Fisher 形状。"""
        # 保存当前参数
        self._old_params = {
            name: param.detach().clone()
            for name, param in model.named_parameters()
            if param.requires_grad
        }
        # 计算邻域 Fisher 形状（扰动样本平均）
        self._fisher_shape = self._compute_neighborhood_fisher(model, task)

    def _compute_neighborhood_fisher(
        self, model: torch.nn.Module, task: Task
    ) -> dict[str, torch.Tensor]:
        """计算邻域平均 Fisher 信息（Fisher-Shaped EWC 核心）。

        对每个参数施加微小扰动 epsilon * N(0, 1)，
        在扰动后的参数点重新计算 Fisher，最后取平均。

        这度量的是参数邻域的敏感性，而非单点敏感性。
        """
        model.train()
        # 初始化邻域 Fisher 为零
        fisher_shape: dict[str, torch.Tensor] = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                fisher_shape[name] = torch.zeros_like(param)

        num_samples = min(self.num_samples, len(task.dataloader.dataset))
        samples_used = 0

        for batch in task.dataloader:
            if samples_used >= num_samples:
                break
            model.zero_grad()
            inputs, labels = batch
            outputs = model(inputs)
            loss = torch.nn.functional.cross_entropy(outputs, labels)
            loss.backward()
            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    fisher_shape[name] += param.grad.detach() ** 2
            samples_used += len(inputs)

        # 取平均得到单点 Fisher
        for name in fisher_shape:
            fisher_shape[name] /= max(1, samples_used)

        # 邻域形状：对每个参数施加扰动，重估 Fisher，取平均
        # 用当前 Fisher 作为基础，扰动后重新估计
        neighborhood_avg: dict[str, torch.Tensor] = {
            name: torch.zeros_like(f)
            for name, f in fisher_shape.items()
        }

        # 保存原始参数
        original_params = {
            name: param.detach().clone()
            for name, param in model.named_parameters()
            if param.requires_grad
        }

        for _ in range(self.neighborhood_samples):
            # 施加扰动
            for name, param in model.named_parameters():
                if name in original_params and param.requires_grad:
                    noise = torch.randn_like(param) * self.perturbation_scale
                    param.data.add_(noise)

            # 扰动后重新计算 Fisher
            perturbed_fisher = self._compute_fisher_perturbed(model, task, num_samples)
            for name in neighborhood_avg:
                neighborhood_avg[name] += perturbed_fisher[name]

            # 恢复原始参数
            for name, param in model.named_parameters():
                if name in original_params and param.requires_grad:
                    param.data.copy_(original_params[name])

        # 取平均
        for name in neighborhood_avg:
            neighborhood_avg[name] /= max(1, self.neighborhood_samples)

        model.eval()
        return neighborhood_avg

    def _compute_fisher_perturbed(
        self, model: torch.nn.Module, task: Task, num_samples: int
    ) -> dict[str, torch.Tensor]:
        """在扰动后的参数点上重新计算 Fisher（不保存/恢复参数）。"""
        fisher: dict[str, torch.Tensor] = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                fisher[name] = torch.zeros_like(param)

        samples_used = 0
        for batch in task.dataloader:
            if samples_used >= num_samples:
                break
            model.zero_grad()
            inputs, labels = batch
            outputs = model(inputs)
            loss = torch.nn.functional.cross_entropy(outputs, labels)
            loss.backward()
            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    fisher[name] += param.grad.detach() ** 2
            samples_used += len(inputs)

        for name in fisher:
            fisher[name] /= max(1, samples_used)
        return fisher

    def state_dict(self) -> dict[str, Any]:
        return {
            "fisher_multiplier": self.fisher_multiplier,
            "num_samples": self.num_samples,
            "neighborhood_samples": self.neighborhood_samples,
            "perturbation_scale": self.perturbation_scale,
            "shape_exponent": self.shape_exponent,
            "fisher_shape": {
                k: v.cpu().numpy() for k, v in self._fisher_shape.items()
            },
            "old_params": {
                k: v.cpu().numpy() for k, v in self._old_params.items()
            },
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.fisher_multiplier = state.get("fisher_multiplier", self.fisher_multiplier)
        self.num_samples = state.get("num_samples", self.num_samples)
        self.neighborhood_samples = state.get("neighborhood_samples", self.neighborhood_samples)
        self.perturbation_scale = state.get("perturbation_scale", self.perturbation_scale)
        self.shape_exponent = state.get("shape_exponent", self.shape_exponent)
        self._fisher_shape = {
            k: torch.from_numpy(v) for k, v in state.get("fisher_shape", {}).items()
        }
        self._old_params = {
            k: torch.from_numpy(v) for k, v in state.get("old_params", {}).items()
        }
