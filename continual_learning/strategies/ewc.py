"""EWC（Elastic Weight Consolidation）正则化策略。

基于 Kirkpatrick et al. (2017)（arXiv:1612.00796）：
使用 Fisher 信息矩阵的对角线近似度量每个参数的重要性，
在训练新任务时对重要参数施加二次惩罚。

关键实证数据：
- permuted MNIST：SGD 97.9%→63.0%（灾难性遗忘）；EWC 96.8%（几乎无遗忘）
- 超参：Fisher multiplier=400、num samples=100

Bayesian PEFT 扩展（Chen & Garner, 2024，doi:10.1109/taslp.2024.3463395）：
- 使用 Laplace 近似（包括对角和 Kronecker 因子化）正则化 PEFT（LoRA）
- 关键：Laplace 近似的精度矩阵即 Fisher 信息矩阵，可无缝替代 EWC 的 Fisher
- 实证：在语言建模与语音合成任务上显著保留预训练知识
"""

from __future__ import annotations

from typing import Any, Optional

import torch

from ..base import ContinualStrategy, StrategyConfig, Task


class EWCTrainingStrategy(ContinualStrategy):
    """EWC 正则化策略。

    EWC 在损失函数中加入 Fisher 信息加权的二次惩罚项：

        L_total = L_base + lambda * sum_i (F_i * (theta_i - theta_old_i)^2)

    其中 F_i 是 Fisher 信息矩阵对角线元素（参数 i 的重要性），
    theta_old 是任务训练完成后的参数。

    Bayesian PEFT 扩展：支持 Laplace 近似（Kronecker 因子化），
    通过 `use_kronecker` 开关启用更精确的近似。
    """

    def __init__(
        self,
        fisher_multiplier: float = 400.0,
        num_samples: int = 100,
        config: Optional[StrategyConfig] = None,
        use_kronecker: bool = False,
        kronecker_damping: float = 1e-5,
    ) -> None:
        super().__init__(config)
        self.fisher_multiplier = fisher_multiplier
        self.num_samples = num_samples
        self.use_kronecker = use_kronecker
        self.kronecker_damping = kronecker_damping
        self._fisher: dict[str, torch.Tensor] = {}
        self._old_params: dict[str, torch.Tensor] = {}
        self._kronecker: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

    def name(self) -> str:
        return "EWC"

    def compute_loss(
        self,
        model: torch.nn.Module,
        batch: Any,
        base_loss: torch.Tensor,
        task: Task,
    ) -> torch.Tensor:
        """在基础损失上加入 EWC 惩罚项。"""
        if not self._fisher and not self._kronecker:
            return base_loss
        if not self._old_params:
            return base_loss

        ewc_loss = torch.tensor(0.0, device=base_loss.device)
        for name, param in model.named_parameters():
            if name in self._fisher and param.grad is not None:
                fisher = self._fisher[name]
                old = self._old_params[name]
                # EWC 惩罚项：0.5 * lambda * sum(F * (theta - theta_old)^2)
                ewc_loss += (fisher * (param - old) ** 2).sum()

        # Kronecker 因子化惩罚（Bayesian PEFT 扩展）
        for name, param in model.named_parameters():
            if name in self._kronecker and param.grad is not None:
                A, B = self._kronecker[name]
                old = self._old_params[name]
                delta = (param - old).reshape(A.shape[0], B.shape[0])
                # Laplace 惩罚：delta^T (A ⊗ B) delta = tr(A delta B delta^T)
                ewc_loss += torch.trace(A @ delta @ B @ delta.t())

        return base_loss + self.fisher_multiplier * ewc_loss

    def after_task(self, model: torch.nn.Module, task: Task) -> None:
        """任务结束后更新 Fisher 信息与旧参数。"""
        # 保存当前参数
        self._old_params = {
            name: param.detach().clone()
            for name, param in model.named_parameters()
            if param.requires_grad
        }
        # 计算 Fisher 信息（使用任务数据的梯度外积近似）
        self._fisher = self._compute_fisher(model, task)

        # Kronecker 因子化近似（Bayesian PEFT 扩展）
        if self.use_kronecker:
            self._kronecker = self._compute_kronecker_factors(model, task)

    def _compute_fisher(
        self, model: torch.nn.Module, task: Task
    ) -> dict[str, torch.Tensor]:
        """使用任务数据计算 Fisher 信息矩阵的对角线近似。

        方法：对每个样本计算损失对参数的梯度平方，然后取平均。
        这是 Fisher 信息对角近似的标准做法（Kirkpatrick et al., 2017）。
        """
        model.train()
        fisher: dict[str, torch.Tensor] = {}
        # 初始化 Fisher 为零张量
        for name, param in model.named_parameters():
            if param.requires_grad:
                fisher[name] = torch.zeros_like(param)

        num_samples = min(self.num_samples, len(task.dataloader.dataset))
        samples_used = 0

        for batch in task.dataloader:
            if samples_used >= num_samples:
                break
            model.zero_grad()
            # 假设数据是 (inputs, labels) 格式
            inputs, labels = batch
            outputs = model(inputs)
            loss = torch.nn.functional.cross_entropy(outputs, labels)
            loss.backward()
            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    fisher[name] += param.grad.detach() ** 2
            samples_used += len(inputs)

        # 取平均
        for name in fisher:
            fisher[name] /= max(1, samples_used)

        model.eval()
        return fisher

    def _compute_kronecker_factors(
        self, model: torch.nn.Module, task: Task
    ) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        """计算 Kronecker 因子化的 Laplace 近似（Bayesian PEFT）。

        对每个矩阵参数 W (m, n)，Kronecker 因子 A (m, m) 和 B (n, n)
        通过梯度外积的期望近似。这是 KFAC 的标准做法。

        注意：本实现是简化的 KFAC 近似，仅支持 2D 参数。
        """
        model.train()
        kronecker: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

        # 收集 2D 参数
        param_shapes = {}
        for name, param in model.named_parameters():
            if param.requires_grad and param.dim() == 2:
                param_shapes[name] = param.shape
                A = torch.zeros(param.shape[0], param.shape[0], device=param.device)
                B = torch.zeros(param.shape[1], param.shape[1], device=param.device)
                kronecker[name] = (A, B)

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
                if name in kronecker and param.grad is not None:
                    g = param.grad.detach()
                    A, B = kronecker[name]
                    # KFAC 近似：A += g @ g^T 的列平均，B += g^T @ g 的行平均
                    # 简化：使用梯度向量的外积近似
                    A += g @ g.t() / (g.shape[1] + 1e-8)
                    B += g.t() @ g / (g.shape[0] + 1e-8)
            samples_used += len(inputs)

        # 取平均并加阻尼
        for name in kronecker:
            A, B = kronecker[name]
            A = A / max(1, samples_used) + self.kronecker_damping * torch.eye(A.shape[0], device=A.device)
            B = B / max(1, samples_used) + self.kronecker_damping * torch.eye(B.shape[0], device=B.device)
            kronecker[name] = (A, B)

        model.eval()
        return kronecker

    def state_dict(self) -> dict[str, Any]:
        return {
            "fisher_multiplier": self.fisher_multiplier,
            "use_kronecker": self.use_kronecker,
            "fisher": {k: v.cpu().numpy() for k, v in self._fisher.items()},
            "old_params": {k: v.cpu().numpy() for k, v in self._old_params.items()},
            "kronecker": {
                k: (a.cpu().numpy(), b.cpu().numpy())
                for k, (a, b) in self._kronecker.items()
            },
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.fisher_multiplier = state.get("fisher_multiplier", self.fisher_multiplier)
        self.use_kronecker = state.get("use_kronecker", self.use_kronecker)
        self._fisher = {
            k: torch.from_numpy(v) for k, v in state.get("fisher", {}).items()
        }
        self._old_params = {
            k: torch.from_numpy(v) for k, v in state.get("old_params", {}).items()
        }
        self._kronecker = {
            k: (torch.from_numpy(a), torch.from_numpy(b))
            for k, (a, b) in state.get("kronecker", {}).items()
        }
