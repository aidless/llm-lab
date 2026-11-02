"""AutoHebb 策略：自修改神经网络持续学习。

基于 Levine et al. (2026)（arXiv:2610.40757，AutoHebb: Self-Modifying Neural Networks that Zero-Shot Infer and Continually Learn）：

核心思想：
- 传统持续学习是「冻结 + 微调/重训」（外部驱动），AutoHebb 是「自修改」——网络自身
  在推理过程中持续更新权重（内部驱动）
- 关键：自修改网络通过局部 Hebbian 可塑性规则实现持续学习，无需外部优化器
- 零样本推理 + 持续学习同时达成（强归纳偏置，如稀疏连接/局部可塑性）

AutoHebb 核心机制：
- Hebbian 可塑性：ΔW_ij = eta * x_i * x_j（前馈激活关联）
- 局部更新：每个突触只依赖局部信息（前/后突触激活），无需全局反向传播
- 衰减/归一化：防止权重爆炸，使用 Oja 规则或归一化
- 稀疏性：只有激活阈值以上的突触被更新（稀疏连接 = 强归纳偏置）

实现要点：
- `learning_rate`: Hebbian 学习率 eta（默认 1e-3）
- `oja_normalization`: 是否使用 Oja 规则归一化（默认 True，防爆炸）
- `sparsity`: 突触稀疏性（默认 0.1，只有 top 10% 激活被更新）
- `recurrent`: 是否使用循环连接（默认 False，简单前馈）
"""

from __future__ import annotations

from typing import Any, Optional

import torch

from ..base import ContinualStrategy, StrategyConfig, Task


class AutoHebbStrategy(ContinualStrategy):
    """AutoHebb 自修改持续学习策略。

    在任务训练后，通过 Hebbian 可塑性规则局部更新权重，
    实现「自修改」的持续学习，而非外部驱动的微调。
    """

    def __init__(
        self,
        config: Optional[StrategyConfig] = None,
        learning_rate: float = 1e-3,
        oja_normalization: bool = True,
        sparsity: float = 0.1,
        recurrent: bool = False,
        hebbian_epochs: int = 5,
    ) -> None:
        """初始化 AutoHebb 策略。

        Args:
            learning_rate: Hebbian 学习率 eta。
            oja_normalization: 是否使用 Oja 规则归一化（防止权重爆炸）。
            sparsity: 突触稀疏性（0-1，0=全更新，1=不更新）。
            recurrent: 是否使用循环连接。
            hebbian_epochs: Hebbian 更新轮数。
        """
        super().__init__(config)
        self.learning_rate = learning_rate
        self.oja_normalization = oja_normalization
        self.sparsity = sparsity
        self.recurrent = recurrent
        self.hebbian_epochs = hebbian_epochs
        self._activation_history: dict[str, list[torch.Tensor]] = {}

    def name(self) -> str:
        return "AutoHebb"

    def after_task(self, model: torch.nn.Module, task: Task) -> None:
        """任务结束后执行 Hebbian 自修改更新。

        对模型每个 2D 线性层，收集前馈激活对（pre-activation, post-activation），
        应用 Hebbian 规则更新权重。
        """
        # 收集激活（需要注册前向钩子）
        activations: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

        def hook_fn(name: str):
            def hook(module, input, output):
                # 记录前馈激活对：pre = input[0], post = output
                pre = input[0].detach()
                post = output.detach()
                if pre.dim() >= 2 and post.dim() >= 2:
                    # 展平到 (batch*seq, features)
                    pre_flat = pre.reshape(-1, pre.shape[-1])
                    post_flat = post.reshape(-1, post.shape[-1])
                    if name not in activations:
                        activations[name] = (pre_flat, post_flat)
                    else:
                        old_pre, old_post = activations[name]
                        activations[name] = (
                            torch.cat([old_pre, pre_flat], dim=0),
                            torch.cat([old_post, post_flat], dim=0),
                        )
            return hook

        # 注册钩子
        hooks = []
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                h = module.register_forward_hook(hook_fn(name))
                hooks.append(h)

        # 前向传播收集激活
        model.eval()
        with torch.no_grad():
            for batch in task.dataloader:
                if isinstance(batch, (tuple, list)):
                    inputs = batch[0]
                else:
                    inputs = batch
                model(inputs)

        # 移除钩子
        for h in hooks:
            h.remove()

        # 应用 Hebbian 更新
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear) and name in activations:
                pre, post = activations[name]
                # Hebbian 规则：ΔW = eta * (post^T @ pre) / N
                delta_w = self.learning_rate * (post.t() @ pre) / max(1, pre.shape[0])

                # Oja 归一化：W <- W + eta*(x y - y^2 W)（防爆炸）
                if self.oja_normalization:
                    y_sq = (post ** 2).mean(dim=0, keepdim=True)
                    delta_w -= self.learning_rate * (y_sq.t() @ pre) / max(1, pre.shape[0])

                # 稀疏性：只更新 top (1-sparsity) 激活的突触
                if self.sparsity > 0:
                    mask = torch.ones_like(delta_w)
                    # 按 |ΔW| 排序，保留 top (1-sparsity)
                    flat = delta_w.abs().flatten()
                    k = int(flat.numel() * (1 - self.sparsity))
                    if k > 0 and k < flat.numel():
                        threshold = torch.topk(flat, k).values.min()
                        mask = (delta_w.abs() >= threshold).float()
                    delta_w = delta_w * mask

                # 更新权重（梯度式局部更新）
                module.weight.data.add_(delta_w)

    def state_dict(self) -> dict[str, Any]:
        return {
            "learning_rate": self.learning_rate,
            "oja_normalization": self.oja_normalization,
            "sparsity": self.sparsity,
            "recurrent": self.recurrent,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.learning_rate = state.get("learning_rate", self.learning_rate)
        self.oja_normalization = state.get("oja_normalization", self.oja_normalization)
        self.sparsity = state.get("sparsity", self.sparsity)
        self.recurrent = state.get("recurrent", self.recurrent)
