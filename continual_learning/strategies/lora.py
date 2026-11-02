"""LoRA（Low-Rank Adaptation）策略。

基于 Hu et al. (2021)（arXiv:2106.09685，ICLR 2022）：
冻结预训练权重，注入低秩分解矩阵，实现参数高效的增量学习。

关键实证数据（GPT-3 175B）：
- 可训练参数降 10,000×
- GPU 内存（VRAM）降 3×（1.2TB → 350GB）
- Checkpoint 大小降 10,000×（350GB → 35MB）
- 训练吞吐提升（32.5 → 43.1 tokens/s per V100）
- rank r=1/2 足够（d=12,288）

LaST-LoRA 扩展（Liu et al., 2025，doi:10.1145/3746260.3762126）：
- 自适应知识复用（adaptive knowledge reuse）：通过潜在子空间追踪
  （latent subspace tracking）识别任务间的共享知识，复用旧适配器
- 防止适配器数量随任务数线性增长
- 关键：相似任务共享低秩子空间，新任务可初始化自旧适配器
"""

from __future__ import annotations

import math
from typing import Any, Optional

import torch
import torch.nn as nn

from ..base import ContinualStrategy, StrategyConfig, Task


class LoRALayer(nn.Module):
    """LoRA 低秩适配层。

    将权重更新表示为两个低秩矩阵的乘积：
        W_adapted = W_original + B * A
    其中 A: (r, in_features), B: (out_features, r)，r << min(in, out)。
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 4,
        alpha: float = 1.0,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha

        # 低秩分解矩阵 A 和 B
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        # 缩放系数（r 与 alpha 的比值）
        self.scaling = alpha / rank

        # 初始化：A 用 Kaiming 均匀分布，B 为零
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor, base_weight: torch.Tensor) -> torch.Tensor:
        """计算 LoRA 适配后的输出。

        Args:
            x: 输入张量。
            base_weight: 原始权重（冻结）。

        Returns:
            适配后的输出。
        """
        # LoRA 路径：x @ A^T @ B^T * scaling
        lora_output = (x @ self.lora_A.t() @ self.lora_B.t()) * self.scaling
        return lora_output


class LoRAModel(nn.Module):
    """LoRA 适配模型包装器。

    将原始模型冻结，并在指定层注入 LoRA 适配器。
    """

    def __init__(
        self,
        base_model: nn.Module,
        target_modules: Optional[list[str]] = None,
        rank: int = 4,
        alpha: float = 1.0,
    ) -> None:
        super().__init__()
        self.base_model = base_model
        self.rank = rank
        self.alpha = alpha

        # 冻结原始模型参数
        for param in base_model.parameters():
            param.requires_grad = False

        # 找到目标层并注入 LoRA
        self.lora_layers: dict[str, LoRALayer] = {}
        self._inject_lora(target_modules)

    def _inject_lora(self, target_modules: Optional[list[str]]) -> None:
        """在指定层注入 LoRA 适配器。"""
        # 默认注入所有 Linear 层
        if target_modules is None:
            target_modules = []
            for name, module in self.base_model.named_modules():
                if isinstance(module, nn.Linear):
                    target_modules.append(name)

        for name, module in self.base_model.named_modules():
            if name in target_modules and isinstance(module, nn.Linear):
                lora_layer = LoRALayer(
                    module.in_features, module.out_features, self.rank, self.alpha
                )
                self.lora_layers[name] = lora_layer
                # 将 LoRA 层注册为子模块
                setattr(self, f"lora_{name.replace('.', '_')}", lora_layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播：原始模型 + LoRA 适配。

        通过 hook 在每个目标 Linear 层后添加 LoRA 贡献，
        使梯度能够流经 LoRA 参数。
        """
        # 保存原始前向函数
        original_forwards = {}

        def make_lora_forward(layer_name: str, lora_layer: LoRALayer):
            """创建带 LoRA 的层前向函数。"""
            original_forward = None

            def lora_forward(module, input, output):
                # 添加 LoRA 贡献：B * A * x * scaling
                lora_contrib = (
                    input[0] @ lora_layer.lora_A.t() @ lora_layer.lora_B.t()
                ) * lora_layer.scaling
                return output + lora_contrib

            return lora_forward

        # 注册前向 hook
        hooks = []
        for name, module in self.base_model.named_modules():
            if name in self.lora_layers:
                lora_layer = self.lora_layers[name]
                hook = module.register_forward_hook(
                    make_lora_forward(name, lora_layer)
                )
                hooks.append(hook)

        # 执行前向传播
        try:
            output = self.base_model(x)
        finally:
            # 移除 hooks
            for hook in hooks:
                hook.remove()

        return output

    def trainable_parameters(self) -> int:
        """计算可训练参数量。"""
        return sum(
            p.numel()
            for p in self.parameters()
            if p.requires_grad
        )


class LoRAStrategy(ContinualStrategy):
    """LoRA 持续学习策略。

    每个任务训练一个新的 LoRA 适配器，冻结预训练权重。
    这将持续学习的"遗忘"问题转化为"适配器管理"问题。

    LaST-LoRA 扩展：支持自适应知识复用（reuse_old_adapter），
    通过潜在子空间追踪识别相似任务，复用旧适配器作为初始化。
    """

    def __init__(
        self,
        rank: int = 4,
        alpha: float = 1.0,
        config: Optional[StrategyConfig] = None,
        reuse_threshold: float = 0.8,
    ) -> None:
        super().__init__(config)
        self.rank = rank
        self.alpha = alpha
        self.reuse_threshold = reuse_threshold
        self.adapters: dict[str, LoRAModel] = {}
        self._subspace_cache: dict[str, torch.Tensor] = {}  # 适配器潜在子空间

    def name(self) -> str:
        return "LoRA"

    def create_adapter(self, base_model: nn.Module, task_name: str) -> LoRAModel:
        """为指定任务创建 LoRA 适配器。"""
        adapter = LoRAModel(base_model, rank=self.rank, alpha=self.alpha)
        self.adapters[task_name] = adapter
        return adapter

    def get_adapter(self, task_name: str) -> Optional[LoRAModel]:
        """获取指定任务的适配器。"""
        return self.adapters.get(task_name)

    def compute_subspace(self, adapter: LoRAModel) -> dict[str, torch.Tensor]:
        """计算适配器的潜在子空间（LaST-LoRA 子空间追踪）。

        对每个 LoRA 层的 A 矩阵做 SVD，取前 rank 个左奇异向量作为子空间基。

        Args:
            adapter: LoRA 适配器。

        Returns:
            子空间字典：{层名: 子空间基矩阵}。
        """
        subspaces = {}
        for name, layer in adapter.lora_layers.items():
            # A: (rank, in_features)，子空间为 A 的行空间
            U, S, Vt = torch.linalg.svd(layer.lora_A.detach(), full_matrices=False)
            subspaces[name] = Vt  # (rank, in_features)
        return subspaces

    def find_similar_adapter(
        self, new_subspaces: dict[str, torch.Tensor]
    ) -> Optional[str]:
        """寻找与新任务最相似的旧适配器（LaST-LoRA 知识复用）。

        通过潜在子空间相似度（余弦相似度）判断任务相似性。
        相似度超过 reuse_threshold 的适配器可复用。

        Args:
            new_subspaces: 新任务的潜在子空间。

        Returns:
            相似适配器名称，无则返回 None。
        """
        best_name = None
        best_score = -1.0

        for task_name, adapter in self.adapters.items():
            old_subspaces = self._subspace_cache.get(task_name)
            if old_subspaces is None:
                continue

            # 计算子空间相似度（取公共层的平均余弦相似度）
            similarities = []
            for layer_name, new_subspace in new_subspaces.items():
                if layer_name in old_subspaces:
                    old_subspace = old_subspaces[layer_name]
                    # 计算子空间基的余弦相似度
                    # 子空间相似度 = ||U_new^T U_old||_F / sqrt(rank)
                    sim = torch.norm(
                        new_subspace @ old_subspace.t()
                    ) / math.sqrt(new_subspace.shape[0])
                    similarities.append(sim.item())

            if similarities:
                avg_sim = sum(similarities) / len(similarities)
                if avg_sim > best_score:
                    best_score = avg_sim
                    best_name = task_name

        if best_score >= self.reuse_threshold:
            return best_name
        return None

    def reuse_old_adapter(
        self, new_adapter: LoRAModel, old_adapter: LoRAModel
    ) -> None:
        """复用旧适配器的知识（LaST-LoRA 初始化）。

        将旧适配器的 LoRA 参数作为新适配器的初始化，实现知识复用。

        Args:
            new_adapter: 新创建的适配器。
            old_adapter: 相似度最高的旧适配器。
        """
        for name, layer in new_adapter.lora_layers.items():
            if name in old_adapter.lora_layers:
                old_layer = old_adapter.lora_layers[name]
                # 用旧参数初始化新参数（部分保留 + 随机扰动）
                with torch.no_grad():
                    layer.lora_A.data = 0.5 * old_layer.lora_A.data + 0.5 * torch.randn_like(layer.lora_A.data)
                    layer.lora_B.data = 0.5 * old_layer.lora_B.data

    def register_subspace(self, task_name: str, adapter: LoRAModel) -> None:
        """登记适配器的潜在子空间。"""
        self._subspace_cache[task_name] = self.compute_subspace(adapter)

    def state_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "alpha": self.alpha,
            "reuse_threshold": self.reuse_threshold,
            "adapters": {
                name: adapter.state_dict() for name, adapter in self.adapters.items()
            },
        }
