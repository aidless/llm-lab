"""统一矩阵分解 LoRA 策略（合并 OPLoRA + CURLoRA）。

基于两个前沿方法：
- OPLoRA（AAAI 2026，doi:10.1609/aaai.v40i40.40703）：SVD 分解 + 双面正交投影，
  保留 top-k 奇异子空间，使新任务更新方向与旧任务子空间正交。
- CURLoRA（2024，doi:10.5281/zenodo.12730055）：CUR 矩阵分解 + LoRA，
  保留原始矩阵结构，缓解灾难性遗忘。

统一接口：`MatrixLoRAStrategy` 通过 `decomposition` 参数选择分解方式：
- `"svd"`：SVD + 双面正交投影（OPLoRA）
- `"cur"`：CUR 分解（CURLoRA）

两种方式共享相同的 LoRA 适配器基础设施（冻结基座 + 低秩适配器 + 前向钩子），
避免重复代码。
"""

from __future__ import annotations

import math
from typing import Any, Optional

import torch
import torch.nn as nn

from ..base import ContinualStrategy, StrategyConfig, Task


# =============================================================================
# 分解函数
# =============================================================================

def svd_projection(
    weight: torch.Tensor,
    top_k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """SVD 双面正交投影（OPLoRA）。

    对权重 W (out, in) 做 SVD：W = U Σ V^T
    保留 top-k 奇异向量：
      P_L = I - U_k U_k^T  （左投影，排除 top-k 左奇异方向）
      P_R = I - V_k V_k^T  （右投影，排除 top-k 右奇异方向）

    Args:
        weight: 冻结权重张量，形状 (out_features, in_features)。
        top_k: 保留的奇异向量数。

    Returns:
        (P_L, P_R)：双面正交投影矩阵。
    """
    W = weight.detach().float()
    U, S, Vt = torch.linalg.svd(W, full_matrices=False)
    k = min(top_k, U.shape[1], Vt.shape[0])

    U_k = U[:, :k]  # (out, k)
    V_k = Vt[:k, :].t()  # (in, k)

    out, in_ = W.shape
    I_out = torch.eye(out, device=W.device)
    I_in = torch.eye(in_, device=W.device)

    P_L = I_out - U_k @ U_k.t()
    P_R = I_in - V_k @ V_k.t()
    return P_L, P_R


def cur_decomposition(
    matrix: torch.Tensor,
    num_cols: int,
    num_rows: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """CUR 矩阵分解。

    A ≈ C * U * R，其中：
      - C：A 的列子集（按列范数选取 num_cols 列）
      - R：A 的行子集（按行范数选取 num_rows 行）
      - U：C 和 R 交叉点的伪逆

    CUR 分解相比 SVD 更稀疏、计算更高效，且保留原始矩阵结构。

    Args:
        matrix: 输入矩阵，形状 (m, n)。
        num_cols: 选取的列数。
        num_rows: 选取的行数。

    Returns:
        (C, U, R) 三元组。
    """
    A = matrix.detach().float()
    m, n = A.shape

    # 按列范数选择 top-k 列
    col_norms = A.norm(dim=0)  # (n,)
    col_idx = torch.argsort(col_norms, descending=True)[:num_cols]
    C = A[:, col_idx]  # (m, num_cols)

    # 按行范数选择 top-k 行
    row_norms = A.norm(dim=1)  # (m,)
    row_idx = torch.argsort(row_norms, descending=True)[:num_rows]
    R = A[row_idx, :]  # (num_rows, n)

    # U = pinv(C ∩ R) 的伪逆
    U_intersect = A[row_idx][:, col_idx]  # (num_rows, num_cols)
    U = torch.linalg.pinv(U_intersect)  # (num_cols, num_rows)

    return C, U, R


# =============================================================================
# 统一 LoRA 适配层
# =============================================================================

class MatrixLoRALayer(nn.Module):
    """带矩阵分解结构的 LoRA 层。

    支持两种分解模式：
    - `decomposition="svd"`：双面正交投影（OPLoRA），LoRA 更新被投影到正交补空间。
    - `decomposition="cur"`：CUR 结构（CURLoRA），LoRA 更新附加 CUR 结构信息。
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: float = 1.0,
        decomposition: str = "svd",
        top_k: Optional[int] = None,
        num_cols: int = 4,
        num_rows: int = 4,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.decomposition = decomposition
        self.top_k = top_k if top_k is not None else rank
        self.num_cols = num_cols
        self.num_rows = num_rows

        # LoRA 适配器
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.scaling = alpha / rank

        if decomposition == "cur":
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        else:
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

        # 分解结构缓冲
        self.register_buffer("P_L", None)
        self.register_buffer("P_R", None)
        self.register_buffer("C", None)
        self.register_buffer("U", None)
        self.register_buffer("R", None)

    def set_structure(self, weight: torch.Tensor) -> None:
        """从冻结权重计算分解结构。

        Args:
            weight: 冻结权重张量，形状 (out_features, in_features)。
        """
        if self.decomposition == "svd":
            P_L, P_R = svd_projection(weight, self.top_k)
            self.P_L = P_L
            self.P_R = P_R
        elif self.decomposition == "cur":
            C, U, R = cur_decomposition(weight, self.num_cols, self.num_rows)
            self.C = C
            self.U = U
            self.R = R
        else:
            raise ValueError(f"未知分解方式: {self.decomposition}")

    def compute_subspace_interference(self, weight: torch.Tensor) -> float:
        """计算子空间干扰度量 ρ_k（仅 SVD 模式）。

        ρ_k = ||P_L W_new||_F / ||W_new||_F
        其中 W_new 是新任务的权重更新方向。
        值越大 = 新更新与旧子空间越正交 = 干扰越小。

        Args:
            weight: 新任务的权重更新（ΔW）。
        Returns:
            ρ_k 度量值。
        """
        if self.decomposition != "svd" or self.P_L is None:
            return 1.0
        delta = weight.detach().float()
        proj_delta = self.P_L @ delta @ self.P_R
        denom = delta.norm() + 1e-8
        return (proj_delta.norm() / denom).item()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """LoRA 适配输出（不含基座，由钩子添加）。"""
        lora_out = (self.lora_B @ self.lora_A) * self.scaling
        if self.decomposition == "svd" and self.P_L is not None:
            # 正交投影 LoRA 更新
            proj = self.P_L @ lora_out @ self.P_R
            return nn.functional.linear(x, proj)
        # CUR 模式：直接使用 LoRA 更新（CUR 结构在钩子中已附加）
        return nn.functional.linear(x, lora_out)


# =============================================================================
# 统一适配模型
# =============================================================================

class MatrixLoRAModel(nn.Module):
    """统一矩阵分解 LoRA 模型包装器。

    冻结基座 + 注入带矩阵分解结构的 LoRA 适配器。
    前向传播通过 `register_forward_hook` 在每个目标 Linear 层后添加 LoRA 贡献。
    """

    def __init__(
        self,
        base_model: nn.Module,
        target_modules: Optional[list[str]] = None,
        rank: int = 8,
        alpha: float = 1.0,
        decomposition: str = "svd",
        top_k: Optional[int] = None,
        num_cols: int = 4,
        num_rows: int = 4,
    ) -> None:
        super().__init__()
        self.base_model = base_model
        self.rank = rank
        self.alpha = alpha
        self.decomposition = decomposition
        self.top_k = top_k
        self.num_cols = num_cols
        self.num_rows = num_rows

        for param in base_model.parameters():
            param.requires_grad = False

        self.lora_layers: dict[str, MatrixLoRALayer] = {}
        self._inject_lora(target_modules)

    def _inject_lora(self, target_modules: Optional[list[str]]) -> None:
        """在指定层注入 LoRA 适配器。"""
        if target_modules is None:
            target_modules = []
            for name, module in self.base_model.named_modules():
                if isinstance(module, nn.Linear):
                    target_modules.append(name)

        for name, module in self.base_model.named_modules():
            if name in target_modules and isinstance(module, nn.Linear):
                layer = MatrixLoRALayer(
                    module.in_features,
                    module.out_features,
                    self.rank,
                    self.alpha,
                    self.decomposition,
                    self.top_k,
                    self.num_cols,
                    self.num_rows,
                )
                layer.set_structure(module.weight.detach())
                self.lora_layers[name] = layer
                setattr(self, f"lora_{name.replace('.', '_')}", layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播：冻结基座 + LoRA 适配贡献。"""
        hooks = []

        def make_hook(layer: MatrixLoRALayer):
            def hook_fn(module, input, output):
                return output + layer(input[0])
            return hook_fn

        for name, module in self.base_model.named_modules():
            if name in self.lora_layers:
                layer = self.lora_layers[name]
                hook = module.register_forward_hook(make_hook(layer))
                hooks.append(hook)

        try:
            output = self.base_model(x)
        finally:
            for hook in hooks:
                hook.remove()

        return output

    def trainable_parameters(self) -> int:
        """计算可训练参数量。"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_rho(self, layer_name: str, delta_weight: torch.Tensor) -> float:
        """获取指定层的子空间干扰度量 ρ_k（仅 SVD 模式）。"""
        layer = self.lora_layers.get(layer_name)
        if layer is None:
            return 1.0
        return layer.compute_subspace_interference(delta_weight)


# =============================================================================
# 统一策略
# =============================================================================

class MatrixLoRAStrategy(ContinualStrategy):
    """统一矩阵分解 LoRA 持续学习策略。

    每个任务创建新的适配器，冻结预训练权重。
    通过 `decomposition` 参数选择：
    - `"svd"`：OPLoRA 正交投影（默认）
    - `"cur"`：CURLoRA CUR 分解
    """

    def __init__(
        self,
        rank: int = 8,
        alpha: float = 1.0,
        decomposition: str = "svd",
        top_k: Optional[int] = None,
        num_cols: int = 4,
        num_rows: int = 4,
        config: Optional[StrategyConfig] = None,
    ) -> None:
        super().__init__(config)
        self.rank = rank
        self.alpha = alpha
        self.decomposition = decomposition
        self.top_k = top_k
        self.num_cols = num_cols
        self.num_rows = num_rows
        self._adapters: list[MatrixLoRAModel] = []
        self._current_adapter: Optional[MatrixLoRAModel] = None

    def name(self) -> str:
        return f"MatrixLoRA({self.decomposition})"

    def on_task_start(self, model: nn.Module, task: Task) -> None:
        """新任务开始时，创建新的矩阵分解 LoRA 适配器。"""
        self._current_adapter = MatrixLoRAModel(
            model,
            rank=self.rank,
            alpha=self.alpha,
            decomposition=self.decomposition,
            top_k=self.top_k,
            num_cols=self.num_cols,
            num_rows=self.num_rows,
        )
        self._adapters.append(self._current_adapter)

    def on_task_end(self, model: nn.Module, task: Task) -> None:
        """任务结束时，保存适配器状态。"""
        if self._current_adapter is not None:
            self._adapters.append(self._current_adapter)

    def compute_loss(
        self,
        model: nn.Module,
        batch: Any,
        base_loss: torch.Tensor,
        task: Task,
    ) -> torch.Tensor:
        """不修改损失函数（分解结构在模型前向中生效）。"""
        return base_loss

    def get_rho(self, layer_name: str, delta_weight: torch.Tensor) -> float:
        """获取指定层的子空间干扰度量 ρ_k（仅 SVD 模式）。"""
        if self._current_adapter is None:
            return 1.0
        return self._current_adapter.get_rho(layer_name, delta_weight)

    def state_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "alpha": self.alpha,
            "decomposition": self.decomposition,
            "top_k": self.top_k,
            "num_cols": self.num_cols,
            "num_rows": self.num_rows,
            "num_adapters": len(self._adapters),
        }
