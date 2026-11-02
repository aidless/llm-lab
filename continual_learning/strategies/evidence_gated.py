"""Evidence-Gated Continual Learning 策略。

基于 Metales & Grabenheinrich (2026)（arXiv:2610.39840，Surprise Alone Learns Lies: Evidence-Gated Continual Learning for LLMs from Streaming Claims）：

核心问题：
- 纯 surprise 驱动（如 gradient-based 选择）会学习不安全知识：假话（false claims）往往产生高 surprise，
  反而被优先吸收 → "Surprise Alone Learns Lies"
- 需要「证据门控」：不仅要 surprise 高，还要有独立证据支持该知识才值得学习

Evidence-Gated 核心机制：
- 对每个候选样本，计算两个分数：
  1. **Surprise Score**：模型对样本的惊讶度（loss / 负对数似然）
  2. **Evidence Score**：独立证据强度（如外部检索、一致率、置信度）
- 门控条件：surprise 高 AND evidence 强 → 才学习；surprise 高但 evidence 弱 → 跳过
- 关键：证据门控解决了「假话被吸收」问题，防止持续学习中毒

实现要点：
- `evidence_threshold`: 证据分数阈值（默认 0.5），低于此阈值的样本即使 surprise 高也不学习
- `surprise_threshold`: surprise 分数阈值（默认 None = 自动从数据分布估计）
- `evidence_estimator`: 证据估计函数（可选，自定义），输入 batch 返回 evidence 分数
"""

from __future__ import annotations

from typing import Any, Callable, Optional

import torch

from ..base import ContinualStrategy, StrategyConfig, Task


class EvidenceGatedStrategy(ContinualStrategy):
    """Evidence-Gated 持续学习策略。

    门控条件：
        learn(sample) = [surprise(sample) > surprise_threshold] AND [evidence(sample) > evidence_threshold]

    防止「surprise 驱动学习假话」：即使 surprise 高，若无独立证据支持，仍跳过。
    """

    def __init__(
        self,
        config: Optional[StrategyConfig] = None,
        evidence_threshold: float = 0.5,
        surprise_threshold: Optional[float] = None,
        evidence_estimator: Optional[Callable] = None,
        gate_mode: str = "hard",
    ) -> None:
        """初始化 Evidence-Gated 策略。

        Args:
            evidence_threshold: 证据分数阈值（默认 0.5）。
            surprise_threshold: surprise 分数阈值（默认 None = 自动估计）。
            evidence_estimator: 自定义证据估计函数，输入 batch 返回 evidence 分数张量。
            gate_mode: "hard"（硬门控，跳过样本）或 "soft"（软加权，权重 = evidence）。
        """
        super().__init__(config)
        self.evidence_threshold = evidence_threshold
        self.surprise_threshold = surprise_threshold
        self.evidence_estimator = evidence_estimator
        self.gate_mode = gate_mode
        self._surprise_history: list[float] = []
        self._evidence_scores: list[float] = []
        self._gated_out: int = 0
        self._total_samples: int = 0

    def name(self) -> str:
        return "EvidenceGated"

    def compute_loss(
        self,
        model: torch.nn.Module,
        batch: Any,
        base_loss: torch.Tensor,
        task: Task,
    ) -> torch.Tensor:
        """在基础损失上应用证据门控。

        对 batch 中每个样本计算 surprise 和 evidence 分数，
        根据门控条件决定是否保留该样本的损失贡献。
        """
        inputs, labels = batch
        # 估计 surprise（负对数似然作为代理）
        outputs = model(inputs)
        log_probs = torch.nn.functional.log_softmax(outputs, dim=-1)
        nll = -log_probs.gather(1, labels.unsqueeze(1)).squeeze(1)
        surprise = nll.detach()

        # 估计 evidence
        if self.evidence_estimator is not None:
            evidence = self.evidence_estimator(batch)
        else:
            # 默认：置信度（softmax 最大概率）作为证据代理
            probs = torch.nn.functional.softmax(outputs, dim=-1)
            evidence = probs.max(dim=-1).values.detach()

        # 自动估计 surprise 阈值（如果没有指定）
        if self.surprise_threshold is None:
            self.surprise_threshold = float(surprise.median()) if len(surprise) > 0 else 0.0

        # 更新统计
        self._surprise_history.extend(surprise.tolist())
        self._evidence_scores.extend(evidence.tolist())
        self._total_samples += len(inputs)

        # 门控：保留满足条件的样本
        if self.gate_mode == "hard":
            keep_mask = (surprise > self.surprise_threshold) & (evidence > self.evidence_threshold)
            self._gated_out += int((~keep_mask).sum().item())
            if keep_mask.sum() == 0:
                # 全部被门控，返回零损失
                return base_loss * 0.0
            # 重新计算加权损失（只保留 keep 的样本）
            kept_loss = torch.nn.functional.cross_entropy(outputs[keep_mask], labels[keep_mask])
            return kept_loss
        else:
            # 软门控：证据权重
            weights = torch.clamp(evidence, min=0.0, max=1.0)
            weighted_loss = (nll * weights).mean()
            return weighted_loss

    def after_task(self, model: torch.nn.Module, task: Task) -> None:
        """任务结束后保存统计信息。"""
        # 无需额外操作，统计已在 compute_loss 中更新

    def state_dict(self) -> dict[str, Any]:
        return {
            "evidence_threshold": self.evidence_threshold,
            "surprise_threshold": self.surprise_threshold,
            "gate_mode": self.gate_mode,
            "gated_out": self._gated_out,
            "total_samples": self._total_samples,
            "gate_rate": (
                self._gated_out / max(1, self._total_samples)
                if self._total_samples > 0
                else 0.0
            ),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.evidence_threshold = state.get("evidence_threshold", self.evidence_threshold)
        self.surprise_threshold = state.get("surprise_threshold", self.surprise_threshold)
        self.gate_mode = state.get("gate_mode", self.gate_mode)
        self._gated_out = state.get("gated_out", self._gated_out)
        self._total_samples = state.get("total_samples", self._total_samples)
