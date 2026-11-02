"""模型合并策略（Model Soups）。

基于 Wortsman et al. (2022)（arXiv:2203.05482，ICML 2022）：
通过权重平均合并多个独立训练的模型，实现无训练的模型能力组合。

关键实证数据：
- ViT-G/14 贪心汤 ImageNet 90.94%（超过 CoAtNet-7 的 90.88%）
- 贪心汤（80.24）优于蒸馏（79.79）
- 均匀汤（Uniform Soup）几乎总是更差
"""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn

from ..base import StrategyConfig


class ModelSoupsMerger:
    """模型合并器（Model Soups）。

    支持两种合并方式：
    - Uniform Soup：均匀平均所有模型权重
    - Greedy Soup：按验证集性能贪心加入模型（性能提升才保留）
    """

    def __init__(self, config: Optional[StrategyConfig] = None) -> None:
        self.config = config or StrategyConfig()
        self.models: list[nn.Module] = []

    def add_model(self, model: nn.Module) -> None:
        """添加一个模型到汤中。"""
        self.models.append(model)

    def uniform_soup(self) -> nn.Module:
        """均匀平均所有模型权重。"""
        if not self.models:
            raise ValueError("No models to merge")

        # 以第一个模型为基准
        soup = self._clone_model(self.models[0])
        state_dict = soup.state_dict()

        # 累加所有模型权重
        for model in self.models[1:]:
            model_state = model.state_dict()
            for key in state_dict:
                state_dict[key] += model_state[key]

        # 取平均
        num_models = len(self.models)
        for key in state_dict:
            state_dict[key] /= num_models

        soup.load_state_dict(state_dict)
        return soup

    def greedy_soup(
        self,
        eval_fn: callable,
        max_models: Optional[int] = None,
    ) -> nn.Module:
        """贪心汤：按验证集性能递增地加入模型。

        Args:
            eval_fn: 评估函数，输入模型返回性能分数。
            max_models: 最多合并的模型数（可选）。

        Returns:
            合并后的模型。
        """
        if not self.models:
            raise ValueError("No models to merge")

        # 按验证集性能排序
        scored = sorted(
            [(m, eval_fn(m)) for m in self.models],
            key=lambda x: x[1],
            reverse=True,
        )

        # 从最佳单模型开始
        best_model = scored[0][0]
        best_score = scored[0][1]
        soup = self._clone_model(best_model)
        current_state = soup.state_dict()

        # 贪心加入模型
        merged_count = 1
        for model, score in scored[1:]:
            if max_models and merged_count >= max_models:
                break

            # 尝试合并
            candidate_state = current_state.copy()
            model_state = model.state_dict()
            for key in candidate_state:
                candidate_state[key] = (
                    candidate_state[key] * merged_count + model_state[key]
                ) / (merged_count + 1)

            # 评估候选
            candidate = self._clone_model(best_model)
            candidate.load_state_dict(candidate_state)
            candidate_score = eval_fn(candidate)

            # 性能提升才保留
            if candidate_score > best_score:
                current_state = candidate_state
                best_score = candidate_score
                merged_count += 1
            else:
                break

        soup.load_state_dict(current_state)
        return soup

    def _clone_model(self, model: nn.Module) -> nn.Module:
        """深拷贝模型（用于合并操作）。"""
        import copy

        return copy.deepcopy(model)
