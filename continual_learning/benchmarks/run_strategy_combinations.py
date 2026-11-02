"""策略组合基准（Strategy Combinations Benchmark）。

检验多个持续学习策略组合的协同效应：
- EWC + Replay
- EWC + Distillation
- Replay + Distillation
- EWC + Replay + Distillation

基于前沿研究_08 的结论：
- 蒸馏最有效保留旧知识
- 回放需与蒸馏结合（避免记忆过拟合）
- EWC 在小型任务有效但 LLM 规模无效

运行：
    python -m continual_learning.benchmarks.run_strategy_combinations
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

# 确保可导入 continual_learning
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from continual_learning.base import Task
from continual_learning.orchestrator import ContinualOrchestrator
from continual_learning.strategies.ewc import EWCTrainingStrategy
from continual_learning.strategies.distillation import DistillationStrategy
from continual_learning.strategies.replay import ReplayStrategy
from continual_learning.strategies.composite import CompositeStrategy


class MLP(nn.Module):
    """简单多层感知机（用于 permuted MNIST）。"""

    def __init__(self, input_dim: int = 784, hidden_dim: int = 256, num_classes: int = 10) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


def get_permuted_mnist(
    num_tasks: int = 5,
    seed: int = 42,
    data_dir: str = "data",
) -> tuple[list[Task], list[torch.Tensor]]:
    """生成 permuted MNIST 任务序列。"""
    from torchvision import datasets, transforms

    # 加载 MNIST
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
    test = datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)

    # 创建任务
    rng = np.random.RandomState(seed)
    tasks: list[Task] = []
    permutations: list[torch.Tensor] = []

    for task_idx in range(num_tasks):
        # 生成随机排列
        perm = torch.randperm(784) if task_idx == 0 else torch.tensor(rng.permutation(784))
        permutations.append(perm)

        # 创建自定义数据集（应用排列）
        train_data = train.data.view(-1, 784)[:, perm].float() / 255.0
        train_labels = train.targets
        test_data = test.data.view(-1, 784)[:, perm].float() / 255.0
        test_labels = test.targets

        from torch.utils.data import TensorDataset, DataLoader

        train_dataset = TensorDataset(train_data, train_labels)
        test_dataset = TensorDataset(test_data, test_labels)

        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

        task = Task(
            name=f"PermutedMNIST_Task{task_idx + 1}",
            dataloader=train_loader,
            eval_dataloader=test_loader,
        )
        tasks.append(task)

    return tasks, permutations


def create_composite_strategy(
    combo: str,
    weights: Optional[list[float]] = None,
) -> CompositeStrategy:
    """创建组合策略。"""
    strategies = []
    if "ewc" in combo:
        strategies.append(EWCTrainingStrategy(fisher_multiplier=400.0, num_samples=100))
    if "replay" in combo:
        strategies.append(ReplayStrategy(memory_size=1000, replay_ratio=0.2))
    if "distill" in combo:
        strategies.append(DistillationStrategy(temperature=2.0, alpha=0.5))

    if weights is None:
        weights = [1.0] * len(strategies)

    return CompositeStrategy(strategies, weights=weights)


def run_experiment(
    combo: str,
    num_tasks: int = 5,
    epochs: int = 3,
    seed: int = 42,
) -> dict:
    """运行一个组合策略实验。"""
    torch.manual_seed(seed)
    np.random.seed(seed)

    # 生成任务
    tasks, _ = get_permuted_mnist(num_tasks=num_tasks, seed=seed)

    # 创建模型
    model = MLP()

    # 创建组合策略
    strategy = create_composite_strategy(combo)

    # 创建编排器
    orchestrator = ContinualOrchestrator(model, strategy)

    # 添加任务
    for task in tasks:
        orchestrator.add_task(task)

    # 运行
    results = orchestrator.run(epochs=epochs)

    return {
        "strategy": combo,
        "metrics": results.to_dict(),
        "accuracies": results.accuracies,
        "task_names": results.task_names,
    }


def main() -> None:
    """主入口。"""
    print("=== 策略组合基准（permuted MNIST）===")
    print("比较组合：ewc+replay / ewc+distill / replay+distill / ewc+replay+distill\n")

    combos = ["ewc+replay", "ewc+distill", "replay+distill", "ewc+replay+distill"]
    all_results = {}

    for combo in combos:
        print(f"\n--- 运行组合：{combo} ---")
        try:
            result = run_experiment(combo, num_tasks=5, epochs=3)
            all_results[combo] = result
            m = result["metrics"]
            print(f"  整体性能: {m['overall_performance']:.2f}")
            print(f"  平均遗忘: {m['average_forgetting']:.2f}")
            print(f"  平均前向迁移: {m['average_forward_transfer']:.2f}")
        except Exception as e:
            print(f"  失败: {e}")
            import traceback
            traceback.print_exc()

    # 保存结果
    output = Path(__file__).parent / "results" / "strategy_combinations_results.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(all_results, indent=2, default=str))
    print(f"\n结果已保存至: {output}")


if __name__ == "__main__":
    main()
