"""真实数据验证：CIFAR-10 持续学习基准（可选，当前不可用）。

⚠️ 当前不可用：CIFAR-10 数据压缩包已损坏（`data/cifar-10-python.tar.gz` 127.2MB，标准 ~170MB，
tarfile 解压失败）。保留此文件作为基准接口示例，若下载完整 CIFAR-10 数据可重新启用。

使用 CIFAR-10 类别增量基准（split CIFAR-10）：
- 10 个类别分为 5 组（每组 2 个类别），作为 5 个连续任务
- 每个任务引入新的类别，模型必须保留旧类别性能
- 评估各策略的遗忘曲线

运行：
    python -m continual_learning.benchmarks.run_split_cifar10

数据下载（~170MB）：
    wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz -P data/
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision import datasets, transforms

# 确保可导入 continual_learning
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from continual_learning.base import Task
from continual_learning.orchestrator import ContinualOrchestrator
from continual_learning.strategies.ewc import EWCTrainingStrategy
from continual_learning.strategies.distillation import DistillationStrategy
from continual_learning.strategies.replay import ReplayStrategy


class CNN(nn.Module):
    """小型 CNN（用于 CIFAR-10 基准）。"""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


def get_split_cifar10(
    num_tasks: int = 5,
    num_samples_per_task: int = 2000,
    seed: int = 42,
    data_dir: str = "data",
) -> tuple[list[Task], dict]:
    """生成 split CIFAR-10 任务序列。

    Args:
        num_tasks: 任务数量。
        num_samples_per_task: 每个任务的训练样本数（默认 2000，控制运行时间）。
        seed: 随机种子。
        data_dir: 数据目录。

    Returns:
        (任务列表, 类别映射)。
    """
    # 加载 CIFAR-10
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    train = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)
    test = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform)

    # 10 个类别分为 num_tasks 组
    classes_per_task = 10 // num_tasks
    torch.manual_seed(seed)

    tasks: list[Task] = []
    class_map = {}

    for task_idx in range(num_tasks):
        start_class = task_idx * classes_per_task
        end_class = start_class + classes_per_task

        # 收集该类别的数据
        train_indices = [i for i, (_, label) in enumerate(train) if start_class <= label < end_class]
        test_indices = [i for i, (_, label) in enumerate(test) if start_class <= label < end_class]

        # 采样训练子集
        np.random.seed(seed)
        if len(train_indices) > num_samples_per_task:
            train_indices = np.random.choice(
                train_indices, num_samples_per_task, replace=False
            ).tolist()

        train_subset = Subset(train, train_indices)
        test_subset = Subset(test, test_indices)

        train_loader = DataLoader(train_subset, batch_size=64, shuffle=True)
        test_loader = DataLoader(test_subset, batch_size=256, shuffle=False)

        task = Task(
            name=f"SplitCIFAR_Task{task_idx + 1}",
            dataloader=train_loader,
            eval_dataloader=test_loader,
        )
        tasks.append(task)

        # 记录类别映射
        class_map[f"Task{task_idx + 1}"] = (start_class, end_class)

    return tasks, class_map


def run_experiment(
    strategy_name: str,
    num_tasks: int = 5,
    epochs: int = 5,
    seed: int = 42,
) -> dict:
    """运行一个策略的实验。"""
    torch.manual_seed(seed)
    np.random.seed(seed)

    # 生成任务
    tasks, _ = get_split_cifar10(num_tasks=num_tasks, seed=seed)

    # 创建模型
    model = CNN()

    # 创建策略
    strategy = None
    if strategy_name == "ewc":
        strategy = EWCTrainingStrategy(fisher_multiplier=400.0, num_samples=100)
    elif strategy_name == "distillation":
        strategy = DistillationStrategy(temperature=2.0, alpha=0.5)
    elif strategy_name == "replay":
        strategy = ReplayStrategy(memory_size=1000, replay_ratio=0.2)

    # 创建编排器
    orchestrator = ContinualOrchestrator(model, strategy)

    # 添加任务
    for task in tasks:
        orchestrator.add_task(task)

    # 运行
    results = orchestrator.run(epochs=epochs)

    return {
        "strategy": strategy_name,
        "metrics": results.to_dict(),
        "accuracies": results.accuracies,
        "task_names": results.task_names,
    }


def main() -> None:
    """主入口。"""
    print("=== Split CIFAR-10 持续学习基准 ===")
    print("比较策略：none / ewc / distillation / replay\n")

    strategies = ["none", "ewc", "distillation", "replay"]
    all_results = {}

    for strategy_name in strategies:
        print(f"\n--- 运行策略：{strategy_name} ---")
        try:
            result = run_experiment(strategy_name, num_tasks=5, epochs=3)
            all_results[strategy_name] = result
            m = result["metrics"]
            print(f"  整体性能: {m['overall_performance']:.2f}")
            print(f"  平均遗忘: {m['average_forgetting']:.2f}")
            print(f"  平均前向迁移: {m['average_forward_transfer']:.2f}")
        except Exception as e:
            print(f"  失败: {e}")
            import traceback
            traceback.print_exc()

    # 保存结果
    output = Path(__file__).parent / "results" / "split_cifar10_results.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(all_results, indent=2, default=str))
    print(f"\n结果已保存至: {output}")


if __name__ == "__main__":
    main()
