"""真实数据验证：permuted MNIST 持续学习基准。

使用 permuted MNIST（Kirkpatrick et al. 2017）作为标准 CL 基准：
- 10 个任务，每个任务是 MNIST 像素的固定随机置换
- 模型必须在保留旧任务性能的同时学习新任务
- 评估各策略（EWC/蒸馏/LoRA/回放/基线）的遗忘曲线

运行：
    python -m continual_learning.benchmarks.run_permuted_mnist
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
from continual_learning.metrics import CLMetrics
from continual_learning.orchestrator import ContinualOrchestrator
from continual_learning.strategies.ewc import EWCTrainingStrategy
from continual_learning.strategies.distillation import DistillationStrategy
from continual_learning.strategies.lora import LoRAStrategy, LoRAModel
from continual_learning.strategies.replay import ReplayStrategy


class MLP(nn.Module):
    """标准 MLP（用于 CL 基准）。"""

    def __init__(self, input_dim: int = 28 * 28, hidden_dim: int = 256, num_classes: int = 10) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


def get_permuted_mnist(
    num_tasks: int = 10,
    num_samples_per_task: int = 5000,
    seed: int = 42,
    data_dir: str = "data",
) -> tuple[list[Task], list[torch.Tensor]]:
    """生成 permuted MNIST 任务序列。

    Args:
        num_tasks: 任务数量。
        num_samples_per_task: 每个任务的训练样本数。
        seed: 随机种子。
        data_dir: 数据目录。

    Returns:
        (任务列表, 置换矩阵列表)。
    """
    # 加载 MNIST
    train = datasets.MNIST(root=data_dir, train=True, download=True)
    test = datasets.MNIST(root=data_dir, train=False, download=True)

    # 转换为张量
    train_data = train.data.float() / 255.0
    train_labels = train.targets
    test_data = test.data.float() / 255.0
    test_labels = test.targets

    torch.manual_seed(seed)
    tasks: list[Task] = []
    permutations: list[torch.Tensor] = []

    for task_idx in range(num_tasks):
        # 生成随机置换（第一个任务用恒等置换）
        if task_idx == 0:
            perm = torch.arange(28 * 28)
        else:
            perm = torch.randperm(28 * 28)
        permutations.append(perm)

        # 应用置换
        X_train = train_data.view(-1, 28 * 28)[:, perm]
        X_test = test_data.view(-1, 28 * 28)[:, perm]

        # 采样训练子集
        num_samples = min(num_samples_per_task, len(X_train))
        indices = torch.randperm(len(X_train))[:num_samples]
        X_train_sub = X_train[indices]
        y_train_sub = train_labels[indices]

        # 创建 DataLoader
        train_dataset = TensorDataset(X_train_sub, y_train_sub)
        test_dataset = TensorDataset(X_test, test_labels)

        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

        task = Task(
            name=f"PermutedMNIST_Task{task_idx + 1}",
            dataloader=train_loader,
            eval_dataloader=test_loader,
        )
        tasks.append(task)

    return tasks, permutations


def run_experiment(
    strategy_name: str,
    num_tasks: int = 5,
    epochs: int = 5,
    seed: int = 42,
) -> dict:
    """运行一个策略的实验。

    Args:
        strategy_name: 策略名称（none/ewc/distillation/lora/replay）。
        num_tasks: 任务数量。
        epochs: 每任务训练轮数。
        seed: 随机种子。

    Returns:
        实验结果（指标 + 任务性能矩阵）。
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    # 生成任务
    tasks, _ = get_permuted_mnist(num_tasks=num_tasks, seed=seed)

    # 创建模型
    model = MLP()

    # 创建策略
    strategy = None
    if strategy_name == "ewc":
        strategy = EWCTrainingStrategy(fisher_multiplier=400.0, num_samples=100)
    elif strategy_name == "distillation":
        strategy = DistillationStrategy(temperature=2.0, alpha=0.5)
    elif strategy_name == "replay":
        strategy = ReplayStrategy(memory_size=1000, replay_ratio=0.2)

    # 创建编排器（LoRA 单独处理，见 run_lora_experiment）
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


def run_lora_experiment(
    num_tasks: int = 5,
    epochs: int = 3,
    seed: int = 42,
) -> dict:
    """运行 LoRA 实验（LoRA 需要特殊处理：冻结基座 + 训练适配器）。"""
    torch.manual_seed(seed)
    np.random.seed(seed)

    # 生成任务
    tasks, _ = get_permuted_mnist(num_tasks=num_tasks, seed=seed)

    # 创建基座模型
    base_model = MLP()

    # LoRA 策略
    strategy = LoRAStrategy(rank=4, alpha=1.0)

    # 冻结基座模型
    for param in base_model.parameters():
        param.requires_grad = False

    # 为每个任务创建适配器并训练
    from continual_learning.strategies.lora import LoRAModel
    import torch.nn.functional as F

    task_accuracies = {}
    eval_accuracies = {}

    for task_idx, task in enumerate(tasks):
        # 创建 LoRA 适配器（包装基座）
        adapter = LoRAModel(base_model, rank=4, alpha=1.0)

        # 训练适配器
        optimizer = torch.optim.Adam(adapter.parameters(), lr=0.001)
        criterion = torch.nn.CrossEntropyLoss()
        adapter.train()

        for epoch in range(epochs):
            for batch in task.dataloader:
                inputs, labels = batch
                optimizer.zero_grad()
                outputs = adapter(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

        # 评估
        adapter.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for batch in task.eval_dataloader:
                inputs, labels = batch
                outputs = adapter(inputs)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        task_accuracies[f"Task{task_idx + 1}"] = correct / max(1, total) * 100.0
        eval_accuracies[f"Task{task_idx + 1}"] = task_accuracies[f"Task{task_idx + 1}"]

    # 计算指标
    op = sum(task_accuracies.values()) / len(task_accuracies)
    # LoRA 每个任务有独立适配器，遗忘近似为 0（冻结基座）
    f = 0.0

    return {
        "strategy": "lora",
        "metrics": {
            "overall_performance": op,
            "average_forgetting": f,
            "average_forward_transfer": 0.0,
        },
        "accuracies": task_accuracies,
        "task_names": list(task_accuracies.keys()),
    }


def main() -> None:
    """主入口。"""
    print("=== Permuted MNIST 持续学习基准 ===")
    print("比较策略：none / ewc / distillation / replay / lora\n")

    strategies = ["none", "ewc", "distillation", "replay", "lora"]
    all_results = {}

    for strategy_name in strategies:
        print(f"\n--- 运行策略：{strategy_name} ---")
        try:
            if strategy_name == "lora":
                result = run_lora_experiment(num_tasks=5, epochs=3)
            else:
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
    output = Path(__file__).parent / "results" / "permuted_mnist_results.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(all_results, indent=2, default=str))
    print(f"\n结果已保存至: {output}")


if __name__ == "__main__":
    main()
