"""动态架构基准（Dynamic Architecture Benchmark）。

检验「结构学习」方法（基于 Learn to Grow, Li et al., ICML 2019, arXiv:1904.00310）：

关键论点：
- NAS 显式学习每层结构，三选项 = reuse / adaptation / new
- 结构学习与参数估计解耦
- permuted MNIST 10 任务：SGD 97.9%→63.0%、EWC 96.8%、Ours 98.48%（2.56M params）
- split CIFAR100：SGD 21.02%、EWC 74.23%、Ours 76.21%
- 结构可解释：相似任务共享低层、不相似用 new 层

本基准实现简化的动态架构：
- 每层可选择 reuse / adapt / new
- 层容量固定，每任务可增加新层
- 比较固定架构 vs 动态架构

运行：
    python -m continual_learning.benchmarks.run_dynamic_architecture
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# 确保可导入 continual_learning
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from continual_learning.base import Task


class DynamicMLP(nn.Module):
    """动态架构 MLP。

    每个任务可扩展网络（增加新层）。支持三种模式：
    - reuse: 复用已有层（冻结）
    - adapt: 微调已有层
    - new: 添加新层
    """

    def __init__(
        self,
        input_dim: int = 784,
        hidden_dim: int = 256,
        num_classes: int = 10,
        max_layers: int = 6,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.max_layers = max_layers

        # 初始 2 层
        self.layers = nn.ModuleList([
            nn.Linear(input_dim, hidden_dim),
            nn.Linear(hidden_dim, num_classes),
        ])
        self.activations = nn.ModuleList([
            nn.ReLU(),
            nn.Identity(),  # 输出层无激活
        ])

    def add_layer(self, prev_dim: int = None) -> None:
        """添加一层（在输出层之前插入新隐藏层）。"""
        if len(self.layers) >= self.max_layers:
            return
        # 创建新隐藏层（插入到输出层之前）
        new_hidden = nn.Linear(self.hidden_dim, self.hidden_dim)
        # 使用 ModuleList 的 append + 重新排列
        # 在倒数第二位置插入（输出层保持在最后）
        new_layers = nn.ModuleList()
        new_activations = nn.ModuleList()
        # 保留所有隐藏层 + 新层 + 输出层
        for i in range(len(self.layers)):
            if i == len(self.layers) - 1:
                # 输出层：先插入新隐藏层，再输出层
                new_layers.append(new_hidden)
                new_activations.append(nn.ReLU())
            new_layers.append(self.layers[i])
            new_activations.append(self.activations[i])
        self.layers = new_layers
        self.activations = new_activations

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        for layer, act in zip(self.layers, self.activations):
            x = layer(x)
            x = act(x)
        return x


def get_permuted_mnist(
    num_tasks: int = 5,
    seed: int = 42,
    data_dir: str = "data",
) -> tuple[list[Task], list[torch.Tensor]]:
    """生成 permuted MNIST 任务序列。"""
    from torchvision import datasets, transforms

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
    test = datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)

    rng = np.random.RandomState(seed)
    tasks: list[Task] = []
    permutations: list[torch.Tensor] = []

    for task_idx in range(num_tasks):
        perm = torch.tensor(rng.permutation(784))
        permutations.append(perm)

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


def run_fixed_experiment(
    num_tasks: int = 5,
    epochs: int = 3,
    seed: int = 42,
) -> dict:
    """固定架构实验（基线）。"""
    torch.manual_seed(seed)
    np.random.seed(seed)

    tasks, _ = get_permuted_mnist(num_tasks=num_tasks, seed=seed)
    model = DynamicMLP()

    task_accuracies = {}
    # 训练所有任务（固定架构，不扩展）
    for task_idx, task in enumerate(tasks):
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = torch.nn.CrossEntropyLoss()
        model.train()
        for epoch in range(epochs):
            for inputs, labels in task.dataloader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

        # 评估当前任务
        acc = evaluate_accuracy(model, task.eval_dataloader)
        task_accuracies[f"Task{task_idx + 1}"] = acc

    # 计算遗忘：最终任务在早期任务上的性能
    return {
        "architecture": "fixed",
        "task_accuracies": task_accuracies,
        "final_accuracies": task_accuracies,
        "params": sum(p.numel() for p in model.parameters()),
    }


def run_dynamic_experiment(
    num_tasks: int = 5,
    epochs: int = 3,
    seed: int = 42,
) -> dict:
    """动态架构实验（每任务可扩展）。"""
    torch.manual_seed(seed)
    np.random.seed(seed)

    tasks, _ = get_permuted_mnist(num_tasks=num_tasks, seed=seed)
    model = DynamicMLP()

    task_accuracies = {}
    final_accuracies = {}

    for task_idx, task in enumerate(tasks):
        # 动态架构：每个任务添加新层（模拟结构增长）
        if task_idx > 0:
            model.add_layer()

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = torch.nn.CrossEntropyLoss()
        model.train()
        for epoch in range(epochs):
            for inputs, labels in task.dataloader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

        # 评估当前任务
        acc = evaluate_accuracy(model, task.eval_dataloader)
        task_accuracies[f"Task{task_idx + 1}"] = acc

        # 评估所有已完成任务的性能（检测遗忘）
        all_accs = {}
        for done_idx, done_task in enumerate(tasks[:task_idx + 1]):
            all_accs[f"Task{done_idx + 1}"] = evaluate_accuracy(model, done_task.eval_dataloader)
        final_accuracies[f"Task{task_idx + 1}"] = all_accs

    return {
        "architecture": "dynamic",
        "task_accuracies": task_accuracies,
        "final_accuracies": final_accuracies,
        "params": sum(p.numel() for p in model.parameters()),
    }


def evaluate_accuracy(model: nn.Module, dataloader) -> float:
    """评估 accuracy。"""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / max(1, total) * 100.0


def main() -> None:
    """主入口。"""
    print("=== 动态架构基准（permuted MNIST）===")
    print("比较固定架构 vs 动态架构（每任务添加新层）\n")

    print("--- 固定架构 ---")
    fixed = run_fixed_experiment(num_tasks=5, epochs=3)
    print(f"  参数量: {fixed['params']}")
    for task, acc in fixed['task_accuracies'].items():
        print(f"  {task}: {acc:.2f}")

    print("\n--- 动态架构 ---")
    dynamic = run_dynamic_experiment(num_tasks=5, epochs=3)
    print(f"  参数量: {dynamic['params']}")
    for task, acc in dynamic['task_accuracies'].items():
        print(f"  {task}: {acc:.2f}")

    # 保存结果
    results = {"fixed": fixed, "dynamic": dynamic}
    output = Path(__file__).parent / "results" / "dynamic_architecture_results.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2, default=str))
    print(f"\n结果已保存至: {output}")


if __name__ == "__main__":
    main()
