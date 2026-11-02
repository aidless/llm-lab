"""连续度量基准（Continuous Metrics Benchmark）。

检验「度量选择对表观涌现/能力的影响」（基于 Emergent Mirage, Schaeffer et al. ICLR 2024）：

关键论点（arXiv:2304.15004）：
- 涌现能力是**度量伪影**（metric artifact），不是模型的基本属性变化
- 非线性/不连续度量（如 accuracy）产生表观涌现
- 线性/连续度量（如 Brier Score / token 误差率）揭示平滑改善
- 两个条件：度量非线性变形 + 测试数据太少

本基准验证：
1. 在 permuted MNIST 上，比较 accuracy（不连续）与 Brier Score（连续）
2. 检查是否出现「表观涌现」或平滑改善
3. 改变测试集大小，验证「测试数据太少」条件

运行：
    python -m continual_learning.benchmarks.run_continuous_metrics
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# 确保可导入 continual_learning
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from continual_learning.base import Task


class MLP(nn.Module):
    """简单多层感知机。"""

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


def compute_accuracy(model: nn.Module, dataloader) -> float:
    """计算 accuracy（不连续度量）。"""
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


def compute_brier_score(model: nn.Module, dataloader) -> float:
    """计算 Brier Score（连续度量）。

    Brier Score = mean(sum_c (p_c - y_c)^2)，其中 y 是 one-hot 真实标签。
    越低越好。与 accuracy 不同，Brier Score 是平滑/连续函数。
    """
    model.eval()
    total_brier = 0.0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            outputs = model(inputs)
            probs = F.softmax(outputs, dim=1)
            # one-hot 标签
            one_hot = F.one_hot(labels, num_classes=probs.size(1)).float()
            brier = ((probs - one_hot) ** 2).sum(dim=1)
            total_brier += brier.sum().item()
            total += labels.size(0)
    return total_brier / max(1, total)


def compute_error_rate(model: nn.Module, dataloader) -> float:
    """计算错误率（1 - accuracy，连续但分段）。"""
    return 100.0 - compute_accuracy(model, dataloader)


def run_experiment(
    num_tasks: int = 5,
    epochs: int = 3,
    seed: int = 42,
) -> dict:
    """运行实验：训练 5 个任务，比较不同度量。"""
    torch.manual_seed(seed)
    np.random.seed(seed)

    tasks, _ = get_permuted_mnist(num_tasks=num_tasks, seed=seed)

    model = MLP()

    # 训练每个任务
    task_metrics = {}
    for task_idx, task in enumerate(tasks):
        # 训练当前任务
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

        # 评估当前任务（用不同度量）
        acc = compute_accuracy(model, task.eval_dataloader)
        brier = compute_brier_score(model, task.eval_dataloader)
        err = compute_error_rate(model, task.eval_dataloader)

        # 不同测试集大小下的 accuracy（验证「测试数据太少」条件）
        acc_small = compute_accuracy_small(model, task.eval_dataloader, frac=0.1)
        acc_tiny = compute_accuracy_small(model, task.eval_dataloader, frac=0.01)

        task_metrics[f"Task{task_idx + 1}"] = {
            "accuracy": acc,
            "brier_score": brier,
            "error_rate": err,
            "accuracy_small": acc_small,
            "accuracy_tiny": acc_tiny,
        }

    return task_metrics


def compute_accuracy_small(model: nn.Module, dataloader, frac: float = 0.1) -> float:
    """用小样本评估 accuracy（验证「测试数据太少」条件）。"""
    # 取前 frac 比例的样本
    all_inputs = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in dataloader:
            all_inputs.append(inputs)
            all_labels.append(labels)
    inputs = torch.cat(all_inputs)
    labels = torch.cat(all_labels)

    n = len(inputs)
    n_sample = max(1, int(n * frac))
    # 固定随机选择
    torch.manual_seed(42)
    idx = torch.randperm(n)[:n_sample]
    inputs_small = inputs[idx]
    labels_small = labels[idx]

    model.eval()
    correct = 0
    with torch.no_grad():
        outputs = model(inputs_small)
        _, predicted = outputs.max(1)
        correct = (predicted == labels_small).sum().item()
    return correct / max(1, n_sample) * 100.0


def main() -> None:
    """主入口。"""
    print("=== 连续度量基准（permuted MNIST）===")
    print("验证 Emergent Mirage：度量选择影响表观涌现\n")

    results = run_experiment(num_tasks=5, epochs=3)

    print(f"{'任务':<8} {'accuracy':>10} {'Brier':>8} {'error_rate':>12} {'acc_10%':>10} {'acc_1%':>8}")
    print("-" * 62)
    for task_name, metrics in results.items():
        print(
            f"{task_name:<8} {metrics['accuracy']:>10.2f} {metrics['brier_score']:>8.4f} "
            f"{metrics['error_rate']:>12.2f} {metrics['accuracy_small']:>10.2f} {metrics['accuracy_tiny']:>8.2f}"
        )

    # 保存结果
    output = Path(__file__).parent / "results" / "continuous_metrics_results.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2, default=str))
    print(f"\n结果已保存至: {output}")


if __name__ == "__main__":
    main()
