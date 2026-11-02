"""少样本涌现（Few-Shot Emergence）基准：验证少样本学习的涌现特征。

GPT-3 (arXiv:2005.14165) 展示 few-shot 显著提升性能。本基准模拟：
- 在不同「任务训练样本数」下，观察模型是否出现「少样本涌现」现象
- 比较：few-shot（每类少量样本）vs full-shot（完整训练）的性能差距
- 验证：性能随样本数增加是否平滑增长（无涌现）还是阶跃式增长（有涌现）

输出：benchmarks/results/few_shot_emergence_results.json
"""

import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from continual_learning.benchmarks.run_permuted_mnist import (
    MLP,
    get_permuted_mnist,
)


def train_model_on_task(
    model: nn.Module,
    task,
    epochs: int = 3,
    lr: float = 0.001,
) -> None:
    """训练模型在单个任务上。"""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    model.train()
    for _ in range(epochs):
        for inputs, labels in task.dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()


def evaluate_model(model: nn.Module, task) -> float:
    """评估模型在任务上的准确率。"""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in task.eval_dataloader:
            outputs = model(inputs)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total if total > 0 else 0.0


def run_few_shot_experiment(
    shot_counts: list[int] | None = None,
    num_tasks: int = 3,
    epochs: int = 3,
    seed: int = 42,
) -> dict:
    """运行少样本涌现实验。

    Args:
        shot_counts: 每类样本数列表（默认 [1, 5, 20, 50, 100, 500]）。
        num_tasks: 任务数。
        epochs: 训练轮数。
        seed: 随机种子。

    Returns:
        结果字典。
    """
    if shot_counts is None:
        shot_counts = [1, 5, 20, 50, 100, 500]

    results = {"config": {
        "num_tasks": num_tasks,
        "epochs": epochs,
        "seed": seed,
        "shot_counts": shot_counts,
    }, "runs": {}}

    # 生成完整任务（用最大样本数）
    tasks, _ = get_permuted_mnist(
        num_tasks=num_tasks,
        num_samples_per_task=5000,
        seed=seed,
    )

    for shots in shot_counts:
        print(f"\n=== 少样本涌现：每类 {shots} 样本 ===")
        accuracies = []

        for task_idx, task in enumerate(tasks):
            # 用完整数据训练，但评估时限制样本数（模拟少样本场景）
            # 实际少样本学习中，训练样本也少。这里用「限制训练样本数」的方式：
            # 从任务数据集中取前 shots × 10 个样本（10 类）
            full_inputs = []
            full_labels = []
            for inputs, labels in task.dataloader:
                full_inputs.append(inputs)
                full_labels.append(labels)
            if not full_inputs:
                continue

            all_inputs = torch.cat(full_inputs, dim=0)
            all_labels = torch.cat(full_labels, dim=0)

            # 限制训练样本
            n_samples = min(shots * 10, all_inputs.size(0))
            subset_inputs = all_inputs[:n_samples]
            subset_labels = all_labels[:n_samples]

            # 创建子任务
            subset_dataset = TensorDataset(subset_inputs, subset_labels)
            subset_loader = DataLoader(subset_dataset, batch_size=32, shuffle=True)

            # 用新模型训练（避免遗忘）
            model = MLP()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            criterion = nn.CrossEntropyLoss()
            model.train()
            for _ in range(epochs):
                for inputs, labels in subset_loader:
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

            # 评估
            acc = evaluate_model(model, task)
            accuracies.append(acc)
            print(f"  Task{task_idx + 1}: acc={acc:.2f}")

        avg_acc = sum(accuracies) / len(accuracies) if accuracies else 0.0
        results["runs"][str(shots)] = {
            "shots_per_class": shots,
            "avg_accuracy": avg_acc,
            "task_accuracies": accuracies,
        }
        print(f"  平均准确率: {avg_acc:.2f}")

    return results


if __name__ == "__main__":
    results = run_few_shot_experiment()
    out_path = Path(__file__).parent / "results" / "few_shot_emergence_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n结果已保存至：{out_path}")
