"""数据配比（Data Ratio）基准：验证 Chinchilla 定律的等比例缩放。

Chinchilla (arXiv:2203.15556)：最优训练是模型参数与 token 数等比例增长
（模型翻倍 → token 翻倍）。本基准在 permuted MNIST 上模拟：
- 固定模型规模，改变数据量（每任务样本数）
- 观察性能如何随数据量变化

输出：benchmarks/results/data_ratio_results.json
"""

import json
import os
import time
from pathlib import Path

from continual_learning.benchmarks.run_permuted_mnist import (
    MLP,
    get_permuted_mnist,
)
from continual_learning.metrics import CLMetrics
from continual_learning.orchestrator import ContinualOrchestrator


def run_data_ratio_experiment(
    data_ratios: list[int] | None = None,
    num_tasks: int = 5,
    epochs: int = 3,
    seed: int = 42,
    base_samples: int = 5000,
) -> dict:
    """运行数据配比实验。

    Args:
        data_ratios: 每任务样本数列表（默认 [1250, 2500, 5000, 10000]）。
        num_tasks: 任务数。
        epochs: 每任务训练轮数。
        seed: 随机种子。
        base_samples: 基准样本数（用于计算倍数）。

    Returns:
        结果字典。
    """
    if data_ratios is None:
        data_ratios = [1250, 2500, 5000, 10000]

    results = {"config": {
        "num_tasks": num_tasks,
        "epochs": epochs,
        "seed": seed,
        "base_samples": base_samples,
        "data_ratios": data_ratios,
    }, "runs": {}}

    for samples in data_ratios:
        print(f"\n=== 数据配比：每任务 {samples} 样本（{samples / base_samples:.2f}× 基准）===")
        tasks, _ = get_permuted_mnist(
            num_tasks=num_tasks,
            num_samples_per_task=samples,
            seed=seed,
        )
        orchestrator = ContinualOrchestrator(MLP(), None)
        metrics = CLMetrics()
        start = time.time()

        for i, task in enumerate(tasks):
            orchestrator.add_task(task)
        results_run = orchestrator.run(epochs=epochs)

        elapsed = time.time() - start
        result = {
            "samples_per_task": samples,
            "ratio": samples / base_samples,
            "overall_performance": results_run.overall_performance(),
            "average_forgetting": results_run.average_forgetting(),
            "task_accuracies": results_run.accuracies,
            "elapsed": round(elapsed, 1),
        }
        results["runs"][str(samples)] = result
        print(f"  OP={result['overall_performance']:.2f}, F={result['average_forgetting']:.2f} ({elapsed:.1f}s)")

    return results


if __name__ == "__main__":
    results = run_data_ratio_experiment()
    out_path = Path(__file__).parent / "results" / "data_ratio_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n结果已保存至：{out_path}")
