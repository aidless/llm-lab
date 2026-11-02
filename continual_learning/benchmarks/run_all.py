"""基准测试套件。

统一运行所有基准测试并汇总结果。

运行：
    python -m continual_learning.benchmarks.run_all 或
    python -m continual_learning.cli benchmark --dataset all
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from .run_permuted_mnist import run_experiment as run_permuted_mnist
from .run_strategy_combinations import run_experiment as run_strategy_combinations
from .run_continuous_metrics import run_experiment as run_continuous_metrics
from .run_dynamic_architecture import (
    run_fixed_experiment,
    run_dynamic_experiment,
)


def run_all_benchmarks(
    num_tasks: int = 5,
    epochs: int = 3,
    seed: int = 42,
    output_dir: str = "results",
) -> dict:
    """运行所有基准测试并汇总。

    Args:
        num_tasks: 任务数量。
        epochs: 每个任务的训练轮数。
        seed: 随机种子。
        output_dir: 输出目录。

    Returns:
        汇总结果 dict。
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    all_results = {}

    # 1. permuted MNIST（5 种策略）
    print("=" * 60)
    print("基准 1：permuted MNIST（5 种策略）")
    print("=" * 60)
    strategies = ["none", "ewc", "distillation", "replay", "lora"]
    all_results["permuted_mnist"] = {}
    for strat in strategies:
        print(f"  策略：{strat}")
        start = time.time()
        result = run_permuted_mnist(strat, num_tasks=num_tasks, epochs=epochs, seed=seed)
        elapsed = time.time() - start
        all_results["permuted_mnist"][strat] = result
        m = result["metrics"]
        print(f"    OP={m['overall_performance']:.2f}, F={m['average_forgetting']:.2f} "
              f"({elapsed:.1f}s)")

    # 2. 策略组合（4 种组合）
    print("\n" + "=" * 60)
    print("基准 2：策略组合（4 种组合）")
    print("=" * 60)
    combos = ["ewc+replay", "ewc+distill", "replay+distill", "ewc+replay+distill"]
    all_results["strategy_combinations"] = {}
    for combo in combos:
        print(f"  组合：{combo}")
        start = time.time()
        result = run_strategy_combinations(combo, num_tasks=num_tasks, epochs=epochs, seed=seed)
        elapsed = time.time() - start
        all_results["strategy_combinations"][combo] = result
        m = result["metrics"]
        print(f"    OP={m['overall_performance']:.2f}, F={m['average_forgetting']:.2f} "
              f"({elapsed:.1f}s)")

    # 3. 连续度量
    print("\n" + "=" * 60)
    print("基准 3：连续度量（accuracy vs Brier Score）")
    print("=" * 60)
    start = time.time()
    cm_results = run_continuous_metrics(num_tasks=num_tasks, epochs=epochs, seed=seed)
    elapsed = time.time() - start
    all_results["continuous_metrics"] = cm_results
    for task, metrics in cm_results.items():
        print(f"  {task}: acc={metrics['accuracy']:.2f}, brier={metrics['brier_score']:.4f}")
    print(f"  ({elapsed:.1f}s)")

    # 4. 动态架构
    print("\n" + "=" * 60)
    print("基准 4：动态架构（固定 vs 动态）")
    print("=" * 60)
    start = time.time()
    fixed = run_fixed_experiment(num_tasks=num_tasks, epochs=epochs, seed=seed)
    dynamic = run_dynamic_experiment(num_tasks=num_tasks, epochs=epochs, seed=seed)
    elapsed = time.time() - start
    all_results["dynamic_architecture"] = {"fixed": fixed, "dynamic": dynamic}
    fixed_accs = list(fixed["task_accuracies"].values())
    dynamic_accs = list(dynamic["task_accuracies"].values())
    print(f"  固定：params={fixed['params']}, accs={[round(a, 2) for a in fixed_accs]}")
    print(f"  动态：params={dynamic['params']}, accs={[round(a, 2) for a in dynamic_accs]}")
    print(f"  ({elapsed:.1f}s)")

    # 保存汇总
    summary_path = output_path / "all_benchmarks_results.json"
    summary_path.write_text(json.dumps(all_results, indent=2, default=str))
    print(f"\n汇总结果已保存：{summary_path}")

    return all_results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="运行所有基准测试")
    parser.add_argument("--tasks", type=int, default=5, help="任务数量")
    parser.add_argument("--epochs", type=int, default=3, help="每个任务的训练轮数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--output-dir", type=str, default="results", help="输出目录")
    args = parser.parse_args()

    run_all_benchmarks(
        num_tasks=args.tasks,
        epochs=args.epochs,
        seed=args.seed,
        output_dir=args.output_dir,
    )
