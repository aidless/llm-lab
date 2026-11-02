"""命令行入口（CLI）。

用法：
    python -m continual_learning.cli train --tasks 3 --epochs 5 --strategy ewc
    python -m continual_learning.cli train --tasks 3 --epochs 5 --strategy distillation
    python -m continual_learning.cli train --tasks 3 --epochs 5 --strategy lora
    python -m continual_learning.cli merge --models model1.pt model2.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from .base import StrategyConfig, Task
from .orchestrator import ContinualOrchestrator
from .strategies.distillation import DistillationStrategy
from .strategies.ewc import EWCTrainingStrategy
from .strategies.lora import LoRAStrategy
from .strategies.model_soups import ModelSoupsMerger
from .strategies.replay import ReplayStrategy


class SimpleMLP(nn.Module):
    """简单多层感知机，用于演示和测试。"""

    def __init__(self, input_dim: int = 28 * 28, hidden_dim: int = 128, num_classes: int = 10) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


def create_task(
    name: str,
    num_samples: int = 200,
    num_classes: int = 10,
    input_dim: int = 28 * 28,
    batch_size: int = 32,
) -> Task:
    """创建一个合成任务用于演示。"""
    from torch.utils.data import DataLoader, TensorDataset

    # 生成合成数据
    torch.manual_seed(42)
    X = torch.randn(num_samples, input_dim)
    y = torch.randint(0, num_classes, (num_samples,))

    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return Task(name=name, dataloader=dataloader)


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="持续学习框架 CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # train 子命令
    train_parser = subparsers.add_parser("train", help="训练持续学习序列")
    train_parser.add_argument("--tasks", type=int, default=3, help="任务数量")
    train_parser.add_argument("--epochs", type=int, default=5, help="每个任务的训练轮数")
    train_parser.add_argument("--strategy", type=str, default="ewc",
                              choices=["ewc", "distillation", "lora", "replay", "none",
                                       "ewc+replay", "ewc+distill", "replay+distill",
                                       "ewc+replay+distill"],
                              help="持续学习策略（支持组合）")
    train_parser.add_argument("--lr", type=float, default=0.001, help="学习率")
    train_parser.add_argument("--seed", type=int, default=42, help="随机种子")
    train_parser.add_argument("--samples-per-task", type=int, default=200,
                              help="每个任务的样本数")
    train_parser.add_argument("--output", type=str, default="metrics.json", help="输出文件")

    # merge 子命令
    merge_parser = subparsers.add_parser("merge", help="合并模型")
    merge_parser.add_argument("--models", type=str, nargs="+", required=True, help="模型文件路径列表")
    merge_parser.add_argument("--method", type=str, default="uniform",
                              choices=["uniform", "greedy"], help="合并方式")
    merge_parser.add_argument("--output", type=str, default="merged_model.pt", help="输出文件")

    # benchmark 子命令
    bench_parser = subparsers.add_parser("benchmark", help="运行基准测试")
    bench_parser.add_argument("--dataset", type=str, default="permuted_mnist",
                              choices=["permuted_mnist", "split_cifar10", "continuous_metrics",
                                       "dynamic_architecture", "strategy_combinations", "all"],
                              help="基准测试类型")
    bench_parser.add_argument("--strategies", type=str, default="none,ewc,replay,distillation,lora",
                              help="逗号分隔的策略列表")
    bench_parser.add_argument("--tasks", type=int, default=5, help="任务数量")
    bench_parser.add_argument("--epochs", type=int, default=3, help="每个任务的训练轮数")
    bench_parser.add_argument("--seed", type=int, default=42, help="随机种子")
    bench_parser.add_argument("--output", type=str, default="benchmark_results.json", help="输出文件")

    # visualize 子命令
    vis_parser = subparsers.add_parser("visualize", help="可视化结果")
    vis_parser.add_argument("--results", type=str, required=True, help="结果 JSON 文件")
    vis_parser.add_argument("--output-dir", type=str, default="figs", help="输出目录")

    return parser.parse_args()


def main() -> None:
    """主入口。"""
    args = parse_args()

    if args.command == "train":
        run_train(args)
    elif args.command == "merge":
        run_merge(args)
    elif args.command == "benchmark":
        run_benchmark(args)
    elif args.command == "visualize":
        run_visualize(args)


def run_visualize(args: argparse.Namespace) -> None:
    """运行可视化。"""
    from .visualization import visualize_all

    print(f"=== 可视化：{args.results} ===")
    visualize_all(args.results, args.output_dir)
    print(f"可视化完成，输出目录：{args.output_dir}")


def run_train(args: argparse.Namespace) -> None:
    """运行持续学习训练。"""
    print(f"=== 持续学习训练：{args.tasks} 个任务，策略={args.strategy} ===")

    # 设置种子
    torch.manual_seed(args.seed)

    # 创建模型
    model = SimpleMLP()
    print(f"模型参数量：{sum(p.numel() for p in model.parameters()):,}")

    # 创建策略（支持组合）
    strategy = None
    if args.strategy == "ewc":
        strategy = EWCTrainingStrategy(fisher_multiplier=400.0, num_samples=100)
        print("策略：EWC（Fisher 重要性 + 二次惩罚）")
    elif args.strategy == "distillation":
        strategy = DistillationStrategy(temperature=2.0, alpha=0.5)
        print("策略：知识蒸馏（Logit 蒸馏）")
    elif args.strategy == "lora":
        strategy = LoRAStrategy(rank=4, alpha=1.0)
        print("策略：LoRA（低秩适配）")
    elif args.strategy == "replay":
        strategy = ReplayStrategy(memory_size=1000, replay_ratio=0.2)
        print("策略：数据回放")
    elif "+" in args.strategy:
        # 组合策略
        from .strategies.composite import CompositeStrategy
        sub_strategies = []
        for name in args.strategy.split("+"):
            if name == "ewc":
                sub_strategies.append(EWCTrainingStrategy(fisher_multiplier=400.0, num_samples=100))
            elif name == "distill":
                sub_strategies.append(DistillationStrategy(temperature=2.0, alpha=0.5))
            elif name == "replay":
                sub_strategies.append(ReplayStrategy(memory_size=1000, replay_ratio=0.2))
        strategy = CompositeStrategy(sub_strategies)
        print(f"策略：组合（{args.strategy}）")
    else:
        print("策略：无（基线）")

    # 创建编排器
    orchestrator = ContinualOrchestrator(model, strategy)

    # 添加任务
    for i in range(args.tasks):
        task = create_task(name=f"Task_{i+1}", num_samples=args.samples_per_task, num_classes=10)
        orchestrator.add_task(task)
        print(f"  已添加任务：Task_{i+1}")

    # 运行训练
    results = orchestrator.run(epochs=args.epochs, learning_rate=args.lr)

    # 输出结果
    metrics = results.to_dict()
    print(f"\n=== 评估结果 ===")
    print(json.dumps(metrics, indent=2))

    # 保存结果
    output_path = Path(args.output)
    output_path.write_text(json.dumps(metrics, indent=2))
    print(f"\n结果已保存至：{output_path}")


def run_merge(args: argparse.Namespace) -> None:
    """运行模型合并。"""
    print(f"=== 模型合并：{len(args.models)} 个模型，方法={args.method} ===")

    merger = ModelSoupsMerger()
    for model_path in args.models:
        model = SimpleMLP()
        state = torch.load(model_path, map_location="cpu")
        model.load_state_dict(state)
        merger.add_model(model)
        print(f"  已加载模型：{model_path}")

    if args.method == "uniform":
        merged = merger.uniform_soup()
    else:
        # 贪心合并需要评估函数，这里使用一个简单示例
        def eval_fn(m: nn.Module) -> float:
            return 0.5  # 示例评估分数

        merged = merger.greedy_soup(eval_fn)

    torch.save(merged.state_dict(), args.output)
    print(f"合并模型已保存至：{args.output}")


def run_benchmark(args: argparse.Namespace) -> None:
    """运行基准测试。"""
    print(f"=== 基准测试：{args.dataset} ===")
    print(f"策略：{args.strategies}")
    print(f"任务数：{args.tasks}，轮数：{args.epochs}，种子：{args.seed}")

    strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]

    if args.dataset == "permuted_mnist":
        from .benchmarks.run_permuted_mnist import run_experiment as run_pm
        all_results = {}
        for strat in strategies:
            print(f"\n--- 策略：{strat} ---")
            result = run_experiment_dispatch(strat, args.tasks, args.epochs, args.seed)
            all_results[strat] = result
            m = result["metrics"]
            print(f"  整体性能: {m['overall_performance']:.2f}")
            print(f"  平均遗忘: {m['average_forgetting']:.2f}")
    elif args.dataset == "all":
        from .benchmarks.run_all import run_all_benchmarks
        print("运行全部基准测试（可能需要较长时间）...")
        all_results = run_all_benchmarks(
            num_tasks=args.tasks,
            epochs=args.epochs,
            seed=args.seed,
            output_dir=str(Path(args.output).parent),
        )
        print("全部基准测试完成")
    elif args.dataset == "continuous_metrics":
        from .benchmarks.run_continuous_metrics import run_experiment as run_cm
        results = run_cm(num_tasks=args.tasks, epochs=args.epochs, seed=args.seed)
        all_results = results
        for task, metrics in results.items():
            print(f"  {task}: acc={metrics['accuracy']:.2f}, brier={metrics['brier_score']:.4f}")
    elif args.dataset == "dynamic_architecture":
        from .benchmarks.run_dynamic_architecture import (
            run_fixed_experiment,
            run_dynamic_experiment,
        )
        fixed = run_fixed_experiment(num_tasks=args.tasks, epochs=args.epochs, seed=args.seed)
        dynamic = run_dynamic_experiment(num_tasks=args.tasks, epochs=args.epochs, seed=args.seed)
        all_results = {"fixed": fixed, "dynamic": dynamic}
        print(f"  固定架构 params={fixed['params']}, accs={fixed['task_accuracies']}")
        print(f"  动态架构 params={dynamic['params']}, accs={dynamic['task_accuracies']}")
    elif args.dataset == "strategy_combinations":
        from .benchmarks.run_strategy_combinations import run_experiment as run_sc
        all_results = {}
        for strat in strategies:
            print(f"\n--- 组合：{strat} ---")
            result = run_experiment_dispatch(strat, args.tasks, args.epochs, args.seed)
            all_results[strat] = result
            m = result["metrics"]
            print(f"  整体性能: {m['overall_performance']:.2f}")
            print(f"  平均遗忘: {m['average_forgetting']:.2f}")
    else:
        print(f"不支持的基准：{args.dataset}")
        return

    # 保存结果
    output_path = Path(args.output)
    output_path.write_text(json.dumps(all_results, indent=2, default=str))
    print(f"\n结果已保存至：{output_path}")


def run_experiment_dispatch(
    strategy: str,
    num_tasks: int,
    epochs: int,
    seed: int,
) -> dict:
    """分派到对应策略实验（支持组合）。"""
    from .benchmarks.run_permuted_mnist import run_experiment as run_pm
    from .benchmarks.run_strategy_combinations import (
        run_experiment as run_sc,
        create_composite_strategy,
    )

    if "+" in strategy:
        return run_sc(strategy, num_tasks=num_tasks, epochs=epochs, seed=seed)
    return run_pm(strategy, num_tasks=num_tasks, epochs=epochs, seed=seed)


if __name__ == "__main__":
    main()
