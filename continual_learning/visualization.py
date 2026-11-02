"""持续学习结果可视化。

生成基准测试结果的可视化图表：
- 策略对比图（整体性能 vs 遗忘）
- 任务准确率曲线（遗忘曲线）
- 连续度量对比图（accuracy vs Brier Score）

运行：
    python -m continual_learning.visualization 或
    python -m continual_learning.cli visualize --results results.json --output figs/
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # 无 GUI 环境
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np

# 设置中文字体（优先 Noto Sans CJK SC）
_available_fonts = {f.name for f in fm.fontManager.ttflist}
_CJK_CANDIDATES = [
    "Noto Sans CJK SC", "Noto Serif CJK SC", "WenQuanYi Zen Hei",
    "SimHei", "Microsoft YaHei", "PingFang SC",
]
for _font in _CJK_CANDIDATES:
    if _font in _available_fonts:
        plt.rcParams["font.sans-serif"] = [_font, "DejaVu Sans"]
        break
plt.rcParams["axes.unicode_minus"] = False


def plot_strategy_comparison(
    results: dict,
    output_path: str = "figs/strategy_comparison.png",
) -> None:
    """绘制策略对比图（散点图：整体性能 vs 遗忘）。

    Args:
        results: 各策略结果 dict（strategy -> {metrics: {overall_performance, average_forgetting}}）。
        output_path: 输出路径。
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    strategies = list(results.keys())
    op_values = [results[s]["metrics"]["overall_performance"] for s in strategies]
    f_values = [results[s]["metrics"]["average_forgetting"] for s in strategies]

    colors = plt.cm.tab10(np.linspace(0, 1, len(strategies)))
    for i, (s, op, f) in enumerate(zip(strategies, op_values, f_values)):
        ax.scatter(op, f, color=colors[i], s=100, label=s, zorder=5)
        ax.annotate(s, (op, f), textcoords="offset points", xytext=(8, 8), fontsize=9)

    ax.set_xlabel("整体性能（OP，%）", fontsize=12)
    ax.set_ylabel("平均遗忘（F，%）", fontsize=12)
    ax.set_title("持续学习策略对比：整体性能 vs 遗忘", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"策略对比图已保存：{output_path}")


def plot_forgetting_curves(
    results: dict,
    output_path: str = "figs/forgetting_curves.png",
) -> None:
    """绘制遗忘曲线（任务准确率随时间变化）。

    支持 accuracies 的两种格式：
    - dict: {Task1: 98.0, ...}（单任务最终准确率）
    - list of lists: 下三角矩阵（task_idx -> [acc_at_task_0, ..., acc_at_task_idx]），
      取对角线元素作为每个任务训练后的准确率。
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    strategies = list(results.keys())
    colors = plt.cm.tab10(np.linspace(0, 1, len(strategies)))

    task_labels = None
    for i, s in enumerate(strategies):
        accs = results[s].get("accuracies", {})
        if not accs:
            continue
        # 兼容 dict 和 list-of-lists 两种格式
        if isinstance(accs, dict):
            task_labels = list(accs.keys())
            values = [accs[t] for t in task_labels]
        else:
            # list of lists：取对角线（每个任务训练后的自身准确率）
            task_labels = [f"Task{i+1}" for i in range(len(accs))]
            values = []
            for idx, row in enumerate(accs):
                if isinstance(row, list) and row:
                    values.append(row[-1] if len(row) <= idx + 1 else row[idx])
                else:
                    values.append(row)
        ax.plot(range(len(values)), values, marker="o", color=colors[i], label=s, linewidth=2)

    ax.set_xlabel("任务索引", fontsize=12)
    ax.set_ylabel("准确率（%）", fontsize=12)
    ax.set_title("遗忘曲线：各任务准确率", fontsize=14)
    if task_labels:
        ax.set_xticks(range(len(task_labels)))
        ax.set_xticklabels(task_labels, rotation=45)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"遗忘曲线已保存：{output_path}")


def plot_continuous_metrics(
    results: dict,
    output_path: str = "figs/continuous_metrics.png",
) -> None:
    """绘制连续度量对比图（accuracy vs Brier Score）。

    Args:
        results: 各任务度量 dict（Task1 -> {accuracy, brier_score}）。
        output_path: 输出路径。
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    tasks = list(results.keys())
    accs = [results[t]["accuracy"] for t in tasks]
    briers = [results[t]["brier_score"] for t in tasks]

    axes[0].plot(range(len(tasks)), accs, marker="o", color="blue", linewidth=2)
    axes[0].set_xlabel("任务索引", fontsize=12)
    axes[0].set_ylabel("Accuracy（%）", fontsize=12)
    axes[0].set_title("Accuracy（不连续度量）", fontsize=14)
    axes[0].set_xticks(range(len(tasks)))
    axes[0].set_xticklabels(tasks, rotation=45)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(range(len(tasks)), briers, marker="s", color="red", linewidth=2)
    axes[1].set_xlabel("任务索引", fontsize=12)
    axes[1].set_ylabel("Brier Score", fontsize=12)
    axes[1].set_title("Brier Score（连续度量）", fontsize=14)
    axes[1].set_xticks(range(len(tasks)))
    axes[1].set_xticklabels(tasks, rotation=45)
    axes[1].grid(True, alpha=0.3)

    fig.suptitle("度量选择对能力评估的影响", fontsize=16)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"连续度量对比图已保存：{output_path}")


def visualize_all(
    results_path: str,
    output_dir: str = "figs",
) -> None:
    """可视化所有结果。

    Args:
        results_path: 结果 JSON 文件路径。
        output_dir: 输出目录。
    """
    with open(results_path) as f:
        results = json.load(f)

    # 检查是否有策略指标
    sample = next(iter(results.values()))
    if isinstance(sample, dict) and "metrics" in sample:
        plot_strategy_comparison(results, f"{output_dir}/strategy_comparison.png")
        plot_forgetting_curves(results, f"{output_dir}/forgetting_curves.png")
    elif isinstance(sample, dict) and "accuracy" in sample:
        plot_continuous_metrics(results, f"{output_dir}/continuous_metrics.png")
    else:
        print("未知结果格式，无法可视化")


def main() -> None:
    """CLI 入口。"""
    import argparse

    parser = argparse.ArgumentParser(description="持续学习可视化")
    parser.add_argument("--results", type=str, required=True, help="结果 JSON 文件")
    parser.add_argument("--output-dir", type=str, default="figs", help="输出目录")
    args = parser.parse_args()

    visualize_all(args.results, args.output_dir)


if __name__ == "__main__":
    main()
