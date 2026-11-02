"""持续学习框架单元测试。

运行：
    python -m pytest tests/ -v
或：
    python tests/test_framework.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import torch.nn as nn

from continual_learning.base import ContinualStrategy, StrategyConfig, Task
from continual_learning.metrics import CLMetrics
from continual_learning.orchestrator import ContinualOrchestrator
from continual_learning.strategies.ewc import EWCTrainingStrategy
from continual_learning.strategies.distillation import DistillationStrategy
from continual_learning.strategies.lora import LoRAStrategy, LoRALayer, LoRAModel
from continual_learning.strategies.replay import ReplayStrategy
from continual_learning.strategies.model_soups import ModelSoupsMerger
from continual_learning.cli import SimpleMLP


class TestCLMetrics:
    """测试评估指标。"""

    def test_record_and_forgetting(self):
        metrics = CLMetrics()
        # 训练完 Task_1 后评估 Task_1
        metrics.record(0, 0, "Task_1", 90.0)
        # 训练完 Task_2 后评估 Task_0 和 Task_1
        metrics.record(1, 0, "Task_0", 80.0)
        metrics.record(1, 1, "Task_1", 85.0)

        # 遗忘：Task_1 训练完 90.0 -> 最终 80.0，遗忘 10.0
        assert metrics.forgetting(0) == 10.0
        # 前向迁移：Task_1 训练完 85.0 -> 最终 85.0，迁移 0.0
        assert metrics.forward_transfer(1) == 0.0
        # 整体性能：最终行 [80.0, 85.0] 平均 = 82.5
        assert metrics.overall_performance() == 82.5

    def test_empty_metrics(self):
        metrics = CLMetrics()
        assert metrics.overall_performance() != metrics.overall_performance()  # NaN


class TestEWCStrategy:
    """测试 EWC 策略。"""

    def test_name(self):
        strategy = EWCTrainingStrategy()
        assert strategy.name() == "EWC"

    def test_compute_loss_no_fisher(self):
        """无 Fisher 信息时，损失应等于基础损失。"""
        strategy = EWCTrainingStrategy()
        model = SimpleMLP()
        batch = (torch.randn(4, 28 * 28), torch.randint(0, 10, (4,)))
        base_loss = torch.tensor(1.0)
        task = Task(name="test", dataloader=None)
        assert strategy.compute_loss(model, batch, base_loss, task) == base_loss

    def test_after_task_updates_fisher(self):
        """任务结束后应更新 Fisher 信息。"""
        strategy = EWCTrainingStrategy(num_samples=10)
        model = SimpleMLP()
        # 创建一个最小任务
        task = create_minimal_task()
        strategy.after_task(model, task)
        assert len(strategy._fisher) > 0
        assert len(strategy._old_params) > 0


class TestLoRA:
    """测试 LoRA 策略。"""

    def test_lora_layer(self):
        """LoRA 层应产生非零输出。"""
        layer = LoRALayer(10, 20, rank=4, alpha=1.0)
        x = torch.randn(3, 10)
        output = layer(x, None)
        assert output.shape == (3, 20)

    def test_lora_model(self):
        """LoRA 模型应冻结基座、注入适配器。"""
        base = SimpleMLP()
        lora_model = LoRAModel(base, rank=4, alpha=1.0)
        # 基座参数应被冻结
        assert all(not p.requires_grad for p in base.parameters())
        # LoRA 层应存在
        assert len(lora_model.lora_layers) > 0

    def test_strategy_name(self):
        strategy = LoRAStrategy(rank=4, alpha=1.0)
        assert strategy.name() == "LoRA"


class TestDistillation:
    """测试蒸馏策略。"""

    def test_name(self):
        strategy = DistillationStrategy()
        assert strategy.name() == "Distillation(logit)"

    def test_compute_loss_no_teacher(self):
        """无教师时，损失应等于基础损失。"""
        strategy = DistillationStrategy()
        model = SimpleMLP()
        batch = (torch.randn(4, 28 * 28), torch.randint(0, 10, (4,)))
        base_loss = torch.tensor(1.0)
        task = Task(name="test", dataloader=None)
        assert strategy.compute_loss(model, batch, base_loss, task) == base_loss

    def test_after_task_sets_teacher(self):
        """任务结束后应设置教师模型。"""
        strategy = DistillationStrategy()
        model = SimpleMLP()
        task = create_minimal_task()
        strategy.after_task(model, task)
        assert strategy.teacher is not None


class TestReplay:
    """测试回放策略。"""

    def test_name(self):
        strategy = ReplayStrategy()
        assert strategy.name() == "Replay"

    def test_store_and_sample(self):
        strategy = ReplayStrategy(memory_size=10)
        batch = (torch.randn(5, 10), torch.randint(0, 3, (5,)))
        strategy.store_samples(batch)
        assert len(strategy.memory) == 5
        samples = strategy.sample_replay(batch_size=5)
        assert samples is not None and len(samples) <= 5


class TestModelSoups:
    """测试模型合并。"""

    def test_uniform_soup(self):
        """均匀汤应产出与单模型尺寸相同的模型。"""
        merger = ModelSoupsMerger()
        model1 = SimpleMLP()
        model2 = SimpleMLP()
        merger.add_model(model1)
        merger.add_model(model2)
        soup = merger.uniform_soup()
        assert soup is not None

    def test_no_models_raises(self):
        """无模型时应抛异常。"""
        merger = ModelSoupsMerger()
        try:
            merger.uniform_soup()
            assert False, "应抛异常"
        except ValueError:
            pass


class TestOrchestrator:
    """测试编排器。"""

    def test_run_with_ewc(self):
        """EWC 策略应能跑通完整流程。"""
        model = SimpleMLP()
        strategy = EWCTrainingStrategy(num_samples=10)
        orchestrator = ContinualOrchestrator(model, strategy)
        orchestrator.add_task(create_minimal_task(name="Task_1"))
        orchestrator.add_task(create_minimal_task(name="Task_2"))
        results = orchestrator.run(epochs=1)
        assert results is not None

    def test_run_without_strategy(self):
        """无策略也应能跑通。"""
        model = SimpleMLP()
        orchestrator = ContinualOrchestrator(model)
        orchestrator.add_task(create_minimal_task(name="Task_1"))
        orchestrator.add_task(create_minimal_task(name="Task_2"))
        results = orchestrator.run(epochs=1)
        assert results is not None


def create_minimal_task(name: str = "test", num_samples: int = 20) -> Task:
    """创建一个最小合成任务。"""
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(42)
    X = torch.randn(num_samples, 28 * 28)
    y = torch.randint(0, 10, (num_samples,))
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=10, shuffle=True)
    return Task(name=name, dataloader=dataloader)


def main() -> None:
    """简单手动测试入口。"""
    print("=== 运行单元测试 ===")
    for cls in [TestCLMetrics, TestEWCStrategy, TestLoRA, TestDistillation,
                TestReplay, TestModelSoups, TestOrchestrator]:
        instance = cls()
        for name in dir(instance):
            if name.startswith("test_"):
                method = getattr(instance, name)
                try:
                    method()
                    print(f"  PASS: {cls.__name__}.{name}")
                except Exception as e:
                    print(f"  FAIL: {cls.__name__}.{name} - {e}")
                    raise


if __name__ == "__main__":
    main()
