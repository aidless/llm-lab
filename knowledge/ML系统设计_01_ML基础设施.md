# ML基础设施：从开发环境到生产平台的工程化之路

## 1. ML开发环境标准化

### 1.1 依赖管理的复现性挑战

Python依赖管理在ML场景下面临的复杂度远超传统软件开发。核心矛盾在于：ML项目同时承载Python包依赖（torch/tensorflow/sklearn等）、系统级库（CUDA/cuDNN/TensorRT）、以及二进制加速库（MKL/OpenBLAS）三层层叠的版本约束。

**依赖管理方案对比**：

| 方案 | 锁定机制 | 关键优势 | 关键缺陷 |
|------|----------|----------|----------|
| Conda | `conda env export > environment.yaml` | 跨平台二进制兼容，conda-forge生态覆盖ML核心库 | 环境解析速度慢，lockfile非确定性锁定 |
| Pip | `pip freeze > requirements.txt` | 生态最广 | 无依赖解析保证，`requirements.txt`不包含传递依赖的传递依赖 |
| Poetry | `poetry.lock` + `pyproject.toml` | 确定性锁定，依赖解析严格 | 对CUDA相关包的`--find-links`支持不原生 |
| uv | `uv.lock` | 解析速度比Poetry快10-100倍，原生支持`--find-links`和索引策略 | 生态较新，部分私有源兼容性问题 |

**Pinning效果的关键差异**：Conda的`explicit`模式可以将每个包的具体构建hash锁定（`conda list --explicit > spec-file.txt`），这才是真正可复现的方案；而`environment.yaml`的`prefix`和`channel`缺失常导致环境漂移。uv的`uv.lock`实现了类似Cargo的精确哈希锁定，是目前最接近Nix级别的Python依赖确定性方案。

### 1.2 GPU依赖的版本兼容矩阵

GPU软件栈的兼容性是ML环境工程化中最棘手的部分。NVIDIA的软件栈存在严格的向下不兼容约束：CUDA版本决定cuDNN/TensorRT的最低要求版本，而PyTorch/TensorFlow等框架可能绑定特定CUDA变体（如`torch==2.1.0+cu121`）。

**Docker + NVIDIA Container Toolkit的容器化实践**：

```dockerfile
# 分层构建策略 - 训练镜像
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04 as base
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3-pip && rm -rf /var/lib/apt/lists/*

FROM base as dependencies
COPY requirements.txt pyproject.toml ./
RUN pip install --no-cache-dir -r requirements.txt

FROM dependencies as runtime
COPY src/ /app/src/
ENV CUDA_VISIBLE_DEVICES=0
ENV NCCL_DEBUG=INFO
```

关键实践要点：
- NVIDIA Container Toolkit通过`nvidia-ctk runtime configure`注册Docker运行时，容器内`nvidia-smi`自动映射宿主机驱动
- CUDA base镜像分`devel`（含编译工具链，用于训练镜像）和`runtime`（仅运行时so，用于推理镜像）两类
- `NVIDIA_DISABLE_REQUIRE=1`可在GPU驱动版本不精准匹配时降级而非崩溃

### 1.3 Jupyter到Production的转化路径

Notebook在原型阶段的便利性与工程化阶段的不可维护性形成根本矛盾。Notebook的工程化缺陷包括：执行状态隐藏（全局变量隐式传递）、无确定性执行顺序、无模块化、无版本控制友好性。

转化路径的核心模式：
1. **Notebook Refactoring Pattern**：将Notebook按功能切分为`feature_engineering.py`、`model.py`、`evaluation.py`模块，Notebook仅作为编排层调用
2. **Papermill参数化执行**：通过`papermill`将Notebook参数化后批量执行，输出标准化`.ipynb`产物
3. **nbconvert自动导出**：CI中将Notebook自动导出为HTML报告并归档至实验追踪系统

### 1.4 远程开发环境

集群场景下，开发环境的代码同步与GPU调试是关键痛点。VS Code Remote SSH和JetBrains Gateway提供两种路径：

- **VS Code Remote SSH**：本地编辑→远程执行，`~/.ssh/config`管理多集群入口，端口转发（`LocalForward 16006 127.0.0.1:6006`）实现TensorBoard本地可视化
- **JetBrains Gateway**：基于`idea`隧道的全功能IDE远程模式，支持远程解释器和远程Jupyter Kernel
- **Rsync-based同步**：`rsync -avz --exclude='.git' --exclude='__pycache__'` + `inotifywait`实现开发机的自动增量同步到集群

远程GPU调试的实践：使用`CUDA_LAUNCH_BLOCKING=1`获取精确的CUDA kernel崩溃堆栈；`TORCH_DISTRIBUTED_DEBUG=DETAIL`打印分布式训练的通信拓扑信息。

---

## 2. 实验管理基础设施

### 2.1 实验追踪工具选型

| 维度 | MLflow | Weights & Biases | Neptune | Aim Stack |
|------|--------|------------------|---------|-----------|
| 部署模式 | 自托管（MLflow Tracking Server） | SaaS | SaaS+自托管 | 自托管（纯本地） |
| Artifact存储 | 本地/S3/GCS/ADL | W&B Cloud | 托管存储 | 本地文件系统 |
| 查询能力 | SQLAlchemy后端SQL查询 | API-based搜索 | REST API | Python SDK查询 |
| 多Run比较 | 内置UI（Parallel Coordinates） | 强（Panels+Reports） | 强（Charts+Comparisons） | 弱（基础表格） |
| 开源程度 | Apache 2.0 | 部分开源 | 非开源核心 | Apache 2.0 |

**选型决策框架**：
- **合规优先**（数据不出域）：Aim Stack（完全本地，无外部依赖）或MLflow（自托管，S3存储）
- **协作优先**（团队可视化分析）：W&B的Reports和Sweeps功能远优于其他方案
- **成本敏感**：MLflow或Aim均为免费开源，W&B Pro按席位收费

### 2.2 实验元数据标准化Schema

统一的实验元数据Schema是跨实验可比较性的基础。推荐的标准化结构：

```yaml
experiment:
  id: uuid
  name: string
  tags:
    project: tmlr-exp001
    task: classification
    backbone: resnet50
    dataset: imagenet-subset-v2
  params:
    optimizer: adamw
    learning_rate: 0.001
    batch_size: 256
    scheduler: cosine
  metrics:
    train/loss: 0.342
    val/accuracy: 0.876
    val/f1: 0.871
    test/accuracy: 0.869
  artifacts:
    model: s3://models/exp-001/model.pt
    checkpoint: s3://models/exp-001/epoch=12-step=15000.ckpt
    predictions: s3://models/exp-001/predictions.parquet
  code:
    commit: a1b2c3d4
    diff_hash: sha256
  environment:
    cuda_version: "12.1"
    python_version: "3.10.12"
    torch_version: "2.1.0+cu121"
```

**自动日志框架设计**：通过Trainer的Callback/Hook机制统一注入日志逻辑。以PyTorch Lightning为例：

```python
class AutoLoggingCallback(Callback):
    def __init__(self, tracker: ExperimentTracker):
        self.tracker = tracker

    def on_train_epoch_end(self, trainer, pl_module):
        metrics = {f"train/{k}": v for k, v in trainer.callback_metrics.items()}
        self.tracker.log(metrics, step=trainer.current_epoch)

    def on_validation_end(self, trainer, pl_module):
        self.tracker.log({
            "val/loss": trainer.callback_metrics["val_loss"],
            "val/accuracy": trainer.callback_metrics["val_acc"]
        })
```

这种CB抽象使得实验日志框架可以与任何追踪后端解耦——后端替换（MLflow→W&B）只需修改`ExperimentTracker`实现，无需修改训练代码。

### 2.3 大规模实验的组织方案

**多维度比较**：当实验规模达到数百次Run时，传统表格比较失效。Parallel Coordinates Plot将超参数维度（lr、batch_size、optimizer、weight_decay）和指标维度（accuracy、f1）映射为平行坐标轴，每条折线代表一次Run，视觉呈现"哪些超参数组合产生高指标"的模式。

**显著性检验集成**：在实验比较中集成统计检验是关键但常被忽略的环节。具体来说，应在多次重复实验中计算指标均值和置信区间（95% CI），对两个实验组执行Welch's t-test或Mann-Whitney U检验，并在实验比较UI中标注p值。

**分布式实验管理**：通过实验Launcher管理大规模超参搜索——将`Optuna`或`Ray Tune`的Trial配置与实验追踪系统的Run一一映射。每次Trial的配置自动写入`params`，每次Trial的结果自动写入`metrics`，搜索空间和Trial结果可在实验追踪UI中直接可视化。

---

## 3. ML Pipeline编排引擎

### 3.1 Pipeline框架对比选型

| 框架 | 架构模式 | DAG表达能力 | Kubernetes原生性 | 学习曲线 |
|------|----------|-------------|------------------|----------|
| Kubeflow Pipelines | Argo Workflows之上的抽象 | 强（条件分支+循环） | 原生 | 陡 |
| TFX | Apache Beam + Orchestrator | 中（顺序+条件） | 通过Kubeflow适配 | 中 |
| ZenML | 元编排器（多后端适配） | 强（DAG+缓存） | 通过Stack适配 | 平 |
| Flyte | 独立引擎+Blob存储 | 强（动态节点+分支） | 原生 | 中 |

**选型建议**：
- 已有Kubernetes集群+需要端到端平台：Kubeflow Pipelines
- 纯Google Cloud环境或TF生态：TFX
- 团队小、需要快速迭代和灵活后端切换：ZenML
- 大规模数据/ML工程团队、需要强类型系统和工作流版本管理：Flyte

### 3.2 Pipeline组件封装

**标准化组件镜像的Dockerfile分层方案**：

```dockerfile
# 基础层 - 系统依赖
FROM python:3.10-slim as base
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 libsm6 libxext6 libxrender-dev && rm -rf /var/lib/apt/lists/*

# 依赖层 - 复用缓存
FROM base as deps
COPY requirements.common.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements.common.txt

# 训练组件层
FROM deps as trainer
COPY requirements.train.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements.train.txt
COPY components/train.py /app/
ENTRYPOINT ["python", "/app/train.py"]

# 评估组件层
FROM deps as evaluator
COPY requirements.eval.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements.eval.txt
COPY components/evaluate.py /app/
ENTRYPOINT ["python", "/app/evaluate.py"]
```

**轻量级Pipeline Runner的Backend适配**：底层抽象接口应该暴露`run_component(component_spec, backend_config)`，由Backend适配层解析为具体的执行后端指令——Local模式直接subprocess调用，Docker模式通过`docker run`挂载数据卷，Kubernetes模式生成`V1Pod`规范通过K8s API提交。

### 3.3 Pipeline元数据与血缘追踪

ML Metadata (MLMD) 是Kubeflow/TFX的血缘追踪核心。其关键数据结构：

```protobuf
message Artifact {
  string id = 1;
  string type = 2;  // Dataset, Model, Metrics, etc.
  string uri = 3;   // s3://bucket/path/to/artifact
  map<string, string> properties = 4;
  map<string, string> custom_properties = 5;
  int64 create_time_since_epoch = 6;
  int64 last_update_time_since_epoch = 7;
}

message Execution {
  string id = 1;
  string type = 2;  // Training, Evaluation, Deployment
  int64 create_time_since_epoch = 3;
  ExecutionState state = 4;
  map<string, string> custom_properties = 5;
}
```

**数据血缘的反向追溯实现**：MLMD通过`Event`消息关联Artifact和Execution——`Event.Type.INPUT`和`Event.Type.OUTPUT`。反向追溯的算法是：
1. 从目标Artifact（如最终模型）出发，找到产生它的Execution（训练）
2. 从该Execution找到所有INPUT类型的Event，获取训练数据集Artifact
3. 递归可追溯至数据集的每批次来源、数据变换Pipeline的上游组件

这种方法在大规模场景下解决了"这个模型是用哪个数据集版本训练的"这个看似简单但手动查证极其困难的核心问题。

---

## 4. Feature Store

### 4.1 离线与在线特征的一致性

Train-Serving Skew是ML系统最隐蔽的问题之一。根本原因在于离线特征工程代码（批处理DataFrame操作）与在线特征工程代码（单行请求时的实时计算）存在差异。

**双环境同一特征定义**：核心方案是建立"特征定义即代码"的原则。特征定义存储在Feature Registry中，离线和在线服务都从同一Registry拉取特征定义：

```python
@feature(
    name="user_purchase_7d_sum",
    description="用户近7天购买金额总和",
    owner="team-a",
)
def user_purchase_7d_sum(transactions: DataFrame) -> DataFrame:
    cutoff = now() - timedelta(days=7)
    return transactions[transactions.timestamp >= cutoff] \
        .groupby("user_id")["amount"].sum().reset_index()
```

离线Pipeline在批处理时调用`user_purchase_7d_sum`将特征写入Offline Store；在线服务在推理时调用同一函数的向量化版本从Online Store获取或实时计算该特征。

**时间戳对齐的挑战**：离线特征使用事件时间（`event_timestamp`），在线特征需要查询到"截至请求时刻的最新值"。两者的时间语义必须统一——`Point-in-Time Join`确保离线训练数据中的每个标签行关联到该标签时刻之前（而非之后）的特征值，避免特征穿越的标签泄漏。

### 4.2 Feature Store方案对比

| 维度 | Feast | Tecton | ByteFeature |
|------|-------|--------|-------------|
| 架构 | Offline Store + Online Store | 全托管 | Offline + Online + Backfill |
| Online Store | Redis/DynamoDB/DataStax | 自研 | Redis/自研 |
| Transformation | 外部处理 | 内置Spark/SQL | Flink原生 |
| Point-in-Time Join | `feast materialize-incremental` + Spark SQL | 自动（Serving-backed） | 自定义接口 |
| 开源协议 | Apache 2.0 | 非开源 | 开源（字节内部） |

**Feast Dual Online Store方案**：当使用Redis+DynamoDB时，Redis处理高QPS的在线推理查询（亚毫秒级延迟），DynamoDB作为Online Store的持久化层（防崩溃丢失）。Feast的`OnlineWriteBatch`同时写入两个后端。

### 4.3 Point-in-Time Join的工程实现

Point-in-Time Join是Feature Store中最复杂的工程挑战。其核心语义是：对标签表中的每一行（`user_id`, `label_timestamp`），找到特征表中在该时间戳之前的最新一条特征值。

在Spark/SQL中的实现示例：

```sql
SELECT
  label.user_id,
  label.label_value,
  feature.feature_value
FROM labels label
LEFT JOIN features feature
  ON label.user_id = feature.user_id
  AND feature.feature_timestamp <= label.label_timestamp
  AND feature.feature_timestamp > label.label_timestamp - INTERVAL 30 DAY
WHERE NOT EXISTS (
  SELECT 1 FROM features f2
  WHERE f2.user_id = feature.user_id
    AND f2.feature_timestamp > feature.feature_timestamp
    AND f2.feature_timestamp <= label.label_timestamp
)
```

**数据骨架（Data Skeleton）构建**：为加速Point-in-Time Join，需构建特征的时间序列索引骨架——将原始特征表物化为按`(user_id, feature_timestamp)`排序的骨架表，在Join时使用窗口函数`ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY feature_timestamp DESC)`仅取最近一条。

### 4.4 特征共享与治理

特征注册表（Feature Registry）是特征复用的核心入口。治理的核心维度：

- **特征目录**：支持按领域（domain）分类浏览——用户特征、物品特征、上下文特征
- **特征文档**：自动从特征定义中的docstring生成
- **特征Owner**：标识特征的生产团队，支持故障时的on-call转接
- **特征版本管理**：当特征逻辑变更时递增`feature_version`，旧版本Pipeline继续使用旧版特征直到主动升级

---

## 5. 模型注册中心与Artifact管理

### 5.1 模型版本管理与Lineage

模型文件的存储采用对象存储（S3/GCS/OSS）组织目录结构。推荐的路径分型方案：

```
s3://ml-platform/models/
  {project_name}/
    {model_name}/
      {version}/
        model.pt               # 模型权重
        model.onnx              # ONNX导出格式
        config.yaml             # 模型配置
        trainer_state.json      # 训练器状态
        metrics.json            # 评估指标汇总
        lineage.json            # 血缘元数据
```

**血缘关联的元数据建模**：`lineage.json`记录模型与实验Run、数据集、Pipeline的直接关联关系：

```json
{
  "model_id": "resnet50-001",
  "model_version": 5,
  "source_run_id": "experiment-run-abc123",
  "source_pipeline_id": "training-pipeline-v2.3",
  "training_dataset_uri": "s3://datasets/imagenet-subset-v2/",
  "training_code_commit": "a1b2c3d4e5f6",
  "base_model_id": "resnet50-v4",
  "derived_models": ["pruned-resnet50-v1", "quantized-resnet50-v1"]
}
```

### 5.2 模型评估与审批工作流

**评估门禁系统**：模型注册到Staging前必须通过质量门禁。门禁规则包含：性能门禁（Accuracy ≥ 0.85，Latency p99 ≤ 50ms）、比较门禁（新版本对比生产版本，F1提升≥0.5%）、公平性门禁（各保护属性分组的TPR差异 ≤ 5%）。

门禁的实现通过自动化评估Pipeline实现——在Pipeline中内置`ModelEvaluation`组件，输出完整的评估比较报告（含Regression Test结果），写入Model Registry的Artifact，门禁结果存储在`model_version.tags["gate_status"]`中。

**Stage Promotion流程**：

```
Staging → (自动门禁通过) → Candidate → (人工审批) → Production → (监控) → Archived
```

- Staging→Candidate：自动门禁触发，通过后CI自动更新Model Registry的stage字段
- Candidate→Production：需要Approver的人工审批（通过Model Registry UI或API的`transition_model_version_stage`接口）
- 审批记录存储在`model_version.tags["approver"]`和`model_version.tags["approval_timestamp"]`中

### 5.3 模型元数据标准

MLflow Model Registry的Schema提供了基础的模型元数据字段（`name`、`version`、`stage`、`description`、`tags`）。对于生产级平台需扩展的自定义Properties：

```yaml
model_version:
  # 内置字段
  name: classifier-v2
  version: 7
  stage: Production
  
  # 自定义Tags扩展
  tags:
    framework: pytorch
    model_family: resnet
    backbone: resnet50
    dataset: imagenet-subset-v2
    training_duration_hours: 12.5
    gpu_type: A100-80GB
    gpu_count: 4
    evaluation_accuracy_test: 0.872
    evaluation_precision: 0.869
    evaluation_recall: 0.875
    author: zhangwei
    approver: lisi
    approval_date: "2025-11-20"
```

**Model Card自动生成**：Model Card是模型的可信信息文件，在模型注册时自动生成。模板包含：模型描述、预期用途、评估结果、公平性分析、训练数据描述、局限性、伦理考量。使用Jinja2模板引擎，在Pipeline的注册步骤中自动填充模板生成Markdown格式的Model Card并上传至Artifact Store。

---

## 6. 部署基础设施

### 6.1 推理服务框架选型

| 框架 | 推理模式 | 核心优势 | 场景定位 |
|------|----------|----------|----------|
| TorchServe | PyTorch原生 | 与PyTorch生态深度集成，Warmup请求避免首次推理慢 | PyTorch模型的快速上线 |
| Triton Inference Server | 多框架（TF/PyTorch/ONNX/TensorRT） | 动态批处理（Dynamic Batching）、并发模型执行、GPU显存池化 | 企业级异构推理集群 |
| BentoML | Python微服务 | 自定义Python处理逻辑灵活、与MLflow集成好 | 需要复杂Pre/Post-processing的推理服务 |
| vLLM | LLM专用 | PagedAttention、Continuous Batching、KV Cache管理、LoRA动态加载 | 大语言模型推理优化 |

**吞吐量对比基准经验数据**：
- Triton Dynamic Batching在ResNet-50推理中（批大小32，A100-80GB）：1080 img/s vs TorchServe 720 img/s（约50%提升）
- vLLM在LLaMA-2-7B推理中：1800 tokens/s vs HuggingFace TGI 900 tokens/s（2x吞吐量），显存节省60%

**CPU vs GPU经济性**：
- 推理延迟容忍度>500ms的场景（批量离线推理）：CPU（Intel Xeon + VNNI/AMX指令集）相比A10 GPU，单位成本吞吐量比可达1:0.6
- 延迟敏感场景（<100ms）：GPU不可替代，但可通过FP16/INT8量化降低单次推理成本

### 6.2 弹性伸缩

**基于请求队列深度的KPA伸缩**：Kubernetes中基于KPA（Knative Pod Autoscaler）的`concurrency-target`配置决定每个Pod的并发请求数上限。推荐配置：`autoscaling.knative.dev/target: "5"`——当请求队列深度超过5时触发扩容。

关键问题在于GPU推理Pod的冷启动。GPU Pod从启动到可服务状态的典型时间线：镜像拉取（30-60s）→ 模型从对象存储加载到GPU显存（10-30s）→ 模型Warmup（5-10s）→ Ready。总冷启动时间约60-100s。

**冷启动优化方法**：
1. **Min-Replicas预留**：设置`minReplicas: 2`保持最少2个预热Pod常驻，应对突发流量
2. **模型预加载（Model Preloading）**：在Pod启动Entrypoint中将模型从S3预加载至GPU显存缓存（使用`torch.cuda.empty_cache()`前的加载顺序确定性）
3. **Pod Lifecycle PreStop Hook**：在Pod缩容前通过PreStop Hook优雅关闭，将当前批次推理完成再下线

### 6.3 推理管线异构部署

复杂推理场景通常包含多个处理阶段：Pre-processing（文本Token化/图像缩放/特征提取）→ Model Inference（GPU）→ Post-processing（结果过滤/格式转换/阈值判定）。

**异构部署的核心原则**：GPU资源专用于Model Inference，Pre/Post-processing运行在CPU上以避免GPU算力浪费。

典型的Pod/Container隔离方案采用Sidecar模式：

```
Pod: inference-pod
├── Container: preprocessor (CPU, resources.requests.cpu=2)
│   └── 入口: gRPC port 50051, 接收请求, 返回预处理Tensor
├── Container: model-server (GPU, resources.requests.nvidia.com/gpu=1)
│   ├── Triton Server监听端口8001
│   └── 从ShareMemory读取预处理Tensor, 执行推理
└── Container: postprocessor (CPU, resources.requests.cpu=1)
    └── 从Triton读取结果, 执行后处理, 返回HTTP响应
```

Container间通信通过Unix Socket或共享内存（`/dev/shm`）实现零拷贝Tensor传递。gRPC的`AsyncService`可以解决异构容器间的背压问题——当GPU繁忙时，CPU端自动限流。

---

## 7. ML平台的可观测性与治理

### 7.1 Pipeline可观测性

**Run失败分析与重试机制**：Pipeline的可观测性核心是聚合所有组件的日志、指标和状态变更到统一视图。

日志聚合方案：每个Pipeline组件的`stdout/stderr`以结构化日志（JSON行格式）输出，通过Fluentd或Vector采集到Elasticsearch。日志行包含`pipeline_run_id`、`component_id`、`timestamp`、`level`、`message`字段，支持在Kibana或Grafana Loki中按`pipeline_run_id`过滤聚合搜索。

重试策略设计：
```yaml
retry_policy:
  max_retries: 3
  backoff: exponential
  initial_interval_seconds: 30
  max_interval_seconds: 300
  retryable_exceptions:
    - ConnectionError
    - TimeoutError
    - ResourceExhaustedError
```

**联合监控的Metric分类**：

| 类型 | 指标 | 采集方式 |
|------|------|----------|
| Pipeline Metrics | Run Duration, Component Success Rate, Data Volume Processed | Pipeline SDK上报到Prometheus |
| Infrastructure Metrics | Pod CPU/Memory/GPU Util, Network I/O, Disk I/O | Node Exporter + DCGM Exporter |
| Model Metrics | 推理延迟p50/p95/p99, 吞吐量QPS, 错误率 | Service Mesh (Istio Envoy) + Prometheus |

GPU指标的细粒度采集依赖NVIDIA DCGM（Data Center GPU Manager）：`DCGM_FI_DEV_GPU_UTIL`（利用率）、`DCGM_FI_DEV_FB_USED`（显存使用）、`DCGM_FI_DEV_POWER_USAGE`（功耗）通过`dcgm-exporter`暴露为Prometheus metrics。

### 7.2 成本分配与资源利用率

**GPU利用率的时间维度分析**：实际生产环境中，GPU利用率（DCGM报告的利用率）通常远低于申请量（K8s `resources.requests`中的GPU数量）。典型的浪费模式有：

1. **过度申请**：训练任务申请4张GPU，但PyTorch DDP配置错误导致仅1张GPU实际被使用
2. **空闲等待**：DataLoader的I/O瓶颈导致GPU计算单元处于空闲等待状态（GPU利用率<30%）
3. **碎片化**：8卡机器上仅申请1卡的任务过多，导致剩余的7卡无法被大任务使用

**成本分摊模型**：
```
单次Run成本 = GPU单价 × 运行时长 × GPU数量 × 利用率系数
利用率系数 = DCGM实际利用率 / 申请量
```

利用率系数<0.3的任务应告警并建议优化DataLoader或减少GPU申请。

### 7.3 ML平台治理策略

**模型访问控制RBAC**：基于角色的访问控制是Model Registry安全治理的基础。推荐的权限模型：

```
roles:
  - name: model_reader
    permissions:
      - model_registry:list
      - model_registry:get
      - model_registry:download
  
  - name: model_developer
    permissions:
      - model_reader.*
      - model_registry:register
      - model_registry:update_description
  
  - name: model_approver
    permissions:
      - model_developer.*
      - model_registry:transition_stage  # Staging→Production
  
  - name: model_admin
    permissions:
      - model_approver.*
      - model_registry:delete
      - model_registry:update_owner
```

**审计日志**：所有模型相关操作需记录审计日志（Event Type + Actor + Resource + Timestamp + Result）。审计日志持久化到不可篡改的存储中（如对象存储的Write-Once模式或专门的审计数据库），支持合规审查时的全量历史追溯。

**数据访问治理**：Feature Store中的特征按灵敏度分级（PII特征需脱敏后才能提供给模型训练），敏感特征的访问需要经过审批并在审计日志中记录每次查询的用户和用途。
