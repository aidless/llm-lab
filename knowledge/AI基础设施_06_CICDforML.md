# AI基础设施：CI/CD for ML——MLOps流水线实践

## 1. MLOps与CI/CD的融合

### 1.1 MLOps的独特挑战：超越传统DevOps

传统DevOps的核心对象是代码（Code）——源代码变更后经过编译、测试、打包即可部署到目标环境，整个交付链的不可变产物是二进制或容器镜像。MLOps之所以不能简单套用DevOps的CI/CD模板，根本原因在于机器学习系统的可交付物并非只有代码，而是由**代码（Code） + 数据（Data） + 模型（Model） + 配置（Configuration）** 四者共同构成[1]。

Sculley等人（2015）在《Hidden Technical Debt in Machine Learning Systems》中首次系统阐述了ML系统的这一特殊性：传统软件的接口是代码层面的函数调用和API契约，而ML系统引入了数据依赖（data dependencies）——一个特征列名的修改、一个数据源URL的变更、一个训练数据集的版本回退，都可能引发模型行为的"静默退化"（silent degradation），且这些变更不会在编译期被捕获[2]。这是MLOps区别于DevOps的第一性原理：**数据是ML系统的隐式代码**。

具体而言，ML系统在交付链中引入了以下传统CI/CD无法覆盖的新维度：

| 维度 | 传统软件 | ML系统 |
|------|----------|--------|
| 主要产物 | 二进制/容器镜像 | 模型权重文件（如pytorch_model.bin） |
| 依赖类型 | 静态库/框架依赖 | 数据版本 + 特征工程逻辑 + 超参数 |
| 测试对象 | 代码逻辑正确性 | 代码 + 数据质量 + 训练收敛性 + 模型指标 |
| 部署风险 | 功能回归 | 功能回归 + 数据漂移 + 概念漂移 |
| 回滚依据 | 代码版本回退 | 模型版本回退 + 数据版本回退 |

### 1.2 CI/CD在ML阶段的扩展：从两阶段到四阶段

传统DevOps的CI/CD定义在ML领域被扩展为四个阶段，这一框架由Google在《Continuous Delivery for Machine Learning》（CD4ML）实践中首次正式化，并由ThoughtWorks的Sato等人进一步阐述[3]：

**CI（持续集成）——ML的三重验证**

ML的CI不再只是"代码编译+单元测试通过"，而是：

1. **代码测试**（Code Tests）：传统单元测试，验证特征工程函数、模型训练脚本、数据预处理逻辑的正确性。这是最接近传统CI的部分，但需要特别注意——ML代码通常涉及随机性（随机种子、shuffle），因此单元测试必须使用固定种子或mock随机操作以确保确定性结果。

2. **数据验证**（Data Validation）：在数据进入训练或推理管道前自动执行数据质量检查。数据模式（schema）——列的预期类型、允许范围、缺失率等——必须在CI阶段被形式化断言。具体实现通常借助Great Expectations的Expectation Suite或TensorFlow Data Validation（TFDV）的schema generation，我们在第2节详细展开。

3. **模型训练验证**（Model Training Validation）：在数据集子集上执行"sanity check"训练，验证训练Pipeline可以完整跑通且loss走向正常。这本质上是ML系统的"编译"——如果训练中断或NaN梯度产生，CI应当失败。

这三个层次必须"全绿"才能进入下一阶段。Google的《MLOps: Continuous Delivery and Automation Pipelines in Machine Learning》技术白皮书特别指出："Any change to the code, data, or configuration should trigger the CI pipeline, not just code changes"[4]。

**CD（持续交付/部署）——模型管道的部署**

ML的CD交付物不仅包含代码的容器镜像，还必须包含以下三个不可分拆的捆绑实体：

- **模型工件**（Model Artifact）：权重文件（PyTorch的pytorch_model.bin、TensorFlow的SavedModel目录、ONNX的model.onnx）
- **预处理代码及其依赖**：tokenizer配置、词汇表文件、特征归一化参数（scaler mean/std）
- **运行时环境定义**：Conda环境描述文件、CUDA版本、Python包锁文件

这三个实体必须版本一致、一同部署。实践中通常使用Model Registry（如MLflow Model Registry）将它们打包为一个"可部署单元"（Deployable Unit），并附上唯一的版本标签。

**CT（持续训练 Continuous Training）——ML独有的新阶段**

CT是指系统自动检测重新训练条件的机制[5]。触发条件包括：

- **数据漂移**（Data Drift）：新流入的生产数据与训练数据的分布偏离超过预设阈值（常用PSI > 0.2作为警戒线）
- **概念漂移**（Concept Drift）：特征与目标之间的关系发生改变（如推荐系统中用户行为模式随季节变化）
- **性能衰减**（Performance Decay）：模型在生产中的监控指标（如AUC、准确率）下降到基线以下
- **新标注数据到达**：人工标注反馈回流，模型可以获得新的监督信号

当这些条件被满足时，CT Pipeline自动触发重新训练——本质上是"Prod监控→触发训练→模型验证→自动部署"的闭环。这是MLOps与传统DevOps最显著的区别：传统软件的"部署"是终点，ML的"部署"是监控-重新训练的起点。

**CD4ML的完整生命周期**

将上述四个阶段串联起来，就构成了CD4ML（Continuous Delivery for Machine Learning）的完整生命周期：

Code Change → CI(Code Tests + Data Validation + Train Smoke Test) → Stage/Staged Deploy → Production Canary → Full Rollout → Monitoring → (drift detected → trigger CT → loop back to CI)

### 1.3 MLOps成熟度四层次模型

Google Cloud在《MLOps Maturity Model》中定义了MLOps实践的四个递进层次[6]，这是业界广泛接受的参考框架：

**Level 0 — 手动过程**

特征：Jupyter Notebook驱动、一人通吃、无管道、无版本控制。典型表现为：数据科学家在本地Notebook中手工运行数据预处理、模型训练和评估，将模型权重手动上传到共享存储，由工程团队手工部署。痛点包括：不可复现（缺少环境锁定和随机种子记录）、手动错误频繁、模型版本与代码版本脱节、从实验到部署的周期以周或月计。

**Level 1 — ML Pipeline自动化（训练自动化）**

特征：引入数据验证、模型训练、模型评估的自动化Pipeline。典型工具为Kubeflow Pipelines、Apache Airflow或Prefect。关键能力包括数据准备和特征工程Pipeline的自动化和可复现、模型训练的自动化触发（定时或事件驱动）、实验跟踪（MLflow Tracking / Weights & Biases）记录每次运行的超参和指标、模型工件的自动存储和版本标记。此时训练环节实现了自动化，但模型部署仍是手动操作。

**Level 2 — CI/CD Pipeline自动化（部署自动化）**

特征：在Level 1的基础上集成CI/CD系统和Model Registry。关键能力包括代码变更触发CI Pipeline（含数据验证和模型训练验证）、训练好的模型自动注册到Model Registry并推进到Staging环境、Staging环境自动执行回归测试和模型验证、通过手动审批或自动门控将模型晋升到Production、生产环境的蓝绿部署或金丝雀部署、模型监控和自动回滚。这是大多数"成熟"MLOps团队实际达到的层次。关键在于**模型注册表作为CI/CD和ML Pipeline的桥梁**——CI产生的模型工件自动注册，CD消费已注册的模型进行部署。

**Level 3 — 完全自动化AutoMLOps**

特征：完整的自动化闭环，人工仅处理异常边缘情况。关键能力包括自动特征工程和AutoML（自动架构搜索、自动超参数调优）、生产监控自动检测漂移并触发Continuous Training、模型A/B实验自动运行统计显著检验后自动决定晋升或回退、自动回滚和自动缩放、自我修复Pipeline（故障自动感知和自动恢复重试）。Level 3在实践中较为罕见，因为它要求组织有极高的ML工程成熟度以及完整的数据治理和监控基础设施。当前业界多数组织的目标是Level 2，少数头部团队（Google、Netflix、Uber等）在向Level 3演进。


## 2. 数据与特征管道CI

### 2.1 数据质量检查框架：Great Expectations的CI集成

数据质量是ML系统的"编译时检查"。在传统软件中，编译器会在代码语法错误或类型不匹配时拒绝构建。在ML系统中，没有"数据编译器"——直到极端离谱的异常值（如年龄列出现负数或缺失率达到90%）在训练中爆炸前，系统不会给出任何警告（这就是为什么Baydin等人（2018）将此类问题称为"silent data corruption"——静默数据损坏[7]）。

**Great Expectations（GX）框架**

Great Expectations是目前最成熟的数据质量断言框架，它将数据测试从Jupyter Notebook中的一次性的临时检查提升为CI中的自动化测试套件。其核心抽象是**Expectation**（期望）——对数据行为的可声明、可度量、可版本化的断言。

典型的Expectation Suite定义示例（在CI中自动执行）：

expectation_suite_name: training_data.validation
expectations:
  - expectation_type: expect_column_to_exist
    kwargs: { "column": "user_id" }
  - expectation_type: expect_column_values_to_be_of_type
    kwargs: { "column": "age", "type_": "int" }
  - expectation_type: expect_column_values_to_be_between
    kwargs: { "column": "age", "min_value": 0, "max_value": 120 }
  - expectation_type: expect_column_values_to_not_be_null
    kwargs: { "column": "user_id" }
  - expectation_type: expect_column_proportion_of_unique_values_to_be_between
    kwargs: { "column": "transaction_id", "min_value": 0.999 }

上述断言在CI阶段的执行逻辑：当Feature Pipeline变更或新的数据批次到达时，CI Pipeline自动加载该Expectation Suite并运行数据验证。如果任何一个Expectation失败（例如某批数据的age列出现-1），CI构建标记为失败，模型训练不会被触发。这是ML CI的核心门禁之一。

**GX + CI Pipeline的具体集成模式**

在Kubeflow Pipelines或GitHub Actions中，数据验证步骤通常作为Pipeline的第一个容器运行：

Step 1: Data Ingestion → Step 2: Data Validation（GX Checkpoint执行）→ failure则CI FAILED阻断下游 → success则Step 3: Feature Engineering → Step 4: Train/Eval

代码层面的CI集成（以GitHub Actions为例）：

steps:
  - uses: actions/checkout@v3
  - name: Validate data with Great Expectations
    run: |
      pip install great_expectations
      great_expectations checkpoint run training_data_checkpoint

此外，TFX（TensorFlow Extended）的ExampleValidator组件提供了对上述逻辑的原生支持——它允许在训练Pipeline中自动生成数据schema、检测异常数据（anomalous data）并进行数据统计漂移检测。TFDV是Google部署在TensorFlow Extended中的一套数据验证机制，它与GX的定位不同：GX面向广义数据质量断言（支持任意数据源），TFDV则是TFX原生集成的组件，专注于TensorFlow生态的数据模式验证和漂移检测[8]。

**Expectation Suite的版本控制与漂移**

值得特别强调的是，Expectation Suite本身也需要版本控制。当生产环境的数据分布发生根本性变化（如业务规则变更导致新特征引入），数据团队需要更新Expectation Suite以匹配新的模式。这个更新本身应当经过代码审查和CI验证——即Expectation Suite的变更也是一个PR→CI→Merge的过程。

### 2.2 特征计算验证：分布一致性SLA

特征管道（Feature Pipeline）是ML系统中最高频变动的组件之一。一个特征工程函数的替换、一个聚合窗口的调整，都可能使特征值与训练时的分布产生偏移。如果这些偏移在模型部署前未被捕获，最终结果是模型在推理时看到的特征值与训练时完全不同——模型表现将不可预测。

**特征分布检验的CI自动化**

在CI中，特征计算的自动测试应包括：

1. **PSI（Population Stability Index）检验**：衡量特征在生产批次和训练集之间的分布差异。PSI计算公式为：

   PSI = Σ (P_train_i - P_prod_i) × ln(P_train_i / P_prod_i)

   其中P_train_i是训练集中特征在第i个分箱的占比，P_prod_i是生产数据对应分箱的占比。经验阈值：PSI < 0.1表示无显著变化，0.1~0.2需关注，>0.2表示显著偏移——此时应中止Pipeline。

2. **KS（Kolmogorov-Smirnov）检验**：对连续型特征的分布一致性进行显著性检验。在CI中设置p-value阈值（通常0.05），如果生产数据与训练数据的分布差异显著（p < 0.05），则CI应当告警。

3. **统计描述对比**：均值、方差、分位数（P50/P95/P99）的对比检查，确保变化在合理范围内。

自动化的CI测试实现示例如下：

`python
import numpy as np
from scipy import stats

def test_feature_psi(reference, production, bins=10, threshold=0.2):
    min_val = min(np.min(reference), np.min(production))
    max_val = max(np.max(reference), np.max(production))
    if max_val - min_val < 1e-10:
        min_val -= 0.5
        max_val += 0.5
    bin_edges = np.linspace(min_val, max_val, bins + 1)
    ref_counts, _ = np.histogram(reference, bins=bin_edges)
    prod_counts, _ = np.histogram(production, bins=bin_edges)
    ref_pct = ref_counts / len(reference) + 1e-10
    prod_pct = prod_counts / len(production) + 1e-10
    psi = np.sum((ref_pct - prod_pct) * np.log(ref_pct / prod_pct))
    assert psi < threshold, f"PSI={psi:.4f} exceeds threshold={threshold}"

def test_feature_ks(reference, production, p_threshold=0.05):
    stat, p_value = stats.ks_2samp(reference, production)
    assert p_value > p_threshold, (
        f"KS test p={p_value:.4f} < {p_threshold}"
    )
`

这两类检验应当成为Feature Pipeline每一次变更的CI门禁。更进一步的实践是：**在CI中维护一个"特征分布基准库"**——每次训练Pipeline运行后，将特征的训练分布统计（直方图分箱计数、分位数、均值/方差）持久化为一个JSON工件，在下次CI运行时作为参考分布加载并进行对比。

**特征新鲜度SLA**

在线ML系统中，特征通常由离线批处理Pipeline（Batch Feature Pipeline）定期计算并写入在线特征存储（如Feast、Redis、DynamoDB）。如果特征更新延迟，推理时将读取到过期的特征值。特征新鲜度（Feature Freshness）SLA定义了特征值从数据源产生到在特征存储中可用的最大允许延迟。

在CI中验证特征新鲜度的做法（伪代码）：

def test_feature_freshness_sla():
    feature_store = get_feature_store()
    for feature_name, tolerated_lag in FEATURE_FRESHNESS_SLA.items():
        last_updated = feature_store.get_last_updated(feature_name)
        actual_lag = datetime.utcnow() - last_updated
        assert actual_lag < tolerated_lag

这个测试在CI中间接验证特征Pipeline的健康状态——如果特征更新延迟持续超过SLA，说明特征Pipeline存在性能瓶颈或故障，应在模型训练前得到修复。

### 2.3 数据版本控制：CI中的数据集一致性门禁

数据集版本控制是ML CI的核心基础设施。与代码版本控制不同，数据集通常很大（从数百MB到数百TB），不适合存储在Git仓库中。DVC（Data Version Control）是目前广泛采用的解决方案，它使用Git管理数据集版本的元数据指针，而实际数据存储在远程存储（S3、GCS、NAS）中。

**DVC在CI中的校验原理**

DVC的工作原理是在Git仓库中为数据集创建一个.dvc文件，该文件包含数据集的MD5哈希值：

`yaml
# data/train.dvc
outs:
- md5: d1b2a5f8e91c3f7e0a2b4c6d8e0f1a2b
  size: 348572134
  path: train.parquet
`

当Git检出到特定commit时，dvc checkout会根据.dvc文件中的md5哈希从缓存或远程存储拉取匹配的数据集。如果在CI中执行dvc status发现当前数据集的MD5与.dvc文件中记录的不一致，这说明数据集被意外替换（可能是手动上传、错误的数据管道输出或存储层的静默损坏）。

CI step示例：
- name: "数据完整性校验"
  run: dvc status
  # 0 - 一致；1 - 不一致（CI标记为失败）

对于使用Delta Lake或HuggingFace Datasets的团队，类似的校验可以通过校验和（checksum）或数据集令牌完成。在HuggingFace Datasets中，可以通过dataset._fingerprint获取数据集的指纹（基于数据内容和元数据计算的哈希值），在CI中将其与注册的版本指纹比对。

**数据集漂移：比代码冲突更难发现的"静默替换"**

实践中经常发生这样的情况：数据工程师在修复生产数据管道时重新运行了ETL作业，但输出路径错误地覆盖了训练数据集。由于数据文件的文件名没有变化，Git中的.dvc文件也没有变化，但内容已经完全不同。更隐蔽的是，当80%的训练数据正确但20%被替换为测试环境的数据时，模型会在CI的smoke test中通过（因为少量数据的替换不会导致训练中断或NaN loss），但模型在生产中的表现会显著下降——这是数据集版本不一致的最危险形式。

解决此问题的推荐做法是：

1. **在CI中不仅校验数据集的MD5哈希，还校验数据集的统计量指纹**——如每列的均值、标准差、分位数、缺失率。将这些统计量编码为一个JSON字典并存储为不可变的数据工件。CI中的校验步骤对比当前数据集的统计量指纹与预期值是否一致。

2. **数据集的Immutable版本化（不可变的版本策略）**：一旦数据集被注册为某个训练运行的输入，该数据集文件标记为"只读"（不可变）。任何对数据集的修改必须创建新版本，而非覆盖旧版本。DVC的分支机制天然支持这一模式——不同的训练实验使用不同的数据版本分支。

3. **在Model Registry中记录训练数据源版本**：MLflow Model Registry允许在模型注册时附加自定义元数据，其中应包含train_data_version字段。在模型部署时，Pipeline自动校验生产环境的当前数据集版本是否与训练时的版本一致——如果不一致且未触发CT Pipeline，则告警。


## 3. 模型训练与实验的CI

### 3.1 模型训练的正确性测试（Smoke Test + Sanity Check）

模型训练的正确性测试是ML CI中最重要的增量——它与传统测试最大的不同在于：训练过程是否成功不是一个简单的布尔值（运行出错/无差错），而是包含多个维度的质量门。

**Smoke Test：小批量训练验证**

完整训练一个生产级模型通常耗时数小时到数天。如果在CI的每次代码提交中都执行完整训练，Pipeline延迟将不可接受。因此，Smoke Test（冒烟测试）是标准做法——使用数据集的一个极小子集（通常1~2个batch的样本量）在CI中运行训练循环，验证：

1. **训练Pipeline可以完整执行**：从数据加载→前向传播→loss计算→反向传播→参数更新→checkpoint保存，全流程无异常抛出。
2. **loss在合理范围内且正常下降**：初始loss不应为NaN或Inf（通常意味着学习率过大、数值不稳定或数据异常）。
3. **梯度不出现NaN**：逐参数检查梯度，如果任意参数的梯度为NaN，训练应立刻中止并触发CI失败。
4. **模型输出形状正确**：前向传播输出形状与预期一致（如分类任务的logits形状为[batch_size, num_classes]）。

PyTorch下的Smoke Test实现：

`python
def test_training_smoke():
    model = MyModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    sample_loader = DataLoader(
        get_mini_dataset(n_batches=2, batch_size=16),
        batch_size=16
    )
    losses = []
    for batch_idx, (inputs, labels) in enumerate(sample_loader):
        optimizer.zero_grad()
        outputs = model(inputs)
        assert outputs.shape == (16, NUM_CLASSES), (
            f"Expected shape (16, {NUM_CLASSES}), got {outputs.shape}"
        )
        loss = criterion(outputs, labels)
        assert not torch.isnan(loss), "Loss is NaN"
        loss.backward()
        for name, param in model.named_parameters():
            if param.grad is not None:
                assert not torch.isnan(param.grad).any(), (
                    f"Gradient NaN in {name}"
                )
        optimizer.step()
        losses.append(loss.item())
    assert losses[-1] <= losses[0] + 1e-3, (
        f"Loss did not converge: {losses[0]:.4f} -> {losses[-1]:.4f}"
    )
`

一个关键的设计原则：**Smoke Test必须具有确定性**。由于使用固定随机种子和小批量数据，每次运行的结果应完全一致，将随机性的影响降至最低。因此设置torch.manual_seed(0)和np.random.seed(0)是必需的。

**全面训练验证：完整数据子集**

Smoke Test验证Pipeline的"完整性"；但仍需一项全面的训练验证——通常在每次代码合并到主分支后触发CI来执行一个在中等规模数据集上的完整训练周期（通常在数据集的重采样2%~10%的子集上）并评估：

- **收敛验证**：训练loss曲线和验证loss曲线是否正常（无发散、无过拟合迹象）。
- **最终指标不降级**：与基线模型在相同测试集上比较AUC/准确率/F1，设定容忍阈值。
- **训练时间基准**：记录训练耗时，防止因代码变更导致的意外性能退化。

### 3.2 模型评估测试与基线对比门控

**自动化评估与基线门控**

模型训练完成后，CI Pipeline自动在Holdout测试集上计算指定的评估指标（Precision、Recall、AUC、准确率、F1等）。关键的门控机制是**新版本模型的指标必须不显著低于注册的基线版本（baseline model）的指标**。

门控阈值的设定需要精细权衡：

- **0.5%的AUC下降硬性阻断**：对于分类任务，通常设定0.5%~1.0%的AUC下降阈值。如果新模型的AUC相对基线下降超过0.5%，CI标记失败。
- **必须考虑随机方差**：由于训练过程的随机性（权重初始化、数据shuffle、GPU的float精度），即使代码不变，两次训练的指标也可能有微小差异。因此门控阈值应大于实验内方差（intra-experiment variance）。一种最佳实践是重复训练3次取均值再对比基线，但这在CI中资源开销过大。折衷方案是使用"容忍区间"（tolerance interval）——基线是基于N次运行的均值和标准差，新模型的评估值必须在均值的±3σ范围内。

MLflow获取基线指标示例：

`python
def test_model_performance_against_baseline():
    mlflow_client = MlflowClient()
    baseline_model = mlflow_client.get_latest_versions(
        "my_model", stages=["Production"]
    )[0]
    baseline_run = mlflow_client.get_run(baseline_model.run_id)
    baseline_auc = baseline_run.data.metrics["val_auc"]
    current_run = mlflow.active_run()
    current_auc = mlflow.get_run(
        current_run.info.run_id
    ).data.metrics["val_auc"]
    tolerance = 0.005
    assert current_auc >= baseline_auc * (1 - tolerance), (
        f"Model regression: AUC {current_auc:.4f} < "
        f"baseline {baseline_auc:.4f}"
    )
`

**多指标联合门控**

实践中应避免单指标门控（因为模型可能AUC没变但召回率大幅下降）。推荐使用**主指标+辅助指标**的联合门控策略：

| 指标类型 | 示例 | 门控要求 |
|---------|------|---------|
| 主指标（Primary） | AUC / Accuracy | 下降<0.5% |
| 辅助指标（Secondary） | Precision, Recall, F1 | 下降<1.0% |
| 差异公平性（Fairness） | Demographic Parity / Equal Opportunity | 差异<2.0% |
| 鲁棒性（Robustness） | OOD检测准确率 | 下降<1.0% |
| 延迟（Latency） | P99推理时间 | 增加<10% |

### 3.3 实验跟踪集成

实验跟踪是现代ML CI的元数据层。它的任务不是验证模型的质量，而是**记录每次CI训练运行的完整上下文，使得历史和未来之间的每一次实验都是可比较、可复现、可追溯的**。

**MLflow Tracking的自动嵌入**

在CI Pipeline中，MLflow Tracking通常以上下文管理器的方式包裹训练代码：

`python
with mlflow.start_run(run_name=f"CI-{commit_sha[:8]}-{datetime.now()}"):
    mlflow.log_param("commit_sha", commit_sha)
    mlflow.log_param("data_version", dvc_lock["data/train.dvc"]["md5"])
    mlflow.log_param("learning_rate", config.learning_rate)
    mlflow.log_param("batch_size", config.batch_size)
    mlflow.log_param("model_architecture", "resnet50")
    for epoch in range(config.epochs):
        train_loss = train_one_epoch(...)
        val_metrics = evaluate(model, val_loader)
        mlflow.log_metric("train_loss", train_loss, step=epoch)
        mlflow.log_metric("val_auc", val_metrics["auc"], step=epoch)
    mlflow.pytorch.log_model(model, "model")
    mlflow.log_artifact("config.yaml")
    mlflow.log_artifact("tokenizer_config.json")
`

**WandB（Weights & Biases）与分布式训练场景**

WandB的自动记录能力更强（支持梯度直方图、GPU利用率、网络拓扑结构可视化等），对于深度神经网络训练场景，WandB提供的梯度日志尤其有价值——梯度范数（Gradient Norm）的异常往往是训练不稳定的早期信号，应在CI中作为隐式门控指标：

wandb.log({"grad_norm": torch.nn.utils.clip_grad_norm_(model.parameters(), 1e10)})

**实验元数据在CI中的使用**

CI系统在训练完成后，应当将所有超参数、训练数据版本、代码版本和评估指标结构化为一个JSON报告作为CI构建的附件：

`json
{
  "run_id": "d1b2a5f8e91c3f7e0a2b4c6d8e0f1a2b",
  "commit_sha": "a3f2b1c4e5d6789f0a1b2c3d4e5f6a7b8c9d0e1f",
  "data_version": "7b3e8f6c7a1d9e0b2f4a5c6d7e8f9a0b",
  "model": "resnet50",
  "hyperparameters": {
    "learning_rate": 0.001,
    "batch_size": 64,
    "optimizer": "adam",
    "epochs": 10
  },
  "metrics": {
    "val_auc": 0.893,
    "val_accuracy": 0.845,
    "val_f1": 0.827,
    "val_loss": 0.312
  },
  "model_registry": {
    "registered": true,
    "stage": "Staging",
    "baseline_compared": true,
    "baseline_auc": 0.891,
    "delta": "+0.002",
    "gate_passed": true
  },
  "resources": {
    "gpu": "A100-40GB",
    "train_time_seconds": 3721,
    "peak_gpu_memory": 28500
  }
}
`

此JSON报告是一个**不可变的实验快照**，应存储在持久化存储中作为模型档案的元数据。这将确保六个月后仍然可以确切地回答"当时那个最好的模型是用什么数据、什么代码、什么超参训练出来的"。


## 4. 模型部署的CD与A/B测试集成

### 4.1 CD管道的环境规划与逐步升级

模型部署的CD（Continuous Delivery/Delivery Pipeline）不同于传统软件部署的核心特征是：**模型在环境之间的晋升必须包含"验证"阶段**——验证的内容不仅是"服务是否正常启动"，更重要的是"模型在新的流量分布下是否仍能保持训练时的指标"。

**主流的环境分级架构**

Dev Environment → Staging Environment → Canary Production → Full Production

各个环境的验证条件：

- **开发环境**：部署后在Dev环境执行模型Serving的健康检查+合约验证（gRPC/REST接口的输入输出模式与tokenizer的兼容性）+ 使用少量生产回放流量（shadow traffic）运行推理并对比模型输出是否合理（无极端异常输出、输出概率分布正常）。

- **预发/暂存环境**：这是CD管道中最关键的环境。预发环境应尽可能接近生产环境的配置（同样的GPU型号、同样的CUDA版本、同样的serving框架版本、同样的内存与存储限制）。在预发环境中执行：
  - **回归测试**：对模型在新数据集上的表现进行自动化评估，对比开发环境指标。
  - **影子流量测试（Shadow Traffic Test）**：将生产环境的实际流量复制一份（shadow）发送到预发部署的模型，但不返回给用户。影子流量测试的核心目的是验证模型在生产数据分布下（而不是Holdout测试集）的推理输出是否稳定、延迟是否在容忍范围内。
  - **负载测试**：使用k6或Locust执行压力测试，确保预发环境的推理延迟P50/P99满足SLA要求（通常比生产环境宽松30%以预留余量）。

- **生产环境**：默认承接全量生产流量。但进入生产的最初阶段应使用金丝雀部署策略。

**从Staging到生产的晋升条件**

条件应量化为一个Checklist，在CI中自动验证，任何一项未通过则晋升中止：

1. 代码和模型工件在Model Registry中以不可变版本注册
2. Staging环境验证全部通过（健康检查、回归测试、负载测试）
3. 性能指标与基线比较无退化（AUC delta < 0.5%）
4. 推理延迟（P50/P99）不高于基线模型10%（或满足SLA）
5. 检查点状态：模型在Staging阶段被签署或审批通过
6. CD Pipeline配置的逐步升级流量扩大路径正确配置

### 4.2 模型上线策略的实现

#### 蓝绿部署（Blue/Green Deployment）

蓝绿部署的核心思路：始终维护两套完整的模型推理环境（Blue—当前生产版本，Green—新版本），通过负载均衡器在两者之间切换流量。

                    Load Balancer (Istio/Nginx+)
        User  ──────→    ├──────→  Blue (旧版本) v1.0.3
                         └──────→  Green (新版本) v1.1.0

切换步骤：
1. 在Green环境中部署新版本模型（与Blue完全相同的GPU/Serving配置）
2. 对Green执行健康检查和shadow traffic验证（与Staging环境一致）
3. 确认Green正常后，将负载均衡器的流量权重从 Blue:100%, Green:0% 切换为 Blue:0%, Green:100%
4. 保留Blue环境在线，作为回滚备份。如果监控发现异常，切换回Blue

在Kubernetes + Istio环境中，蓝绿部署通过VirtualService和DestinationRule实现：

`yaml
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: model-serving
spec:
  hosts:
  - model-service
  http:
  - route:
    - destination:
        host: model-service
        subset: green
      weight: 100
    - destination:
        host: model-service
        subset: blue
      weight: 0
`

蓝绿部署的优势：零停机部署、部署和回滚都可在秒级切换。劣势：需要双倍资源（GPU），在大模型推理场景下成本翻倍。

#### 金丝雀部署（Canary Release）

金丝雀部署将新模型部署到少量副本（Pod）上，承接生产流量的一个微小比例进行验证。这是大模型推理中最常用的策略——因为它消耗的额外GPU资源远低于蓝绿部署（仅需原规模的5%~10%）。

     User Traffic → Router →  v1 (95%) / v1.1 Canary (5%)

金丝雀部署的关键指标监控期：

- **延迟（Latency）**：P50/P99推理延迟与基线模型的偏差，若延迟显著增加（>20%），则触发自动回滚。
- **错误率（Error Rate）**：HTTP 4xx/5xx响应比例。若错误率超过基线模型的2倍，自动回滚。
- **成功率（Success Rate）**：推理请求的成功完成率。
- **输出分布监测**：新模型输出的概率分布与旧模型的一致性（使用KL散度对比两个模型的softmax输出分布）。

当金丝雀在预设监控窗口内（通常30分钟~2小时）指标表现正常，CD Pipeline自动将流量比例逐步提升（5% → 25% → 50% → 100%）。每一档的提升都保留等待条件验证的时间窗口。

#### A/B测试集成：Feature Flag驱动的流量路由

蓝绿和金丝雀解决的是"部署风险"问题——如何安全地引入新版本。A/B测试回答的是"哪个版本更好"的问题——需要用户级别的随机分割流量，确保同一用户始终路由到同一模型版本（sticky session），直到统计检验（如paired t-test或Bayesian A/B test）得出置信结论[9]。

A/B测试通常在模型serving端（如TorchServe、NVIDIA Triton Inference Server）之前引入一个**路由层**，该路由层使用**特征标识**（Feature Flag / Experiment ID）来决定每个请求应该由哪个模型版本处理：

Request → Feature Flag Router → user_id % 2 → Model A (v1) / Model B (v2)

商业级Feature Flag系统（如LaunchDarkly、Flagsmith或自研的Feature Flag Server）允许动态调整分流比例而不需要重新部署：

`python
def predict(request):
    user_id = request.get_user_id()
    experiment = feature_flag_client.get_variation(
        "model_ab_test_v2", user_id, default="control"
    )
    if experiment == "treatment":
        return model_v2.predict(request)
    else:
        return model_v1.predict(request)
`

A/B实验的维护周期通常为1~4周，期间需要持续的指标监控，并在实验结束时进行统计显著检验。如果treatment组的指标显著优于control组，则模型晋升为全量上线；如果无显著差异或显著更差，则模型标记为"实验失败"（实际上避免了劣质模型的部署，本身就是CD的价值）。

### 4.3 回滚自动化

**回滚触发器**是监控系统检测到异常后自动触发的回滚机制。回滚在模型部署中的自动化级别应当高于传统软件，因为模型上线后可能以我们尚未预见的方式失败（数据分布迁移、特征失效等）。这些失败的一个共同特征是：**在生产流量中暴露给用户的错误无法被CI的预发验证完全覆盖**。

自动回滚的逻辑链：

Kubernetes Metrics Server（采集请求延迟P99/错误率/GPU利用率）→ Prometheus（告警规则评估）→ 规则引擎判定是否超过阈值 → 超过阈值则触发自动回滚脚本

自动回滚脚本示例（Kubernetes场景）：

kubectl rollout undo deployment/model-serving -n production

自动回滚的判定指标应包括：

| 指标 | 默认阈值 | 说明 |
|------|---------|------|
| 错误率 | > 5% | HTTP 5xx响应比例 |
| P99延迟 | > 基线150% | 99分位推理延迟 |
| 请求吞吐量 | < 基线50% | QPS下降过半 |
| 模型输出分布 | KL散度 > 0.1 | 输出概率分布异常偏移 |
| GPU内存溢出 | OOM计数 > 0 | 显存超限导致OOM |
| 健康检查 | 连续3次失败 | 模型Serving失去响应 |

这些指标的组合阈值应设计为"AND/OR"逻辑链——例如（错误率>5% AND P99延迟>基线150%）或（GPU OOM > 0）任一满足即触发回滚。

回滚后需要执行的关键步骤：
1. **写入回滚事件到审计日志**：记录时间戳、触发指标值、回滚前后的模型版本。
2. **暂停CT Pipeline**：避免回滚后立刻触发重新训练。
3. **通知on-call工程师**：通过PagerDuty/OpsGenie发送告警。
4. **保留崩溃现场**：新模型版本的Pod日志和监控数据不应删除，用于后续RCA。


## 5. 模型注册表与版本管理

### 5.1 MLflow Model Registry：模型工件的全生命周期管理

模型注册表（Model Registry）是MLOps CI/CD Pipeline中连接"持续集成"与"持续交付"的枢纽。CI Pipeline将训练好的模型及其元数据注册到Model Registry；CD Pipeline从Model Registry拉取已批准的模型进行部署。MLflow Model Registry是目前业界最广泛采用的实现方案。

**模型版本管理**

MLflow Model Registry的版本管理机制围绕以下核心概念构建：

- **模型注册（Registered Model）**：一个命名实体（如"resnet50-fashion-mnist"），对应一个模型的一组版本。
- **版本（Version）**：每次注册新模型工件时自动生成的递增版本号。每个版本关联唯一的run_id、artifact_uri和model flavor。
- **阶段（Stage）**：每个版本在其生命周期中经历的不同阶段，MLflow预定义了三个阶段：Staging（暂存）、Production（生产）和Archived（归档）。一个版本不能同时在两个阶段。

阶段的转换是模型部署的核心工作流：

`python
from mlflow.tracking import MlflowClient

client = MlflowClient()

# 步骤1：将新训练的模型注册为版本v3，初始状态为None（无阶段）
client.create_registered_model("fraud_detection_xgb")
client.create_model_version(
    name="fraud_detection_xgb",
    source="runs:/d1b2a5f8e91c3f7e/model",
    run_id="d1b2a5f8e91c3f7e"
)

# 步骤2：CI验证通过后，将v3推进到Staging
client.transition_model_version_stage(
    name="fraud_detection_xgb",
    version=3,
    stage="Staging"
)

# 步骤3：Staging测试通过+人工审批后，晋升为Production
# 注意：晋升时需自动将当前Production版本降级为Archived
client.transition_model_version_stage(
    name="fraud_detection_xgb",
    version=3,
    stage="Production"
)
# 同时降级旧版本
client.transition_model_version_stage(
    name="fraud_detection_xgb",
    version=2,
    stage="Archived"
)
`

### 5.2 模型版本化的完整资产

一个模型版本不仅仅是权重文件。以下是需要一起版本化的完整资产清单：

**核心资产**

| 资产类型 | 文件示例 | 说明 |
|---------|---------|------|
| 模型权重 | pytorch_model.bin / tf_model.h5 / model.onnx | 训练得到的参数文件 |
| 模型定义 | model_architecture.py / config.json | 网络结构代码及配置 |
| Tokenizer | tokenizer_config.json, vocab.txt, merges.txt | 文本预处理依赖 |
| 特征归一化 | scaler.pkl / normalizer_params.json | 在线推理时特征变换所需参数 |
| 标签映射 | label_encoder.pkl / class_mapping.json | 模型输出到业务标签的映射 |

**环境依赖**（可复现运行的基础）

- requirements.txt：精确锁定Python包版本（pip freeze > requirements.txt）
- conda.yaml：Conda环境描述文件，包括CUDA版本和cuDNN版本
- Dockerfile：包含基础镜像tag和CUDA运行时版本

**元数据**

模型的元数据是MLOps体系中的"数据血统"（lineage）部分，是审计和故障诊断的基础：

`python
client.set_model_version_tag(
    name="fraud_detection_xgb",
    version=3,
    key="train_data_version",
    value="d1b2a5f8e91c3f7e0a2b4c6d8e0f1a2b"
)
client.set_model_version_tag(
    name="fraud_detection_xgb",
    version=3,
    key="train_code_commit",
    value="a3f2b1c4e5d6789f"
)
client.set_model_version_tag(
    name="fraud_detection_xgb",
    version=3,
    key="hyperparameters",
    value='{"lr": 0.001, "batch_size": 64, "epochs": 10}'
)
client.set_model_version_tag(
    name="fraud_detection_xgb",
    version=3,
    key="val_auc",
    value="0.893"
)
`

### 5.3 模型晋升的CI/CD集成

模型晋升是MLOps中关键的"人机协作"节点。CI/CD系统应当实现以下自动化流程：

1. **自动登记**：CI Pipeline中训练完成并获得Validate+Label的模型工件自动注册到Model Registry的一个名为"candidates"的暂存模型的初始版本。
2. **自动验证**：Model Registry的Webhook或CI系统轮询检测新版本后，自动在Staging环境触发回归测试。
3. **审批门控**：Staging测试通过后，系统通过Slack/Email通知审批人。审批人通过MLflow UI或API手动审批晋升。
4. **自动部署**：晋升到Production后，CD Pipeline自动触发部署到生产环境。

这一流程确保了劣质模型永远不会进入Staging环境，更不会触及Production——劣质模型在CI中的性能对比阶段就已经被阻断。

### 5.4 模型版本的回滚与归档

当生产中出现问题时，回滚到之前的Production版本是标准操作。MLflow Model Registry的回滚流程：

`python
# 步骤1：识别要回滚的目标版本（当前Production的前一个版本）
current_prod = client.get_latest_versions("my_model", stages=["Production"])[0]
prev_version = current_prod.version - 1  # 简单场景，实际可能更复杂

# 步骤2：将当前版本降级为Archived
client.transition_model_version_stage(
    name="my_model", version=current_prod.version, stage="Archived"
)

# 步骤3：将目标版本晋升为Production
client.transition_model_version_stage(
    name="my_model", version=prev_version, stage="Production"
)
`

但回滚之后的关键问题是：**回滚的模型在当前的推理数据分布下是否还能正常工作？** 因为数据可能已经漂移，回滚到的旧版本模型可能是在数月前训练的，当时的数据分布与现在完全不同。因此，回滚后应自动触发CT Pipeline，将当前生产数据纳入训练集重新训练一个新版本。


## 6. 模型推理的CI/CD

### 6.1 模型推理服务化验证

模型部署到推理服务（Serving）后，面临的第一个问题是：**新模型版本在推理框架上能否正常加载和推理**？这个问题看似简单，但在实践中却频繁引入事故——原因包括PyTorch/TensorFlow版本不兼容、ONNX导出时算子的静态图固化导致动态shape不匹配、GPU驱动版本与CUDA运行时库不匹配等。

**合约验证测试**是推理CI的第一步。它验证模型Serving端（如TF Serving、NVIDIA Triton Inference Server、TorchServe）的输入输出契约：

`python
# tests/test_serving_contract.py
import requests
import numpy as np

SERVING_URL = "http://localhost:8501/v1/models/my_model:predict"

def test_serving_health_check():
    """模型Serving的健康检查端点"""
    resp = requests.get("http://localhost:8501/v1/models/my_model")
    assert resp.status_code == 200
    model_metadata = resp.json()
    assert "model_spec" in model_metadata
    assert model_metadata["model_spec"]["name"] == "my_model"

def test_serving_prediction_shape():
    """验证推理输出形状与训练时一致"""
    sample_input = np.random.randn(1, 3, 224, 224).astype(np.float32)
    payload = {
        "instances": sample_input.tolist()
    }
    resp = requests.post(SERVING_URL, json=payload)
    assert resp.status_code == 200
    predictions = np.array(resp.json()["predictions"])
    assert predictions.shape == (1, 1000), (
        f"Expected (1, 1000), got {predictions.shape}"
    )

def test_serving_latency_sla():
    """推理延迟SLA验证：P50 < 50ms, P99 < 200ms"""
    latencies = []
    for _ in range(100):
        start = time.time()
        requests.post(SERVING_URL, json=payload)
        latencies.append((time.time() - start) * 1000)
    p50 = np.percentile(latencies, 50)
    p99 = np.percentile(latencies, 99)
    assert p50 < 50, f"P50 latency {p50:.1f}ms exceeded 50ms"
    assert p99 < 200, f"P99 latency {p99:.1f}ms exceeded 200ms"
`

**GPU内存占用验证**是大模型推理独有的CI需求。新模型版本在加载时可能消耗比预期更多的显存，导致与同一GPU上运行的其他模型或进程冲突：

`python
def test_gpu_memory_limit():
    """验证模型加载后显存占用不超出容器限制"""
    import pynvml
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    gpu_memory_mb = mem_info.used / 1024 / 1024
    limit_mb = 80 * 1024  # 80GB A100, 容器限制80%
    assert gpu_memory_mb < limit_mb, (
        f"GPU memory {gpu_memory_mb:.0f}MB exceeds {limit_mb}MB"
    )
`

### 6.2 推理管道的负载测试与SLA验证

模型上线前，需要用接近真实生产流量的负载对推理服务进行压力测试。负载测试的核心目标是验证新模型在预期峰值QPS下的延迟和吞吐量是否满足SLA。

**k6负载测试脚本示例**：

`javascript
import http from 'k6/http';
import { check, sleep } from 'k6';

export const options = {
  stages: [
    { duration: '2m', target: 100 },  // 逐步上升到100 QPS
    { duration: '5m', target: 100 },  // 维持100 QPS 5分钟
    { duration: '2m', target: 200 },  // 上升到200 QPS
    { duration: '5m', target: 200 },  // 维持200 QPS
    { duration: '2m', target: 0 },    // 逐步下降
  ],
  thresholds: {
    http_req_duration: ['p(95)<200', 'p(99)<500'],
    http_req_failed: ['rate<0.01'],
  },
};

const PAYLOAD = JSON.stringify({
  instances: [[1.0, 2.0, 3.0, /* ... */]]
});

export default function () {
  const res = http.post(
    'http://model-serving:8501/v1/models/my_model:predict',
    PAYLOAD,
    { headers: { 'Content-Type': 'application/json' } }
  );
  check(res, {
    'status is 200': (r) => r.status === 200,
    'response time < 500ms': (r) => r.timings.duration < 500,
  });
  sleep(1);
}
`

此脚本在发布流水线的CI阶段执行，如果p(95)延迟超过200ms或错误率超过1%，则Pipeline标记为失败并阻止晋升。

**对于大语言模型推理的负载测试需要考虑的特殊维度**：

- **TTFT（Time To First Token）**：首token生成延迟，影响用户体验的关键指标
- **TPOT（Time Per Output Token）**：每输出token的延迟
- **吞吐量（Tokens/sec）**：模型每秒生成的token数
- **KV Cache命中率**：连续请求序列中KV cache的复用比例

这些指标应纳入负载测试的SLA框架，与标准p50/p99延迟并行验证。

### 6.3 推理管道的回滚

当模型推理新版本的监控指标超出预设阈值时，CD系统应自动触发回滚。回滚在推理侧比在模型侧更强调**流量路由的原子性切换**——因为推理服务承载着最终用户的请求，回滚延迟每增加一秒都可能产生显著影响。

**Kubernetes原生回滚**

最直接的回滚方式是使用kubectl rollout undo回滚Deployment的镜像tag到旧版本：

`ash
# 自动回滚脚本（由Prometheus Alertmanager webhook触发）
kubectl rollout undo deployment/fraud-model-serving -n production
kubectl rollout status deployment/fraud-model-serving -n production
`

但kubectl rollout undo的问题是回滚的是整个Deployment，包括代码环境的所有变更，而不仅仅是模型版本。更精确的回滚应当使用**蓝绿部署的金丝雀扩展的反向切换**——在Service层面将流量指向旧版的Pod集合。

**基于Istio的流量回滚**

`ash
# 将流量从新版本(v2)切回旧版本(v1)，无需重新部署
kubectl apply -f - <<EOF
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: model-serving
spec:
  hosts:
  - model-service
  http:
  - route:
    - destination:
        host: model-service
        subset: v1
      weight: 100
EOF
`

这个操作的原子性高于kubectl rollout undo——因为在流量切换过程中，旧模型的Pod一直保持运行状态（如果是蓝绿部署），因此切换后立刻可用，无模型加载延迟。

**自动回滚的决策逻辑**

CD系统应实现以下回滚决策树：

1. 持续监控推理流水线的黄金指标（延迟、错误率、吞吐量）
2. 如果任何指标超过"Warning"阈值，触发告警但不回滚（给人工判断时间）
3. 如果任何指标超过"Critical"阈值，自动回滚：将Service流量100%切回稳定版
4. 在25%金丝雀阶段发现问题的回滚影响面小于1%用户，因此回滚门槛可以更低
5. 回滚完成后，自动暂停CT Pipeline，避免重新训练循环在已破坏的数据上运行


## 7. MLOps平台工具链

### 7.1 Kubeflow：基于Kubernetes的ML工作流

Kubeflow是Google开源的MLOps平台，原生构建于Kubernetes之上。其核心组件Kubeflow Pipelines（KFP）提供了ML管道的DAG编排能力，使得数据预处理、模型训练、模型评估、模型部署等各阶段可以独立容器化并组合为可复现的工作流。

**Kubeflow Pipelines的DAG组件化架构**

KFP的核心抽象是**组件（Component）**和**管道（Pipeline）**。每个组件是一个独立的容器化任务，通过inputs/outputs与其他组件交换artifact和参数：

`python
from kfp import dsl, compiler

@dsl.component(
    base_image="python:3.9",
    packages_to_install=["pandas", "scikit-learn"]
)
def validate_data(dataset_path: str) -> str:
    """数据验证组件：使用Great Expectations套件"""
    import great_expectations as ge
    df = ge.read_csv(dataset_path)
    results = df.expect_column_values_to_be_between("age", 0, 120)
    assert results["success"], "Data validation failed"
    return f"{dataset_path}.validated"

@dsl.component(
    base_image="python:3.9",
    packages_to_install=["pytorch", "mlflow"]
)
def train_model(
    validated_data: str,
    learning_rate: float,
    epochs: int
) -> str:
    """模型训练组件：接收验证后的数据路径和超参数"""
    # 训练逻辑
    model_uri = "runs:/<run_id>/model"
    return model_uri

@dsl.pipeline(
    name="ml-training-pipeline",
    description="端到端ML训练Pipeline"
)
def ml_pipeline(
    dataset_path: str = "data/train.csv",
    learning_rate: float = 0.001,
    epochs: int = 10
):
    validate_task = validate_data(dataset_path=dataset_path)
    train_task = train_model(
        validated_data=validate_task.output,
        learning_rate=learning_rate,
        epochs=epochs
    )

compiler.Compiler().compile(ml_pipeline, "ml_pipeline.yaml")
`

每个组件完全独立容器化，这意味着：

- **依赖隔离**：不同组件可以使用不同的基础镜像和Python依赖版本
- **独立扩缩**：计算密集的训练组件可以使用GPU节点池，数据验证组件可以使用CPU节点
- **失败重试**：单个组件失败不会影响整个Pipeline状态，可以配置自动重试

**KFP + CI/CD的集成模式**

KFP管道可以通过Tekton（Tekton是基于Kubernetes的CI/CD框架）或Argo Workflows触发。典型集成方式是将KFP管道的触发集成到GitHub Actions中——当代码合并到主分支后，GitHub Action调用KFP客户端提交新的Pipeline Run：

`python
# ci/trigger_pipeline.py — 在GitHub Action中执行
import kfp

client = kfp.Client(host="https://your-kfp-instance.com")
run = client.create_run_from_pipeline_func(
    pipeline_func=ml_pipeline,
    arguments={
        "dataset_path": "data/train.csv",
        "learning_rate": 0.001,
        "epochs": 10
    },
    experiment_name=f"CI-commit-{commit_sha}"
)
`

**KServe（原KFServing）——模型推理的Kubernetes原生部署**

KServe是Kubeflow生态中的模型推理组件，提供Serverless风格的模型部署能力。它支持TF Serving、Triton Inference Server、TorchServe、ONNX Runtime等多种推理框架的自动部署，并内置了金丝雀部署、自动缩放和监控功能：

`yaml
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: fraud-detection
spec:
  predictor:
    pytorch:
      storageUri: "s3://models/fraud-detection/v3"
      resources:
        limits:
          nvidia.com/gpu: "1"
    canaryTrafficPercent: 10  # 金丝雀流量比例
`

KServe对CI/CD Pipeline的价值在于：它将模型部署从"手工操作"简化为"声明式配置+自动编排"，使得CD Pipeline可以仅通过修改storageUri的模型路径来完成模型版本更新。

### 7.2 MLflow的多功能模块体系

MLflow是目前社区采纳率最高的MLOps工具之一，其四大核心功能模块覆盖了从实验到部署的全流程。

**MLflow Tracking——实验跟踪**

我们已在3.3节详细讨论。其核心价值在于：**CI/CD Pipeline的每次运行都产生一条可追溯、可比较的实验记录**。当团队需要审计为什么某个模型版本出现问题时，Tracking模块提供了从生产模型反向回溯到训练代码commit的全部元数据链。

**MLflow Projects——代码打包**

MLflow Projects是ML代码的可重复运行打包格式。它定义了一个MLproject.yaml文件来描述运行环境和入口点：

`yaml
# MLproject
name: fraud-detection
conda_env: conda.yaml
entry_points:
  main:
    parameters:
      data_file: { type: string, default: "data/train.csv" }
      learning_rate: { type: float, default: 0.001 }
    command: "python train.py --data {data_file} --lr {learning_rate}"
  validate:
    parameters:
      data_file: { type: string }
    command: "python validate.py --data {data_file}"
`

当ML项目以MLflow Projects格式打包后，任何环境（本地、远程集群、Databricks）都可以使用相同的命令运行：mlflow run . --entry-point main。这与Docker容器的可移植性理念类似。

**MLflow Models——统一模型打包格式**

MLflow Models定义了一个统一的模型打包格式（称为MLmodel flavor），使得不同框架训练出的模型可以使用统一的接口进行推理：

`yaml
# MLmodel
time: "2024-01-15T10:30:00"
flavors:
  python_function:
    loader_module: mlflow.pytorch
    data: model.pkl
    env: conda.yaml
  pytorch:
    pytorch_version: "2.0.1"
    model_data: model.pkl
    code: model_architecture.py
`

统一的pyfunc flavor使得CD Pipeline中的部署模块无须关心底层框架——无论是PyTorch、TensorFlow还是XGBoost，部署代码都可以统一调用mlflow.pyfunc.load_model()。

**MLflow Model Registry——模型版本管理**

已在第5节全面讨论。Model Registry是MLflow体系的"枢纽"——它将Tracking（实验数据）、Projects（代码打包）和Models（模型格式）串联为一个完整的MLOps工作流。

### 7.3 Vertex AI Pipelines：全托管ML管道

Google Cloud的Vertex AI Pipelines提供了Serverless（无服务器）的ML管道编排能力，是Google内部MLOps实践的外部化[10]。与Kubeflow Pipelines相比，Vertex AI Pipelines的优势在于：

- **完全托管**：无需管理Kubernetes集群，Pipeline Runner由Google Cloud自动管理
- **原生Google Cloud集成**：可以直接使用BigQuery进行数据验证、AutoML进行自动模型搜索、Cloud Storage进行artifact存储
- **预定义组件库**：提供了250+预构建组件，涵盖数据处理、训练、评估、部署等各个环节

**Vertex AI Pipelines的工作流定义**使用YAML/Python SDK（与KFP兼容）：

`python
from google.cloud.aiplatform import pipeline_jobs
from google_cloud_pipeline_components import aiplatform as gcc_aip

@dsl.pipeline(
    name="vertex-ml-pipeline",
    pipeline_root="gs://my-bucket/pipeline_root/"
)
def vertex_pipeline(bq_table: str = "project.dataset.table"):
    # 步骤1：使用BigQuery进行数据验证
    bq_query_op = dsl.ContainerOp(
        name="data-validation",
        image="gcr.io/my-project/data-validator:latest",
        arguments=["--table", bq_table]
    )
    # 步骤2：使用AutoML训练
    automl_op = gcc_aip.AutoMLTabularTrainingJobRunOp(
        project="my-project",
        display_name="fraud-detection-automl",
        dataset=bq_query_op.outputs["dataset"],
        target_column="is_fraud",
        budget_milli_node_hours=1000
    )
    # 步骤3：模型部署到Endpoint
    deploy_op = gcc_aip.ModelDeployOp(
        model=automl_op.outputs["model"],
        machine_type="n1-standard-4",
        min_replica_count=1,
        max_replica_count=5
    )

pipeline_jobs.PipelineJob(
    display_name="fraud-detection-pipeline",
    template_path="vertex_pipeline.json",
    enable_caching=True
).submit()
`

### 7.4 工具链选型对比

| 维度 | Kubeflow | MLflow | Vertex AI | 自研方案 |
|------|---------|--------|-----------|---------|
| 部署方式 | 自托管K8s | 轻量级服务器 | Serverless | 完全定制 |
| 基础设施成本 | 高（需K8s集群） | 低 | 按使用付费 | 极高 |
| 实验跟踪 | 有限 | 强 | 强（通过Vertex Experiments） | 定制 |
| 模型注册表 | 有限（结合MLflow） | 强 | 强（Vertex Model Registry） | 定制 |
| CI/CD集成 | 通过Tekton | 通用（任意CI） | Cloud Build原生 | 定制 |
| 推理部署 | KServe强大 | 有限 | Vertex Endpoint强大 | 定制 |
| 学习曲线 | 陡峭 | 平缓 | 中等 | 极陡峭 |
| 多框架支持 | 全部（容器化） | 全部（flavor） | 主要框架 | 取决于实现 |
| 大模型推理 | 支持（Triton） | 有限 | 支持（Vertex Prediction） | 取决于实现 |

几乎所有存在规模级ML实践的组织都不会只选择一个平台，而是采用**混合策略**：MLflow作为实验跟踪和模型注册的元数据中枢（轻量级、框架无关），Kubeflow或Vertex AI作为实际管道的编排引擎，并结合自有的CI/CD系统（GitHub Actions/GitLab CI）完成数据和代码的CI验证。

## 参考文献

[1] D. Sculley, G. Holt, D. Golovin, et al. "Hidden Technical Debt in Machine Learning Systems." Advances in Neural Information Processing Systems (NeurIPS), 2015. https://papers.nips.cc/paper/2015/hash/86df7dcfd896fcaf2674f757a2463eba-Abstract.html

[2] D. Sculley, G. Holt, D. Golovin, et al. "Machine Learning: The High Interest Credit Card of Technical Debt." SE4ML: Software Engineering for Machine Learning (NeurIPS Workshop), 2014.

[3] D. Sato, A. Wider, and C. Windheuser. "Continuous Delivery for Machine Learning." ThoughtWorks Technology Radar, 2019. https://martinfowler.com/articles/cd4ml.html

[4] Google Cloud Architecture Framework. "MLOps: Continuous Delivery and Automation Pipelines in Machine Learning." Google Cloud Architecture Framework, 2021. https://cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning

[5] E. Breck, S. Cai, E. Nielsen, et al. "The ML Test Score: A Rubric for ML Production Readiness and Technical Debt Reduction." IEEE International Conference on Big Data, 2017. https://research.google/pubs/pub46555/

[6] Google Cloud. "MLOps Level 0, 1, 2, 3: MLOps Maturity Model." 2021. https://cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning

[7] A. G. Baydin, B. A. Pearlmutter, A. A. Radul, et al. "Automatic Differentiation in Machine Learning: A Survey." Journal of Machine Learning Research, 2018. https://jmlr.org/papers/v18/17-468.html

[8] N. Polyzotis, S. Roy, S. E. Whang, et al. "Data Lifecycle Challenges in Production Machine Learning: A Survey." ACM SIGMOD Record, 2018. https://dl.acm.org/doi/10.1145/3299887.3299891

[9] R. Kohavi, A. Deng, B. Frasca, et al. "Online Controlled Experiments at Large Scale." ACM International Conference on Knowledge Discovery and Data Mining (KDD), 2013. https://exp-platform.com/online-controlled-experiments-at-large-scale/

[10] E. Breck, N. Polyzotis, S. Roy, et al. "Data Validation for Machine Learning." Conference on Systems and Machine Learning (SysML), 2019. https://mlsys.org/Conferences/2019/doc/2019/167.pdf

[11] S. Schelter, D. Lange, P. Schmidt, et al. "Automating Large-Scale Data Quality Verification." Proceedings of the VLDB Endowment, 2018. https://www.vldb.org/pvldb/vol11/p1781-schelter.pdf

[12] K. Hsieh, A. Phanishayee, O. Mutlu, et al. "The Non-IID Data Quagmire of Decentralized Machine Learning." International Conference on Machine Learning (ICML), 2020. https://proceedings.mlr.press/v119/hsieh20a.html

[13] C. Munnelly, S. Jain. "Continuous Training: The Next Frontier in Machine Learning Operations." Proceedings of the ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 2020.

[14] R. Kumar, M. Bhardwaj. "Kubeflow for Machine Learning: From Lab to Production." O'Reilly Media, 2020.

[15] A. Chen, J. T. Dudley. "MLflow: A Platform for the Machine Learning Lifecycle." Proceedings of the Conference on Machine Learning and Systems (MLSys), 2020. https://mlflow.org/docs/latest/index.html
