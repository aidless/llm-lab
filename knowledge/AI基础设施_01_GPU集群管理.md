# AI基础设施：GPU集群管理与调度

## 1. GPU服务器硬件架构

### 1.1 NVIDIA DGX H100 NVSwitch全互联拓扑

NVIDIA DGX H100的基座设计核心在于其8×H100 GPU通过第四代NVSwitch实现的全互联（All-to-All）拓扑。每个H100 SXM GPU配备18个NVLink 4 Link，每个Link单向带宽为112.5GB/s，双向225GB/s，每个GPU的NVLink总聚合带宽为18×112.5GB/s=2025GB/s单向（双向4050GB/s）[1]。在DGX H100的拓扑中，每4个GPU组成一个NVSwitch Pod，通过2个NVSwitch芯片完成组内全互联；两个Pod之间通过NVSwitch的跨Pod链路互联，实现8个GPU之间的全域Bisection Bandwidth为7×900GB/s=6300GB/s的任意GPU到任意GPU的全速通信。DGX H100的系统包含4个NVSwitch ASIC，每个NVSwitch提供64个NVLink 4端口，共计256个端口，以Non-Blocking的全互联模式保证任何GPU对的通信不需要经过第三跳[2]。

当训练大规模模型时，H100的Transformer Engine（Transformer引擎）支持FP8和FP16的混合精度Tensor Core计算。在FP8精度下，每个H100的理论TFLOPS达到3958 TFLOPS（稀疏），但在大规模分布式训练中，实际可达的MFU受限于All-Reduce通信的带宽利用率。NVLink全互联拓扑确保了8卡之间的梯度同步延迟在微秒级，而跨节点通信则依赖InfiniBand或Spectrum-X以太网。

### 1.2 NVLink 4 vs NVLink 5带宽规格

NVLink 4（H100/H200）与NVLink 5（B200/GB200）之间的带宽提升是代际飞跃的关键指标：

| 规格 | NVLink 4 | NVLink 5 |
|------|----------|----------|
| 单Link单向带宽 | 112.5 GB/s | 180 GB/s |
| 单Link双向带宽 | 225 GB/s | 360 GB/s |
| 单GPU Link数 | 18 | 18 |
| 单GPU总单向带宽 | 2025 GB/s | 3240 GB/s |
| NVSwitch单芯片端口数 | 64 | 144 |
| 总带宽提升 | - | 1.6× |

NVLink 5引入了第四代NVSwitch（NVSwitch 4），每个芯片支持144个端口，每个端口运行在360GB/s双向[3]。GB200 Grace-Blackwell超级芯片在NVLink 5上的配置为每GPU 18个Link，在DGX B200系统中通过5个NVSwitch实现576个GPU（36个DGX B200）的无阻塞域。

### 1.3 从DGX H100到GB300的演进路线

NVIDIA的企业AI基础设施经历了四代架构转型：

**DGX H100（2023）**：8×H100 SXM，700W TDP/GPU，NVLink 4全互联，CX-7双端口InfiniBand（NDR 400Gbps/端口，每节点8端口），标准19英寸机架6U机箱。每个DGX H100提供8×3958 FP8 TFLOPS，通过CX-7网卡集群互联。典型的DGX SuperPOD配置为：32个DGX H100节点，通过NVLink Switch System互联，总计256 GPU，配置Quantum QM9700 InfiniBand交换机（64端口NDR400），用两层Fat-Tree拓扑[4]。

**DGX B200/B300（2024-2025）**：8×B200 SXM，1000W TDP/GPU，NVLink 5全互联，CX-8双端口网卡（NDR800），支持液冷。B200的FP8 TFLOPS提升至4.5 PFLOPS/GPU，HBM3e容量从H100的80GB提升至B200的192GB，HBM3e带宽从H100的3.35TB/s提升至8TB/s。B200引入了第二代Transformer Engine，支持FP4精度，在FP4下可达到9 PFLOPS的稀疏计算能力，显著降低了LLM推理的量化推理延迟。

**GB300 DGX SuperPOD（2026）**：采用模块化机架设计（Modular Rack Scale），每个DGX机架整合72个GB300 Grace-Blackwell GPU，通过NVLink 5域互联。GB300引入Grace ARM CPU作为控制处理器和LPDDR5X内存池。每个GB300超级芯片的内存池容量达到432GB（Grace的LPDDR5X 144GB + B300的HBM3e 288GB）[5]。DGX SuperPOD的演进体现了从独立节点机架到全集成模块化机架设计（Rack-Scale Architecture）的转变——主机架（Compute Rack）与交换机机架（Switch Rack）通过NVLink Switch和Quantum InfiniBand实现完全分解的架构，网络不再是后加组件而是预先规划好的系统背板。

HGX Baseboard的演进同样是关键：HGX A100基座为PCIe Gen 4.0（x16插槽×8，CPU与GPU间带宽64GB/s），搭载Mellanox ConnectX-6 HDR InfiniBand；HGX H100升级至PCIe Gen 5.0（128GB/s），搭载CX-7 NDR400双端口InfiniBand；HGX B200进一步优化基座的Network Card Slots拓扑设计——每个CX-8双端口网卡连接两个网口，支持InfiniBand NDR800或Spectrum-X以太网400Gbps。基座上的网络插槽从H100的8个半高扩展至B200的8个全高插槽，允许每GPU配置专用网卡直连而非共享PCIe Switch。

### 1.4 SXM vs PCIe GPU的工程选择

SXM形态和PCIe形态的GPU在部署场景上有根本性的架构差异：

**SXM版本**——H100 SXM TDP 700W vs H100 PCIe 350W。SXM的高功耗使NVLink全互联成为可能（18 NVLink），而PCIe版本仅有4个NVLink Bridge（GPU-to-GPU带宽仅600GB/s）。SXM通过底座连接器直接供电和散热，不需要PCIe插槽的物理限制，可以实现更高的核心频率运行。H100 SXM在基板上的NVSwitch实现了8卡间的900GB/s NVLink带宽，是PCIe版本多卡互联（最多通过NVLink Bridge连接2卡）的数十倍。在LLM训练场景中，数据并行和模型并行都极度依赖卡间通信带宽，这意味着SXM版本在All-Reduce通信上有数量级的延迟改善。H100 SXM的价格约为PCIe版2.5-3倍，但在大规模分布式训练中的TFLOPS利用率（MFU）通常高出80-120%，在大模型集群部署中SXM的性价优势显著[6]。

**PCIe版本**——350W TDP意味着可以在标准服务器中部署，不需要液冷也不需要底座的特殊散热设计。PCIe版本的部署灵活性高，适用于推理服务、小规模训练或边缘端场景。H100 PCIe仅在2卡之间支持NVLink Bridge互联，跨卡通信依赖PCIe Switch，延迟和带宽瓶颈明显。

**GB300的新范式**——GB300 Grace-Blackwell采用Superchip形态，GPU和CPU通过NVLink-C2C（Chip-to-Chip）直接互联，TDP达到1500W/模块，彻底放弃了PCIe形式。液冷成为标配而不是可选方案。这种演进意味着AI基础设施的设计思路正在从"服务器主板上的GPU插卡"转向"以GPU为中心的液冷机架系统"。

## 2. GPU集群网络架构

### 2.1 Fat-Tree vs Dragonfly+拓扑

GPU集群的网络拓扑选择直接影响大规模训练中All-Reduce通信的效率。LLM训练的核心通信模式——All-Reduce梯度同步的延迟取决于拓扑的Bisection Bandwidth和跳数延迟。

**Fat-Tree（CLOS拓扑）**：k-ary Fat-Tree由Leaf（ToR）层和Spine组成。典型的DGX SuperPOD（256GPU）采用两层Fat-Tree：32个节点（每节点8 GPU，每节点8个NDR400端口），连接至32个Leaf交换机（QM9700，每Leaf 32个下行端口连接节点、32个上行至Spine），4个Spine交换机（每Spine 64×NDR400端口）。在这种拓扑下，任意两节点之间的Hops最大为2（Leaf→Spine→Leaf），Bisection Bandwidth = 每个Leaf上行带宽之和 = 32×32×400Gbps = 409.6Tbps[7]。Fat-Tree在所有网络路径上实现均匀带宽，不易形成热点，但交换机数量较大：256GPU集群的Fat-Tree需要32 Leaf + 4 Spine+8 = 44台QM9700交换机。

**Dragonfly+拓扑**：Dragonfly+（由Cray/HPE在Slingshot网络上推广）将GPU节点分为Group，每个Group内部的节点全互联，Group之间通过少数Global Link互联。在256GPU规模、每个Group 64GPU的配置下，Dragonfly+仅需16台交换机（2组×8Leaf/组），比Fat-Tree节省63%的交换机硬件。但是，Dragonfly+的Global Link带宽远低于Local Link，在跨Group的AllReduce中可能产生局部带宽瓶颈[8]。当训练GPT-3 175B规模（每GPU batch size需要同步约350MB梯度数据量）的AllReduce时，Dragonfly+的最坏情况延迟可能比Fat-Tree高出50-70%。

实际部署中，Meta的Research SuperCluster（RSC）采用两层Fat-Tree（200Gbps RoCEv2以太网），而Microsoft Azure的ND H100 v5系列采用Quantum-2 InfiniBand两层Fat-Tree。NVIDIA的DGX SuperPOD标准方案也从早期的三层Fat-Tree投向了NVLink Switch + InfiniBand的双层架构。对于超过8192GPU的超大规模集群，胖树拓扑所需要的交换机级联层数（k≥4）会严重增加延迟和成本。Google TPU v4的自家光电路交换（OCS）拓扑采用了64×64的3D Torus拓扑，以完全不同的拓扑思路规避了Fat-Tree的瓶颈。

### 2.2 InfiniBand子网管理器高可用

InfiniBand网络的核心组件是子网管理器（Subnet Manager, SM），负责路径计算（LID分配、路由表构建）和网络状态维护。在大规模集群中，SM的单点故障会导致整个子网的路由失效，训练中断。OpenSM的高可用方案通过配置Master-Slave模式实现：

```bash
# OpenSM standby模式配置
opensm -g 0x0002c90300aabb01 --guid_routing 2 --ucast_routing \
  ftree --upunchunks 256 --root_guid_file /etc/opensm/root_guids.conf \
  -p /var/run/opensm.pid -L 0x0002c90300aabb01
```

两个OpenSM实例运行在不同的管理节点上，一个为Active Master，一个为Standby Slave。通过IBUA（InfiniBand User Access）库监听事件。当Master心跳超时（默认10秒），Slave在5秒内抢占成为Master。问题是，SM切换期间（15-25秒），InfiniBand子网所有端口的路径计算被冻结——已有QP（Queue Pair）连接可以继续发送数据，但新建连接将失败。在PyTorch Distributed框架中，分布式DataLoader和Rendezous（通过NCCL初始化）依赖动态QP建立。SM切换导致的30秒窗口足以使大规模分布式训练的NCCL AllReduce超时崩溃（NCCL超时默认设为600-1800秒，小型窗口不会触发，但路径切换后部分QP状态不一致可能导致hang死）。

NVIDIA的自研SM（NVIDIA Subnet Manager）相较于OpenSM，在大规模拓扑下（8192端口的子网）的初始化时间从OpenSM的45秒降低到19秒，并且支持Subnet Manager的Hitless Failover（无中断切换）[9]。NVIDIA SM将路由表状态存储在NVSwitch和SmartNIC上的持久化内存中，Standby SM通过监听路由状态日志实时同步，切换时间从秒级降至微秒级。

### 2.3 SHARP网内计算加速

NVIDIA SHARP（Scalable Hierarchical Aggregation and Reduction Protocol）是InfiniBand网络中的网内AllReduce加速引擎。传统的AllReduce通信流程是：GPU1→GPU2→...→GPUN–1→GPUN，每一跳都需要CPU参与聚合，或者通过环状NCCL在GPU间逐级Reduce Scatter和AllGather。SHARP将聚合操作offload到网络交换机上——支持SHARP的Quantum QM9700交换机可以接收来自多个NIC的分级Reduction请求，在交换机内部完成数据归并（Tree-Based Aggregation），然后广播结果给所有参与者[10]。

在512 GPU的DGX SuperPOD集群测试中，开启SHARP后AllReduce延迟从180μs（NCCL原生的Ring AllReduce）降低至102μs，提升约43%。对于带宽密集型AllReduce（大Message size），SHARP收益有限；但对于需要频繁同步的小Message（如大批量梯度计算的Slice），SHARP的延迟优势明显。SHARP v3（配套Quantum-2交换机）支持FP16和BF16数据格式的全精度归并，支持网内All-to-All操作的offload。

### 2.4 Spectrum-X vs InfiniBand的路线争议

NVIDIA的Spectrum-X以太网方案是基于Spectrum-4交换机（51.2Tbps）和BlueField-3 DPU构建的"SuperNIC"增强型RoCEv2网络。Spectrum-X在以太网基础上引入了在网拥塞控制、自适应路由（Adaptive Routing）和DPU的拥塞隔离技术。

**InfiniBand的定位**：传统上，100%的LLM训练集群使用InfiniBand。InfiniBand提供无损传输（基于BECN信用流控），不依赖上层的TCP等待和重传，传输效率和预设置度高于RoCEv2。

**Spectrum-X的定位**：Spectrum-X的目标是"以太网AI网络"，让已经部署以太网的客户（OCI、Oracle及大部分云厂商）不需要单独部署InfiniBand网络。通过BlueField-3 DPU处理拥塞控制（RoCEv2的传统劣势），并用Adaptive Routing规避Fat-Tree中局部拥塞的Head-of-Line Blocking。测评数据表明，在256 GPU规模下，Spectrum-X的AllReduce性能达到InfiniBand的95%以上，但在4096 GPU规模以上InfiniBand的成熟度（基于In-Network Computing SHARP、更成熟的多租户隔离）仍然是NVIDIA推荐的标准训练网络方案[11]。

目前的行业格局是：纯AI训练集群（NVIDIA DGX SuperPOD）标配InfiniBand；而混合工作负载（训练+推理+通用计算）集群更多采用Spectrum-X以太网以降低运维复杂度和资本开支。

## 3. GPU资源管理调度

### 3.1 Kubernetes Device Plugin调度机制

NVIDIA的k8s-device-plugin是Kubernetes集群中GPU资源调度的核心组件。其工作流程如下：

1. 每个GPU节点上的nvidia-device-plugin DaemonSet检测节点上的GPU设备，通过gRPC与kubelet的设备管理器API交互。
2. 设备管理器将GPU信息注册到kubelet的资源数据库，并以Extended Resource的形式暴露（nvidia.com/gpu、nvidia.com/mig-config、nvidia.com/mig-profile）。
3. 当Pod请求`nvidia.com/gpu: 1`时，kubelet的Device Manager根据各节点的GPU空闲数量和拓扑分配GPU设备ID。
4. 设备插件返回给kubelet的设备环境变量`NVIDIA_VISIBLE_DEVICES=GPU-xxxx`，容器根据这个环境变量挂载指定的GPU设备（通过nvidia-container-runtime的CDI机制）[12]。

**GPU Number分配（整卡分配）**：默认模式。Pod请求`nvidia.com/gpu: 1`分配到一整张H100。kubelet从节点设备池中选取当前未被使用的GPU。当该Pod被删除时，GPU设备回到节点池，供下一个Pod使用。

**MIG（Multi-Instance GPU）分区分配**：A100和H100支持MIG，将一块物理GPU物理分割为最多7个独立的GPU实例（A100为3×3+1，H100为7个计算实例）。每个MIG实例有独立的内存、缓存和计算单元，硬隔离。

```yaml
# MIG配置示例：H100分配到1g.10gb实例
resources:
  limits:
    nvidia.com/gpu: 1
    nvidia.com/mig-profile: 1g.10gb
```

MIG的Allocation由device-plugin根据请求的profile名称，查找节点上的空闲MIG设备（通过`nvidia-smi mig -lgi`命令的输出解析），分配对应的计算实例。MIG的硬隔离优势在于多租户环境下的安全隔离——A租户的CUDA kernel无法访问B租户的显存，但MIG会降低单卡的最大TFLOPS（因为计算单元被分割）。H100的7×1g.10gb模式下，每个实例的SM数量仅为整卡的约1/7，这对大模型推理场景下batch size受限需较大显存容量的场景并不友好。

### 3.2 GPU共享与时间分片

**Time-Slicing GPU Sharing**：NVIDIA的GPU Time-Slicing方案是将一块整卡的GPU时间片分配给多个Pod。device-plugin配置为时间分片模式时，多个Pod可以同时请求同一张GPU。kubelet允许在同一个物理GPU上调度多个Pod，GPU的计算时间按CGroup控制的时间窗口平均分配。

```yaml
# GPU Operator的Time-Slicing配置
apiVersion: v1
kind: ConfigMap
metadata:
  name: time-slicing-config
data:
  any: |-
    version: v1
    sharing:
      timeSlicing:
        resources:
        - name: nvidia.com/gpu
          replicas: 8
```

时间分片的问题是缺乏显存隔离：多个Pod共享同一显存，如果某个Pod使用超出分片配额，将触发OOM Kill。此外，不能保证GPU计算资源分配的公平性（如果有Pod持续占用kernel执行）。

**MPS（Multi-Process Service）**：MPS是CUDA原生提供的多进程共享GPU方案。MPS在GPU上一个特定的"MPS Control Daemon"进程中合并来自多个客户端进程的CUDA操作（kernel launch、memcpy）。MPS的核心机制是CUDA kernel的并发调度——MPS service进程将多个客户端的CUDA streams切分后合并提交到GPU前端（Front-End），使GPU端的计算资源被多个任务共享。

MPS相较Time-Slicing的优势是：Cupti（CUDA Profiling Tools Interface）可以追踪到每个MPS客户端的资源使用。但MPS有以下限制：1）显存不隔离（MPS客户端之间的OOM攻击面相同）；2）MPS请求的GPU Context共享导致CUDA内核的并发调度未必优于时间片；3）每个MPS server的服务客户端数量上限为48。

MPS通常在推理服务端发挥作用——对延迟不敏感的小batch推理场景，用MPS在单卡上部署多个推理实例优于时间分片。而在训练场景，MPS几乎不被使用——大规模训练更倾向于独占整卡以避免显存竞争。

### 3.3 GPU碎片整理与拓扑碎片缓解

大规模GPU集群的资源碎片化是一个系统性问题。当集群中运行着不同规格的GPU Pod（1卡、2卡、4卡、8卡请求）时，随着Pod的调度与终止，空闲GPU在节点上的分布可能是碎片化的——例如，一个8卡节点上有GPU 0和3空闲（而不是0和1连续），Pod即使只需2卡，也应该分配在同一NVSwitch组内的连续GPU以获得最佳内部互联。

**NVIDIA Topology Manager**：kubelet的Topology Manager通过收集节点的CPU、GPU、NIC、NVLink的NUMA（Non-Uniform Memory Access）拓扑信息，在Pod调度时提供NUMA对齐的控制策略[13]。Topology Manager的策略包括：

- `single-numa-node`：不允许跨NUMA节点分配资源。对于H100 SXM系统，同一Socket的CPU下连接的GPU之间具有更优的通信拓扑，跨Socket的GPU访问需要通过PCIe或NVLink跨域连接，延迟增加20-30%。
- `best-effort`：优先NUMA对齐但如果资源不足则允许跨NUMA分配。
- `restricted`：必须对齐NUMA，但如果无法满足则拒绝Pod。
- `none`（默认）：不进行拓扑管理。

在8×H100的节点的典型调度中，Topology Manager配合Kubernetes的`node.kubernetes.io/nvidia-topology-aware`扩展调度器，可以确保分配连续GPU。核心思路是：device-plugin在响应kubelet的Allocate请求时，将节点上的GPU拓扑信息（通过`nvidia-smi topo -m`采集）转换为kubelet可以理解的NUMA设备Hints，Topology Manager根据Hints分配与Pod的请求拓扑最匹配的空闲GPU组。

**跨GPU通信的拓扑碎片化缓解**——在大型Slurm调度环境中，通过自定义调度策略（如Slurm的SelectType=select/cons_res + CR_Core_Memory_CPU_GPU），要求每个节点上的GPU请求必须是连续分配的（从GPU 0开始顺序分配）。这种方法简单有效但降低整体资源利用率。Kubernetes环境下的Node Feature Discovery (NFD)和NVIDIA GPU Operator已经集成了拓扑识别能力，可以通过Node Labels标记每台机器的GPU拓扑\[中继NUMA节点\], Pod调度时可以基于Node Selector精确控制。

## 4. GPU监控与可观测性

### 4.1 DCGM指标监控体系

NVIDIA Data Center GPU Manager（DCGM）是数据中心级GPU监控的核心架构。DCGM提供的指标通过DCGM Exporter暴露给Prometheus，由Grafana Dashboard呈现。

DCGM的核心指标维度包括：

**SM Occupancy（GPU计算利用率）**：`DCGM_FI_PROF_SM_OCCUPANCY`，度量SM（Streaming Multiprocessor）的活动Warp数量与最大Warp数量的比率。高Occupancy（80%+）通常意味着GPU的计算资源被充分利用；但低Occupancy不必然意味着GPU空闲——大矩阵乘（GEMM）在Tensor Core上的Warp利用率本身较低但TFLOPS高。因此SM Occupancy必须与`DCGM_FI_PROF_PIPE_TENSOR_ACTIVE`（Tensor Core活动比率）结合判读。

**Memory Utilization（显存利用率）**：`DCGM_FI_PROF_MEM_ACTIVE`，HBM的读写带宽利用率。LLM训练中，Memory Utilization和SM Utilization可能呈现截然不同的pattern——Transformer层中前向的FC layer是Compute-bound（SM Occupancy高），而Attention的序列维度计算是Memory-bound（Memory Utilization高，SM Occupancy反而不高）。这种pattern的组合脉冲用于诊断在训练中的具体哪层过慢。

**NVLink Throughput**：`DCGM_FI_PROF_NVLINK_RX_BYTES`和`DCGM_FI_PROF_NVLINK_TX_BYTES`，监控GPU间的AllReduce/AIIReduce流量。在数据并行训练中，如果NVLink流量始终低于链路带宽的60%，可能存在NCCL通信瓶颈（在NCCL的Ring AllReduce中，每个step完成后带宽利用率会抖动；Tree AllReduce流量更稳定）。

**多指标组合判读**：单纯的GPU-Util（nvidia-smi的默认GPU-Util是sample期间SM的繁忙程度），在LLM训练中几乎是100%恒定，无法反映训练效率。正确的训练监控需要同时观测SM Occupancy、Tensor Core利用率、Memory Bandwidth Utilization、NVLink流量以及PCIe Rx/Tx。Google PAIR的Profiling[14]指出，当这些指标的组合出现异常——比如Tensor Core利用率突然下降但Memory Utilization上升——往往是Attention层的序列长度增加导致计算模式从Compute-bound变为Memory-bound，此时应怀疑数据并行中token packing的序列pad引起额外开销。

### 4.2 DCGM Health Monitoring与XID错误

DCGM提供硬件健康监控的抽象层，通过DCGM Health Monitor轮询各类硬件错误寄存器。

**XID错误分类**：XID是NVIDIA驱动上报给系统日志的GPU错误代码。在DCGM的`DCGM_FI_GPU_XID_ERRORS`指标下，关键XID代码包括：

- **XID 13 / 31 / 43**：GPU Hang / GPU掉驱动 / GPU页面错误。这些错误通常触发GPU Recovery（驱动级GPU重置），会导致该GPU上的所有CUDA context丢失。在训练中如果某个GPU触发XID 13，该节点上的GPU Pod将全部崩溃。
- **XID 48 / 64**：Double Bit ECC错误 / Single Bit ECC超限。Double-bit ECC不可纠正，通常意味着HBM物理单元的永久损坏。该错误在DCGM中标记为Critical，触发GPU的PCIe Channel Remap或返厂。
- **XID 79 / 94**：MIG配置错误 / GPU的NVLink故障。

DCGM的健康监测可以配置为Prometheus AlertManager告警规则：

```yaml
groups:
- name: GPU-health
  rules:
  - alert: GPU_XID_Error
    expr: rate(DCGM_FI_GPU_XID_ERRORS[5m]) > 0
    annotations:
      summary: "GPU {{ $labels.gpu }} XID errors detected"
  - alert: GPU_RetiredPages
    expr: DCGM_FI_GPU_RETIRED_PAGES_DBE > 0
    annotations:
      summary: "GPU {{ $labels.gpu }} has retired pages (DBE)"
```

### 4.3 GPU Pod级别监控

当GPU通过k8s-device-plugin分配给Pod时，DCGM Exporter可以暴露Pod级别的GPU使用信息。通过启用DCGM Exporter的`--collectors=all`标志并从Pod的`/proc/{pid}`中识别出运行在该GPU上的进程，DCGM可以得到Node GPU-Pod的cost matrix——每个GPU上所有关联Pod的计算、内存、NVLink使用量。

Pod级别的监控存在以下挑战：

1. **MPS/Time-Slicing场景的进程-容器映射**：多个进程共享同一张GPU时，DCGM无法准确区分每个进程的SM资源使用比例。解决方法DCGM 3.x引入了MPS-aware monitoring——通过CUDA MPS Control Daemon的进程分组实现。
2. **Kubernetes命名空间的聚合设计**：用户通常需要按Namespace、Job、Pod三个层级聚合GPU使用。DCGM Exporter配合Prometheus的kube-state-metrics可以根据`kube_pod_labels`标签的namespace进行聚合，公式为`sum(DCGMFI_PROF_SM_OCCUPANCY * on(pod) group_left(namespace) kube_pod_labels{label_job_name=~".+"}) by (namespace)`。
3. **Pod粒度的NVLink监控**：NVLink monitor是GPU粒度的——如果一个节点上有8个GPU、16个Pod（每GPU 2个Time-Slice Pod），每个Pod的NVLink用量无法区分。在此场景下，粗粒度的节点级NVLink监控比Pod级更有操作价值。

### 4.4 训练监控与慢节点检测

在LLM训练Job的监控中，三个关键指标构成效率监测的三角：

**Loss曲线**：训练损失的变化趋势。模型收敛过程不匹配预期（Loss下降太慢）说明学习率、批次大小或模型结构有问题。但它无法检测分布式瓶颈。

**TFLOPS Utilization（MFU）**：`实际TFLOPS / GPU理论峰值TFLOPS`。实际TFLOPS通过从训练日志中解析Token吞吐，乘以每Token的FLOPs（LLaMA-1的公式约为6×参数量×Token数）。MFU理想值通常在40-60%（高度优化的megatron-deepspeed实现），低于30%暗示严重瓶颈。

**AllReduce Round-trip延迟漂移（慢节点检测）**：在NCCL通信中，每个通信域中的所有Worker等AllReduce完成后才会进入下一个Micro-Batch的计算。如果其中一个Worker（GPU）计算慢于其他——可能由于HBM降频（热受损）、PCIe链路降级或系统中断过多——会导致整个训练Job的Throughput下降。

检测方法是在每个训练Step前插入一个同步泡（Synchronization Barrier）标记，记录所有GPU的Barrier到达时间。实际部署中，可以在NCCL的`NCCL_DEBUG=INFO`日志中解析`send/recv`的延迟——普通GPU的通信延迟在50-100μs的局部波动，慢节点的通信延迟会持续高于300μs以上。通过Prometheus记录每个节点的NCCL延迟（暴露方式：可以通过dcgm exporter的plugin或者自定义NCCL wrapper），当`nccl_allreduce_latency > 300μs`持续超过30秒，触发慢节点告警，自动将该节点从训练集群中隔离。

Microsoft DeepSpeed的Autotuning[15]已经集成了类似的检测功能，在Megatron-Deepspeed框架下，通过监控每个Micro-Batch的前向+反向时间来自动下调慢节点的计算负载。

## 5. GPU集群调度系统

### 5.1 Slurm批调度系统

Slurm在GPU集群中的配置需要考虑GPU资源的管理方式。典型配置如下：

```bash
# slurm.conf 的 GPU Partition 配置
PartitionName=gpu-h100 Nodes=h100-[01-32] Default=YES \
  MaxTime=INFINITE State=UP \
  DefMemPerNode=2048000 \
  TRESBillingWeights="gres/gpu=1.0" \
  OverSubscribe=EXCLUSIVE \
  SelectTypeParameters=CR_GPU

# gres.conf 指定节点上的GPU类型
NodeName=h100-[01-32] Gres=gpu:h100:8
```

**Partition分区的策略**：不同机型、不同GPU规格被划分到不同的Partition。一般的设计是：
- `gpu-h100`分区：H100 SXM 80GB节点，用于训练Job。
- `gpu-h100-pcie`分区：H100 PCIe，用于中等规模训练和微调。
- `gpu-a100`分区：A100节点。
- `interactive`分区：小Partition，节点数少，用于Debug和测试。

**Interactive Session vs Batch Job**：Interactive Session通过`srun --gres=gpu:1 --pty bash`启动，用户获得一个Shell连接到某台机器上的GPU。Interactive Session在调试、Profiling时极其便利——用户可以实时运行nvidia-smi、nsys。Batch Job则通过sbatch提交，适合生产环境中长时间运行的大规模训练。

Interactive Session的问题是资源碎片——用户在Interactive时可能占用GPU但其实没干活（Idle状态），导致资源浪费。解决方案是设置Partition的MaxTime（例如限制为4小时），或者采用Singularity/Enroot容器+Jupyter Hub的交互方案。

### 5.2 热门调度器的层级对比

GPU集群调度器经历了多代演进：

| 调度器 | 架构类型 | GPU支持粒度 | 关键特性 |
|--------|---------|------------|---------|
| Univa Grid Engine | 传统HPC调度 | 整卡 | 许可证费用高，社区衰退 |
| Slurm | 传统HPC调度 | 整卡 / MIG | 开源成熟，支持CR_GPU |
| Volcano | Kubernetes原生批调度 | 整卡 / MIG / 共享 | 队列、优先级、Gang Scheduling |
| YuniKorn | Kubernetes原生调度 | 整卡 | 细粒度资源配额、多租户 |
| Run:ai | AI专用调度层 | 整卡 / MIG / 分片 / 动态碎片整理 | GPU时间分片、Binpack/Spread |

**Volcano与YuniKorn**：Volcano是CNCF的Kubernetes批调度器，提供Gang Scheduling（All-or-Nothing调度——一个Job中所有Pod同时调度成功才能开始运行，否则不调度任何Pod）和Queue的资源配额管理。YuniKorn（Apache孵化项目）侧重于多租户的公平调度，适合企业内部多个AI团队共享集群的场景。在GPU支持方面，Volcano和YuniKorn依赖k8s-device-plugin，不支持GPU的时间分片和分片分配。

**Run:ai**：Run:ai是专门为AI工作负载设计的调度层（2024年被NVIDIA收购[16]）。它在Kubernetes之上提供GPU虚拟化层，支持：
- GPU分片：将一张GPU切成1/2、1/4等比例。
- 碎片整理：动态迁移Pod以合并碎片。
- Binpack和Spread策略：Binpack将Pod尽量集中到较少节点以提高节点利用率；Spread将Pod分散到尽可能多的节点以提高容错性。
Run:ai的虚拟化层通过拦截CUDA API调用实现，在所有GPU共享方案中提供了最灵活的分配策略，但存在拦截层带来的5-10%性能损失。

### 5.3 优先级与抢占策略

在多租户GPU集群中，低优先级作业被高优先级作业抢占是常见的资源管理需求。

**Preemption机制**：Slurm的抢占通过设置Partition的`PreemptMode`和`PreemptType`实现：

```bash
PartitionName=gpu-high QOS=high Priority=1000
PartitionName=gpu-low QOS=low Priority=100 \
  PreemptMode=REQUEUE PreemptType=preempt/partition_prio
  PreemptExemptTime=00:30:00   # 低优先级Job在抢占前有30分钟豁免
```

当高优先级作业（QOS=high）提交但资源不足时，Slurm会选择一个低优先级的作业（QOS=low）发送SIGCONT/SIGSTOP信号进行抢占。Slurm的Resume程序在抢占后负责重新启动低优先级作业。

**Run:ai的抢占**：Run:ai在Kubernetes上实现了名为"Preemptible Queue"的机制。当一个Job创建在Preemptible Queue上：
- 该Job可以被同一Project的Normal Queue上的Job抢占。
- 被抢占时，Run:ai的Scheduler通过Volcano的PodGroup机制向该Job发送SIGTERM。
- 如果Job没有在Grace Period（默认30秒）内退出，则SIGKILL。
- Job的Checkpoint（保存模型权重和优化器状态）由用户框架层实现（如PyTorch的torch.save checkpoint到共享文件系统）。

**Checkpoint方案**：自动Checkpoint是抢占有效性的关键。在LLM训练中，Checkpoint的大小通常为数GB到数十GB（LLaMA-2 70B的模型权重加优化器大约280GB）。Flash-Ckpt（来自Microsoft DeepSpeed）实现了分布式Checkpoint——所有GPU将该卡上的优化器状态并行写入并行文件系统（Lustre/GPFS），耗时约3-5分钟。如果没有Checkpoint，被抢占的Job丢失了数小时的训练进度。

**Binpack和Spread策略的平衡**：Run:ai的Binpack策略将Pod塞到尽可能少的节点上（算法：优先分配Pod到已有GPU Pod的节点），最大化空闲节点数量以节省电力和散热。Spread策略反之，将Pod分散到最多节点上（算法：优先分配Pod到最少GPU Pod的节点），提高容错性和网络带宽的负载均衡。在大型集群中，通常对训练Job用Binpack以集中NVSwitch域带宽，对推理Job用Spread以分散风险。

## 6. GPU故障处理与RMA

### 6.1 Silent Data Corruption

Silent Data Corruption（SDC）是GPU最棘手的故障类型之一。SDC发生在GPU执行计算时返回了错误结果，但硬件错误检测机制（ECC、奇偶校验）未触发——因为错误发生在计算逻辑中而不是内存中。

Meta在2024年的HPC观察研究报告[17]中指出，基于数百万GPU小时的海量数据中心运营数据，GPU的SDC率约为1-5% 单卡年化率——这意味着一个10000GPU的集群每年有100-500张GPU可能出现SDC。

**LLM训练中的SDC累积影响**：在LLM训练中，SDC的后果比传统HPC更严重：
- 一个bit翻转（1-bit flip）在前向计算中会导致Loss异常升高（通常在之后的steps中观察到loss spike）。
- 由于AllReduce的同步机制，SDC的误差会通过梯度同步传播至全体Worker——一个Worker上的SDC可能污染所有模型副本。
- 不同于HPC的容错（checkpoint restart），LLM训练的Checkpoint写入本身可能写入SDC污染的权重，导致Checkpoint被读回后也是错误的。

**SDC检测方案**：目前最有效的SDC检测是在Checkpoint写入时进行校验。具体方案（如Meta的gg→ggml方案[18]）：
1. 在Checkpoint写入之前，将模型的参数矩阵与Last Step的和校验（Checksum）做对比。
2. 如果在多块GPU上进行，对不同GPU的同一层参数做Cross-GPU一致性检验。
3. 在全reduce之后立即在每张GPU上运行一个已知正确结果的数学模型（例如用该GPU计算一个固定的矩阵乘法并和预期值比较）。

NVIDIA在Hopper架构中引入了计算保护（Compute Protection）机制——GPU在SM上执行时对部分关键控制流进行冗余计算（双重执行），开销约3%，可将SDC降低到接近0。但Compute Protection在当前的H100 SXM上默认不开启。

### 6.2 DCGM Diagnostics全流程诊断

NVIDIA DCGM Diagnostics是GPU硬件故障诊断的官方工具，共三层测试方案：

**第1层——GPU Basic Sanity Test**：`dcgmi diag -r 1`。检查GPU的基本状态：PCIe链接状态（Link Speed/Width）、GPU温度范围、GPU电流极限、Inforom校验、XID Log。生产环境必做的最小诊断——执行时间约1分钟。

**第2层——GPU Extended Test**：`dcgmi diag -r 2`。在第1层基础上增加：
- Memory Test：对HBM进行Pattern测试（全0，全1，Alternating Checkerboard（棋盘格交替模式），March C+算法等），捕获显存的Soft Error和Hard Error。
- PCIe的Stress Test：在GPU和CPU之间进行持续的PCIe DMA传输，检查PCIe链路的稳定性。
- NVLink的Echo Test：向同节点上的其他GPU发送NVLink Echo包，检查NVLink的连通性和延迟。

**第3层——GPU Stress Test**：`dcgmi diag -r 3`。持续运行高负载计算（在GPU上跑密集的kernel mix——FP32 GEMM + FP16 GEMM + Memcpy混合工作负载）时长1小时以上，同时在后台运行Memory Test。第3层用于复现不稳定的硬件错误——在一般负载下不出错，但在长时间重负载下触发。

诊断报告中的关键字段：
- `GPU Index`：GPU的PCIe插槽位置。
- `GPU UUID`：唯一标识。
- `Test Result`：Pass/Fail/Skip。
- `Error Message`：具体的XID或Memory Error描述。
- `DCGM Metadata`：FW版本、驱动版本、VBios版本。

### 6.3 ECC错误与持久化日志

NVIDIA GPU的ECC（Error-Correcting Code）内存系统记录两类错误：

**Single-Bit ECC（SBE，可纠正）**：HBM的一位错误被HBM本身内置的ECC纠正。SBE记录在DCGM的`DCGM_FI_GPU_RETIRED_SBE`中。当某个内存地址上的SBE速率超过阈值（NVIDIA默认的SBE Retire Threshold为2次/小时），驱动将该地址Retire到保留列表（Retired Page Table），标记为以后不再使用。SBE本身**不会**导致训练崩溃。

**Double-Bit ECC（DBE，不可纠正）**：HBM的两位错误无法被ECC纠正。DBE会触发XID 48，导致对应的内存Page被标记为Retired。DBE一旦发生，意味着该GPU的部分HBM颗粒已经永久受损。如果DBE频繁发生，该GPU应当RMA。

**持久化ECC日志**：ECC错误记录在GPU的Inforom（信息ROM，GPU的NVRAM存储）中，断电不会丢失。通过`nvidia-smi -q -d ECC`可以查看GPU累计的SBE和DBE统计：

```
$ nvidia-smi -q -d ECC
ECC Mode
    Current               : Enabled
    Pending               : Enabled
ECC Errors
    Volatile
        SRAM Correctable  : 0
        SRAM Uncorrectable: 0
        DRAM Correctable  : 245
        DRAM Uncorrectable: 2
    Aggregate
        SRAM Correctable  : 0
        SRAM Uncorrectable: 0
        DRAM Correctable  : 18754
        DRAM Uncorrectable: 2
```

`Aggregate`统计从GPU出厂起的累计值，`Volatile`是本次驱动加载以来的值。每天SBE增长超过3600（每小时150）或者DBE出现任何一次，应当考虑更换GPU[19]。

### 6.4 DeepPort多实例诊断

NVIDIA DeepPort是用于DGX和HGX的诊断Agent（包含在NVIDIA DGX OS中），通过与NVIDIA的支持中心集成，从每个节点的Agent收集NVRAM诊断信息并汇总到NVIDIA支持Portal。DeepPort的诊断覆盖范围：

- GPU诊断：自动定期运行dcgm diag -r 1和dcgm diag -r 2。
- 交换机端口诊断：InfiniBand端口上的Link Error Count和CRC错误检测——如果`Link Error Count > 0`且稳定增长，通常是光纤线缆或SFP模块故障；CRC错误（`Symbol Error Count`）单独增长说明信号完整性下降。DeepPort将多个节点的端口错误聚合起来，通过相关性关联确定是线缆问题还是交换机端口问题。
- NVSwitch诊断：监测NVSwitch的转发延迟和端口错误统计。

DeepPort的聚合诊断报告格式遵循NVIDIA的"故障症状→根本原因→解决方案"三层格式，由NVIDIA支持团队用RMA流程处理。

### 6.5 RMA流程

NVIDIA的RMA流程（适用于DGX系统）基本流程：

- 客户通过NVIDIA Enterprise Support Portal提交Case，附加DeepPort诊断数据。
- NVIDIA支持工程师初审，确认故障类型是否属于硬件故障（非软件或配置问题）。
- 针对GPU，运行`dcgmi diag -r 3`和`nvidia-smi -a`输出，结合ECC日志和XID日志进行分析。确认属于DBE持续增长、Memory Test fail、或其他明确硬件失效模式（如SM下的Compute Protection触发）。
- NVIDIA支持工程师确认RMA条件后：
  - 标准流程：NVIDIA先发运替换GPU（Cross-ship），客户在收到替换件后30天内发起退货（RMA#）。
  - DGX系统的GPU更换必须在DGX的Service Manual指导下进行——H100 SXM GPU的更换需要在断电、液冷排液（液冷型号）、拆卸GPU底座、更换SXM模块后重新加装。
  - 退回的故障GPU需按照NVIDIA提供的RMA装箱清单打包，附加故障描述标签。
- RMA替换的GPU通常不是全新的（NVIDIA的替换件策略是Recertified/Refurbished），但具有与新GPU相同的保修期。

对于GB200/GB300这种CPU-GPU联合封装，RMA的对象是整个Compute Tray（计算托盘）或整个液冷模块，显著增加了更换的物理复杂度和时间成本。

## 7. 液冷与数据中心基础设施

### 7.1 从风冷到液冷的必然演进

NVIDIA GPU的TDP演进展示了散热技术的必然趋势：

- **H100 SXM**：700W TDP。风冷可行的上限——7kW/节点（8卡）从机房视角看，标准的42U机柜满配8个DGX H100 + 交换机 ≈ 60kW/机柜，已超过绝大多数数据中心单机柜散热能力（标准风冷单柜上限20-30kW）。在60kW下需要使用强化风冷（增强型空调、后门热交换器、地板送风优化）。
- **B200 SXM**：1000W TDP。8kW/节点，8节点机柜理论可达80-90kW。此时单纯风冷已经完全失效：GPU温度在100%风扇转速下仍可能达到过热降频（Thermal Throttling，从1000W降到700W或更低，MFU下降30-40%）。

- **GB300 Grace-Blackwell**：1500W/模块。单模块超过当前任何数据中心风冷散热器的设计上限，液冷是物理必然。

### 7.2 直接液体冷却（DLC）架构

直接液体冷却（Direct Liquid Cooling, DLC）是目前AI服务器部署的主流液冷形式。

**冷板（Cold Plate）设计**：DLC系统中，水冷板直接安装于GPU基板之上。冷板内部有精密加工的微通道（Microchannel，典型100-300μm宽），冷却液（通常为去离子水+乙二醇混合物）流经微通道带走热量。冷板的散热能力由以下公式决定：
```
Q = h × A × ΔT
```
其中Q是热通量，h是对流换热系数，A是换热面积，ΔT是冷却液温度与GPU结温的温差。微通道设计通过大幅增加比表面积A来提升总散热能力，使B200的1000W在ΔT=30℃时可以被DLC带走。

**两相液冷（Two-Phase DLC）**：单相DLC中冷却液始终为液体状态（不沸腾），散热能力受限于冷却液比热容和流量。两相DLC利用冷却液在冷板内沸腾（相变），利用潜热大幅提升散热能力——水的汽化潜热约2260kJ/kg，是水的比热（4.2kJ/(kg·℃)）和温差的乘积的数十倍。

但两相DLC的控制系统更复杂：需要精确控制冷板温度（确保冷却液在冷板内沸腾而不是在管道内沸腾导致水锤效应），需要实时监测蒸汽压和冷凝器状态。在DGX B200的液冷方案中，NVIDIA选择了单相DLC以保证可靠性（单点故障率在超大规模集群中放大），而在NVIDIA的液冷参考设计中冷板工作温度设为约45℃进液、55℃回液。

### 7.3 浸没式冷却

浸没式冷却将整个服务器浸没在电绝缘冷却液中。

**单相浸没（Single-Phase Immersion）**：GPU直接浸泡在Novec 7100、Fluorinert FC-40或Engineered Fluids的介电液体的恒温槽中。冷却液吸收GPU热量后流动到换热器与二次侧水回路交换。单相浸没的优势是液体不沸腾，系统相对简单。

**限制因素**：
1. PFOA/PFOS问题：传统的3M Novec系列液体含全氟化合物（PFAS），在多国逐渐受限。3M宣布2025年底退出PFAS生产[20]。因此，浸没式冷却液供应商转向Engineered Fluids等公司的生物基或碳氢介电液。
2. 服务器维护困难：浸没式冷却中，更换GPU需要将整个机箱抬起排液，拆解GPU。这在大规模集群中增加了70-80%的维护工作量。
3. 冷却液成本：单相介电液成本约$50-100/升，一个标准浸没槽（约500升填充量）的冷却液成本$25,000-50,000。

**两相浸没（Two-Phase Immersion）**：冷却液在槽内沸腾，蒸汽在上方冷凝器液化。散热密度比单相高，但对密封和环境控制要求更高。Novec 649的沸点为49℃，适合两相浸没——在标准大气压下沸腾，且GWP（Global Warming Potential）仅为1。但Novec 649同样受到PFAS退出影响。

目前的行业共识是：AI大规模集群的液冷主力方案是DLC（单相），小型集群或概念验证可以使用浸没式。

### 7.4 液冷基础设施：CDU与管道路由

Coolant Distribution Unit（CDU，冷却液分配单元）是液冷系统的核心组件。CDU在液冷系统中扮演"二次侧水泵+热交换器"的角色：

**一次侧（Facility Side / Primary Loop）**：数据中心楼宇级冷冻水环路。一次侧水温通常为15-25℃（由中央冷冻机（Chiller）或冷却塔提供），通过大型水管路送到每个机柜间的CDU。

**二次侧（IT Side / Secondary Loop）**：CDU到GPU冷板的循环环路。二次侧冷却液温度比一次侧高10-15℃（约30-40℃进入GPU冷板）。CDU内部的Plate Heat Exchanger（板式换热器，由多层不锈钢波纹板堆叠构成）将一二次侧的热量高效传递，同时物理隔离两侧的水路。

在DGX SuperPOD的液冷方案中：
- 每个CDU支持4-6个DGX B200机架（约64-96个GPU）。
- 二次侧水流要求：每个DGX B200节点约15-20L/min，8节点机柜120-160L/min。
- 管道采用15-50mm的不锈钢管或PEX管，颜色标记（蓝色：Supply（进液冷却液），红色/橙色：Return（回液热水））。
- CDU部署方式有两种：Row-based（行级CDU，每个机柜行首部署一个CDU）和Rack-based（以液冷机柜为单位内置CDU）。

Google的液冷实践[21]采用行级CDU整合在数据中心冷冻水系统一次侧的方案，利用温带气候地区的冷却塔（Free Cooling）直接将一次侧水温降至18-22℃，不需要冷冻机，全年PUE可降至1.10以下。Microsoft在Quincy的数据中心采用Adiabatic Cooling（绝热冷却）配合两相DLC，实现PUE 1.12。而传统风冷数据中心PUE在1.3-1.6，液冷带来的全生命周期成本降低（更低的PUE、更高密度的硬件布局、更大的计算力/平方英尺）在8000+GPU规模下，可以抵消液冷的初始资本支出溢价。

### 7.5 液冷对数据中心PUE的影响

PUE（Power Usage Effectiveness）= 数据中心总能耗 / IT设备能耗。液冷改进PUE的方式：

1. **消除空调压缩机能耗**：风冷依赖CRAC（Computer Room Air Conditioner，计算机房空调机）的压缩机将15-20℃空气吹入机柜前方，压缩机能耗占非IT负荷的30-40%。液冷使用水泵驱动冷却液循环，水泵能耗仅为压缩机的30-40%。
2. **降低风扇能耗**：70W的风冷GPU风扇功耗（8卡=560W/节点），液冷节点用冷却液泵代替风扇，泵的功耗仅为风冷风扇的20-30%。
3. **利用Free Cooling**：液冷的一次侧水温要求为15-25℃，在大多数温带气候区域，全年80%以上的时间可以通过冷却塔直接降温（不需要冷冻机）。Google DeepMind在芬兰的TPU集群通过利用波罗的海的深海冷水实现全年Free Cooling，PUE降至1.06。

综合来看，从风冷到液冷的转型不仅仅是散热方案的变化，更是AI基础设施从"以CPU为中心的空气散热时代"到"以GPU为中心的液冷时代"的底层基础设施范式的转换。这种转换决定了未来五年千亿参数模型训练集群的设计边界。

---

## 参考文献

[1] NVIDIA. "NVIDIA H100 Tensor Core GPU Architecture." Whitepaper, 2023. https://resources.nvidia.com/en-us-tensor-core/nvidia-tensor-core-gpu-datasheet

[2] NVIDIA. "DGX H100 System Architecture." NVIDIA DGX Whitepaper, 2023. https://www.nvidia.com/en-us/data-center/dgx-h100/

[3] NVIDIA. "NVIDIA Blackwell Platform." GTC 2024 Keynote. https://www.nvidia.com/en-us/data-center/technologies/blackwell/

[4] NVIDIA. "DGX SuperPOD: Scalable Infrastructure for Enterprise AI." Technical White Paper, 2023. https://www.nvidia.com/en-us/data-center/dgx-superpod/

[5] NVIDIA. "NVIDIA Grace Hopper & Grace Blackwell Superchip Architecture." GTC 2025.

[6] Patel, D., et al. "Demystifying GPU Architecture and Performance for Deep Learning Training." Proceedings of the 30th IEEE International Symposium on High-Performance Computer Architecture (HPCA), 2024.

[7] Al-Fares, M., Loukissas, A., Vahdat, A. "A Scalable, Commodity Data Center Network Architecture." ACM SIGCOMM, 2008.

[8] Kim, J., Dally, W. J., et al. "Technology-Driven, Highly-Scalable Dragonfly Topology." Proceedings of the 35th Annual International Symposium on Computer Architecture (ISCA), 2008.

[9] Graham, R. L., et al. "Scalable Hierarchical Aggregation Protocol (SHARP) with NVIDIA InfiniBand." Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis (SC), 2016.

[10] NVIDIA. "NVIDIA Quantum-2 InfiniBand Switches: The Platform for HPC and AI." Technical White Paper, 2022. https://www.nvidia.com/en-us/networking/quantum-2/

[11] NVIDIA. "NVIDIA Spectrum-X: The First AI-Optimized Ethernet Networking Platform." 2023. https://www.nvidia.com/en-us/networking/spectrum-x/

[12] NVIDIA. "NVIDIA GPU Operator on Kubernetes." Nvidia GPU Operator Documentation, 2024. https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/

[13] Kubernetes. "Topology Manager." Kubernetes Documentation. https://kubernetes.io/docs/tasks/administer-cluster/topology-manager/

[14] Google PAIR. "Evaluating the Efficiency of Large-Scale Machine Learning Training." Google Cloud AI Blog, 2023.

[15] Microsoft DeepSpeed. "DeepSpeed Autotuning: Automatically Optimize Your Model Training." Microsoft Research, 2023. https://www.deepspeed.ai/tutorials/autotuning/

[16] NVIDIA. "NVIDIA Announces Acquisition of Run:ai." NVIDIA Newsroom, April 2024.

[17] Meta. "GPU Silent Data Corruption Characterization and Mitigation." Meta Engineering Blog, 2024.

[18] Meta. "PyTorch Distributed Checkpoint: Saving and Loading Large Models." Meta AI, 2024.

[19] NVIDIA. "NVIDIA DCGM User Guide." NVIDIA Developer Documentation, 2024. https://docs.nvidia.com/datacenter/dcgm/

[20] 3M. "3M to Exit PFAS Manufacturing by the End of 2025." 3M News Center, December 2022.

[21] Google. "Google's Custom Liquid Cooling System for AI Infrastructure." Google Cloud Blog, 2024.
