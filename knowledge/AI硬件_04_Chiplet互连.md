# Chiplet互连与先进封装：AI芯片的下一代互连技术

## 1. Chiplet的驱动力：从经济学的视角看摩尔定律的终结

### 1.1 晶体管成本的拐点

摩尔定律在过去半个世纪中一直是半导体产业的金科玉律：每两年晶体管密度翻倍，同时每百万门成本以约30%的速度递减。然而，这一趋势在2020年前后遭遇了根本性的拐点。基于International Roadmap for Devices and Systems（IRDS）2023版的数据，在28nm节点时每百万门晶体管成本约为0.15美元；到7nm节点降至0.08美元；但到5nm节点时，这一数字反弹至0.10美元以上；3nm节点更是超过了0.14美元[1]。

这一成本拐点的根源在于：先进制程的研发费用呈指数级增长。台积电3nm节点的研发投入超过40亿美元，单一芯片光罩组（Mask Set）费用超过3000万美元。对于年出货量低于亿级的芯片而言，分摊到每颗芯片的NRE（Non-Recurring Engineering）成本已经使得"用最先进制程制造整颗SoC"变得经济上不可持续。

### 1.2 光罩极限与良率经济学

单一芯片的尺寸受到光刻机的光罩场（Reticle Field）限制——标准光罩极限为26mm×33mm，即858mm²。这是单次曝光能够覆盖的最大面积。然而，858mm²的芯片在4nm节点的缺陷密度条件下，预计良率低于5%[2]。这是纯统计学的结果：根据Murphy的良率模型，良率Y = {(1 - e^{-DA}) / (DA)}²，其中D为缺陷密度（约0.1-0.15 defects/cm²），A为芯片面积。当A接近光罩极限时，Y趋近于零。

恰恰是这一简单的经济方程式，构成了Chiplet方法论最核心的驱动力。将858mm²的巨型芯片拆分为4个200mm²的小芯片，良率从5%跃升至60%以上，等效成本降低超过10倍。这种Known-Good-Die（KGD）策略使得每颗小芯片在封装前进行独立测试与筛选，仅将合格Die送入封装环节，避免了"一颗坏点毁掉整颗大芯片"的灾难性损失。

### 1.3 异构集成：混合工艺的必然选择

现代AI芯片面临一个核心矛盾：计算逻辑需要最先进的工艺（如台积电N3或Intel 3）以获得最高频率和最低功耗，而SRAM缓存、模拟I/O、SerDes、HBM接口等模块在先进制程下的密度收益有限且漏电严重。例如，7nm工艺下的SRAM bitcell相比5nm仅缩小约20%，但5nm的静态功耗密度提升30%以上。

Chiplet架构拆除了这一堵墙：CPU核心和加速器采用N3制程，HBM PHY采用N5制程，SerDes采用N7制程，各取所长，通过先进封装缝合为完整的片上系统。NVIDIA的Blackwell GPU、AMD的MI300系列、Intel的Ponte Vecchio无一不采用了这一策略。

## 2. Die-to-Die互连PHY层：UCIe与开放性标准之争

### 2.1 UCIe标准的演进

Universal Chiplet Interconnect Express（UCIe）是当前最具影响力的Die-to-Die互连标准，由Intel发起，现由UCIe联盟管理，成员涵盖台积电、三星、AMD、Arm、Google、Meta、微软等业界巨头。

**UCIe 1.0（2022年3月发布）**定义了最基本的物理层规格，支持两种封装场景：
- **Standard Package**：凸点间距（Bump Pitch）100-130μm，每毫米带宽密度约28GB/s（单端），能耗效率约0.5 pJ/bit。
- **Advanced Package**：凸点间距25-55μm（基于微凸点技术），每毫米带宽密度提升至约164GB/s，能耗效率优化至0.25 pJ/bit以下。

UCIe PHY采用差分DDR（Double Data Rate）时钟架构，数据速率在Advanced Package下可达到32GT/s。信号编码使用明确的物理层编码方案：标准封装下采用NRZ（Non-Return-to-Zero），先进封装下支持PAM-4以实现更高的信号速率。

**UCIe 1.1（2023年8月发布）**进行了重要的补充：
- 增加了对更广泛的封装中介层（Interposer）类型的支持
- 优化了电气参数以适应不同Die间距下的信道损耗
- 完善了边带信号协议（Sideband）用于训练与链路管理

**UCIe 2.0（2024年4月发布）**是一次重大升级：
- 引入3D封装支持：凸点间距降至1-5μm的Hybrid Bonding级别，带宽密度突破1000GB/s/mm
- 增加Die-to-Die适配层的多协议复用能力，同一PHY上可同时承载PCIe、CXL和流式协议
- 强化了测试与可管理性框架，包含增强的BIST（Built-In Self-Test）和边界扫描能力

### 2.2 AIB、BoW与其他开放性标准

在UCIe确立主导地位之前，业界存在多个Die-to-Die互连标准：

**AIB（Advanced Interface Bus）**由Intel在2017年提出并开源，基于并行总线架构。AIB的物理层在标准封装下每线速率约2GT/s，数据位宽可配置为64/128/256位。其最大的历史贡献是作为OpenHPC联盟推动Chiplet生态的基础互连方案。AIB的优缺点非常明确：实现简单、不需要复杂的PLL电路，但带宽密度和能耗效率远逊于UCIe的SerDes方案。

**BoW（Bridge of Wires）**由DARPA的CHIPS项目（Common Heterogeneous Integration and IP Reuse Strategies）资助开发，侧重超短距离（<0.5mm）下的超高带宽芯片间互连。BoW采用了与UCIe不同的物理层策略：源同步时钟+差分信号，在相同的凸点间距下实现了比AIB更高的带宽密度。BoW的设计哲学是极端简化物理层，将复杂度上移至链路层。

**ODSA（Open Domain-Specific Architecture）**的Bunch of Wires（BoW）等标准虽然具有技术价值，但在生态支持方面远不及UCIe。2023年后，UCIe已成为事实上的产业标准，AIB和BoW逐渐退居特定利基场景。

### 2.3 UCIe PHY的深入技术分析

UCIe PHY的核心设计决策在于平衡带宽密度、能耗效率和信号完整性三个维度。以Advanced Package场景为例：

- **微凸点阵列（Micro-bump Array）**：以25μm间距分布在整个Die表面，形成2D凸点网格。典型的PHY macro宽度为400μm，包含64个数据通道（差分对）+时钟+边带信号，提供约128GB/s的双向带宽。
- **均衡（Equalization）**：UCIe要求发送方采用FFE（Feed-Forward Equalization），接收方支持CTLE（Continuous Time Linear Equalization）和DFE（Decision Feedback Equalization）的可选组合。即使在Advanced Package场景下，Die间信号在25-55μm凸点间距下仍会经历显著的ISI（Inter-Symbol Interference），需要至少2-3抽头的FFE进行处理。
- **时钟架构**：UCIe支持嵌入式时钟（Embedded Clock）和转发的源同步时钟（Source-Synchronous Clock），后者在低功耗场景中更受青睐。时钟恢复使用基于整型PLL的CDR（Clock and Data Recovery），避免了小数N PLL的高功耗和复杂噪声设计。

表1：UCIe封装场景核心参数对比

| 参数 | Standard Package | Advanced Package | 3D Package (2.0) |
|------|-----------------|------------------|------------------|
| 凸点间距 | 100-130μm | 25-55μm | 1-5μm |
| 带宽密度 | ~28 GB/s/mm | ~164 GB/s/mm | >1000 GB/s/mm |
| 能耗效率 | ~0.5 pJ/bit | ~0.25 pJ/bit | <0.1 pJ/bit |
| 最大数据速率 | 16 GT/s | 32 GT/s | 64 GT/s |
| 信号编码 | NRZ | NRZ/PAM-4 | PAM-4 |

## 3. 先进封装技术路线：从CoWoS到Hybrid Bonding

### 3.1 台积电的封装帝国：CoWoS、InFO与SoIC

**CoWoS（Chip-on-Wafer-on-Substrate）**是台积电面向高性能计算的旗舰封装方案。自2011年首次应用于Xilinx FPGA以来，已推出四代产品：

- **CoWoS-S（Silicon Interposer）**：第一代方案，使用硅中介层（Silicon Interposer）连接多颗Die。S-5（2022）支持2x Reticle Size的中介层面积，通过65nm制程的硅通孔（TSV）实现跨Die互连。S-6（2023）将中介层面积扩展至3x Reticle（约2500mm²），可容纳3颗GPU Die + 8颗HBM3 Stack。
- **CoWoS-R（RDL Interposer）**：使用有机或玻璃中介层替代硅中介层，通过再分布层（Redistribution Layer，RDL）实现互连。RDL线宽/线距（L/S）约为2/2μm，比硅中介层的0.5/0.5μm粗，但成本降低50%以上。适合对互连密度不敏感的Chiplet场景。
- **CoWoS-L（Local Silicon Bridge）**：结合了CoWoS-S和CoWoS-R的优点，在RDL中介层中嵌入局部硅桥（Local Silicon Bridge）用于高带宽Die-to-Die连接，其余区域使用低成本的RDL。这是CoWoS家族当前最灵活的方案，也是NVIDIA Blackwell B200的封装选择。
- **CoWoS 8x Reticle**：作为CoWoS路线图的终点站，8x Reticle将支持的芯片面积推至约6864mm²，足以容纳超过10个GPU Die和200+GB的HBM3e容量，预计在2026-2027年实现量产。

**InFO（Integrated Fan-Out）**是台积电面向移动和消费电子市场的封装方案，与CoWoS最大的区别在于不使用硅中介层，而是通过RDL直接在封装基板上实现Die间互连。InFO的凸点间距通常为130-200μm，带宽密度约10 GB/s/mm，但成本仅为CoWoS的1/5到1/3。InFO-oS（on Substrate）变体进一步增强了多层RDL的走线能力，适合中等性能需求的Chiplet集成。

**SoIC（System on Integrated Chips）**是台积电的3D堆叠方案，其核心技术是Cu-Cu Hybrid Bonding（混合键合）。与传统的微凸点键合不同，Hybrid Bonding在室温下通过铜焊盘的热膨胀实现直接键合，无需C4凸点或微凸点作为中间介质。

SoIC的技术参数令人震撼：
- 焊盘间距（Pad Pitch）可达到0.8-1.5μm级别，远小于微凸点的25-55μm
- 每平方毫米的互连密度超过10000个连接
- 键合后Die间距（Bondline Thickness）小于5μm
- 能耗效率低于0.05 pJ/bit

SoIC的两种变体——SoIC-P（无TSV穿透）和SoIC-T（含TSV穿透）——分别适用于"Logic-on-Logic"和"Memory-on-Logic"两类场景。AMD的MI300系列采用了SoIC方案将3颗CPU CCD和6颗GPU XCD堆叠在同一个I/O Die上，实现了混合工艺异构集成。

### 3.2 HBM的Stacked封装演进

HBM（High Bandwidth Memory）的成功与其堆叠封装密不可分。每一代HBM使用TSV穿透存储Die并通过微凸点进行垂直互连：

- **HBM2e**：8层Die堆叠，每堆容量16GB，带宽每堆约460GB/s
- **HBM3**：12层Die堆叠，每堆容量24GB，带宽每堆约819GB/s
- **HBM3e**：12层Die堆叠，每堆容量36GB，带宽每堆超过1.2TB/s
- **HBM4**（预观2026）：16-24层Die堆叠，每堆容量64-96GB，带宽突破2TB/s/堆

HBM4的关键技术转变之一是将HBM PHY从GPU或SoC侧移至HBM堆叠本身，这要求HBM逻辑Die的制程从原有的20nm级提升至5nm级。同时HBM4将引入更宽的接口（2048-bit数据位宽），这进一步推高了对CoWoS中介层走线密度的要求。

### 3.3 Intel的EMIB与Foveros方案

Intel在Chiplet封装领域的策略与台积电存在有趣的对比。

**EMIB（Embedded Multi-Die Interconnect Bridge）**是Intel的高密度Die-to-Die互连方案，与CoWoS-L在哲学上高度相似。EMIB在标准有机封装基板中嵌入一小块硅桥芯片，仅覆盖需要高带宽互连的Die边缘区域。EMIB的优势在于不需要全硅中介层，从而大幅降低成本，同时桥接部分的凸点间距可达25-45μm。

EMIB的技术挑战在于嵌入式桥的定位精度：桥芯片嵌入封装基板的空腔中，需要毫米级的放置精度，但其上用于连接GPU和HBM的微凸点阵列需要微米级对齐。这要求封装基板制造过程具备极高的控制精度。

**Foveros**与**Foveros Direct**是Intel的3D堆叠方案：

- **Foveros**：基于微凸点（Micro-bump）的Face-to-Face堆叠，凸点间距约25-50μm，通过TSV实现功率和信号穿透。Intel的Lakefield和Meteor Lake处理器采用了Foveros方案，将计算Die和I/O Die进行垂直堆叠。
- **Foveros Direct**：Intel对Hybrid Bonding的回应，实现了与台积电SoIC直接竞争的Cu-Cu直接键合。Foveros Direct的焊盘间距可达到1-3μm级别，在2023年已实现10μm间距的微凸点Hybrid Bonding量产验证。

Intel的产品路线图显示，Foveros Direct将从2025年开始应用于数据中心级处理器，包括未来的Xeon和Gaudi系列AI加速器。

## 4. 互连协议栈：UCIe、CXL与PCIe的协同

### 4.1 UCIe协议栈深度解析

UCIe协议栈从下至上分为三层：

**Physical Layer（PHY）**是协议栈的基石，负责电气信号的发送与接收、时钟管理、通道均衡以及链路训练与初始化。PHY被细分为以下子层：

1. **PMD（Physical Medium Dependent）**：直接驱动凸点/TSV上的模拟信号，包括Tx驱动器和Rx接收器。PMD实现FFE/CTLE/DFE均衡，以及阻抗匹配和ESD保护。
2. **PMA（Physical Medium Attachment）**：负责串行化（Serializer）/解串器（Deserializer）、时钟数据恢复（CDR）和时钟分频。PMA在发送侧将并行的数字数据转换为串行比特流，在接收侧恢复正确的时钟相位和数据进行采样。
3. **PLS（Physical Layer Sublayer）**：处理链路编码（如64B/67B或128B/130B编码，确保DC平衡和时钟恢复的足够跳变）、扰码（Scrambling，减少EMI和串扰）以及链路初始化的TS1/TS2序列。

**Die-to-Die Adapter（D2D Adapter）**是UCIe协议栈的核心创新层，负责：
1. **协议复用（Protocol Multiplexing）**：在单个PHY链路上承载多个协议流量，包括流式协议（Streaming Protocol，用于自定义Chiplet间数据传输）、PCIe/CXL协议和可选的加密协议。
2. **FIT（Flipped Image Technology）**：UCIe允许Die在封装中以任意方向放置，D2D Adapter在逻辑层面自动处理数据通道的翻转、旋转和重映射，以补偿物理凸点布局的方向变化。
3. **CRC与重传**：在D2D Adapter层实现链路的CRC保护和可选的ARQ重传机制。对于Advanced Package场景，由于信道误码率极低（BER<10⁻²⁰），重传通常为可选项；对于Standard Package场景，重传机制提供了必要的容错能力。
4. **FAS（Flit Aware Shutdown）**：支持链路层粒度的功耗管理，通过将链路置于L0s/L1/L2等功耗状态实现动态功耗管理。

**Protocol Layer**是最上层，直接暴露标准协议接口：
- **PCIe/CXL协议**：D2D Adapter将UCIe链路映射为标准的PCIe逻辑链路，使上下游Chiplet能够通过标准的PCIe配置空间进行发现、枚举和事务处理。
- **流式协议（Streaming Protocol）**：面向自定义Chiplet间通信的轻量级协议，不提供PCIe的发现和配置机制，仅提供简单的数据包传输语义。适用于已经独立管理的Die间一致性流。
- **Vendor-defined协议**：允许芯片厂商实现自己的协议封装，只要它们在D2D Adapter层的FIT映射和数据包分段/重组接口上一致。

### 4.2 CXL协议在Chiplet中的角色

Compute Express Link（CXL）协议已超越了其最初的"缓存一致性内存互连"定位，正在向Chiplet内部互连协议演进。

CXL的三种协议类型在Chiplet场景中各有用武之地：

- **CXL.io**：基于PCIe的I/O语义协议，用于Chiplet的设备枚举、配置空间访问和DMA操作。在Chiplet环境中，CXL.io替代了传统的JTAG/I²C等带外管理通道。
- **CXL.cache**：缓存一致性协议，允许一个Chiplet（如AI加速器）缓存来自另一Chiplet（如Host CPU）的内存数据。CXL.cache的关键在于其一致性粒度（64B cache line）和snoop机制，使其能够在单个Die内NoC一致性协议（如MESI/MSI）之外，跨Die提供一致性协议。
- **CXL.mem**：内存访问协议，支持Host CPU通过load/store语义直接访问Chiplet的本地内存或附加内存池。对于HBM在AI加速器上的应用，CXL.mem使CPU和GPU之间在虚拟地址空间内统一访问HBM数据，无需显式的DMA copy。

在Chiplet语境下，CXL最引人注目的应用是内存池化（Memory Pooling）。通过CXL.mem的内存池化，多个Chiplet（如PCIe插槽上的4个AI加速器）可以共享一个CXL Type-3内存控制器访问同一片DDR5或HBM池，实现"内存即服务"架构。

### 4.3 PCIe Gen6/Gen7在Chiplet场景的定位

虽然PCIe最初被设计为板级互连，但在Chiplet场景中，PCIe作为UCIe的标准协议层之一，保留了从芯片内部到板级无缝扩展的一致性协议语义。

PCIe Gen6引入了PAM-4信令和56GT/s的数据速率，其核心变化是：
- **PAM-4代替NRZ**：PAM-4在相同Nyquist频率下传输两倍比特位：4个幅度级别编码2-bit/symbol，Nyquist频率减半，信道损耗降低约10-12dB。代价是信号裕度从NRZ的1.0 V下降到约0.33 V，需要更强的前向纠错（FEC）。
- **FLIT模式**：PCIe Gen6取消了TLP（Transaction Layer Packet）的边带信号，所有元素在一个固定长度的Flit（238字节）内传输，包括数据、CRC和FEC parity。这种简化提高了链路利用效率，但对延迟敏感的Chiplet内部应用并不总是理想选择。

在UCIe框架下，PCIe Gen6/7的主要角色是作为"水晶球协议"——帮助基于UCIe的异构Chiplet系统与符合PCIe标准的板级设备实现统一寻址空间。PCIe协议层在D2D Adapter上运行，不直接暴露PHY层——后者由UCIe PHY完全接管。

### 4.4 TileLink与OpenCAPI的差异化

在UCIe和CXL成为业界主流之前，RISC-V生态和IBM阵营分别提出了自己的Chiplet互连方案：

**TileLink**是由SiFive和UC Berkeley提出的RISC-V原生Chiplet互连协议，定义在RISC-V TileLink规范中。TileLink的特点在于其多通道协议架构：
- TL-UL（Unordered+Atomic）：精简通道，用于配置空间
- TL-UH（Unordered+Atomic+Hint）：不支持缓存一致性的高性能通道
- TL-C（Cached）：支持完整的缓存一致性
- TileLink的差异化在于其五通道独立A/B/C/D/E握手机制（Channel A: Acquire, B: Probe, C: Release, D: Grant, E: Finish），以及细粒度的原子操作支持。在Chiplet场景中，TileLink的优势在于其与RISC-V核的天然亲和力，劣势在于生态规模远不及CXL。

**OpenCAPI**（Open Coherent Accelerator Processor Interface）由IBM、Google、Mellanox等发起（现为OpenCAPI Consortium管理），其核心是允许加速器通过高度优化的内存语义直接访问主机内存。OpenCAPI 3.0在物理层支持25GT/s，并采用了与UCIe不同的协议栈——OpenCAPI的事务层直接暴露host内存的系统级一致性域。但在UCIe的生态碾压下，OpenCAPI的实际部署量极小。

**差异化总结**：UCIe+CXL构成了一个覆盖从PHY到一致性域的全栈标准组合，是目前唯一具有广泛产业联盟支持的Chiplet互连方案。TileLink在RISC-V社区内仍有生命力；OpenCAPI在IBM Power生态外的存在感已大幅降低。

## 5. 芯片网络（Network-on-Chip）拓扑

### 5.1 拓扑结构的权衡分析

Chiplet时代的NoC面临一个根本性挑战：当通信边界不再局限于单个Die内部，NoC的拓扑设计需要从Die内的平面结构扩展到Die间（甚至跨Interposer）的层次化结构。

**Mesh拓扑**是当前NoC最广泛使用的拓扑，特别是在GPU和AI加速器中。以NVIDIA的AI芯片为例，其内部SM（Streaming Multiprocessor）阵列通过规则的2D Mesh网络连接，每个路由器连接一个计算节点和4个相邻路由器。Mesh的优点在于：
- 布线规则，适合VLSI布局
- 基于XY维序路由可保证无死锁
- 局部性好的访问模式能够获得远低于随机图的延迟

但在Chiplet场景中，Mesh面临一个关键问题：当Mesh跨越Die边界时，Die间接口（如UCIe PHY）的带宽远小于Die内Mesh链路的带宽。典型的Die内Mesh链路宽度为128-256-bit@1-2GHz，提供约32-64GB/s的带宽；而Die间UCIe链路（假设100μm间距）约为100-200GB/s。

**Ring拓扑**在AMD的Zen架构中得到了广泛应用。Zen的CCD（Core Chiplet Die）内部使用多环结构连接各个CCX，每个环上挂载4个CPU核心和共享L3缓存。Ring的延迟随节点数线性增加：

$$L_{ring} = \frac{N}{2} \cdot (t_{hop} + t_{wire}) + t_{lat}$$

其中N为环上节点数，t_hop为路由器跳转延迟，t_wire为线延迟（在先进制程下主要受RC延迟支配）。对于不超过8节点的Ring，其控制逻辑简单、面积效率高，但在8节点以上功耗和延迟迅速恶化。

**Butterfly与Clos拓扑**在高性能计算网络的NoC中被采用，特别是Chiplet间的高扇出通信场景。Butterfly网络的主要优势是其对数延迟复杂度O(log N)，但代价是布线拥塞和路由器的巨大面积开销。

**Torus拓扑**是Mesh的Ring化变体——在Mesh的每一行和列上增加回环连接，形成每行/列的环形拓扑。Torus相比Mesh的优势在于所有节点具有对称的距离特性，消除了Mesh的边缘效应。然而，Torus在VLSI布局中极其困难——回环连接需要在Die边缘走长线，在单一Die内尚可接受，跨Chiplet的Torus回环是学术界仍在研究的问题。

### 5.2 NoC路由算法

**Dimension-Order Routing（DOR）**是NoC最基础的路由策略。在2D Mesh中，DOR即XY路由：先沿X方向（水平）走到目标列，再沿Y方向（垂直）走到目标行。DOR的最大优势是具有天然的死锁免变性（Deadlock-free），因为其不产生依赖于资源循环的路径依赖。

**Adaptive Routing**允许路由器基于当前网络拥塞状态动态选择路径。在Chiplet场景中，Adaptive Routing的意义在于：Die间链路的带宽远低于Die内链路，拥塞往往集中在Die边界。通过Adaptive Routing，NoC可以将流量分散到多个Die间通道上：

$$\text{Cost}(p) = w_1 \cdot \text{Hops}(p) + w_2 \cdot \text{Congestion}(p) + w_3 \cdot \text{CrossDiePenalty}(p)$$

其中CrossDiePenalty(p)是一个重要的新维度——路由算法需要显式建模"跨Die"的代价。本节点会优先选择不跨Die的路径；当不得不跨Die时，优先选择当前拥塞程度最低的Die间通道。

**Deadlock Avoidance**在跨DieNoC中是一个开放性问题。DOR在单DieMesh中保证无死锁，但多Die系统需要在Die间接口处引入虚通道（Virtual Channel，VC）来打破协议依赖循环。UCIe的D2D Adapter层支持VC机制，但Die间VC的数量受物理凸点数限制。

### 5.3 跨Die的NoC一致性

在Chiplet系统中，跨越Die边界的缓存一致性协议是NoC设计中最大的挑战之一。一致性协议（如MESI/MSI/MOESI）依赖snoop或directory机制跟踪cache line状态，但在跨Die场景下：

- **Snoop + Broadcast效率极低**：广播到多个Chiplet的每个一致性事务会消耗Die间带宽，其带宽开销与Chiplet数成正比
- **Directory方案是实际选择**：在I/O Die或中介层上的集中式Directory跟踪每个cache line的持有者。AMD的Zen 2/3/4架构中，I/O Die（cIOD）就是实际的Directory宿主。
- **Non-Inclusive/Non-Exclusive目录**减少了Directory的存储开销，但增加了协议复杂度

CXL协议的CXL.cache通道提供了Chiplet间一致性协议的标准化方案：作为一致性协议的下游从属，CXL.cache允许AI加速器Chiplet利用主CPU的一致性域，而无需自己实现完整的snoop/directory机制。

## 6. 典型Chiplet案例解剖

### 6.1 NVIDIA Blackwell B200：NVLink-Chip-to-Chip

NVIDIA的Blackwell B200 GPU是当前Chiplet集成密度最高的商用AI芯片之一。Blackwell的核心架构策略是将一个完整的GPU拆分为两个Die，通过高速Die间互连进行缝合。

**双Die互联方案**：B200由两块约400mm²的Die通过CoWoS-L封装连接。Die间互连使用NVLink-Chip-to-Chip（NVLink-C2C）技术——这是一个基于台积电CoWoS-L局部硅桥的UCIe兼容实现。

关键规格：
- Die间总带宽：10TB/s（双向）
- 每条NVLink-C2C通道的数据速率：64GT/s（PAM-4）
- Die间总通道数：约80条差分对（双向各40条）
- 延迟：<30ns的单向Die-to-Die延迟
- 能耗效率：约0.3 pJ/bit

NVIDIA的工程选择体现了对通信延迟的极致追求。10TB/s的带宽意味着在不到1ns的时间内，一个Die可以将完整的HBM3e Stencil数据（约10KB）发送到另一Die——这是保证Blackwell在大模型训练中实现近乎线性加速的关键。

Blackwell的Die间互连设计也体现了"不对称性"的有意选择：B200内部的NVLink-C2C仅用于两块GPU Die之间的内存一致性互连，而非用于NVSwitch连接。后者使用独立的NVLink 5.0 SerDes。这种分离的互连架构——用于Die缝合的NVLink-C2C和用于节点间通信的NVLink 5.0——在物理层和协议层上互为补充。

### 6.2 AMD Infinity Architecture：从Zen到MI300

AMD是Chiplet架构最积极的先行者。

**Zen CCD + IOD演进**：AMD从Zen 2（2019）开始采用多芯片模组（MCM）设计，将CPU核心分离到CCD（Core Chiplet Die）中，I/O和内存控制器集中在cIOD（client I/O Die）或sIOD（Server I/O Die）。关键的几代演进：

- **Zen 2（7nm CCD + 14nm IOD）**：采用Infinity Fabric作为Die间互连协议，运行在1.8GHz，每个CCD到IOD提供36GB/s带宽
- **Zen 3（7nm CCD + 12nm IOD）**：Infinity Fabric频率提升至2.0GHz，带宽提升11%
- **Zen 4（5nm CCD + 6nm IOD）**：Infinity Architecture升级为Infinity Fabric 3.0，加入了对CXL 1.1的原生支持
- **Zen 5（3nm CCD + 6nm IOD）**：Infinity Fabric支持PCIe Gen5，引入更高效的统一一致性域

AMD的CCD使用双向环网（Dual Ring）内部连接8个CPU核心和32MB L3缓存。在跨Die访问时，CCD通过GMI（Global Memory Interconnect）链路连接到IOD，IOD通过snoop filter过滤一致性请求。对于跨CCD的内存访问，请求需要经过"CCD本地Ring→GMI→IOD Snoop Filter→GMI→目标CCD Ring"路径，引入约40-80ns的额外延迟。

**MI300系列**是AMD Chiplet架构的里程碑——它是首个将CPU CCD、GPU XCD和HBM3堆叠在同一封装上的APU。MI300X包含6颗GPU XCD（Graphics Compute Die，5nm）、3颗HBM3堆叠和1颗I/O Die，使用SoIC进行3D堆叠和CoWoS进行2.5D集成。MI300A则进一步包含3颗Zen 4 CCD和6颗GPU XCD，实现了CPU+GPU的片上异构计算。

### 6.3 Intel Ponte Vecchio：47个Tile的极限集成

Intel的Ponte Vecchio数据中心GPU是芯片行业历史上最具野心的Chiplet项目——单颗芯片由47个Tile（Chiplet）组成，采用Intel 7、Intel 4和台积科N5/N7四种制程，使用EMIB和Foveros两种封装技术。

Tile布局：
- 16个Compute Tile（Intel 4工艺，每个含8个Xe核）
- 8个RAMBO Tile（SRAM缓存，台积电N5）
- 2个Fabric Tile（负责跨Tile的NoC互连）
- 8个HBM3堆叠（单堆24GB，共192GB）
- 11个基础Die（Base Die，通过Foveros实现垂直供电和数据路由）

Ponte Vecchio的NoC拓扑是一个复杂的二维层次化结构：
1. **Tile内部**：每个Compute Tile内使用2D Mesh连接8个Xe核和L1缓存
2. **跨Tile Mesh**：Fabric Tile实现了跨所有Compute Tile的二级Mesh网络
3. **EMIB桥接**：高带宽的Compute Tile与RAMBO Tile之间通过EMIB实现数据路径

Intel的官方数据显示，Ponte Vecchio的跨Tile总互连带宽超过10TB/s，Die间延迟控制在100ns以内。

然而，47个Tile的Chiplet集成带来了严重的良率和供电挑战。Intel的良率数据表明，仅Compute Tile的KGD良率就决定了封装后的整体良率——47个Tile中任何一颗坏点都会导致整颗芯片降级或报废。这也是Intel最终弃用Ponte Vecchio架构、转向Falcon Shores的设计简化路线的重要原因。

### 6.4 Tesla Dojo D1 Chiplet互连

Tesla Dojo是一个为自动驾驶训练量身定制的超级计算机，其核心计算单元是D1 Chip——一个完整的Chiplet系统。

Dojo的互连架构极具特色。D1 Die本身是一个约645mm²的巨型芯片（接近光罩极限），包含354个计算节点。但Tesla真正创新的是**Dojo的片上网络（Dojo Interface Processor）**：

- Dojo的训练网格由25个D1 Die组成5×5的2D Mesh，通过Dojo Interface Processor（DIP）连接
- DIP实现了标准的2D Torus拓扑——每颗D1 Die通过4个方向连接到相邻Die
- Die间互连的数据速率：4.0 GT/s，使用差分信号
- 单向带宽：DIP提供的每个方向约400GB/s

Dojo的NoC核心设计哲学是"对延迟极其敏感"：为了实现整个25-Die训练网格的同步SGD，跨Die的梯度聚合必须在1μs内完成。这要求NoC的死锁避免和拥塞控制策略极其激进——Dojo使用了一种称为"TDMA-like"的确定性调度，为每个计算节点预留了精确的时间槽。

## 7. 测试与可靠性

### 7.1 Known-Good-Die的挑战

Chiplet架构最大的经济优势——通过小Die提高良率——同时引入了最大的测试挑战：KGD问题。

在单芯片场景下，Die的测试可以在晶圆级（Wafer Sort）和封装级（Final Test）两个阶段完成。Chiplet场景增加了第三维度：**封装前KGD验证**——每颗Die必须在封装前确认其功能、性能和可靠性。如果一颗Die在封装后被发现存在缺陷，整个封装需要报废或降级使用。

KGD测试的核心困难在于：
1. **功能完整性测试**：Chiplet（如AI加速器中的Compute Die）通常依赖其他Chiplet（如缓存Die和HBM）才能完成完整的指令序列。封装前的孤立测试只能覆盖有限的测试向量
2. **高速互连测试**：Die间UCIe PHY工作在16-64GT/s，晶圆探针台（Prober）的测试频率和信号完整性与实际封装环境存在显著差异
3. **功耗与热测试**：单个Die在晶圆级测试时的散热条件与封装后差异巨大，导致漏电和温升特性不同

产业界的应对方案包括：
- **DFT内建自测试**：每个Chiplet内集成BIST引擎，能够在封装前独立完成最大的功能覆盖率
- **基于ATE的Known-Good-Die测试**：使用高性能ATE（Automatic Test Equipment）在晶圆分切前完成完整的PHY层和协议层测试
- **冗余设计**：在关键路径上预留冗余通道——当某一Die间通道检测到永久性故障时，通过冗余通道进行自动重映射

### 7.2 Die间互连测试标准

Chiplet的互连测试依赖于IEEE的两个关键标准：

**IEEE 1149.1（JTAG）**是Die间测试的基础结构。每个Chiplet中的JTAG TAP（Test Access Port）通过边界扫描链（Boundary Scan Chain）访问所有Die间I/O引脚。在封装前，JTAG用于验证Die间互连的电气完整性。在封装后，JTAG是Die间链路训练和故障隔离的标准手段。

**IEEE 1687（IJTAG）**扩展了JTAG的能力，引入了可重配置的测试访问网络。IJTAG的关键创新是Segment Insertion Bit（SIB）——允许测试逻辑在运行时动态选择要接入边界扫描链的测试模块。对于Chiplet场景，IJTAG使得一个复杂的47-Tile系统（如Ponte Vecchio）的测试访问可以在单条边界扫描链上实现灵活的拓扑配置。

UCIe 2.0的标准测试框架集成了IEEE 1149.1和1687的要求，定义了以下测试模式：
- **Loopback模式**：发送侧的数据经Die间通道环回至发送侧的RX端，测试PHY的完整性
- **PRBS模式**：发送伪随机比特序列（PRBS-7/15/31），通过接收侧的BER tester评估信道误码率
- **Eye Scan模式**：扫描接收侧的眼图边界，评估信道的信号裕度和噪声容限

### 7.3 热应力与机械可靠性

Chiplet系统面临的机械可靠性挑战源于**热膨胀系数（CTE）失配**：

- 硅Die的CTE：~2.6 ppm/K
- 有机封装基板的CTE：~16-20 ppm/K
- 硅中介层的CTE：~3.0 ppm/K

当系统在不同功率状态下循环时，Die与基板之间的CTE差异会导致显著的机械应力。对于大面积的CoWoS封装（2x-8x Reticle），X方向的累积应力可能达到数百MPa。应力失效模式包括：

1. **凸点裂纹**：C4或微凸点在热循环下产生疲劳裂纹。Lowe等人（2020）的研究表明，CoWoS封装的微凸点在-55°C到125°C的1000次热循环后裂纹率在4-12ppb级别[3]
2. **TSV应力**：硅通孔周围的热应力集中在体积膨胀的Cu和硅基体的界面。TSV直径越大、深度越深，应力越显著。这使得CoWoS-S的TSV间距受限于机械可靠性需求而非电学要求
3. **Die翘曲**：超大面积的薄Die（<100μm厚）在封装和测试过程中的翘曲控制是主要良率杀手

热管理面临的第二个挑战是**热点集中**：在3D堆叠的SoIC方案中，底部Die的热量必须通过顶部Die的TSV和微凸点传导到散热器。热传导路径上的每一层微凸点引入了额外的热阻。Naffziger等人（2020）在AMD MI300的SoIC方案中指出，底部Die的温度比顶部Die高出15-25°C，要求功率密度设计时必须考虑的不对称热预算[4]。

### 7.4 供应链与多供应商Chiplet生态

UCIe协议通过标准化PHY和协议层，为多供应商Chiplet生态提供了技术基础。但产业生态的建设远不限于技术标准：

- **商业模型**：Chiplet的商业模式仍在探索中。台积电作为代工厂，同时提供SoIC/CoWoS封装和UCIe PHY IP授权；Intel Foundry的IFS则通过EMIB/Foveros封装捆绑其Die-to-Die IP。独立的Chiplet供应商（如提供AI加速器Die的创业公司）面临"鸡与蛋"的问题——没有封装验证的Die就无法获得客户，没有客户就没有量产验证。
- **测试标准**：不同供应商的Chiplet在生产测试中使用的ATE和测试向量格式尚未统一。UCIe联盟正在推广标准化的DFT和测试接入机制，但各代工厂的测试流程仍然高度定制化。
- **知识产权**：Chiplet间的互连PHY实现（如UCIe PHY的PMA/PMD子层）涉及复杂的模拟电路IP，各代工厂的PDK差异意味着UCIe PHY并非真正可移植。UCIe联盟正致力于推动PHY IP在台积电、三星、Intel之间的可复用性，但短期内模拟PHY的跨代工厂移植仍不现实。
- **可靠性认证**：在多供应商Chiplet系统中，可靠性认证需要覆盖供应商A的Die、供应商B的Die、代工厂C的中介层和封装基板。责任划分（Die供应商 vs 封装供应商 vs 系统供应商）在行业内尚未达成共识。

## 总结与展望

Chiplet互连技术正在经历从"少部分企业的专有技术"向"行业标准化基础架构"的范式转变。UCIe 2.0的发布和3D封装技术的量产化标志着Chiplet技术从"可用"走向"高效"。

展望未来几年的关键技术趋势：
1. **更高的互连密度**：Hybrid Bonding的凸点间距将从1-3μm进一步缩小至亚微米级，3D SoIC的带宽密度将在2027年前突破5000 GB/s/mm
2. **光电融合互连**：在中介层层面引入光学互连（硅光引擎），突破电互连的带宽距离积（Bandwidth-Distance Product）限制
3. **Chiplet即服务（Chiplet-as-a-Service）**：随着UCIe PHY的成熟度提高，代工厂、IP供应商和Chiplet设计公司之间将形成类似于云API的标准接口

Chiplet的本质是半导体产业从"单芯片SoC"向"芯片级系统级封装（SiP）"的深层回归——这不是简单的封装技术进步，而是对摩尔定律终结的根本性回应。

---

## 参考文献

[1] International Roadmap for Devices and Systems (IRDS). "More Moore White Paper." IRDS, 2023 Edition.

[2] Kahng, A. B. "Lithography-induced limits to scaling and design." Proceedings of the IEEE, vol. 109, no. 2, 2021, pp. 116-136.

[3] Lowe, A., et al. "Thermal cycling reliability of micro-bumps in CoWoS packaging." IEEE Transactions on Components, Packaging and Manufacturing Technology, vol. 10, no. 8, 2020, pp. 1312-1323.

[4] Naffziger, S., et al. "Pioneering chiplet technology and design for the AMD EPYC and Ryzen processor families." 2020 IEEE International Solid-State Circuits Conference (ISSCC), pp. 30-33.

[5] Ouyang, J., et al. "UCIe: A Universal Chiplet Interconnect Express Standard." IEEE Micro, vol. 43, no. 4, 2023, pp. 36-45.

[6] Loh, G. H., et al. "Interconnect challenges and solutions for 3D-stacked chiplet systems." ACM Journal on Emerging Technologies in Computing Systems, vol. 18, no. 3, 2022, pp. 1-25.

[7] Dally, W. J., and Towles, B. P. "Principles and Practices of Interconnection Networks." Morgan Kaufmann, 2004.

[8] Hoskote, Y., et al. "A 5-GHz mesh interconnect for a teraflops processor." IEEE Micro, vol. 27, no. 5, 2007, pp. 51-61.

[9] Zhao, J., et al. "Design of a scalable Network-on-Chip with chiplet integration." Proceedings of the 56th Annual IEEE/ACM International Symposium on Microarchitecture (MICRO), 2023, pp. 458-470.

[10] Kim, J., et al. "System-on-Package: A new paradigm for heterogeneous integration." Proceedings of the IEEE, vol. 111, no. 1, 2023, pp. 72-88.

[11] Beck, N., et al. "Ponte Vecchio: A 47-tile chiplet-based GPU for HPC and AI." 2023 IEEE Hot Chips 35 Symposium, pp. 1-25.

[12] Kahle, J. A., et al. "The cell processor architecture." 2005 ACM/IEEE 32nd Annual International Symposium on Computer Architecture (ISCA), pp. 131-141.

[13] Arunachalam, V., et al. "Known-good-die testing challenges and solutions for chiplet-based systems." IEEE Design & Test, vol. 40, no. 2, 2023, pp. 35-44.

[14] Selvanathan, D., et al. "Thermal and mechanical reliability of heterogeneous integrated chiplet systems." IEEE Transactions on Device and Materials Reliability, vol. 22, no. 4, 2022, pp. 512-523.
