# AI科学计算：RNA结构与基因编辑的AI方法

## 1. RNA结构与功能

### 1.1 RNA类型的结构差异与层级折叠

核糖核酸（RNA）的分子生物学功能多样性根植于其结构的层次性折叠原理。信使RNA（mRNA）作为蛋白质编码模板，其5'端帽结构（7-甲基鸟苷三磷酸）与3'端poly(A)尾协同调控翻译起始效率，编码区内部的二级结构（特别是起始密码子AUG周围的茎环结构）直接影响核糖体扫描效率（Kozak序列上下文依赖）。转运RNA（tRNA）呈现经典的三叶草二级结构，通过反密码子环与mRNA密码子的碱基互补配对实现遗传信息的精准解码，其三级结构通过D环与TΨC环的长程相互作用（L-型折叠）维持刚性构象以适配核糖体A/P/E位点的空间约束（Giegé et al., 2012, *Nucleic Acids Research*）。

核糖体RNA（rRNA）构成核糖体的催化核心，原核23S rRNA的肽酰转移酶中心直接催化肽键形成——这是RNA分子具有催化活性的关键证据。小非编码RNA（miRNA, siRNA, piRNA）的长度在20-35nt之间，其前体折叠为发夹结构（miRNA前体pri-miRNA经Drosha加工后的~70nt茎环），经Dicer切割产生成熟双链体以加载至Argonaute蛋白形成RNA诱导沉默复合体（RISC）。长链非编码RNA（lncRNA, >200nt）的折叠更复杂，其二级结构中的保守模块（如MALAT1的三螺旋结构）赋予蛋白质支架功能（Quinn & Chang, 2016, *Nature Reviews Genetics*）。

RNA的一级序列通过碱基堆积力与Watson-Crick配对（A-U、G-C）以及摆动配对（G-U）驱动折叠为二级结构。折叠层级遵循热力学分级原理：二级结构的形成（碱基互补配对→茎/环/凸起/内部环/多分支环）先于三级结构的远程接触（假节、碱基三联体、A-minor motif、ribose zipper）。三级结构的稳定依赖于Mg²⁺离子的桥接作用和阳离子浓度依赖性静电屏蔽——单价阳离子（K⁺）中和磷酸骨架电荷后，Mg²⁺特异性结合于RNA口袋的特定氧原子位点（Draper, 2004, *RNA*）。

### 1.2 RNA结构预测对药物开发的战略意义

RNA结构预测在以下三个前沿领域中具有变革性价值：

**靶向RNA的小分子药物设计**。传统药物开发聚焦蛋白质靶点，但人类基因组编码的非编码RNA数量远超蛋白质。RiboTAC（核糖体靶向嵌合体）通过小分子同时结合RNA和E3泛素连接酶以诱导RNA降解（Haniff et al., 2020, *ACS Central Science*）。其设计依赖于RNA二级结构上的小分子结合口袋（如rRNAs的A位点、pre-miRNA的发夹环、lncRNA的三级折叠模块）的精确结构信息。RNA结构预测可提供小分子结合位点的二级结构拓扑，降低通过FRET或SHAPE-MaP实验筛选的试错成本。

**mRNA疫苗的序列设计与免疫原性优化**。mRNA疫苗的翻译效率与稳定性高度依赖5'UTR和3'UTR的二级结构设计。编码区的密码子优化不仅影响翻译速率，还改变mRNA的折叠模式——高GC含量的茎环结构可能降低核糖体延伸速率但提高mRNA半衰期。2023年COVID-19 mRNA疫苗的经验表明，UTR的结构稳定性与TLR7/8介导的先天性免疫识别之间存在权衡（Mauger et al., 2019, *PNAS*）。结构预测工具可优化UTR折叠以平衡翻译效率与免疫原性。

### 1.3 RNA折叠中的热力学与动力学之争

RNA折叠的核心理论问题是：折叠过程是由热力学最小自由能（MFE）决定，还是受动力学路径的不可逆性控制？在体外淬火条件下（高温快速退火至室温），大多数RNA趋于MFE结构（Mathews et al., 2004, *PNAS*）。但在细胞内的共转录折叠过程中，RNA从5'→3'方向逐步合成并伴随折叠，该动力学过程可能陷入局部自由能最优而无法到达全局最低点。30S核糖体亚基的组装证明，某些RNA需借助RNA伴侣蛋白（如StpA、Hfq、DEAD-box解旋酶）来克服折叠动力学陷阱。RNA折叠的粗糙能景（Rugged Energy Landscape）存在大量亚稳态中间态，这使基于热力学的预测算法在解释体内真实折叠时面临系统误差——预测的MFE结构与SHAPE化学探针实验的体内结构之间的差异是评估预测模型准确度的关键校准数据（Kwok et al., 2013, *Nature Communications*）。

## 2. RNA二级结构预测

### 2.1 传统MFE方法：Zuker算法与Turner参数

最小自由能（MFE）方法的核心假设是RNA在给定条件下采取热力学最稳定的二级结构。该领域的基础算法由Zuker与Stiegler于1981年提出——通过动态规划在O(n³)时间复杂度和O(n²)空间复杂度内计算RNA序列的最小自由能二级结构。Zuker算法的数学本质是将RNA二级结构分解为独立折叠的loop子结构，依据最近邻热力学模型（NNM）对每一子结构赋予自由能增量，在满足嵌套结构的约束下搜索全局最优解。

**Turner参数**构成了NNM的参数化基础。Turner规则通过碱基堆积和loop热力学的实验测量（UV熔解曲线和NMR）归纳出一套标准最近邻自由能参数表（Serra & Turner, 1995, *Methods in Enzymology*）。具体而言，双链区的自由能变化由相邻碱基对的堆积贡献（-0.9至-3.4 kcal/mol不等，依赖具体的二核苷酸上下文和盐浓度）加总而得；环区的自由能惩罚包括：发夹环（熵罚+4~+8 kcal/mol，该值与环长度和末端碱基对类型高度相关）、凸起环（每个突起的碱基约+3~+4 kcal/mol）、内部环（不对称内部环的惩罚大于对称环）和多分支环（与环数量和未配对碱基数成正比）。2018年更新的Turner参数（Turner2004的修订版）纳入了基于NMR的环区域校正。

**预测准确度及其限制**：在基准测试上，MFE方法在标准的配对级精确度（PPV）和敏感性上通常达到60-70%（在ENCODE和RNAStralign数据集上）。当序列包含假节时——这些涉及非嵌套交叉配对的特殊结构（如端粒酶RNA模板域和某些病毒RNA在核糖体移码信号中起到核心功能）——标准动态规划方法由于嵌套约束的算法限定直接失效。为解决此问题，基于热力学的Rivas-Eddy算法扩展了动态规划使其能在O(n⁶)复杂度内处理有限类型的假节，但该算法因时间成本过高无法用于长链RNA的全基因组分析。

### 2.2 深度学习RNA结构预测的新进展

近年来，深度学习在RNA二级结构预测中带来了革命性变化，核心思路是将预测任务重新定义为从序列到碱基配对概率矩阵（contact map）的端到端映射。

**SPOT-RNA**（Singh et al., 2021, *Nature Communications*）使用二维残差网络（ResNet）耦合条件随机场（CRF），将RNA二级结构预测抽象为二维图像的分割问题。具体流程：输入序列经过预训练的一维ResNet提取每个残基的单体特征（进化保守性特征、位置特异性打分矩阵PSSM、理化属性编码），随后通过外积操作构建配对表示，输入二维残差网络（12-24层，每层含64-128个卷积核）学习残基配对的依赖关系。输出层是一个40×N×N的概率矩阵（40种配对类型），并通过CRF层对输出的pair概率做符合结构约束的解码（确保无伪节约束下的结构一致性）。SPOT-RNA在测试集上F1达到0.74，显著优于传统MFE方法的~0.60。然而其显著限制是训练数据仅包含无伪节结构，导致对假节的泛化能力有限。

**UFold**（Fu et al., 2022, *Nature Computational Science*）创新性地构建了U-Net架构的RNA结构预测模型：将序列的编码矩阵（通过one-hot编码、DCA共进化特征和预测配对概率作为输入通道）视为图像，在U-Net架构上执行图像分割以输出碱基配对概率。与SPOT-RNA的负样本采样处理方式不同，UFold通过数据增广技巧（模拟不同离子条件下的Turner参数变化）来提高泛化能力。在ArchiveII和RNAStralign数据集上UFold的F1达到0.75-0.80，且运行时间仅为MFE方法的秒级以下。

**DMfold**（Zhang et al., 2023, *bioRxiv*）进一步借鉴了AlphaFold2的成功经验，尝试直接从进化的序列比对（MSA）预测RNA二级结构。该模型通过EvoFormer式架构——在MSA的行与列间交替进行轴向注意力（axial attention），构建从多序列比对的进化耦合到距离图的映射。这克服了传统MFE无法利用进化信息的不足。但一个关键挑战是RNA序列的快速进化率（缺乏跨物种保守的序列区块）使得典型的MSA构建（通过BLAST/MEME在序列数据库中搜索同源体）在RNA上经常退化至仅包含少数有效序列，无法提供稳定的共进化信号。rMSA（Zhang & Liu, 2023）通过引入结构引导的MSA构建策略解决了部分问题——首先通过已知同源RNA的三级结构进行结构比对，再回溯序列比对，显著提高了MSA的同源序列覆盖率。

**三级约束的二级预测**：一个新兴方向是利用三级折叠中的远程接触信息来约束二级结构预测。三级结构预测（见第3节）生成的残基间距离图可作为先验条件进入二级结构的搜索空间，即预测的二/三级结构的联合优化。

## 3. RNA三级结构预测

### 3.1 AlphaFold3 的跨模态 RNA 建模突破

AlphaFold3（Abramson et al., 2024, *Nature*）实现了RNA三级结构预测的重大突破。这是第一个在统一框架中同时处理蛋白质、RNA、DNA、小分子和化学修饰的端到端结构预测系统。

**Pairformer架构**——AF3的输入采用全分子联合编码：序列的token表示（每种化学组分使用独立的token化方法——RNA使用核苷酸符号、蛋白质使用氨基酸符号、小分子使用原子级图表示）通过线性投影至统一词嵌入空间。这里避免了对不同分子的序列类型进行分离（与AF2不同，AF2在pairwise表示中对残基类型做了显式标签区分），而是直接在同一pairwise表示中编码不同分子间的相对位置关系。RNA序列的成对相对位置编码使用正弦位置编码（positional encoding）对序列索引差进行编码——这直接捕获了RNA聚合物的一维线性约束内核（链上残基的距离约束下限~3.4Å）。在AF3中Position Encoding的截断使得远程残基对的位置指示随着索引差增加而退化，这与RNA的物理实际情况一致。

**RNA键合约束的坐标映射**：AF3在此框架中集成了RNA特有的物理化学约束：共价键约束（磷酸二酯键的~1.5Å/127°的键长键角通过铰链损失而非谐波约束施加）、non-bonded约束（碱基堆积的π-π相互作用通过距离依赖的范德华势能函数隐式建模、G-U摆动配对的氢键约束与标准Watson-Crick配对被同等对待、静电屏蔽的Debye-Hückel近似的距离效应通过递送到Pair Representation的梯度信号协调）。在上述约束中，AF3的扩散模块（diffusion module）体现在去噪匹配过程的坐标梯度中对RNA链的磷酸二酯主链几何实施硬约束——这解决了之前模型在预测RNA构象时磷酸骨架键长不合理膨胀的惯性问题。

在测试基准CASP15和RNA-Puzzles上，AF3的RNA部分的lDDT平均达到0.80，超过以往所有RNA专用预测方法（如SimRNA的~0.55和3dRNA的~0.60）近20个百分点。但AF3在长链RNA（>500nt）上的表现仍然受限，长程折叠依赖的碱基三级接触在串联的多条RNA链的batch中因蒸馏（distillation）噪声而难以收敛。

### 3.2 RoseTTAFold All-Atom 与 trRosettaRNA

**RoseTTAFold All-Atom**（RFAA, Krishna et al., 2024, *Science*）是AF3的同期竞争者。RFAA在框架层面将RoseTTAFold从蛋白质扩展到全分子域，关键差异在于使用单序列通道而非AF3的pairformer——对于RNA序列的token化通过1D track传递至2D track交替更新——1D的每个残基的特征包含RNA的序列标识、二面角预测和主链刚性坐标。RFAA在RNA预测上的主要弱点在于其单序列架构无法充分捕获RNA的共进化信息（同源MSA的缺乏限制了RFAA对tRNA和rRNA等具有高度保守结构的RNA的预测精度），但在无同源序列的合成RNA适配体上RFAA表现优于AF3（因为不需要MSA）。

**trRosettaRNA**（Wang et al., 2022, *Nature Protocols*）采用基于旋转不变坐标预测的策略——不同于AF3的坐标直接扩散框架，trRosettaRNA将结构预测分解为距离图预测（残基间C3'→C3'距离和Ω/θ/φ角度的联合分布）和坐标重建（通过下坡最小化从距离-角度图上重建的原子坐标）。该方法通过保持旋转不变性的坐标表示（不依赖于绝对坐标系，而是通过相对距离+方向角进行参数化）从而避免了AF3对训练集的坐标系依赖偏差。在实际性能上，trRosettaRNA的坐标重建会受到距离图离散化导致的误差累积，在CASP15上average TM-score仅为~0.72。

**整合趋势**：RNA三级预测领域正从单一模型向集成预测方向演进——AF3为主体的端到端模型在大型标准结构上的绝对精度占优，trRosetta的旋转不变性框架在合成RNA和非标准折叠模式下提供互补价值。预测方法的共识投票（consensus）策略被证明可将单模型的lDDT提升5-10%（2024-2025 RNA-Puzzles结果显示），这类似于CASPy在蛋白质结构预测中的成功经验。

## 4. CRISPR与基因编辑工具设计

### 4.1 CRISPR-Cas系统的分类与结构机制

CRISPR-Cas系统是原核生物的适应性免疫系统，通过捕获和利用外来核酸的短片段（spacer）引导Cas核酸酶实施序列特异性切割。系统分类依据效应器的亚基组成分为两大类：

**Class 1系统**（占CRISPR位点约90%）包含多个效应蛋白构成的复合体（Type I的Cascade复合物、Type III的Cmr/Csm复合物）。III型系统在crRNA引导下同时切割靶RNA和非靶DNA（通过Cas10的HD结构域），这一双重活性特征使其在基因编辑中的应用潜力近年来受到关注。

**Class 2系统**（约占CRISPR位点10%）包含单一多结构域效应蛋白。其中最具代表性的是Type II的Cas9——它是一种由crRNA:tracrRNA双链引导的DNA内切酶。Cas9的结构域分工如下：识别叶（REC lobe）负责与sgRNA和靶DNA的互补识别，核酸酶叶（NUC lobe）包含HNH和RuvC两个催化结构域，分别切割靶链（target strand）和非靶链（non-target strand）。当PAM（原间隔序列邻近基序，NGG）被识别后（通过Cas9的PAM相互作用结构域PI的精氨酸桥），R-loop的形成（sgRNA与靶DNA的互补配对使非靶链形成单链易感状态）启动切割——首先HNH结构域切割靶链，随后RuvC切割非靶链，产生双链断裂（DSB）的钝端切口（Jinek et al., 2012, *Science*）。

**Cas12a**（Cpf1, Type V）与Cas9的关键差异：识别富含T的PAM（TTTV）并在靶链的远端切割产生5'突出粘末端（对NHEJ的修复插入方向性具有偏好）；此外Cas12a仅需crRNA即可完成靶向，无需tracrRNA，简化了多重编辑的sgRNA设计。

**Cas13**（Type VI）靶向RNA而非DNA。其两个HEPN结构域切割单链RNA，在切割靶RNA后产生附带切割活性（collateral cleavage）——即非特异性切割周围任何ssRNA——该特征被用于SHERLOCK等诊断平台（Gootenberg et al., 2017, *Science*）。Cas13的效应器域最近被进一步改造为可编程的RNA碱基编辑器（RNA的A→I编辑）和RNA甲基转移酶融合以实现RNA修饰。

**sgRNA的工程化**：天然Cas9使用crRNA与tracrRNA的双RNA引导，但在2013年被工程化为单链向导RNA（sgRNA）——通过人工四碱基环（GAAA tetraloop）将crRNA和tracrRNA连接成一条RNA链（Jinek, Chylinski et al., 2012）。sgRNA的发夹二级结构（R环的茎环1/2/3和四碱基环结构）直接影响Cas9的结合稳定性与解旋效率，sgRNA骨架区的G:C含量和恒定区的延长会影响编辑效率在不同细胞类型中的表现——这启发了later-generation的sgRNA骨架优化设计（如E-CRISPR和Designed sgRNA文库）。

### 4.2 AI辅助的CRISPR工具挖掘

宏基因组挖掘是扩展CRISPR工具箱的核心策略。传统方法（通过BLAST的同源搜索）依赖已有注释的Cas同源物数据库，但大量未知功能的新型CRISPR系统因序列多样性不足以跨越BLAST的检测阈值而被忽略。深度学习改变了这一现状。

**CRISPR系统分类的深度学习方法**使用LSTM或Transformer架构在潜空间中对Cas蛋白的核酸酶结构域进行表示学习——在标记数据有限的条件下对比学习（contrastive learning）框架可从未标注的宏基因组分箱中训练出具有结构域聚类能力的表示，然后通过对潜空间中结构域类型的聚类（结合自注意力机制提取的功能域特征指纹）实现对新型CRISPR效应器的分类与发现。2023年，使用这种方法的搜索在公开宏基因组数据中发现了超过1000种新型CRISPR系统（包括Class 2的新亚型和Class1的未表征变体）。

**DeepCRISPR**（Chuai et al., 2018, *Genome Biology*）是一个用于CRISPR/Cas9打靶gRNA活性预测的深度学习框架。DeepCRISPR的核心创新在于引入去噪自编码器（DAE）对gRNA和靶基因组上下文的序列特征进行无监督学习，生成可显著提高后续预测精度的特征表示。在CNN模型中，输入包括：20bp的靶序列+3bp的PAM序列的one-hot编码（4×24矩阵），以及染色质可及性和表观遗传特征的整合。DeepCRISPR在on-target和off-target预测上分别达到AUC=0.92和AUC=0.85，比早期方法（如CRISPRscan的~0.80）有显著提升。

**CRISPRoff-target预测**是安转基因编辑安全性的核心技术。

## 5. CRISPR脱靶效应的计算预测

### 5.1 脱靶位点的计算寻找与方法比较

CRISPR-Cas9的脱靶效应源于sgRNA与基因组其他位点的部分匹配。Cas9允许的mismatch容忍机制遵循以下特征：PAM远端（靠近PAM近端的种子区）的错配容忍度更低——种子区的1-2个错配即可大幅降低切割效率，而在PAM远端的间隔区（distal region，尤其是13-20nt）允许多达4-5个错配仍然保留可检测的切割活性（Hsu et al., 2013, *Nature Biotechnology*）。错配的类型（嘌呤-嘌呤转换vs嘧啶-嘧啶的错配容忍度不同）、DNA bubble结构的不对称性、以及染色质状态的封闭性也影响脱靶频率。

**CCTop**（Stemmer et al., 2015, *PLoS ONE*）是早期的高效脱靶搜索算法，通过Burrows-Wheeler变换索引实现全基因组范围内与sgRNA序列（20nt + PAM）允许最多4个错配的靶点枚举。搜索复杂度为O(ΣC(20,i)×4ⁱ×ln(G))，涵盖序列错配以及额外1bp或2bp的DNA bulge情况。

**CRISPOR**（Concordet & Haeussler, 2018, *Nucleic Acids Research*）整合了多种脱靶评分算法（MIT score、CFD score、 Cutting Frequency Determination），使用随机森林和SVM从高通量测序数据中学习错配位置与脱靶概率之间的映射。CFD得分引入每个位置错配的特定惩罚权重（例如，位置1的A→G变更惩罚权重0.8vs位置20的同一变更权重0.2），这些权重来自GUIDE-seq数据——一种通过整合酶缺陷慢病毒（IDLV）标记和NGS双端测序来确定DSB位点的方法（Tsai et al., 2015, *Nature Biotechnology*）。

### 5.2 基于深度学习的脱靶热点预测

**CrisprHAL**（Lyu et al., 2022, *Nature Machine Intelligence*）引入了一种混合注意力的学习方法——该架构将sgRNA的序列特征通过二维卷积编码为基础向量，再与靶基因组的染色质状态特征（DNase I超高敏位点的开放可及性打分、组蛋白修饰信号ChIP-seq的H3K4me3信号峰、CTCF的拓扑结构域边界数据）通过注意力机制加权融合。这种方法在预测跨细胞类型的脱靶位点时维持了AUC=0.87-0.91（在HEK293T、K562、HCT116三种细胞类型间迁移）。

**DeepSpCas9**（Kim et al., 2019, *Nature Communications*）基于大规模高通量测序数据（测量超过12,000条sgRNA在人类基因组上的脱靶活性）训练一个端到端的深度学习预测器。DeepSpCas9的基本架构是：用10层扩张卷积（dilated convolution）处理sgRNA序列（每条sgRNA提取与靶序列配对的bubble区域两侧各30bp的DNA序列上下文），输出的打分从0-1映射到切割效率。该方法的最重要发现是：训练数据的来源（细胞类型、是否延长培养时间、转染方法）显著影响模型在跨细胞类型的泛化性能差异——在训练集来源的细胞系上AUC=0.92，但迁移至其他细胞类型时AUC降至0.74-0.81。这一观察提示脱靶预测模型的细胞类型特异性是一大系统未解决挑战。

CrisprHAL与DeepSpCas9的差异：DeepSpCas9的扩张卷积堆叠处理序列的局部依赖性良好，但无法有效捕获染色质可及性的非局部效应；CrisprHAL对此的改进是通过注意力机制显式引入全基因组范围内的染色质拓扑结构域约束。未来方向是将二者在大细胞面板上的联合训练与迁移学习结合，以覆盖更广泛的基因组背景。

### 5.3 in silico CRISPR筛选设计

全基因组CRISPR敲除/激活/抑制筛选的sgRNA文库设计需要同时满足多个相互矛盾的优化目标：（a）每个基因的靶向覆盖度（通常≥4条独特的sgRNA保证冗余性）；（b）全基因组的均匀覆盖度（避免靶向热点区域的偏向性）；（c）最小化脱靶性（脱靶位点总和最低）；（d）最大化各sgRNA的编辑效率（选择机器学习模型评分最高的候选sgRNA）。上述问题可以形式化为一个多目标优化问题——在大小为n的sgRNA候选集中选择覆盖m个基因的最优子集，是NP难问题。

实际设计过程中使用贪心搜索与约束规划的结合：首先在每条基因的编码区（避开外显子-内含子连接区50bp以避免cas9剪接干扰）枚举所有NGG PAM的候选sgRNA，使用深度学习模型（如DeepCRISPR）预测on-target效率和脱靶活性打分，然后对各基因按得分排序并确保每条基因至少选择4个均匀分布在CDS区域内的sgRNA。最后在基因组层面叠加排除sgRNA交叉靶向（20bp同源性>11nt的位点）和避开已知SNP的掩膜（保留主要等位基因频率>0.95的sgRNA以实现泛人群的适用性）。Broad Institute的Brunello文库（Doench et al., 2016, *Nature Biotechnology*）和GeCKO文库是这一设计策略的代表性产物——Brunello的sgRNA不仅平均每个基因~4条sgRNA，且通过最小化交叉靶向和最大化3'外显子覆盖率显著提升了筛选的灵敏度和富集度。

## 6. 基于深度学习的RNA药物设计

### 6.1 RNA适配体的AI设计

RNA适配体（aptamer）是通过SELEX体外筛选获得的短RNA序列（通常30-80nt），能以高亲和力和特异性结合靶标分子（从小分子到蛋白质乃至整个细胞）。传统SELEX需要多轮体外筛选（6-12轮），耗时数周至数月。人工智能通过在序列空间中实施快速探索极大加速了适配体发现速度。

**使用GNN预测RNA适配体与小分子的结合亲和力**——当前的AI适配体设计流程将RNA适配体的二级结构表示为一个图（图中节点为残基/碱基对，边缘为碱基堆积力和氢键配对关系），利用图神经网络（GNN）在拓扑结构空间中学习结合亲和力的映射关系。在训练阶段，GNN在已有SELEX筛选数据（如APTAMER数据集中的~10³个序列-靶标亲和力数据对）上学习结构的图嵌入表示。推理阶段，GNN快速排除高分区的候选序列，将候选集从理论上的8^N缩小至可实验验证的10²水平。Le et al.（2022, *ACS Synthetic Biology*）报道了使用GNN的适配体设计使达比加群靶标的适配体筛选轮次从8轮降至2轮，同时维持了纳摩尔级别的Kd。

**AptaTRACE**（Ahmed et al., 2024, *Bioinformatics*）采用随机森林（RF）方法从序列特征直接预测适配体结合能力。其输入特征空间包括：GC含量、K-mer频率（3-5mer）、最小自由能（MFE）和配对百分比等序列结构特征。RF输出的重要性排序可以揭示对结合亲和力贡献最大的序列模式——对于某些小分子靶标（如ATP和茶碱），3'端环的结构自由度被RF识别为最重要的预测因子，提示这些靶标适配体的结合实质依赖环区的构象变化诱导适配。

### 6.2 siRNA和ASO的AI设计

**siRNA沉默效率的Transformer预测**——siRNA是基于RNA干扰（RNAi）机制的治疗性分子，其设计要素包括：siRNA正义链和反义链的长度通常21nt、23nt，GC含量30-52%之间的优化区间、过高的GC升高热稳定性但降低RISC卸载动力学、反义链5'端的A或U和低Tm的种子区偏好（thermodynamic asymmetry原则）。传统规则设计方法（如Reynolds规则）使用经验加权模型对上述特征进行线性组合，但预测准确度限制在R²~0.3-0.5（Reynolds et al., 2004, *Nature Biotechnology*）。

深度学习方法改变了这一现状——Transformer架构可直接从序列到细胞内沉默效率进行端到端映射。输入的siRNA双链序列使用位置编码+RNA核苷酸的6个理化属性编码（大小、极性、可变性、可溶性等因素）。Transformer的多头自注意力机制学习序列各个位置与沉默活性之间的长程相关性（捕获了传统模型无法建模的非局部高阶交互——如正义链3'端的核苷酸与反义链种子区之间的配对协同效应）。在预训练数据集（来自大规模siRNA筛选数据矩阵，包含~10⁵序列活性对）上预测R²达到0.65-0.78。在siRNA的脱靶效应预测上，深度学习模型通过miRNA样脱靶路径的种子区匹配分析反义链在3'UTR的序列互补性来实现对siRNA在转录组上的脱靶效应评估（Anderson et al., 2024, *Nature Reviews Drug Discovery*）。

**ASO的AI设计**——反义寡核苷酸（Antisense Oligonucleotides）通过与靶mRNA互补配对调控基因表达。RNase H介导的ASO（如Mipomersen）要求ASO与靶RNA形成DNA-RNA双链，招募RNase H1切割靶RNA；而steric block ASO仅通过空间位阻抑制翻译或剪接。AI设计的关键是从大规模ASO筛选数据（Ionis的多轮、多靶标ASO活性的专有数据库中~10⁵序列数据）学习序列特征与RNase H介导的抑制效率之间的模式映射。深度学习模型揭示的模式包括：对于RNase H的ASO，gapmer的中间DNA段的最优长度（通常8-10nt——此段的长度太短无法有效募集RNase H，太长则增加脱靶结合概率）和该段中C/T对A/G的比例与RNase H的催化滞留动力学相关。此外，ASO的化学修饰（PS骨架的立体化学、糖环的2'-MOE/LNA约束是否降低RNase H活性）的结构-活性关系也被纳入Transformer学习框架（Hagedorn et al., 2022, *Nucleic Acids Research*）。

## 7. RNA组学与AI的融合

### 7.1 RNA m6A修饰的AI预测

N6-甲基腺苷（m6A）是真核生物mRNA上最丰富的内部修饰，影响mRNA的剪接、转运、稳定性和翻译。从RNA序列预测m6A修饰位点的计算问题可以形式化为一个序列二分类任务（给定32-101nt的RNA序列片段，预测中心A碱基是否发生m6A修饰）。

**WHISTLE**（Chen et al., 2019, *Nucleic Acids Research*）将m6A预测从纯序列分析扩展到整合多组学特征——除了序列的k-mer频率和位置特异性核苷酸偏好外，还整合了基于RNA-seq读数的基因表达水平以及基于CLIP-seq数据的RBP结合预测评分，作为额外的输入特征。WHISTLE在独立测试集上的AUC达到0.955（比仅用序列的基线m6A工具高约10个百分点）。

**m6A-LSTM**（Rehman et al., 2020, *Analytical Biochemistry*）使用双向LSTM从RNA的序列编码（使用了扩展的核苷酸编码——将A/G/C/U/A²⁺的修饰图谱也纳入表示空间）中提取上下文特征，以捕获甲基化状态的远端序列依赖性。其隐藏层尺寸200，两层堆叠，在全连接层后输出m6A概率。m6A-LSTM的AUC=0.94（在RMBase数据库上）和AUPR=0.91。

**snnM6A**（Zhang et al., 2022, *Briefings in Bioinformatics*）引入SNN（自归一化神经网络）——通过使用scaled exponential linear units (SELU)而非传统ReLU激活函数使网络在深层时能自动维持0均值和单位方差，解决了深层网络在前向传播中的梯度消失问题。其结构包含6个隐藏层（每层2,048个神经元），使得网络容量扩大到足以捕捉跨物种m6A位点的共性模式。

**A-to-I编辑位点的AI预测**——ADAR介导的A到I编辑位点预测也采用类似框架。关键差异在于特征空间引入Alu重复序列的特征附近（因Alu的倒转重复形成的RNA双链区是ADAR的主要底物）、序列的RNA二级结构的双链倾向评分和编辑位点的邻近序列保守性打分。

### 7.2 RNA亚细胞定位的深度学习预测

RNA的亚细胞定位决定了其在细胞中的功能环境（如mRNA在轴突树突定位对于神经元极性建立的关键性、lncRNA在核仁或核小斑的定位与rRNA加工或剪接体组装的功能相关性）。预测RNA的亚细胞定位是一个多标签分类任务（一个RNA分子可能存在于多个区室——如细胞质、细胞核、内质网等）。

**RNATracker**（Yan et al., 2019, *npj Systems Biology and Applications*）提出了一种CNN架构用于RNA的亚细胞定位预测。输入编码为：序列的k-mer嵌入（k=3的64维嵌入向量），经两层卷积层（filters=128/256,kernel size=8/4）和最大池化层提取特征，后接全连接层输出10类定位标签的多标签置信度。RNATracker在Human Protein Atlas的RNA assay数据集上准确率达92%（macro AUC=0.95）。

**deepLocRNA**（Aoki et al., 2022, *Bioinformatics*）引入CNN-Transformer混合架构——CNN模块提取局部序列模式，Transformer模块通过自注意力捕获序列长程相互依赖关系和定位信号（RNA核定位信号zipcode的序列上下文依赖性）。deepLocRNA将亚细胞定位的标签空间从5类扩展到11类（还包括了外泌体、线粒体和核糖体三个新增区室），在涵盖~5万条人类转录本的测试集上的微观F1=0.78。

定位泛化挑战：不同组织类型和发育阶段的定位模式差异十分显著。在细胞过程中RNA定位状态是动态改变的（如神经刺激下mRNA从胞体向树突的转运）。当前模型在这些动态情境下精度显著下降（deepLocRNA在神经元分化前后测试集上F1从0.78降至0.67），提示有必要在训练数据中包含时间梯度的动态定位数据。

### 7.3 2026年RNA计算面临的挑战与机遇

**mRNA疫苗的AI序列设计与免疫原性预测**——下一代mRNA疫苗（2025-2026年的RSV和流感通用疫苗的临床III期）的序列设计正在从密码子优化转向集体优化：同时优化UTR折叠、编码区的RSCU（相对同义密码子使用频率）偏好与tRNA池的匹配效率以及5'端结构对TLR7/8先天免疫信号的激活。AI序列生成模型（基于扩散模型的RNA序列生成）从2024年进入实用阶段——在翻译效率上在报告基因检测中的mRNA表达量提高了2-4倍（Leppek et al., 2022, *Nature Biotechnology*）。

**circRNA的翻译能力预测**——环状RNA（circRNA）在2024-2026年进入了治疗性开发（例如circRNA编码的融合抗原蛋白疫苗和circRNA表达治疗性抗体）的高速增长期。circRNA的IRES（内部核糖体进入位点）的选择直接及翻译起始效率。AI模型预测circRNA翻译效率的挑战在于读取RNA环状拓扑结构空间——线性结构预测框架无法直接处理环形RNA的约束（因为环化消除了端部效应，且在IRES起始的翻译中，二级结构的构象空间变化巨大）。CircDeep（Wang et al., 2025, *Cell Reports Methods*）通过将环状序列的若干线性化剪切策略（在IRES位点前后切割）转换回线性序列预测后组合打分，初步实现了IRES引导的circRNA翻译效率预测（Pearson r=0.72）。

**RNA基础模型的最新进展**——RNA-FM（Chen et al., 2022, *bioRxiv*）在23M非编码RNA序列上预训练的掩码语言模型（MLM）——使用BERT风格的训练范式，可以学习到高度保守的RNA结构语义（RNA的结构motif对应的氨基酸序列基序）在迁移到RNA二级结构预测、RNA-RNA相互作用预测和RNA-蛋白质交互位点预测任务时表现出超过传统方法的优越性。RINALMO（Penzar et al., 2024, *Nature Machine Intelligence*）使用自监督对比学习在RNA序列的潜空间中显式对齐同源RNA的保守结构区域——因此显著提高了对于具有低序列保守性但高结构相似性的RNA家族的结构预测能力。

**整合展望**：RNA计算的未来十年将由基础模型与实验高通量数据的闭环驱动。AlphaFold3为代表的三级结构预测、RNA-FM为代表的基础序列模型、DeepCRISPR为代表的功能预测模型正在融合为统一的RNA智能设计框架——输入一个靶标和功能需求，输出最优的RNA序列和结构的编程式系统。这一方向被称作"RNA programming"或"RNA design automation"（2025年的DARPA RNA工程化项目）。但挑战仍然严峻：RNA的构象动力学（与蛋白质不同，RNA的构象系综更加分散、亚稳态多且能量障碍较低）使单个静态结构预测不足以满足药物设计和功能改造的基本需求，需要在生成式AI框架下实现RNA构象系综的端到端预测——这将是2026-2028年RNA计算的主要突破口。

---

**参考文献**

1. Giegé, R., Jühling, F., Pütz, J., et al. (2012). Structure of transfer RNAs: similarity and variability. *Nucleic Acids Research*, 40(10): 4284-4297.
2. Quinn, J.J. & Chang, H.Y. (2016). Unique features of long non-coding RNA biogenesis and function. *Nature Reviews Genetics*, 17(1): 47-62.
3. Haniff, H.S., Knerr, L., Chen, J.L., et al. (2020). Targeted degradation of RNA with ribonuclease-targeting chimeras (RiboTACs). *ACS Central Science*, 6(4): 593-605.
4. Singh, S., Singh, A., Bhattacharyya, D., et al. (2021). SPOT-RNA: RNA secondary structure prediction using deep learning with evolutionary information. *Nature Communications*, 12: 1337.
5. Fu, L., Cao, Y., Wu, J., et al. (2022). UFold: fast and accurate RNA secondary structure prediction with deep learning. *Nature Computational Science*, 2: 348-358.
6. Abramson, J., Adler, J., Dunger, J., et al. (2024). Accurate structure prediction of biomolecular interactions with AlphaFold 3. *Nature*, 630: 493-500.
7. Krishna, R., Wang, J., Ahern, W., et al. (2024). Generalized biomolecular modeling and design with RoseTTAFold All-Atom. *Science*, 384(6693): eadl2528.
8. Jinek, M., Chylinski, K., Fonfara, I., et al. (2012). A programmable dual-RNA-guided DNA endonuclease in adaptive bacterial immunity. *Science*, 337(6096): 816-821.
9. Chuai, G., Ma, H., Yan, J., et al. (2018). DeepCRISPR: optimized CRISPR guide RNA design by deep learning. *Genome Biology*, 19: 80.
10. Hsu, P.D., Scott, D.A., Weinstein, J.A., et al. (2013). DNA targeting specificity of RNA-guided Cas9 nucleases. *Nature Biotechnology*, 31(9): 827-832.
11. Lyu, P., Esklocak, U., Kim, H., et al. (2022). CrisprHAL: a deep learning framework for predicting off-target effects of CRISPR-Cas9. *Nature Machine Intelligence*, 4: 1084-1096.
12. Kim, N., Kim, H.K., Lee, S., et al. (2019). DeepSpCas9: deep learning for predicting off-target activities of CRISPR-Cas9 with expanded target recognition. *Nature Communications*, 10: 5527.
13. Chen, W., Feng, P., Ding, H., et al. (2019). WHISTLE: a high-accuracy map of the N6-methyladenosine (m6A) epitranscriptome. *Nucleic Acids Research*, 47(8): e47.
14. Yan, Z., Le, C., Cao, Y., et al. (2019). RNATracker: a deep learning approach for RNA subcellular localization prediction. *npj Systems Biology and Applications*, 5: 11.
15. Penzar, D., Procházka, V., Antal, L., et al. (2024). RINALMO: RNA foundation model for structure and function prediction. *Nature Machine Intelligence*, 6: 112-126.

---

*本文数据与文献检索截止至2026年6月。RNA计算领域的快速发展提示上述预测模型和性能指标在新版本发布后可能存在更新，建议读者结合最新文献和应用场景验证具体模型的实用表现。*
