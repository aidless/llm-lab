# LLM应用架构：向量数据库与检索增强生成系统

## 1. RAG的动机与基本框架

### 1.1 静态知识的根本困境

大语言模型（LLM）的本质是一个在固定数据集上训练的参数化系统。预训练完成后，模型的所有"知识"被冻结在数十亿至数千亿个权重矩阵中——这就是所谓**参数知识（Parametric Knowledge）**。这种知识存储方式存在一个根本性的缺陷：**知识截止日期（Knowledge Cutoff）**。当用户询问2023年之后的事件（如"2024年诺贝尔文学奖得主是谁"），一个训练数据截止于2023年底的模型只能给出错误的猜测或直接拒绝回答。

更为深刻的问题是：即便不计代价地重新训练或微调模型以注入新知识，参数更新的代价也极为高昂。GPT-4级别模型的一次全量训练成本高达数千万美元，而频繁更新还面临灾难性遗忘（Catastrophic Forgetting）的风险——新知识的注入会不可逆地扰动已经存储的旧知识。这种**静态权重与动态世界之间的矛盾**，构成了RAG诞生的核心理论动因[1]。

### 1.2 RAG的三阶段形式化定义

Lewis等人（2020）在经典的RAG论文中首次将检索增强生成形式化为一个端到端的概率框架[2]。设有一个预训练的序列到序列（Seq2Seq）模型 $p_{\theta}(y|x)$，其中 $x$ 为输入，$y$ 为输出。RAG引入了一个非参数化的检索索引 $\mathcal{Z}$（由文档 $z$ 构成的向量知识库），将生成过程重新定义为：

$$p_{\text{RAG}}(y|x) = \sum_{z \in \text{top-}k(p_{\eta}(\cdot|x))} p_{\eta}(z|x) \cdot p_{\theta}(y|x, z)$$

其中 $p_{\eta}(z|x)$ 是检索器（Retriever），$p_{\theta}(y|x, z)$ 是生成器（Generator）。这一定义揭示了RAG的三个核心阶段：

**阶段一：Indexing（索引构建）**。原始文档集合 $\mathcal{D} = \{d_1, d_2, ..., d_m\}$ 首先被分割成若干语义完整的块（Chunk）$\{c_1, c_2, ..., c_n\}$。每个块通过嵌入模型 $f_{\text{enc}}: \mathbb{R}^{|c_i|} \to \mathbb{R}^d$ 映射为 $d$ 维稠密向量 $v_i = f_{\text{enc}}(c_i)$。所有向量被存储到向量数据库中，并建立高效的近似最近邻搜索索引。索引过程是离线的、一次性的，但可以随文档更新进行增量维护。

**阶段二：Retrieval（检索）**。给定用户查询 $q$，系统首先计算查询向量 $v_q = f_{\text{enc}}(q)$，然后在向量索引中搜索与 $v_q$ 最相似的 Top-$K$ 个文档块。相似度度量通常采用**内积（Inner Product）** $\langle v_q, v_i \rangle$、**L2欧氏距离** $\|v_q - v_i\|_2$ 或**余弦相似度** $\cos(v_q, v_i) = \frac{v_q \cdot v_i}{\|v_q\|\|v_i\|}$。实际系统中，余弦相似度由于具有归一化的稳定范围 $[-1, 1]$ 而成为最广泛采用的度量。检索结果集 $\mathcal{R} = \{r_1, r_2, ..., r_K\}$ 构成了生成阶段的上下文。

**阶段三：Generation（生成）**。检索到的文档块与原始查询拼接构造为增强后的Prompt $x' = \text{concat}(q, r_1, r_2, ..., r_K)$。生成器 $p_{\theta}(y|x')$ 利用这个包含外部知识的上下文来产生最终回答。此阶段的关键在于Prompt模板的设计：检索文档的排列顺序、长度控制、以及如何标记检索和非检索内容。

### 1.3 RAG的信息流框架

从系统架构视角，RAG的完整信息流可表述为六级流水线：

$$
\text{Query} \xrightarrow{(1)} \text{Embed} \xrightarrow{(2)} \text{Retrieve} \xrightarrow{(3)} \text{Rank} \xrightarrow{(4)} \text{Augment} \xrightarrow{(5)} \text{Generate}
$$

其中Rank步骤（重排序）是可选的增强环节。这六个步骤构成了一个完整的**检索增强生成系统流（Retrieval-Augmented Generation Pipeline）**，其核心思想是将LLM从一个封闭的静态知识系统扩展为一个开放的、可实时访问外部知识库的动态系统。正如Gao等人（2023）在其综合性RAG综述中所指出的，RAG本质上是在解决LLM的**幻觉问题（Hallucination）**和**知识时效性问题（Knowledge Temporal Grounding）**——前者通过提供事实上下文约束生成，后者允许系统在推理时访问更新后的知识库[3]。

## 2. 向量嵌入与索引结构

### 2.1 嵌入模型的选择与权衡

嵌入模型是将文本映射到语义空间的核心组件。当前主流的嵌入模型主要分为两条技术路线：

**开源Sentence-BERT系列**。基于Transformer的双编码器架构，通过Siamese或Triplet网络进行对比学习训练。代表性模型包括 `multi-qa-MiniLM-L6-cos-v1`（384维）、`all-MiniLM-L6-v2`（384维）和 `all-mpnet-base-v2`（768维）。这些模型的优势在于低成本的自部署、无API延迟、数据不外传，且针对特定领域可以进行微调。Reimers与Gurevych（2019）在原始Sentence-BERT论文中证明，这种架构在保持BERT级语义理解能力的同时，将相似度搜索的速度提升了数万倍[4]。

**闭源商业嵌入API**。OpenAI的 `text-embedding-3-large` 提供1536维和3072维两种规格，在MTEB（Massive Text Embedding Benchmark）上取得了当时最优的成绩。Cohere的Embed v3系列和Voyage AI的嵌入API也提供了针对特定语言和领域优化的嵌入模型。商业API的优势在于持续更新的模型质量和多语言能力，但代价是每百万tokens的调用成本、额外的网络延迟（通常15-50ms/次调用），以及对数据隐私控制的减弱。

嵌入维度的选择是一个关键的**质量-延迟权衡**。更高维度（如3072维）理论上能捕获更精细的语义差异，但同时也带来三个问题：(1) 存储成本与维度呈线性增长；(2) 高维空间中的距离度量趋于均匀化（"维度灾难"的一部分表现）；(3) ANN搜索的计算延迟与维度呈近似线性关系。Mussmann等人（2017）的理论研究表明，在极高维度上，任何距离度量都会失去区分能力[5]。实践经验表明，对于大多数RAG应用，768维的嵌入已经能够提供足够的语义分辨能力。

### 2.2 索引结构的选择与原理

向量索引决定了检索的速度与精度。三种主要的索引结构值得深入理解：

**FLAT（暴力全库搜索）**。这是最原始的搜索方式：对查询向量与库中的每一个向量计算距离，返回最相似的K个结果。其数学复杂度为 $O(N \cdot d)$，其中 $N$ 是向量总数，$d$ 是维度。FLAT的精度和召回率均为100%（因为执行了精确的KNN搜索），但当 $N$ 超过百万量级时，单次查询的延迟会达到不可接受的水平（数十毫秒至数百毫秒）。FLAT适用于小规模知识库（通常 $N < 10^4$）或对精度有严格要求的场景。

**IVF（倒排文件索引，Inverted File Index）**。IVF通过**K-means聚类**将向量空间划分为 $nlist$ 个Voronoi单元，每个单元以质心 $c_j$ 表示。搜索时，首先计算查询向量 $q$ 到各质心的距离，只搜索最近的 $nprobe$ 个单元内的向量。设每个单元平均包含 $N / nlist$ 个向量，则搜索复杂度降至 $O(nlist \cdot d + nprobe \cdot (N/nlist) \cdot d)$。当 $nprobe \ll nlist \ll N$ 时，加速比可达100倍以上。IVF的召回率通过调节 $nprobe$ 参数控制——$nprobe$ 越大，搜索的范围越广，召回越接近精确搜索。实践中通常用Recall@1作为IVF精度与速度平衡的指标。

**HNSW（Hierarchical Navigable Small World，分层可导航小世界图）**。HNSW由Malkov与Yashunin（2020）提出，是目前最流行的ANN索引算法[6]。其核心思想是构建一个多层的图结构：底层（Layer 0）包含所有向量结点，随着层数升高，结点密度呈指数衰减（通常通过 $1/\ln(M)$ 的概率决定每个结点的最高层）。搜索过程从顶层开始，在每层执行**贪婪搜索（Greedy Search）**：从入口结点出发，不断移动到当前最近邻结点的最近邻，直到无法更近；然后将该层找到的最相似结点作为下一层的入口，继续向下搜索。

HNSW的建图参数包括 $M$（每个结点的最大连接数，控制图密度和内存）和 $efConstruction$（建图时的动态候选列表大小，控制建图质量）。搜索参数 $ef$ 控制搜索时的候选列表大小，$ef$ 越大搜索精度越高但延迟也越大。HNSW的关键优势在于其搜索复杂度仅为 $O(\log N)$ 级别，在百万至千万级数据上可达到亚毫秒级延迟，同时保持95%以上的Recall@10。

在实际系统中，**IVF+HNSW的混合索引**策略也常见：先用IVF定位到少数候选单元，再在候选集内部使用HNSW图进行高效搜索。Milvus等向量数据库提供了这种多级索引的组合支持。

## 3. 文档分块技术

### 3.1 固定大小分块

固定大小分块（Fixed-size Chunking）是最直接的分块策略：将文档按照固定的token数量切割成连续的片段。实现上，使用分词器（Tokenizer，如 `cl100k_base`、`SentencePiece` 或 `tiktoken`）对整个文档进行编码，然后按固定步长 $S$ 分割token序列。

关键设计参数是**分块大小（Chunk Size）$S$** 和**重叠步长（Chunk Overlap）$O$**。重叠机制的核心动机是：当文档边界恰好切断了某个语义完整的句子或段落时，重叠区域可以作为"缓冲区"，确保关键信息不会因边界而丢失。设总token数为 $T$，则生成的块数为 $\lceil (T - O) / (S - O) \rceil$。

重叠参数的选择至关重要。若 $O$ 过小（如 $O < 10\% S$），则边界信息的暴露量不足；若 $O$ 过大（如 $O > 50\% S$），则冗余信息膨胀、存储成本和检索混淆度显著增加。标准推荐值为 $O = 10\% - 20\% S$，例如对 $S=512$ 的分块设置 $O=50$ 至 $O=100$ tokens的重叠。

固定分块的优点是简单、可预测、处理速度快；缺点是它会无视自然语义边界——一个完整的句子可能被切成两半，分别落入相邻的两个块中。

### 3.2 语义分块

语义分块（Semantic Chunking）试图解决固定分块破坏语义完整性的问题。其基本思想是：通过检测文档中语义密度的变化点来确定分块边界。

一种实现方式是**基于嵌入相似度的分割**：将文档逐句（或逐段）编码为向量，计算相邻句子对 $(\text{sent}_i, \text{sent}_{i+1})$ 之间的余弦相似度，标记相似度显著下降的位置（即语义"跳变点"）作为候选切分点。设 $\delta_i = \cos(v_i, v_{i+1})$，当 $\delta_i < \mu - \alpha \sigma$（其中 $\mu$ 和 $\sigma$ 是所有 $\delta$ 的均值和标准差，$\alpha$ 是敏感度超参数）时，认为该位置发生了主题或语义转折。

另一种方法利用**TextTiling算法**（Hearst，1997），一种无需训练的无监督文本分割算法[7]。TextTiling通过计算相邻句子块之间的词汇重复模式（词频分布的余弦相似度）来检测主题转换点。该算法假定：同一主题内的用语有较高的词汇重复率，而主题转换处词汇会显著变化。

近年来，利用LLM直接识别主题边界的策略也开始出现——向模型提供文本片段并询问"此处是否应作为段落分割点"。这种方法的优势在于可以利用LLM对深层语义结构的理解，但代价是额外的推理成本和延迟。

### 3.3 基于文档结构的分块

结构化文档（如Markdown、HTML、LaTeX）具有天然的语义分块线索。**基于文档结构的分块（Document Structure-based Chunking）**利用这些线索来保证分块的语义完整性：

- 对Markdown文档，根据 `#` / `##` / `###` 等标题层级划分章节；对HTML文档，根据 `<h1>` - `<h6>`、`<section>`、`<article>` 等标签分割。
- 每个章节作为一个独立的语义块，保持段落结构完整。
- 对于过长的章节，可以在其内部继续使用固定大小或语义分块，但保证不会跨标题切割。

这种方式特别适合结构化知识库（如维基百科、技术文档、说明书）的索引构建，因为它的分块天然对齐了人类写作的逻辑单元。

### 3.4 代码分块

对于代码知识库，分块需要尊重编程语言的语法结构。LangChain的 `RecursiveCharacterTextSplitter` 提供了按语言分块的能力，其底层实现利用了编程语言的抽象语法树（AST）规则：

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter
# Python代码分块：优先按函数/类定义分割，再按行分割
splitter = RecursiveCharacterTextSplitter.from_language(
    language="python", chunk_size=500, chunk_overlap=50
)
```

这种分块策略确保每个代码块在语法上完整——不会出现函数定义被切断、括号不匹配等导致语法错误的情况。对支持的语言（Python、JavaScript、Go、Rust等），LangChain会使用语言特定的分隔符列表（如 `\nclass `、`\ndef `、`\n\t` 等），按从高到低的优先级尝试分割。

### 3.5 分块策略的评估

选择最优分块策略需要定量评估。核心指标分为两类：

**检索质量指标**：Recall@K（检索到的相关文档占全部相关文档的比例）、MRR（Mean Reciprocal Rank，第一个相关文档排名的倒数均值）。这些指标衡量的是分块策略是否将相关信息放在了可检索的位置。

**生成质量指标**：Faithfulness（忠实度，生成的回答是否完全基于检索到的文档内容，无幻觉信息）、Answer Relevancy（答案相关性，回答是否针对用户的问题而非提供无关信息）。

这些指标的关系是：分块过大（如超过LLM的上下文窗口）会导致关键信息被截断或压缩；分块过小则导致上下文不完整，使LLM缺乏理解所需的背景信息。经验表明，对于大多数问答场景，256-512 tokens的分块大小能取得最优平衡[3]。

## 4. 检索后处理

### 4.1 重排序（Reranking）

向量检索的第一阶段（通常使用双编码器 Bi-Encoder）本质上是一个**近似检索**：它有效地将候选集从整个知识库缩小到Top-$K'$（如 $K'=100$），但由于双编码器分别编码query和文档，它们之间的交互是间接的——只有最后一步的向量点积。这种架构无法捕捉query与文档之间的细粒度交互信息。

**Cross-Encoder重排序器**解决了这一问题：它将query和文档拼接为一个序列 $[\text{CLS}]\ q \ [\text{SEP}]\ d \ [\text{SEP}]$，输入一个编码器-分类器架构，输出一个相关性分数 $s(q, d) \in [0, 1]$。由于Cross-Encoder允许query与文档的所有token进行直接注意力交互，它能更准确地判断相关性。Cohere的Rerank API和开源的 `cross-encoder/ms-marco-MiniLM-L-6-v2` 都是这一范式的代表。

重排序的代价是它需要为每个候选对执行一次完整的Transformer前向传播——复杂度为 $O(K' \cdot (L_q + L_d) \cdot d)$，其中 $L_q$ 和 $L_d$ 是query和文档的长度。因此，重排序器通常只对Bi-Encoder检索出的Top-100候选进行操作，输出排序后的Top-$K$（如 $K=5$ 或 $K=10$）。

Nogueira与Cho（2019）在原始Cross-Encoder重排序论文中证明，仅需对Top-1000候选进行重排序，即可在MS MARCO数据集上将MRR@10从约0.27提升至0.36以上[8]。在RAG场景下，重排序通常将检索的Precision@K提升15%-30%。

### 4.2 互惠排名融合

当系统使用多种检索策略（如稠密向量检索+BM25稀疏检索）时，需要一种机制将多个排序结果合并为一个统一的列表。**互惠排名融合（Reciprocal Rank Fusion, RRF）** 是一种简单而有效的融合方法[9]。

设共有 $m$ 种独立的检索策略，对文档 $d$，每种策略 $i$ 给出一个排序位置 $r_i(d)$（若未检索到则设 $r_i(d) = \infty$）。RRF融合后的分数为：

$$S(d) = \sum_{i=1}^{m} \frac{1}{k + r_i(d)}$$

其中 $k$ 是一个平滑常数（通常设为60，用以控制排序靠前文档的权重优势）。最终的文档列表按 $S(d)$ 降序排列。

RRF的优势在于：(1) 它无需训练，无需调整权重；(2) 它对不同策略的打分尺度无关——只需排名位置；(3) 它能有效利用不同策略的互补性——稠密检索擅长捕捉语义相似但词汇不重叠的匹配，而BM25擅长精确词汇匹配。

**混合检索（Hybrid Search）** 策略，即Dense + Sparse检索的RRF融合，已成为生产级RAG系统的标准实践。Karpukhin等人（2020）在DPR论文中展示了混合检索相对于纯稠密或纯稀疏检索的显著优势[10]。

### 4.3 最大边际相关性重排

**最大边际相关性（Maximum Marginal Relevance, MMR）** 由Carbonell与Goldstein（1998）提出，用于在信息检索中平衡相关性与多样性[11]。其核心思想是：在已经选择了 $i-1$ 个文档的情况下，选择第 $i$ 个文档时，不仅要考虑该文档与query的相关性，还要考虑该文档与已选文档集合的相似性（即冗余度）。

MMR的数学形式为：
$$\text{MMR} = \arg\max_{d \in \mathcal{R} \setminus \mathcal{S}} \left[ \lambda \cdot \text{Sim}(d, q) - (1 - \lambda) \cdot \max_{s \in \mathcal{S}} \text{Sim}(d, s) \right]$$

其中 $\mathcal{R}$ 是候选文档集合，$\mathcal{S}$ 是已选文档集合，$\lambda \in [0, 1]$ 是相关性与多样性之间的权衡参数。当 $\lambda = 1$ 时，MMR退化为纯相关性排序；当 $\lambda = 0$ 时，只考虑多样性，完全忽略相关性。

MMR在RAG中的价值在于：如果Top-$K$ 检索结果中包含多个语义相似的文档（例如来自同一网页的连续段落），直接将它们全部注入prompt会导致信息冗余——LLM从多个相似来源中很难获得新的、互补的信息。MMR通过消除冗余，确保注入prompt的K个文档在相关性得到保证的前提下尽可能多样，从而提供给LLM更丰富的信息视角。

## 5. 高级RAG模式

### 5.1 Self-RAG：自省式检索增强

传统的RAG框架中，检索是固定的：无论查询是否真正需要外部知识，系统总是检索Top-$K$文档并将它们注入prompt。这导致了一个根本性问题：当LLM本身已经足够知识来回答时（例如"请解释牛顿第二定律"），不必要的检索不仅增加了延迟和token消耗，还可能引入与生成内容矛盾的、噪声性的检索结果。

Asai等人（2023）提出的**Self-RAG**通过引入特殊的**Reflection Token**机制解决了这一问题[12]。Self-RAG的核心创新是训练LLM输出两类反射标记：

1. **检索标记（Retrieval Token）**：在每个生成段（Segment）之前，模型输出一个三元决策——{`Retrieve`, `NoRetrieve`, `Continue`}。若模型判断当前段需要外部知识支撑（例如模型不确定的事实陈述），则触发检索；若判断不需要（例如常识性或创意性内容），则跳过检索。

2. **相关性标记（IsRel Token）**：当检索触发后，模型对检索到的每个passage评判其与当前生成段的相关性，输出 {`Relevant`, `Irrelevant`} 二分类评分，过滤不相关的检索结果。

3. **支持标记（IsSup Token）**：生成段中每个事实陈述后，模型评判该段是否被检索到的文档所支持，输出 {`Fully supported`, `Partially supported`, `No support`}。

4. **实用度标记（IsUse Token）**：对整个生成段，模型自评其对回答用户问题的有用度。

这些Reflection Token的引入使Self-RAG具备了自我监控和自我修正的能力。在多个开放域问答和事实验证数据集上，Self-RAG显著优于标准RAG，在保持生成质量的同时将幻觉率降低了20%-30%。

### 5.2 Graph RAG：基于知识图谱的全局检索

标准RAG基于向量相似度进行检索，在回答全局性、跨文档的综合问题时往往力不从心。例如，"公司的研发部门在2024年发布了哪些涉及AI安全的产品？"——这类问题需要跨越多个文档、整合分散信息，而向量检索倾向于返回与query在语义上最接近的单个文档片段，缺乏全局综合能力。

Microsoft Research（2024）提出的**Graph RAG**通过引入知识图谱来克服这一局限[13]。其工作流程分为两个阶段：

**索引阶段**：(1) 将文档分割为文本块；(2) 使用LLM从每个块中提取实体（如人物、公司、产品名）和关系（如"发布"、"收购"、"隶属于"）；(3) 构建知识图谱 $\mathcal{G} = (\mathcal{V}, \mathcal{E}, \mathcal{T})$，其中 $\mathcal{V}$ 是实体节点集合，$\mathcal{E}$ 是关系边集合，$\mathcal{T}$ 是文本属性（上下文描述）；(4) 对图谱进行社区检测（Community Detection，如Leiden算法），生成从叶子社区到根社区的层次化摘要。

**查询阶段**：(1) 用户query被映射到知识图谱中的相关实体和关系；(2) 从相关实体出发，在图谱中执行宽度优先遍历，收集关联的社区摘要；(3) 将这些社区摘要作为增强上下文注入LLM生成最终回答。

Graph RAG的一个关键概念是**社区摘要的分层结构**：叶级别的社区摘要包含特定话题的详细信息（如"2024年Q1发布的AI安全工具"），而高层次社区摘要则提供跨主题的综述（如"公司2024年的AI产品布局"）。在问答时，系统可以同时利用多个层级的摘要来提供既有广度又有深度的回答。

在Microsoft发布的基准评测中，Graph RAG在跨文档多跳问答任务上的准确率比朴素RAG高出30%-60%，特别是在需要全局理解的"总结性"问题上表现尤为突出。

### 5.3 Agent集成RAG：自主检索代理

传统RAG中，检索是一个"一次性的"、与对话上下文解耦的步骤。但在复杂的多轮对话中，用户的真实信息需求往往是逐渐明确的，检索策略也需要动态调整。

**Agent集成RAG（Agentic RAG）** 将LLM的函数调用（Function Calling）能力与RAG流程结合，使模型能够自主决定检索的时机、内容和方式。其核心能力包括：

1. **意图驱动的检索决策**：Agent根据对话状态决定是否需要检索。例如，若用户说"继续刚才的话题"，Agent需要理解"刚才的话题"是什么，然后针对性地检索。

2. **查询重写（Query Rewriting）**：原始用户query可能不适合直接作为向量检索的输入。例如，用户问"他们的CEO是谁"——"他们"的指代需要在前文解析。Agent通过LLM将query重写为自包含的检索表达式："苹果公司2024年的CEO是谁"。

3. **多步检索（Iterative Retrieval）**：Agent在一次回答中进行多次检索，每次的query依赖于前一次检索的结果。这种"思考-检索-再思考"的循环使系统能够处理复杂的多跳问题。

Yao等人（2022）提出的ReAct范式（Reasoning + Acting）为Agentic RAG提供了理论基础[14]——LLM在推理轨迹（Reasoning Trace）和行动（Action，即检索函数调用）之间交替，形成了一个可解释的推理-检索循环。

## 6. RAG系统评估与优化

### 6.1 三维评估框架

RAG系统的评估需要涵盖三个独立维度，每个维度衡量系统的不同方面：

**维度一：Context Relevancy（上下文相关性）**。衡量检索到的文档与用户query之间的相关程度。理想情况下，检索到的每个文档都应该完全相关，且不包含噪声信息。设检索到的文档集合为 $\mathcal{R}$，人工标注的相关文档集合为 $\mathcal{R}^*$，则Context Precision（CP）为 $|\mathcal{R} \cap \mathcal{R}^*| / |\mathcal{R}|$。低CP意味着检索结果中混入了大量不相关信息，这些噪声会干扰生成阶段。

**维度二：Answer Faithfulness（答案忠实度）**。衡量生成回答是否严格基于检索到的文档内容，而非模型自身的"幻觉"知识。评估方法通常是将回答拆解为一系列原子声明（Atomic Claims），逐一验证每个声明是否能从检索文档中推导出来。Faithfulness分数 = 可推导的声明数 / 总声明数。这是RAG系统最关键的评估指标——即使检索完全相关，生成器也可能产生不受检索文档约束的幻觉。

**维度三：Answer Relevance（答案相关性）**。衡量生成回答是否针对用户的实际问题。即使回答是事实正确且忠实于文档的，如果它回答的是另一个问题而非用户的问题，它在实际使用中仍然是无用的。

这三个维度构成了一个互相制约的三角：过度追求上下文相关性可能导致检索结果过于窄化，损害生成回答的全面性；过度强调忠实度可能使回答变得机械或遗漏隐含信息的推理；而过度追求相关性可能使回答遗漏关键细节。RAG系统的优化本质上是在这三维度之间寻找最优平衡。

### 6.2 黄金数据集与RAGAS

评估RAG系统需要高质量的基准数据集。Gold QA三元组 $\{(q_i, c_i, a_i)\}$ 的构建通常由人工标注完成——标注员首先提交一个query $q_i$，然后从知识库中标记出相关文档片段 $c_i$，最后撰写一个独立的参考答案 $a_i$。这个三元组构成了评估的"黄金标准"。

**RAGAS（RAG Assessment）** 是Shahul等人（2023）提出的自动评估框架[15]。RAGAS利用LLM作为评估器，避免了昂贵的人工标注，实现了大规模的自动评估。其核心评估流程包括：

- **Faithfulness评估**：将生成回答 $a$ 拆解为原子声明集合 $\{s_1, s_2, ..., s_n\}$；对每个 $s_i$，构造一个验证prompt："以下声明是否可以从给定的检索文档中推导出来？"；LLM的二元判断用于计算Faithfulness分数。
- **Answer Relevancy评估**：让LLM根据回答 $a$ 反向生成若干伪query $\{q'_1, ..., q'_m\}$，计算这些伪query与真实query $q$ 之间的嵌入余弦相似度。若伪query与真实query在语义上相近，则说明回答是相关的。

RAGAS的自动化评估虽然不如人工评估精确，但在大型迭代开发中提供了宝贵的快速反馈循环。实践经验表明，RAGAS的评分与人工评估的一致性能达到0.7-0.8的Spearman相关系数，足以支持系统的渐进式改进决策。

### 6.3 标准Benchmark

**RGB（Retrieval-Augmented Generation Benchmark，中文版）**：由清华大学和百度联合发布的中文RAG基准测试集[16]。RGB覆盖了单跳问答、多跳问答、事实验证、实体链接等任务类型，并提供了标准的评估脚本和指标计算代码。RGB对于中文RAG系统的发展具有重要的推动作用。

**KILT（Knowledge Intensive Language Tasks）**：由Facebook AI Research发布的综合知识密集型任务基准[17]。KILT涵盖了5类知识密集型任务（事实验证、开放域问答、实体链接、关系提取和对话），包含5个数据集共约4.7万条测试样本。KILT的一个关键设计是所有任务共享同一个维基百科知识源，确保检索系统和生成系统在不同任务上的可比性。

**CRUD RAG Benchmark**：由Lyu等人（2024）提出的一种根据信息需求类型对RAG任务进行分类的框架[18]。CRUD代表四种任务类型：
- **Create（创建）**：生成新内容（如写报告、编写代码）
- **Read（读取）**：回答基于已有事实的问题（如标准问答）
- **Update（更新）**：修改或补充已有信息（如编辑文档、更新知识）
- **Delete（删除）**：去除或纠正信息（如去重、删除过时内容）

不同类型的RAG任务对检索策略、分块大小、生成方式的要求截然不同。CRUD分类框架为RAG系统的差异化设计和针对性优化提供了理论基础。

## 7. 生产级RAG系统架构

### 7.1 多层架构设计

一个生产级的RAG系统通常设计为多层架构，每层解决系统中的一个关键关注点：

**数据层（Data Layer）**：数据层负责文档的摄取和预处理。这是整个RAG系统的输入起点，也是决定检索质量的基础。

文档摄取（Document Ingestion）使用专门的解析工具将各种格式的原始文档转换为可供后续处理的纯文本。主流工具包括：
- **Unstructured.io**：支持PDF、HTML、Word、PowerPoint、电子邮件等数十种格式的文档解析，提供分区（Partitioning）和清理（Cleaning）功能，能提取文档中的标题、段落、表格、列表等结构化元素。
- **Docling（IBM Research）**：专为文档AI设计的解析工具，支持PDF、DOCX、PPTX等格式，保留文档的布局结构（如多栏、表格、页眉页脚），输出为适合编码的Markdown格式。
- **PDF解析器（如PyMuPDF、pdfplumber、LlamaParse）**：专注于PDF的高效文本和表格提取。

解析后的文档进入**切分（Chunking）层**，根据第3节讨论的策略进行分块，然后发送给**嵌入（Embedding）层**进行向量编码。

对于多模态文档（包含图片、表格、图表的PDF），**多模态RAG**架构需要在嵌入阶段支持多模态的混合嵌入——将文本块、图片的视觉嵌入（通过CLIP或类似模型）、表格的结构化表示分别编码，存储在同一向量空间中，或为不同模态建立平行的索引结构。

**缓存层（Cache Layer）**：缓存层通过语义缓存（Semantic Cache）显著降低RAG系统的延迟和成本。其工作原理是：当一个新的query $q$ 进入系统时，首先将其嵌入为 $v_q$，在语义缓存中搜索是否有与 $v_q$ 高度相似的已缓存query（余弦相似度超过阈值 $\tau$，通常为0.95-0.98）。若命中，直接返回缓存中的答案，跳过整个检索-生成的流程。

语义缓存对高频重复查询（如常见FAQ、系统状态查询）效果显著。在典型的企业知识库场景中，语义缓存可以命中20%-40%的查询，将平均响应延迟从2-3秒降低到50-100毫秒，同时节省相应的LLM调用成本。缓存淘汰策略通常采用LRU（最近最少使用）结合TTL（过期时间），确保缓存的答案是时效且内存可控的。

### 7.2 RAG系统的推理增强

2025-2026年，RAG系统发展的一个重要趋势是将推理时间（Inference-time）的推理能力与检索增强相结合，形成更强大的推理-检索混合范式：

**推理分阶段RAG（Reasoning-Stage RAG）**：LLM在生成回答时，将推理过程分解为多个阶段，每个阶段根据当前推理状态决定是否需要检索。例如，在回答"2024年全球GDP增长最快的三个经济体是哪些？"时，系统首先检索2024年各国GDP数据，然后利用推理能力进行排名和分析，最后生成回答。这种"检索-推理-再检索"的迭代过程使系统能够处理需要多步推理和多次知识查询的复杂问题。

**迭代检索（Iterative Retrieval）**：与单次检索不同，迭代检索允许系统在生成过程中根据需要多次触发检索。这通常通过Agent框架实现——LLM在生成过程中维护一个"待验证"的事实列表，对每个待验证事实触发一次针对性的检索。这种方法对需要对多个独立事实进行逐一验证的场景特别有效。

**HyDE（假设文档嵌入，Hypothetical Document Embeddings）**是Gao等人（2022）提出的一种有趣的检索增强技术[19]。其核心思路是：在检索时，首先让LLM根据用户的query $q$ 生成一个假设性的伪文档 $\tilde{d}$——即"如果有一篇完美的回答文档，它应该长什么样"。然后将 $\tilde{d}$ 的嵌入作为检索query进行向量搜索。

HyDE的理论直觉是：query $q$ 本身可能处于"问句空间"而非"文档空间"，直接嵌入query进行检索存在语义空间的错配。而伪文档 $\tilde{d}$ 位于"文档空间"中，它与真实相关文档之间的嵌入距离比query与文档之间的距离更近。通过这种"query → 文档 → 嵌入"的间接路径（query2doc），HyDE在多个基准上显著提升了检索召回率。HyDE的一个关键前提是需要一个足够强大的生成器来产生逼真的伪文档——若伪文档质量不佳，反而可能引入噪声。

### 7.3 2026年RAG发展的新趋势

截至2026年，RAG系统的发展呈现以下趋势：

**RAG与Agent的深度融合**：RAG不再是一个独立的检索-生成模块，而是Agent系统中的一个"工具"。Agent根据推理轨迹决定何时调用RAG工具、调用后如何利用检索结果，以及是否需要多个工具的组合调用。

**Graph RAG的生产化**：Graph RAG正在从学术研究快速走向生产部署。微软在GraphRAG开源项目中的实践表明，通过工程优化（如LLM提取实体时的批量处理、图谱构建的并行化、查询阶段的社区摘要缓存），Graph RAG的成本已降至可接受的企业级水平。

**RAG评估的标准化**：RAG评估正在从论文中的临时基准向工业标准演进。RAGAS的不断迭代、RGB中文基准的推广、以及CRUD分类框架的采用，正在逐步形成一套完整的RAG评估体系。

**多模态RAG的成熟**：随着多模态嵌入模型（如CLIP、ImageBind）和多模态LLM（如GPT-4V、Gemini）的成熟，支持文本、图片、表格、图表、代码等多模态内容的RAG系统正在成为标准。

---

**参考文献**

[1] Borgeaud, S., Mensch, A., Hoffmann, J., et al. (2022). Improving Language Models by Retrieving from Trillions of Tokens. *ICML 2022*.

[2] Lewis, P., Perez, E., Piktus, A., et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. *NeurIPS 2020*.

[3] Gao, Y., Xiong, Y., Gao, X., et al. (2023). Retrieval-Augmented Generation for Large Language Models: A Survey. *arXiv preprint arXiv:2312.10997*.

[4] Reimers, N. & Gurevych, I. (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. *EMNLP-IJCNLP 2019*.

[5] Mussmann, S., Liang, P., & Ermon, S. (2017). On the Relationship between Data Efficiency and Error for Uncertainty Sampling. *AISTATS 2017*.

[6] Malkov, Y. A. & Yashunin, D. A. (2020). Efficient and Robust Approximate Nearest Neighbor Search Using Hierarchical Navigable Small World Graphs. *IEEE TPAMI*, 42(4), 824-836.

[7] Hearst, M. A. (1997). TextTiling: Segmenting Text into Multi-paragraph Subtopic Passages. *Computational Linguistics*, 23(1), 33-64.

[8] Nogueira, R. & Cho, K. (2019). Passage Re-ranking with BERT. *arXiv preprint arXiv:1901.04085*.

[9] Cormack, G. V., Clarke, C. L. A., & Buettcher, S. (2009). Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods. *SIGIR 2009*.

[10] Karpukhin, V., Oğuz, B., Min, S., et al. (2020). Dense Passage Retrieval for Open-Domain Question Answering. *EMNLP 2020*.

[11] Carbonell, J. & Goldstein, J. (1998). The Use of MMR, Diversity-Based Reranking for Reordering Documents and Producing Summaries. *SIGIR 1998*.

[12] Asai, A., Wu, Z., Wang, Y., et al. (2023). Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection. *ICLR 2024*.

[13] Edge, D., Trinh, H., Cheng, N., et al. (2024). From Local to Global: A Graph RAG Approach to Query-Focused Summarization. *arXiv preprint arXiv:2404.16130*.

[14] Yao, S., Zhao, J., Yu, D., et al. (2022). ReAct: Synergizing Reasoning and Acting in Language Models. *ICLR 2023*.

[15] Shahul, E. S., et al. (2023). RAGAS: Automated Evaluation of Retrieval Augmented Generation. *arXiv preprint arXiv:2309.15217*.

[16] Zhu, Y., et al. (2023). Retrieval-Augmented Generation Benchmark (RGB) for Chinese. *arXiv preprint*.

[17] Petroni, F., Piets, G., Rocktäschel, T., et al. (2020). KILT: a Benchmark for Knowledge Intensive Language Tasks. *ACL 2021*.

[18] Lyu, Y., Li, Z., Niu, Z., et al. (2024). CRUD-RAG: A Comprehensive Chinese Benchmark for Retrieval-Augmented Generation. *arXiv preprint arXiv:2401.17043*.

[19] Gao, L., Ma, X., Lin, J., & Callan, J. (2022). Precise Zero-Shot Dense Retrieval without Relevance Labels. *ACL 2023*.
