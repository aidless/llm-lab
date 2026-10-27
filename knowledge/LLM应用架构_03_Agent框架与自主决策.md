# LLM应用架构：AI Agent框架与自主决策系统

## 1. Agent的基本范式

### 1.1 感知-思考-行动-观察循环的形式化描述

AI Agent（智能体）在LLM语境下被重新定义为：以大型语言模型为推理核心，驱动自主决策循环的系统。其基本运作模型可形式化为四步闭环——**感知（Perception）→ 思考（Thinking）→ 行动（Action）→ 观察（Observation）**。

设环境状态为 $E_t$，Agent的内部状态为 $S_t$，则在时间步 $t$：

1. **感知（Perception）**：$P_t = f_{perceive}(E_t)$，Agent从环境中获取观察信息，将其转化为LLM可处理的文本表示。这包括用户输入、API返回、环境状态变更等任何形式的外部信号。

2. **思考（Thinking）**：$T_t = f_{think}(P_t \oplus S_t)$，LLM根据当前感知和内部状态（记忆、历史轨迹）生成推理链条。这是Agent的"大脑"环节——LLM通过自回归生成对当前情境的分析、下一步计划的推理，以及工具选择的理由。

3. **行动（Action）**：$A_t = f_{act}(T_t)$，将思考结果转化为可执行的动作——可以是调用外部工具（API/Function Call）、生成文本回复、或更新内部状态。

4. **观察（Observation）**：$O_t = f_{observe}(Execute(A_t), E_{t+1})$，执行动作后环境返回新的状态变化，Agent将反馈纳入下一轮循环：

$$E_{t} \xrightarrow{\text{Perception}} P_t \xrightarrow{\text{LLM Reasoning}} T_t \xrightarrow{\text{Action}} A_t \xrightarrow{\text{Execution}} O_t \xrightarrow{\text{Update}} E_{t+1}$$

这一循环在概念上可追溯到Russell与Norvig的经典Agent定义[1]，但LLM的引入使得"思考"环节具备了前所未有的开放语义推理能力。

### 1.2 思考模式：CoT / ReAct / Plan-After-Play

LLM Agent的核心差异在于思考环节的策略选择。当前主流范式可分为三类：

**Chain-of-Thought（CoT）**：由Wei等人（2022）提出，通过在Prompt中注入"Let's think step by step"等触发词，引导LLM生成中间推理步骤[2]。其数学形式为：给定输入 $x$，LLM生成推理链 $z_1, z_2, ..., z_n$，然后输出最终答案 $y$：

$$P(y|x) = \sum_{z} P(z|x) \cdot P(y|z,x)$$

CoT的局限在于它仅产生推理链而不与环境交互——Agent停留在"纯思考"层面，无法执行外部操作来验证假设或获取新信息。

**ReAct（Reasoning + Acting）**：Yao等人（2023）提出的ReAct模式将推理与行动交织为Thought→Action→Observation交替循环[3]。每个时间步的生成可表示为：

$$T_t \sim P_{LLM}(T_t | P_t, S_t, H_{<t})$$
$$A_t = \arg\max P_{LLM}(A | T_t)$$
$$O_t = \text{Execute}(A_t)$$

其中 $H_{<t} = \{(T_1, A_1, O_1), ..., (T_{t-1}, A_{t-1}, O_{t-1})\}$ 为完整的交互历史。ReAct的有效性在WebShop和ALFWorld等交互式任务中得到验证——在ALFWorld上，ReAct的Pass@1准确率达到71%，远高于Reason-only的26%和Act-only的44%[3]。这一差距表明：推理与行动的交替不仅是互补，更产生了1+1>2的协同效应。

**Plan-After-Play**：这是一种高级模式，Agent先通过探索（Play）积累环境信息，再制定完整计划（Plan）执行。典型的实现是SayCan的"affordance-aware planning"范式——Agent通过预训练模型评估每个动作在当前环境中的可行性（affordance），然后基于此进行规划[4]。这种模式适合长周期、高风险的任务场景，避免在执行中途因信息不足而做出错误决策。

### 1.3 Agent的基本构成模块

一个完整的Agent系统由三个核心模块构成：

**Profile（角色配置）**：Agent的System Prompt定义了其身份、行为约束和能力边界。这一模块的本质是**行为锚定**——通过精心设计的角色描述，将LLM的通用能力约束到特定任务域。Profile的设计直接影响Agent的"人格一致性"，例如一个"严格的代码审查Agent"与"创意写作助手Agent"在处理同一段代码时的输出会有本质差异。Xi等人（2023）在综述中将Profile定义为Agent的"行为控制层"[5]。

**Memory（记忆系统）**：Agent的记忆分为短期和长期两个层次。短期记忆等价于LLM的上下文窗口——在当前Session中维护的交互历史 $H_t$。其容量受限于 $C_{context}$（如GPT-4的128K tokens）。长期记忆则通过外部存储实现，典型方案包括：

- **Vector DB记忆**：将Agent的历史经验通过embedding函数 $e: \text{Text} \to \mathbb{R}^d$ 编码为向量，存入Chroma/Pinecone等向量数据库。检索时通过相似度搜索 $k\text{-NN}(e(q), DB)$ 获取相关经验。
- **Graph DB记忆**：以知识图谱形式存储实体间关系，支持多跳推理。

Park等人（2023）的Generative Agents研究展示了长期记忆对Agent行为连贯性的关键作用——250个Agent在一个模拟小镇中依靠记忆流系统维持了可信的社交行为[6]。

**Planning（规划模块）**：任务分解是将复杂目标递归拆解为可执行子任务的过程。不同层次的规划策略包括：

- **CoT式线性规划**：逐步骤推理，适合顺序依赖的任务。
- **ToT（Tree of Thoughts）**：Yao等人（2023）提出在每一步思维分支时展开多个候选，使用BFS/DFS搜索最优路径[7]。
- **GoT（Graph of Thoughts）**：Besta等人（2023）进一步突破树结构的局限，允许思维的合并与融合[8]。

## 2. ReAct框架的底层实现

### 2.1 从CoT到ReAct的范式跃迁

ReAct框架的核心理念在于打破纯推理（CoT）与纯行动（Act-only）之间的壁垒。理解ReAct需要先理解它的两个前身：

**纯推理模式（Reason-only / CoT）**：LLM生成一段推理链，直接输出答案。系统流程为：
```
Input → [LLM CoT Reasoning] → Final Answer
```
这种方式在数学推理（GSM8K）、常识推理等封闭任务上表现优异，但在需要与环境交互的开放任务中无能为力——因为LLM无法通过推理获取训练数据之外的世界知识。

**纯行动模式（Act-only）**：LLM直接输出行动指令，不生成中间推理。系统流程为：
```
Input → [LLM Action Generation] → Execute → Observation → ...
```
这种方法在简单交互任务上可行，但缺乏推理指导的行动容易在复杂场景中陷入局部最优或死循环。

**ReAct的融合**：ReAct将上述两种模式有机结合，形成如下循环：

```
Input → [Thought → Action → Observation → Thought → ...] → Final Answer
```

ReAct的核心创新在于：**Thought不仅是对环境的分析，更是对后续Action的推理依据**。Yao等人（2023）的实验数据极具说服力[3]：

| 任务 | Reason-only | Act-only | ReAct |
|------|-------------|----------|-------|
| ALFWorld (Pass@1) | 26% | 44% | 71% |
| WebShop (Success Rate) | 不适用 | 62.6% | 75.3% |
| HotPotQA (F1) | 27.2% | 不适用 | 34.5% |

在ALFWorld上，ReAct相比Act-only提升了27个百分点，相比Reason-only提升了45个百分点。这种提升来自于ReAct特有的**错误纠正能力**——当推理方向错误时，环境的Observation可以提供修正信号；当行动失败时，Thought可以重新分析原因并调整策略。

### 2.2 ReAct循环的停机时机

ReAct循环的实现本质上是一个 `while` 循环：

```
def react_loop(llm, tools, max_iterations):
    iteration = 0
    finished = False
    
    while iteration < max_iterations and not finished:
        thought = llm.generate_thought()      # Step 1: 推理
        action = llm.generate_action(thought) # Step 2: 行动决策
        
        if action.type == "Finish":
            finished = True
            return action.result
        
        observation = execute(action, tools)  # Step 3: 执行并获取观察
        iteration += 1
    
    return "Max iterations reached"
```

停机条件的判断由两部分组成：

1. **显式停机**：Action类型被LLM标记为"Finish"，通常在LLM认为已经获得足够信息完成任务时触发。LangChain等框架通过在Tool定义中内置一个特殊的 `Finish` tool来实现。

2. **隐式停机**：Agent在Observation中检测到终止条件（如任务成功信号、重复循环检测），这需要在System Prompt中明确约定"当你观察到X状态时应当触发Finish"。

实际工程中，还需要引入**最大迭代保护**（preventing infinite loops）——通常设置在15-30次之间，超过后触发fallback策略（如返回已收集的部分结果或报错）。

### 2.3 ReAct中的Prompt工程

ReAct高层的实现依赖于精密的Prompt设计。其核心由三部分组成：

**Few-shot专家轨迹示例**：在System Prompt中固定嵌入2-3个高质量的ReAct示例轨迹。每个示例展示了完整的Thought→Action→Observation循环。关键设计原则：

1. **推理过程的显式结构化**：使用标记语言约束Thought的格式，例如：
   ```
   Thought: 我需要查找数据库中2024年的销售数据。这个查询需要统计总销售额，所以应该使用SUM函数。
   Action: sql_query: SELECT SUM(revenue) FROM sales WHERE year = 2024
   Observation: 12345678
   Thought: 得到了总销售额12,345,678美元。现在我需要确认用户是否还需要其他信息。
   Action: Finish
   ```

2. **agent_scratchpad的设计**：这是ReAct实现中最容易被忽视但最关键的技术细节。agent_scratchpad是一个实时更新的文本缓冲区，记录了当前会话中已经执行的所有Thought→Action→Observation序列。每次LLM被调用时，agent_scratchpad都会被注入到Prompt中，确保Agent知道"我已经做了哪些事、得到了什么结果"。它的作用类似于循环中的 **状态保持变量**：

   ```
   System Prompt: [任务描述 + 工具定义 + few-shot示例]
   
   User Input: [原始用户请求]
   
   Agent Scratchpad:
   Thought 1: 分析问题...
   Action 1: tool_a(params)
   Observation 1: 结果A
   Thought 2: 基于结果A，我需要进一步...
   Action 2: tool_b(params)  
   Observation 2: 结果B
   
   Current iteration:
   Thought 3: (LLM从这里继续生成)
   ```

   这种设计巧妙地解决了LLM无状态的问题——通过每次调用时将完整的执行轨迹作为前缀注入，使得LLM可以在不维护显式状态的情况下实现连贯的多步推理。

实现ReAct时还有两个关键参数：**temperature**（通常设定较低值0-0.3以确保行动决策的确定性）和 **stop tokens**（设定如"Observation:"等标记来控制生成长度）。

## 3. 规划与反思机制

### 3.1 Plan-and-Execute的规划分解

Plan-and-Execute模式是ReAct的进化版本，它将规划与执行解耦为两个独立组件：

**Planner**：接收用户请求，生成一个多步骤的任务计划（Plan）。其核心输出不是具体的执行指令，而是**子任务的抽象列表**——每个子任务对应一个高层次的步骤描述，而非具体的工具调用。形式化地：

$$Plan = Planner(Query) = [Task_1, Task_2, ..., Task_n]$$

其中每个 $Task_i$ 是自然语言描述的子目标，如"收集2024年各季度销售数据"而不是"调用sql_query('SELECT ...')"。

Planner的设计有两种风格：
- **单次规划（One-shot Planning）**：LLM一次性生成所有子任务。优点是简单高效，适合结构清晰的任务；缺点是无法根据中间结果调整计划。
- **动态规划（Dynamic Planning）**：Planner在执行过程中被重新调用，根据已完成的子任务结果调整后续计划。这更接近人类的规划方式——先做粗略规划，在执行过程中细化。

**Executor**：从Plan中取出一个子任务 $Task_i$，进入类似ReAct的循环来执行它。每个子任务的执行也是一个完整的感知-思考-行动-观察循环。

Plan-and-Execute相比纯ReAct的优势在于：
1. **上下文效率**：每个子任务只需关注其局部上下文，避免了长序列任务的上下文污染
2. **错误隔离**：单个子任务的失败不会影响整个计划的其余部分
3. **并行潜力**：无依赖的子任务可以并行执行

### 3.2 Tree of Thoughts：思维树搜索

ToT（Tree of Thoughts）由Yao等人（2023）提出，是对CoT的突破性扩展[7]。CoT的推理是线性的——一条路径走到黑。ToT则引入了**树搜索**的概念，在每个推理步骤展开多个候选思路，系统性地探索推理空间。

**数学建模**：
- 思维状态：$s_i$ 表示推理过程中的第 $i$ 步状态
- 行动空间：$A(s_i) = \{a_1, a_2, ..., a_k\}$ 是在状态 $s_i$ 下所有可能的下一步思维
- 状态转移：$s_{i+1} = f(s_i, a)$，其中 $f$ 由LLM实现

**搜索策略**：
- **BFS（广度优先搜索）**：每层保留Top-K个最有前景的节点。适合推理树的宽度不大但需要探索多种可能性的场景。
- **DFS（深度优先搜索）**：优先深入探索一条路径，遇到死胡同时回溯。适合推理深度较大的场景。

ToT通过引入**评估函数** $V(s_i)$ 来判断当前思维状态的价值——LLM被要求对每个中间状态进行打分（如1-10分），指导搜索方向。

ToT的最震撼结果来自24点游戏实验：CoT的成功率仅7.3%，而ToT达到了74%[7]——10倍的提升说明了在需要搜索和回溯的推理任务中，结构化搜索相比线性推理的绝对优势。

### 3.3 Self-Consistency & Graph of Thoughts

**Self-Consistency**：由Wang等人（2022）提出，是一种简单而有效的推理增强策略[9]：

1. 对同一个问题，用CoT采样 $N$ 次推理路径（通常在5-20次之间）
2. 对每条路径的输出结果进行多数投票（majority voting）
3. 选择得票最多的答案作为最终输出

$$FinalAnswer = \arg\max_{y} \sum_{i=1}^{N} \mathbb{I}[CoT_i(x) = y]$$

Self-Consistency在数学推理任务（GSM8K）上将准确率从CoT单次的71.8%提升至85.6%[9]。这种提升的代价是推理成本的线性增长（$N$ 倍），但好处是不需要额外的训练或复杂的搜索逻辑。

**Graph of Thoughts（GoT）**：Besta等人（2023）的工作是对思维结构的最彻底解放[8]。GoT将推理建模为**有向图** $G = (V, E)$，其中：

- $V$：思维节点集合，每个节点代表一个推理步骤的状态
- $E$：边集合，表示思维之间的依赖和转换关系

GoT相比于ToT的关键创新在于**思维的合并与融合**。在ToT中，不同分支的思维是平行的，互相之间不交流；而在GoT中，可以将两个分支的推理结果进行merge操作，生成更综合的思维。例如，在文档合并任务中，一个分支分析文档A的结构，另一个分支分析文档B的结构，两个结果可以融合成一个合并方案。

GoT定义了四种基本操作：
1. **分解（Decomposition）**：将一个思维分裂为多个子思维
2. **细化（Refinement）**：对单个思维进行优化更新
3. **聚合（Aggregation）**：将多个相关思维合并为一个综合思维
4. **生成（Generation）**：从现有思维产生新的相邻思维

这种灵活的操作集使得GoT能够在复杂推理任务（如排序网络合成、文档合并）上取得比CoT/ToT更好的效果[8]。

## 4. 反思与自我改进

### 4.1 Reflexion：经验驱动的Agent反射机制

Shinn等人（2023）提出的Reflexion框架是最具代表性的Agent反思机制[10]。其核心思想是：Agent不仅仅执行任务，还会从执行失败中**提炼经验**，并在未来的尝试中使用这些经验。

Reflexion的三阶段工作流：

1. **执行阶段（Act）**：Agent执行任务，记录完整的执行轨迹
2. **反思阶段（Reflect）**：当Agent遭遇失败或错误时，LLM被提示分析执行轨迹，总结失败原因，生成经验文本。这个过程被称为**verbal reinforcement learning**——因为"奖励信号"是自然语言描述的经验，而非数值化的reward。
3. **记忆阶段（Store）**：经验文本被存入一个短期缓存（episodic buffer）

在下一轮任务开始时，episodic buffer中的经验被处理并注入到Agent的Prompt中，作为额外的few-shot示例或上下文约束。

Reflexion的设计揭示了一个深刻的洞察：**LLM Agent的学习不是通过梯度更新，而是通过Prompt的演变**。每一次失败都被编码为自然语言经验，修改了Agent的推理前提。这种"外挂式的经验积累"使得Agent可以持续改进而不需要重新训练模型。

在AlfWorld和HotPotQA等任务上，Reflexion将ReAct的成功率分别提升了22%和12%[10]。

### 4.2 Self-Evaluation与Self-Critique

Self-Evaluation机制要求LLM对自己的输出进行元评估。标准实现是**Multi-dimension Scoring**：

```
你是一个评估者。请对自己的回答在以下维度打分（1-10分）：
1. 答案完整性：是否回答了用户所有需求
2. 逻辑一致性：推理链是否自洽  
3. 证据充分性：每个结论是否有足够支撑
4. 语言清晰度：表达是否无歧义

请输出JSON格式评分和改进建议。
```

这种自我评估的数学本质是让LLM计算 $P(output | question, evaluation\_criteria)$，其中evaluation_criteria是对"好输出"的约束。这等价于一个**可微的隐式奖励函数**——无需显式建模，而是通过Prompt的自然语言描述来定义。

**Self-Refine**（Madaan等人，2023）进一步将Self-Evaluation与迭代优化结合[11]：Agent生成初始输出 → 自我反馈 → 根据反馈改进 → 循环直到收敛。这个过程可以形式化为：

$$y_0 = LLM(x)$$
$$fb_t = LLM_{feedback}(x, y_t)$$
$$y_{t+1} = LLM_{refine}(x, y_t, fb_t)$$

在对话生成、代码优化等任务中，Self-Refine经过3-5轮迭代后可以显著提升输出质量[11]。

### 4.3 CRITIC：外部知识验证的步进纠正

Gou等人（2023）提出的CRITIC框架代表了反思机制的另一个重要方向——**借助外部工具验证**来纠正LLM的推理错误[12]。

CRITIC的工作流程：
1. LLM生成一个初始推理路径或答案
2. CRITIC模块分析每一步，识别可能需要验证的声明
3. 对于每个需要验证的声明，CRITIC调用外部工具（如搜索引擎、计算器、代码解释器）获取验证信息
4. 如果发现错误，LLM在验证反馈的基础上修正推理
5. 重复直到所有步骤通过验证或达到最大循环次数

CRITIC的关键创新在于它将"思考"与"验证"分离开来。LLM负责创造性的推理假设，而外部工具负责事实性的验证。这种**分工**缓解了LLM在事实准确性和推理深度之间的权衡。

在Freebase QA和WikiQA等事实密集型任务上，CRITIC将准确率提升了15-25%[12]。

## 5. Agent记忆系统

### 5.1 短期记忆与动态上下文管理

LLM的上下文窗口是Agent短期记忆的物理边界。当前最先进的模型（如GPT-4-128K、Claude-200K）提供了前所未有的窗口长度，但Agent的执行轨迹仍然可能在长周期任务中超出窗口限制。

**滑动窗口策略**是最直接的短期记忆管理方案：

$$Window_t = \text{Last } K \text{ tokens of } \{Interaction_1, ..., Interaction_t\}$$

其中 $K < C_{context}$。这种方法简单高效，但会遗忘中间部分的关键信息。

**MemGPT式的上下文分页**（Pagination）是更精致的方案[13]。其核心思想：

1. **上下文监控**：持续跟踪当前上下文窗口的利用率，设定阈值（如80%）
2. **记忆分层**：当上下文接近阈值时，将部分内容归档到持久层：
   - 最近交互 → 保留在主上下文
   - 较旧但不重要的交互 → 压缩为摘要后保留
   - 关键信息（如工具执行结果） → 按重要性排序，选择性保留
3. **检索触发**：当Agent需要访问已归档的信息时，自动触发记忆检索（Memory Retrieval）

MemGPT的类比是操作系统的虚拟内存——LLM的上下文窗口是物理内存，持久存储是磁盘，Agent的检索机制是MMU（内存管理单元）。

### 5.2 长期记忆的Vector DB存储

长期记忆的核心技术是向量数据库。Agent将每次执行的结果（包括成功经验和失败教训）通过embedding模型 $e$ 编码为向量：

$$v_i = e(experience\_text_i) \in \mathbb{R}^d$$

存入向量数据库（如Chroma、Pinecone、Weaviate、Qdrant）。

当Agent面临新的决策时，使用当前状态 $q$ 进行检索：

$$Retrieved = k\text{-NN}(e(q), DB) = \{v_{i_1}, v_{i_2}, ..., v_{i_k}\}$$

检索到的经验文本被注入到Prompt中，作为Agent决策的参考。这种机制使Agent具备了**跨Session的经验积累能力**——一个Agent在Session A中学会的技巧可以在Session B中被自动唤醒。

向量检索的检索质量取决于三个因素：
1. **Embedding质量**：模型需要能够捕捉任务语义的相似性
2. **分块策略**：过长或过短的经验文本都会影响检索精度，通常需要200-500 token的分块粒度
3. **检索数量**：$k$ 的取值需要在信息充足与上下文污染之间平衡，通常取3-10

### 5.3 知识图谱结构记忆

Graph记忆是比Vector DB更结构化的记忆形式。它以 **实体-关系-实体** 三元组 $(h, r, t)$ 为基本存储单元，构建关于Agent环境的知识网络。

一个典型的Agent知识图谱包含：
- **人物节点**：用户身份、偏好、互动历史
- **位置节点**：当前环境、可用资源
- **物品节点**：工具、文档、数据源
- **关系边**："属于"、"依赖"、"拥有"、"位于"等

知识图谱的核心优势在于**多跳推理**（multi-hop reasoning）：

$$(x) \xrightarrow{located\_in} (Room A) \xrightarrow{contains} (Database) \xrightarrow{has\_schema} (Schema) \xrightarrow{has\_table} (Sales Table)$$

这种链式推理在纯向量检索中很难实现——向量检索擅长语义相似性，但不擅长关系链的追踪。

在实际系统中，Graph记忆和Vector DB记忆通常是互补的：**Vector DB提供语义检索**，**Graph DB提供关系推理**。Agent先通过向量检索找到相关的记忆片段，再通过图谱的关系路径进行深层次的推理。

## 6. 多Agent协作

### 6.1 角色分工与任务分解

多Agent系统的首要设计原则是**角色分工**（Role Specialization）。与单Agent不同，多Agent系统通过定义多个专业化角色，将复杂任务分解为多个子任务，由不同Agent协作完成。

一个典型的三Agent协作模式：

```
[Task Decomposition Agent] → 分析用户请求，分解为子任务列表
         ↓
    [Worker Agent(s)] → 执行每个子任务，调用工具和API
         ↓
    [Review Agent] → 检查Worker的输出质量，提供反馈
```

这种架构的关键优势在于：
1. **关注点分离**：每个Agent只需要精通其负责的领域，降低了单个Agent的Prompt复杂度
2. **质量门禁**：Review Agent的存在形成了天然的质量关卡
3. **并行加速**：多个Worker Agent可以并发执行无依赖的子任务

**MetaGPT**（Hong等人，2023）将这种角色分工推向了极致[14]。MetaGPT模拟了一个软件公司的完整产品开发流程，定义了多个专业化角色Agent：

- Product Manager（PM）：定义产品需求和PRD
- Architect：设计技术架构
- Project Manager：分配任务和追踪进度
- Engineer：编码实现
- QA Engineer：测试和质量保证

每个角色Agent的输出格式被严格定义为结构化的文档模板（如SRDOutput——Software Requirement Document Output），通过标准化的信息传递实现角色间的协作。这种**模板化通信**确保了信息在不同角色间传递时不会丢失关键细节[14]。

### 6.2 Agent间通信机制

多Agent系统的生命力在于通信。当前有两种主流的Agent间通信范式：

**自然语言Message Passing**：Agent之间通过自然语言直接对话。AutoGen（Wu等人，2023）实现了典型的Two-Agent对话框架[15]：

1. **Sender Agent** 生成一条消息
2. **Receiver Agent** 接收并处理消息
3. Receiver可以回复、要求澄清或完成任务
4. 过程持续迭代直到任务完成或达到终止条件

这种方式的优点是灵活——Agent可以通过自然语言协商任务细节，无需预定义的通信协议。缺点是可靠性——自然语言可能导致歧义，且长对话可能发散。

**结构化Message Bus**：Agent通过预定义的消息格式进行通信。MetaGPT使用的SRDOutput就是典型例子——PM Agent的输出是一个结构化的JSON文档，Architect Agent可以直接解析这个结构进行架构设计。

结构化通信的优势：
1. **类型安全**：每个字段有明确的语义定义
2. **可追踪**：消息的传递可以被审计和回放
3. **可组合**：不同Agent的输出可以像管道一样串联

### 6.3 Discussion与Voting机制

多Agent讨论与投票是提高决策质量的重要机制。其理论基础是**群体智慧**（Wisdom of the Crowd）——多个独立决策者的聚合判断通常优于单个决策者。

**G3框架**（Du等人，2023）提出的Multi-Agent Debate是最具代表性的实现[16]：

1. 多个Agent（通常3个）分别独立生成初始答案
2. Agent们阅读彼此的答案和推理过程
3. 每个Agent基于其他Agent的观点修正自己的判断
4. 经过多轮辩论后，通过投票（Majority Voting）或共识（Consensus）选择最终答案

在事实性任务上，G3的3-Agent辩论将准确率提升了约18%，且在幻觉减少方面尤为显著[16]。实验表明，辩论轮次在2-3轮时达到最佳效果，更多的轮次边际效益递减。

**Democratic Consensus机制**通过多数投票抑制幻觉的原理可以用以下框架理解：

设每个Agent的正确答案概率为 $p$（假设Agent独立同分布），则 $n$ 个Agent多数投票的正确答案概率为：

$$P_{majority}(n) = \sum_{k=\lceil n/2 \rceil}^{n} \binom{n}{k} p^{k} (1-p)^{n-k}$$

当 $p > 0.5$（Agent略优于随机），$n$ 越大则 $P_{majority}$ 越接近1。例如，$p=0.6$ 时单Agent准确率60%，3-Agent投票提升到64.8%，5-Agent投票提升到68.3%。

## 7. Agent框架的演进与未来方向

### 7.1 LangGraph：Agent编排框架的图论实现

LangGraph（LangChain团队开发）是目前最具代表性的Agent编排框架之一。它将Agent的工作流建模为**有向状态图**（StateGraph）：

$$G = (V, E, S_0, F)$$

其中：
- $V = \{Node_1, ..., Node_m\}$：节点集合，每个节点可以是一个Agent、一个Tool调用、或一个条件判断
- $E \subseteq V \times V \times \text{Condition}$：边集合，包含状态转移的条件
- $S_0$：初始状态
- $F \subseteq V$：终止节点集合

LangGraph的核心运行机制：

**Node（节点）**：每个Node是一个可调用对象 $f: S_t \to S_{t+1}$，接收当前状态并返回更新后的状态。Node可以是：
- Agent Node：包含LLM推理和决策逻辑
- Tool Node：执行具体的外部工具调用
- Conditional Node：根据当前状态决定下一步跳转

**Edge（边）**：定义状态转移的方向和条件。LangGraph支持：
- 无条件边：始终执行的转移
- 条件边：基于状态的路由选择，如 `if state.confidence < 0.5 then route_to_refine else route_to_finish`

**State（状态）**：贯穿整个图的数据对象，包含Agent执行过程中需要的所有信息——用户输入、中间结果、执行轨迹、错误信息等。

LangGraph的关键能力是**Checkpoint-based Human-In-The-Loop**：

```
Agent Execution
    ↓ (pause at checkpoint)
State Snapshot → Human Review → Modify State or Approve
    ↓ (resume)
Continue Execution
```

Checkpoint机制允许Agent在执行的中途暂停，人类可以检查当前状态、修改Agent的决策或注入额外信息，然后恢复执行。这种机制在以下场景中至关重要：
- **高风险的业务决策**：需要人类审批后才能执行
- **歧义处理**：Agent不确定时暂停等待人类指导
- **调试**：开发阶跟踪Agent的中间状态

### 7.2 RAG与Memory-Augmented Agent的融合

2026年的Agent框架发展呈现出两个明确趋势：

**趋势一：Agent与RAG的深度融合**

早期的Agent和RAG（Lewis等人，2020[17]）是相对独立的研究方向。当前的发展表明，两者的融合是不可逆转的——Agent需要RAG来获取外部知识，RAG需要Agent来执行多步检索推理。

Agent-RAG融合的典型模式：
1. **检索增强的规划**：Agent在制定计划时，通过RAG检索相关的历史方案和最佳实践
2. **检索增强的执行**：在执行过程中遇到不确定时，Agent自动触发知识库检索
3. **检索增强的反思**：Agent反思失败时，检索类似的失败案例作为参考

**趋势二：Multi-Turn Tool Planning的长期执行**

Agent正在从单轮工具调用（Single-turn Tool Use）向多轮工具规划（Multi-turn Tool Planning）演进。早期的Toolformer（Schick等人，2023）[18]展示了LLM自主学习工具调用的能力，但每次调用都是独立决策。当前的趋势是Agent需要管理持续数小时甚至数天的工具操作流程。

Multi-Turn Tool Planning面临的关键挑战：
- **任务持久化**：长期执行的Agent需要能够在会话中断后恢复
- **中间结果管理**：数十次工具调用的中间结果需要有效组织和存储
- **动态重规划**：当环境发生变化时，Agent需要调整后续的工具调用计划

### 7.3 Agent Evaluation与Safety

随着Agent能力的增强，对Agent行为的安全约束变得越来越重要。

**Preemptive Observation机制**：在Agent的每个思考步骤中嵌入安全检查：

```
Thought: (Agent分析当前状态)
Safety Check: (在行动前进行风险评估)
Action: (通过安全检查后执行)
```

具体实现包括：
1. **输出的前处理拦截**：在Agent生成Action后执行前，由一个Safety Model评估Action的安全性
2. **行动边界检查**：定义Agent的行动域白名单，超出边界的行动自动拒绝
3. **敏感操作分级**：高风险操作（如写文件、执行代码）需要额外的确认步骤

**LLM内在安全约束的多层叠加**：当前的安全架构通常采用多层防御：

- **层1（System Prompt）**：在Agent的System Prompt中嵌入安全规则
- **层2（Output Guardrails）**：使用独立的Guard Model检查Agent输出
- **层3（Human Oversight）**：关键决策点的HITL审核
- **层4（Audit Trail）**：所有Agent行动的不可篡改日志

Anthropic等机构的研究表明，多层安全约束能够将Agent的违规行为率降低约93%[19]，但仍无法完全消除风险——这构成了Agent产业化部署中最重要的未解决问题之一。

---

### 参考文献

[1] Russell, S. J., & Norvig, P. (2010). *Artificial Intelligence: A Modern Approach* (3rd ed.). Prentice Hall.

[2] Wei, J., Wang, X., Schuurmans, D., et al. (2022). Chain-of-Thought Prompting Elicits Reasoning in Large Language Models. *NeurIPS 2022*.

[3] Yao, S., Zhao, J., Yu, D., et al. (2023). ReAct: Synergizing Reasoning and Acting in Language Models. *ICLR 2023*.

[4] Ahn, M., Brohan, A., Brown, N., et al. (2022). Do As I Can, Not As I Say: Grounding Language in Robotic Affordances. *CoRL 2022*.

[5] Xi, Z., Chen, W., Guo, X., et al. (2023). The Rise and Potential of Large Language Model Based Agents: A Survey. *arXiv:2309.07864*.

[6] Park, J. S., O'Brien, J. C., et al. (2023). Generative Agents: Interactive Simulacra of Human Behavior. *UIST 2023*.

[7] Yao, S., Yu, D., Zhao, J., et al. (2023). Tree of Thoughts: Deliberate Problem Solving with Large Language Models. *NeurIPS 2023*.

[8] Besta, M., Blach, N., Kubicek, A., et al. (2023). Graph of Thoughts: Solving Elaborate Problems with Large Language Models. *arXiv:2308.09687*.

[9] Wang, X., Wei, J., Schuurmans, D., et al. (2022). Self-Consistency Improves Chain of Thought Reasoning in Language Models. *ICLR 2023*.

[10] Shinn, N., Cassano, F., Gopinath, A., et al. (2023). Reflexion: Language Agents with Verbal Reinforcement Learning. *NeurIPS 2023*.

[11] Madaan, A., Tandon, N., Gupta, P., et al. (2023). Self-Refine: Iterative Refinement with Self-Feedback. *NeurIPS 2023*.

[12] Gou, Z., Shao, Z., Gong, Y., et al. (2023). CRITIC: Large Language Models Can Self-Correct with Tool-Interactive Critiquing. *ICLR 2024*.

[13] Packer, C., Wooders, S., Lin, K., et al. (2023). MemGPT: Towards LLMs as Operating Systems. *arXiv:2310.08560*.

[14] Hong, S., Zheng, X., Chen, J., et al. (2023). MetaGPT: Meta Programming for A Multi-Agent Collaborative Framework. *ICLR 2024*.

[15] Wu, Q., Bansal, G., Zhang, J., et al. (2023). AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation. *arXiv:2308.08155*.

[16] Du, Y., Li, S., Torralba, A., et al. (2023). Improving Factuality and Reasoning in Language Models through Multiagent Debate. *ICML 2024*.

[17] Lewis, P., Perez, E., Piktus, A., et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. *NeurIPS 2020*.

[18] Schick, T., Dwivedi-Yu, J., Dessì, R., et al. (2023). Toolformer: Language Models Can Teach Themselves to Use Tools. *NeurIPS 2023*.

[19] Anthropic. (2023). Constitutional AI: Harmlessness from AI Feedback. *arXiv:2212.08073*.
