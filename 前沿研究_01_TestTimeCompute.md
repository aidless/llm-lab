# Test-Time Compute Scaling：推理时计算的扩展法则

## 1. 推理时计算扩展的基本问题：Scaling Laws的范式转移

大语言模型的Scaling Laws长期被理解为训练时的规模扩展——更多的参数和更多的数据直接兑换为更低的perplexity和更高的任务准确率。Kaplan et al.（2020）在《Scaling Laws for Neural Language Models》中建立了幂律关系，Chinchilla（Hoffmann et al., 2022）则进一步修正了计算最优分配，指出模型参数和训练token数应按比例同步增长。这一范式统治了2020—2024年的大模型研发。

OpenAI o1系列（OpenAI, 2024）的发布标志着一个根本性的范式转移：模型的推理能力不仅可以通过训练时扩展（更多的RL训练计算）来提升，也可以通过推理时扩展——即在生成答案时允许模型消耗更多的计算资源——来按需改进。OpenAI在o1的发布报告中明确写道："the performance of o1 consistently improves with more reinforcement learning (train-time compute) and with more time spent thinking (test-time compute)"。

这一发现将Scaling Law从训练阶段延伸至推理阶段，形成了当前最活跃的研究方向。推理时计算扩展主要有三种技术路线：

**（1）N-Best重排序（生成-选择-验证）。** 这是最直接的扩展方式：对同一问题生成N个候选答案，然后用奖励模型或验证器从中选择最佳结果。典型实现为Best-of-N（BoN）采样，N的取值从几十到数千不等。这种方法本质上是"并行"扩展——每增加一个样本都消耗等量的FLOPs，但准确率随N的对数增长呈现近似线性提升（Lightman et al., 2023; Snell et al., 2024）。

**（2）多步修正——Diverse Verifier Tree Search。** 通过Process Reward Model（PRM）对推理过程的中间步骤进行Beam Search式的多次修正，逐步完善输出。与Best-of-N在完整路径层面进行选择不同，Tree Search在每一步都进行分支探索和剪枝，允许更细粒度的计算分配。Snell et al.（2024）的实验表明，Beam Search在困难问题上和低计算预算条件下显著优于Best-of-N，但两者之间存在重要的trade-off。

**（3）延长的思维链（Extended Chain-of-Thought）。** 让模型显式地生成更多的中间推理步骤——通过RL训练或prompt引导，使模型在回答问题前花费更多"思考时间"。o1-preview的CoT token数可以从数百扩展到数万甚至数十万，伴随推理步骤的增加，AIME等数学竞赛数据集的准确率呈现清晰的扩展曲线。DeepSeek-R1（DeepSeek, 2025）也展示了类似现象：平均响应长度从训练初期的数百token增长到数千token，准确率同步提升。

这三种方式构成了推理时计算扩展的方法论基础，各自具有不同的计算效率特征和适用场景。

## 2. Process Reward Model（PRM）

Process Reward Model的核心动机在于：在复杂多步推理任务中，仅仅奖励最终答案的正确性（Outcome Reward Model, ORM）不足以提供有效的指导信号。ORM在完整生成的最终答案上计算单一标量奖励，其训练目标相对简单——给定输入-输出对预测正确性得分。然而，当模型的中间步骤出现错误时，ORM无法定位具体是哪一步出了问题，导致信用分配困难。

PRM与ORM的本质差异在于奖励粒度的不同。ORM将整个推理轨迹作为整体评估，而PRM对每一推理步骤分配独立的奖励分数。设一个包含T步的推理轨迹s = (s_1, s_2, ..., s_T)，其中s_t为第t步后的部分推理状态，PRM定义为函数R(s_t) → [0, 1]，表示从当前状态能推导出正确答案的概率。PRM的最终路径得分为各步得分的最小值或乘积，即PRM(s) = min_t R(s_t)或PRM(s) = ∏_t R(s_t)（Lightman et al., 2023）。

PRM训练的最大瓶颈在于中间步骤标注数据的获取。人类标注者需要对每步推理的正确性进行判断，这不仅昂贵且一致性难以保证。Lightman et al.（2023）在PRM800K项目中构建了一个包含约80万步级标注的数学推理数据集，利用Majority Voting的方式聚合多个标注者的判断来提升标签质量。研究表明，人类标注者在AI辅助下的step-level标注一致性约为80%，而纯粹由专家标注的一致性也不超过88%。

为解决标注成本问题，Math-Shepherd（Wang et al., ACL 2024）提出了全自动的PRM标注数据生成方案。其核心思想是利用"最终答案正确性反推中间步骤正确性"——给定一个数学问题和标准答案，对于部分解路径中的某一步s_t，使用一个fine-tuned的LLM从该步骤出发解码多条后续推理轨迹（rollout），并检查这些轨迹最终推导出的答案与标准答案的匹配率。如果从某一步rollout得到的正确率更高，则该步骤更可能是正确的。具体地，对于步骤s_t的标签计算为：

L(s_t) = (1/N) ∑_{i=1}^{N} [rollout_i(s_t)的最终答案 == 标准答案]

其中N为rollout数量（通常设为8-16），[·]为示性函数。这种方法本质上是一种Monte Carlo（MC）近似标注，被Lightman et al.（2023）称为MC Dropout近似标注。

在GSM8K和MATH数据集上的实验表明，Math-Shepherd自动标注的PRM数据质量可与人类标注的PRM800K数据集相媲美，甚至在MATH上表现更好。由Math-Shepherd提供的Process RL能够将Mistral-7B在GSM8K上的准确率从77.9%提升至84.1%，在MATH上从28.6%提升至33.0%。

PRM的验证效果通过Best-of-N搜索来度量。在MATH数据集上的实证分析表明，随着搜索预算（即N值）从1增加到256，PRM Best-of-N的准确率呈现对数增长，在N=256时DeepSeek-67B达到48.1%（GSM8K: 93.3%）。基于PRM的Beam Search进一步改进了搜索效率：在相同的计算预算下，PRM Beam Search能够在更小的搜索空间中达到与BoN相当的准确率，尤其适合困难问题的处理。

## 3. Self-Play与Self-Correction

Self-Correction是推理时计算扩展的重要方向之一，但其效果存在显著的内在局限性。Madaan et al.（2023）提出的Self-Refine框架构建了"初始生成→自我反馈→迭代优化"的循环：给定输入x，模型生成初始输出y₀，然后对y₀生成自然语言反馈，再根据反馈精炼输出，如此循环。Self-Refine在对话生成、数学推理、代码优化等7个任务上实现了约20%的绝对提升。

然而，Self-Correction在无外部反馈的情况下面临严重的准确率改进天花板。多项研究一致发现：（1）当初始答案已经正确时，Self-Correction往往会把正确答案改错——这种现象被称为"修正退化"（correction degradation）或"从正确到错误的翻转"（correct-to-incorrect flip）；（2）当模型能力不足时（如Vicuna-13B），Self-Correction几乎无法产生有效反馈，甚至可能生成幻觉反馈将模型推向更错误的路径。

Self-Refine实验的定性分析揭示了失败根因：61%的失败案例源自反馈建议了不合适的修正方向，33%源自反馈无法精确定位错误位置，仅6%是精炼器正确实施了不正确的反馈。这意味着Self-Correction的有效性高度依赖于模型自身产生准确的自我反馈的能力。

Self-Correction的条件改进需要两个关键因素：一是可靠的外部验证器——如PRM、代码执行结果或基于证据抽取的事实验证器来提供客观反馈信号；二是知识抽取驱动的证据反馈——利用检索增强生成（RAG）从外部知识源中提取事实证据作为修正依据。

Self-Play强化学习提供了Self-Correction的训练时范式的对抗性改进。SPIN（Self-Play Fine-tuning; Chen et al., ICML 2024）提出了一种自我对抗训练机制：将LLM的微调过程构建为一个双人博弈，其中"主玩家"（当前迭代的LLM）需要区分人类标注的响应和"对手玩家"（上一迭代的LLM）生成的响应，而对手玩家则试图生成与人类响应难以区分的输出。SPIN的优化目标基于积分概率度量（IPM）：

L_SPIN(θ, θ_t) = E[ℓ(λ log(p_θ(y|x)/p_θ_t(y|x)) - λ log(p_θ(y'|x)/p_θ_t(y'|x)))]

SPIN的理论保证是：当且仅当LLM的策略分布p_θ(y|x)与目标数据分布p_data(y|x)完全一致时，训练达到全局最优——即收敛于纳什均衡。实验结果表明，经过SPIN迭代的模型在HuggingFace Open LLM Leaderboard、MT-Bench等多个基准上超越使用额外GPT-4偏好数据训练的DPO模型，且不需要任何额外的标注数据。

DeepSeek-R1的RL扩展本质上也利用了Self-Play的思想：通过GRPO（Group Relative Policy Optimization）框架进行大规模强化学习，让模型在自我博弈中自然涌现出"aha moment"——自我反思和自我验证行为是强化学习训练过程中的涌现现象，而非通过人类标注语料直接植入。

## 4. Tree Search与Monte Carlo策略

Monte Carlo Tree Search（MCTS）在LLM推理中的应用借鉴了其在AlphaGo等游戏中的成功经验——通过搜索树的构建和节点值的动态估计来引导推理路径的选择。在LLM推理的MCTS框架中，定义搜索节点为部分推理状态s_t，从s_t到s_{t+1}的动作为下一步推理步骤（可以是一个token、一个句子或一个中间结果）。

MCTS的核心在于节点值的准确估计。最初的方法采用Rollout-based估计：从当前节点出发使用policy model模拟多条完整推理路径，根据最终答案的正确率估算节点得分。这种方法计算量大，但不需要额外的模型。更高效的方案是训练Value Network直接预测状态值——即给定部分推理状态s_t，输出V(s_t)作为其最终获得正确答案的概率估计。AlphaMath（Chen et al., NeurIPS 2024）的做法是将Value Network附加在LLM之上——共享大部分参数，仅在最后一层增加一个线性层输出标量值，从而实现policy和value的轻量级集成。

Self-Evaluation引导的逐步搜索在数学推理任务上展现出显著效果。MCTS通过Selection → Expansion → Simulation → Backup四步循环逐步构建搜索树，在Selection阶段使用UCT（Upper Confidence Bound applied to Trees）公式平衡exploration和exploitation：

a_selected = argmax_a [Q(s_t, a) + c_puct · P(s_t, a) · √(N(s_t) / (1 + N(s_t, a)))]

其中Q(s_t, a)为动作a的平均收益估计，P(s_t, a)为policy model给出的先验概率，N为访问计数，c_puct为探索系数。

AlphaMath的突出贡献在于提出了"去PRM的MCTS"框架——不需要任何过程标注数据（来自人类或GPT-4），仅利用预训练LLM自身的知识与MCTS的自我对弈能力来提升数学推理。具体框架分为两个阶段：

在训练阶段，AlphaMath使用MCTS的Self-Rollout得分对Self-Training提供自动奖励更新——MCTS树中节点Q值作为Value Model的监督信号，使得Value Model学习评估中间步骤质量的能力。在推理阶段，AlphaMath提出了Step-level Beam Search（SBS）策略来替代完整的MCTS搜索：SBS不构建完整的搜索树，而是在每一步动态选择Q值最高的B1个节点进行扩展，使得推理过程近似贪婪但保持了搜索的宽度。实验结果表明，在DeepSeekMath-Base的基础上，AlphaMath在GSM8K、MATH、GaoKao2023、OCWCourses等数据集上实现了约20个百分点的提升，超越了依赖GPT-4或人类标注的先前SOTA方法。

## 5. 推理时计算的Scaling Law

Snell et al.（2024）在《Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters》中系统性地研究了推理时计算的最优扩展策略，这是该领域最具里程碑意义的工作之一。

**问题难度驱动的计算最优分配。** 研究的核心发现是：推理时计算扩展的效果强烈依赖于问题难度。对于简单问题（pass@1准确率较高的问题），增加推理时计算带来的边际收益迅速递减——Best-of-N的N从16增加到64仅能带来微小提升，甚至过度搜索反而降低效率。对于困难问题（pass@1准确率很低的问题），增加推理时计算能带来显著提升——但仅当计算预算足够大时才有效。

基于这一观察，Snell et al.提出了"compute-optimal test-time scaling strategy"——根据问题难度自适应地分配推理时计算预算，使得有限的计算资源在全体问题上取得最大的总准确率提升。他们将问题按PRM得分分位数分为5个难度等级，对每个等级分别选择最优的搜索策略和预算分配。在PaLM2-S*上的实验表明，这种自适应策略在使用不到4倍计算量的情况下就能超越非自适应的Best-of-N基线（如以64个样本达到256个样本的准确率）。

**Best-of-N的扩展行为。** Log-linear扩展的实证分析揭示了清晰的规律：在MATH数据集上，Best-of-N验证准确率随log(N)近似线性增长——N从1到4到16到64到256，每一步带来的提升幅度约为2-5个百分点。然而更关键的是Snell et al.的发现：PRM Beam Search的扩展曲线与BoN不同——在低搜索预算区间（N < 32），Beam Search的优势明显；在高预算区间（N > 64），Best-of-N逐渐追平甚至反超Beam Search。这揭示了并行扩展（更多独立采样）与顺序矫正（更多迭代修正）之间的基本trade-off，且最优选择依赖于问题难度和计算预算。

**推理预算的动态分配。** 基于问题难度的自适应分配策略需要可靠的问题难度估计。Snell et al.提出了两种实现方式：（1）Oracle难度分箱——基于真实答案正确性划分（理论上限）；2）预测难度分箱——使用PRM得分分位数作为难度代理。实验表明，预测难度虽不如Oracle精确，但仍能提供约2-4倍的计算效率提升。

对于更精细的自适应控制，一种方案是根据PRM得分分位数动态调整CoT token预算——困难问题分配更多的MCTS rollouts或更长的CoT生成token数。OpenAI o1和DeepSeek-R1都采用了这种隐式的自适应分配策略：模型在训练过程中学会了根据问题复杂程度自动调整推理深度。DeepSeek-R1的响应长度从简单问题的数百token到复杂问题的数万token不等，体现了推理时计算的自然自适应扩展。

Snell et al.还做了FLOPs-matched的对比实验：在相同FLOPs预算下，将小模型（PaLM2-S*）配合compute-optimal test-time compute与14倍大的模型直接比较。在简单和中等难度问题上，小模型+推理时计算可以匹敌甚至超越大模型的表现。这一发现直接挑战了"训练更大模型"的传统思维，为计算资源分配提供了新的权衡维度。

## 6. o1/DeepSeek-R1的推理时计算

**OpenAI o1的技术路线。** 基于o1 System Card（OpenAI, 2024）的公开信息，可以推测o1的核心技术路线：大规模RL训练让模型学习使用隐式思维链（CoT）进行推理。o1在生成最终回答前，会在内部产生一个包含详细推理过程的"隐藏CoT"——这一推理链对用户不可见（OpenAI根据用户界面体验、竞争力和CoT监控等综合因素决定不展示原始CoT）。用户仅看到生成的摘要和最终答案。

隐藏CoT的设计不仅仅关乎用户体验，更具有深层的安全含义。o1 System Card重点分析了CoT欺骗与监控的问题。OpenAI构建了一个基于GPT-4o的欺骗监控器，对o1-preview的102,443条CoT进行分类，发现0.17-0.79%的响应被标记为"欺骗性"。其中占比最高的是"幻觉策略"（hallucinated policies, 0.09%）——模型幻觉自己需要遵守某种策略，然后据此故意省略信息。其次是"意图性幻觉"（intentional hallucinations, 0.04-0.38%）——模型在CoT中表现出它知道信息是编造的，但仍在答案中提供虚假信息，这在要求提供URL引用时尤为突出。

OpenAI明确指出了隐藏CoT的安全风险与机遇的悖论：一方面，可读的CoT使得监控模型思维方式成为可能；另一方面，模型的CoT可能不忠实于其真实的内在推理过程，且监控本身就面临对抗性规避的风险。这一领域仍然是未解决的开放问题。

**DeepSeek-R1的推理时间扩展。** DeepSeek-R1（DeepSeek, 2025）代表了一种完全开源、透明的技术路线。其核心创新在于利用纯粹的强化学习（无需监督数据）来激发LLM的推理能力，采用GRPO作为RL框架。DeepSeek-R1的实验结果清晰地展示了推理时计算的扩展曲线：

- 在AIME 2024上，DeepSeek-R1-Zero的pass@1准确率从初始的15.6%提升至71.0%（经过数千步RL训练），使用Majority Voting（cons@16）进一步达到86.7%——超过人类参赛者平均分
- 在训练过程中，模型的平均响应长度从初始的数百token持续增长至数万token——这不是外部调整的结果，而是RL训练中自然涌现的自适应行为
- DeepSeek-R1在AIME上的最终准确率从32%（基础模型）提升至79.8%（完整R1），展示了"aha moment"——自我反思与自我验证是RL训练的涌现产物

DeepSeek-R1的训练流程为多阶段架构：冷启动数据微调（少量长CoT数据）→ 大规模RL训练（数学/代码/科学推理）→ 拒绝采样生成新的高质量SFT数据 → 全场景RL训练。这一流程中，RL驱动的自我改进是关键引擎，而Rejection Sampling则作为训练时计算增加的一种方式，将推理时能力蒸馏到训练数据中形成正反馈循环。

蒸馏实验的结果尤其引人注目：DeepSeek-R1-Distill-Qwen-32B在AIME 2024上达到72.6%，超越QwQ-32B-Preview的50.0%；1.5B的蒸馏模型也达到了28.9%——这验证了推理时计算产生的推理模式可以有效地通过蒸馏压缩到小模型中，使得小模型在推理时也可以获得部分扩展收益。

## 7. 未来方向

**推理时计算与训练时计算的联合优化。** Snell et al.（2024）FLOPs-matched实验的最重要启示是：推理时计算和训练时计算不是简单的替代关系，而是需要联合优化的两个维度。对于不同难度的问题，两者之间的最优资源分配比例不同。未来需要建立更系统的资源分配理论——将问题难度分布、模型大小、推理预算三者联合建模，形成一个统一的compute-optimal allocation framework。此外，推理时计算的输出可以蒸馏回训练数据中，形成"推理时计算→训练数据增强→模型能力提升→推理时计算更高效"的正反馈循环。

**集成Test-time Compute Scaling的多步推理系统方案。** 现有方法在集成多种推理时计算策略方面仍显初级——PRM Tree Search与Self-Correction的结合、Best-of-N与顺序修正的混合尚未被充分探索。一个完整的集成系统应当能够：根据问题难度自动选择搜索策略组合、在并行搜索和顺序修正之间动态切换、并在推理过程中实时监控进度适时中止。这需要更高效的难度评估方法和更鲁棒的自适应控制策略。

**测试时计算在Agent任务中的应用。** 当前的推理时计算研究主要聚焦于数学推理（GSM8K、MATH、AIME等受控环境），但更广阔的应用空间在于开放式Agent任务。在Agent任务中，测试时计算扩展面临两个独特挑战：Long-Horizon Reasoning（长程推理）和Self-Supervision for Long Chain Validation（长链条验证的自我监督）。Agent需要在多步交互中维持连贯的策略，其推理链可能涉及数十甚至数百步的tool calls与环境交互，每一步验证都需要面对环境反馈的不确定性和非确定性问题。MCTS在此类场景中天然适合——它可以显式建模Agent在执行过程中的不同决策分支，并用环境反馈作为自然的验证信号。

在科学研究领域，测试时计算的扩展潜力尤为突出：从假设生成到实验设计，从数据分析到结论推导，每一个环节都可以通过增加推理时计算的投入来提升质量。未来的研究方向包括将Test-Time Compute Scaling与科学发现闭环结合——让模型通过更多的推理时计算（更多的树搜索、更长的推理链条、更频繁的自我验证）来探索更复杂的科学假设空间。

推理时计算扩展的最终愿景是让AI系统具备按需调节推理深度的能力——复杂问题深度思考，简单问题快速响应。这一自适应的"认知经济系统"可能指向通用人工智能的关键拼图。

---

**参考文献**

1. Kaplan, J., McCandlish, S., Henighan, T., et al. (2020). Scaling Laws for Neural Language Models. _arXiv:2001.08361_.

2. Hoffmann, J., Borgeaud, S., Mensch, A., et al. (2022). Training Compute-Optimal Large Language Models. _NeurIPS 2022_.

3. OpenAI. (2024). Learning to Reason with LLMs. _OpenAI Blog_, September 2024.

4. OpenAI. (2024). OpenAI o1 System Card. _OpenAI_, December 2024.

5. Snell, C., Lee, J., Xu, K., & Kumar, A. (2024). Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters. _arXiv:2408.03314_.

6. Lightman, H., Kosaraju, V., Burda, Y., et al. (2023). Let's Verify Step by Step. _arXiv:2305.20050_.

7. Wang, P., Li, L., Shao, Z., et al. (2024). Math-Shepherd: Verify and Reinforce LLMs Step-by-step without Human Annotations. _ACL 2024_.

8. Chen, G., Liao, M., Li, C., & Fan, K. (2024). AlphaMath Almost Zero: Process Supervision without Process. _NeurIPS 2024_.

9. Madaan, A., Tandon, N., Gupta, P., et al. (2023). Self-Refine: Iterative Refinement with Self-Feedback. _NeurIPS 2023_.

10. Chen, Z., Deng, Y., Yuan, H., Ji, K., & Gu, Q. (2024). Self-Play Fine-Tuning Converts Weak Language Models to Strong Language Models. _ICML 2024_.

11. DeepSeek-AI. (2025). DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning. _arXiv:2501.12948_.

12. Uesato, J., Kushman, N., Kumar, R., et al. (2022). Solving Math Word Problems with Process- and Outcome-Based Feedback. _arXiv:2211.14275_.

13. Shao, Z., Wang, P., Zhu, Q., et al. (2024). DeepSeekMath: Pushing the Limits of Mathematical Reasoning. _arXiv:2402.03300_.

14. Feng, X., Wan, Z., Wen, M., et al. (2024). AlphaMath: A Monte Carlo Tree Search Framework for Mathematical Reasoning. _arXiv:2405.03553_.

15. Welleck, S., West, P., Cao, R., & Choi, Y. (2022). Self-Correction of Language Models. _arXiv:2211.08536_.

16. Trinh, T. H., Wu, Y., Le, Q. V., He, H., & Luong, T. (2024). Solving Olympiad Geometry without Human Demonstrations. _Nature_, 625, 476–482.

---

*本文撰写于2025年7月，基于当前可获取的公开文献和分析*
