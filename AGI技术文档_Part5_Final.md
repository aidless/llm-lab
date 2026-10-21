# 人工通用智能（AGI）技术综述：最终增补

## AGI 研究的关键未解决问题清单

以下是在 Part 1-4 基础上，对 AGI 研究中最关键未解决问题的汇总与评级：

### Level 1：核心理论问题（AGI 的科学基础）

1. **人类通用智能的实质是计算的吗？** 即使我们接近形式化人类认知的某些方面，但"通用智能"是否可以完全在计算框架中实现？这是计算主义的根本问题，也是 AGI 项目的"形而上学"基础。

2. **什么是理解的可靠测试？** 目前没有可接受的"理解测试"。"图灵测试"——一个有 70 年历史的启发——是不充分的，但没有一个替代方案获得广泛共识。

3. **是否存在 AGI 的唯一算法？** 还是 AGI 必须由多个具有异构机制的子组件构成？"统一 vs 多元"的争论在 AGI 社区中没有共识。

### Level 2：工程与结构性问题（AGI 的系统设计）

4. **知识表示的最优形式**：符号级的、亚符号级的（分布式的、连续的）、还是混合的？每种形式都有自己的优势和弱点。一个跨模态统一知识表示仍不可预见。

5. **学习与推理的关系**：学习产生新的知识、规则和新结构；推理操纵现有的知识。但它们的相互关系——推理为学习提供"注意力方向"，而学习为推理提供"规则"——仍然难以形式化。

6. **持续学习与灾难性遗忘**：当 AGI 系统学习新任务或适应新环境时，它将面临灾难性遗忘（catastrophic forgetting）的问题——旧的知识被新知识"重写"。这个问题在深度学习中已经被广泛研究（Kirkpatrick et al., 2017），但还没有干净的解决方案。

7. **缩放极限与人机协作的超越**：在数据瓶颈和计算成本的双重约束下，缩放策略能走多远？在缩放不再具有成本效益后，我们将使用什么样的替代方法？

### Level 3：评估与方法论问题（如何知道 AGI 即将到来）

8. **基于基准的评估是否具有预测性**？在一组基准上表现良好的系统是否必然在 AGI 任务上表现良好？还是基准分数只是在"迷惑我们"？

9. **如何设计反饱和基准**？我们如何才能创建一个基准框架——它有尽量少的"饱和"信号，并持续测试系统能力边界？

10. **人类比较的校准问题**：如何在一个日益复杂、专业化的基准环境中适当校准人类的表现？

### Level 4：安全与价值问题（AGI 对齐）

11. **从人类数据中对齐的价值**：通过"人类反馈"（RLHF）来对齐的系统是否具有"正确的"价值观？在实践中，由有限标注员提供的反馈可能会引入偏见、不一致和不稳定性。

12. **价值演化**：如果 AGI 的目标函数是静态的（在训练完成后被冻结），那么如果用它来操作持续进化的人类社会——人类价值观在代际间深刻变化——它会变得过时吗？

13. **可扩展的监督**：随着 AI 系统在更复杂、更高级的任务上"超越"了人类，人类如何保持对齐的监督能力？这是"对齐的终极问题"——也是 Bostrom 所说"控制问题"的变体。

## AGI 时间线预测元分析

基于 RAND 2026 年报告、arXiv:2405.10313（"How Far Are We From AGI"）、行业专家调查以及第二篇 ARC-AGI 报告的数据，以下是 AGI 时间线预测的元分析：

| AGI 层级 | 最乐观（10% 概率） | 中枢估计 | 最悲观（90% 概率） | 来源 |
|---------|------------------|--------|------------------|------|
| Level 1 (Emerging) | 2026-2027 | 2027-2030 | 2033 年之后 | 多家机构共识 |
| Level 2 (Competent) | 2028-2030 | 2033-2047 | 2060+ | RAND, 专家调查 |
| Level 3 (Expert) | 2033-2038 | 2045-2060 | 2100+ | RAND |
| Level 4 (Virtuoso) | 2040-2050 | 2060-2090 | 从未 | 推测性 |
| Level 5 (ASI) | 2045-2060 | 2070+ | 从未 | 高度推测性 |

**中心预测**：Level 2（Competent AGI）作为"许多人认为 AGI 真正到来"的标志，最可能出现在 2033-2047 年之间。这个范围兼顾了"缩放将继续带来能力进步"和"需要根本性突破"两种观点。

**需要注意**：所有的预测都有很大的不确定性。ARC-AGI-3 的结果——展示了"在交互式设置中的意外弱点"——可能比许多人预期的更慢地达到 Level 2。相反，如果在组合泛化方面取得根本突破，Level 2 可能比现在预期的更快到来。

## AGI 情景分析

### 情景一：渐进式整合（概率约 40%）

在这个情景中，AGI 将通过渐进整合当前技术（LLM+高级推理+认知架构）的方式实现。2020 年代的多代理、复合系统在 2030 年代进化到更集成的"通用认知架构"。Level 2 AGI 在 2035-2045 年间实现。这种情景以"迭代工程进步"为特征，而不是单一的"突破"。

经济影响是渐进的，但真正的生产力增长——使大多数行业发生根本性转变——需要到 Level 3 级别。

### 情景二：快速突破（概率约 15%）

在这个情景中，一些新的、目前无法预见的技术突破——在组合泛化、持续学习、符号-亚符号集成方面——在一个单一代理中实现 Level 2-3 的 AGI。

经济影响将"如闪电般"发生，在 5-10 年内使大多数职业过时。这可能会引发大规模的社会动荡和体制转换。

快速突破情景具有很高的风险：如果价值对齐不完美，并且系统以"代理级"或以上的能力运行，对齐的失败可能具有严重的存在后果。

### 情景三：缩放缓慢（概率约 30%）

在这个情景中，缩放定律被证明不具有扩展到人类级别 AGI 的能力。尽管在特定任务上持续、渐进的改进，但"通用智能"仍然非常微弱。Level 2 可能需要几十年的时间，发生在 2050 年代或更晚。

社会影响是有管理的和渐进的，与 AI 的实际能力保持"软匹配"。这种情景允许充分的时间来发展治理和反思框架。

### 情景四：AGI 不可能（概率约 15%）

在这个情景中，证明或反对 AGI 的不可实现性被"实现"——智能需要生物基础（如意识、体会或某种未知的生物物理过程），而这些不能被在硅基计算系统中复制。

这种情景的概率被大多数研究者认为是很小的。但如果它发生了，它将导致实质性的、关于 AI 发展的基础假设的"范式转变"。

## 最终反思："AGI"这个词是否已经过时了？

一个根本的问题：在 2020 年代后期——当"AI"系统已经能够写论文、编码、做数学、编程、创造——"AGI"这个词是否已经失去了它的分析价值？

也许。在本文的分析中，我们使用了 Morris 的层级框架和 Chollet 的技能-获取效率框架来定位当前的 AI 系统——这可能是比单维度的"AGI 是或否"更有用的分类方法。

如果我们继续使用"AGI"这个术语，我们应该用一个清晰的、多维度的框架来界定它——类似于 Morris 的 "Levels of AGI" 或一篇新的综述论文所提议的评估框架。

总的来说，AGI 研究正在走向"后冲击"时代——建立在深度学习巨大成功的基础上，但越来越认识到需要根本性的新技术和能力。无论我们是否在"AGI 的道路上"，研究正在变得越来越有趣、越来越多样化、越来越具有影响力。真正的 AGI——如果它即将到来——将是工程、科学和人文的杰出成就。

---

## 最终参考文献索引

[1] Wang, P. (2019). "On Defining Artificial Intelligence." JAGI.

[2] Morris, M.R., et al. (2023). "Levels of AGI." arXiv:2311.02408.

[3] Chollet, F. (2019). "On the Measure of Intelligence." arXiv:1911.01547.

[4] Hutter, M. (2005). "Universal Artificial Intelligence."

[5] Goertzel, B. (2023). "OpenCog Hyperon Framework."

[6] Schmidhuber, J. (2009). "Ultimate Cognition à la Gödel."

[7] Chollet, F. & Knoop, M. (2026). "ARC-AGI-3 Technical Report." arXiv:2603.24621.

[8] Knoop, M. (2025). "ARC-AGI Prize 2025 Results Analysis."

[9] RAND Corporation (2026). "AGI Forecasting."

[10] Curtis, C. & Regli, W. (2025). "Neuro-Symbolic AI: A Systematic Review." arXiv:2501.05435.

[11] Kirkpatrick, J., et al. (2017). "Overcoming Catastrophic Forgetting in Neural Networks." PNAS 114(13):3521-3526.

[12] Russell, S. (2019). "Human Compatible."

[13] Bostrom, N. (2014). "Superintelligence."

[14] Searle, J. (1980). "Minds, Brains, and Programs."

[15] Harnad, S. (1990). "The Symbol Grounding Problem."

[16] Laird, J. (2012). "The SOAR Cognitive Architecture."

[17] Fodor, J. & Pylyshyn, Z. (1988). "Connectionism and Cognitive Architecture."

[18] Brooks, R. (1991). "Intelligence Without Representation."

[19] Kahneman, D. (2011). "Thinking, Fast and Slow."

[20] Baddeley, A. (2000). "The Episodic Buffer." Trends in Cognitive Sciences.

[21] Spelke, E. (2000). "Core Knowledge." American Psychologist.

[22] Piaget, J. (1936). "The Origins of Intelligence in Children."

[23] Cowan, N. (2001). "The Magical Number 4." BBS.

[24] Miller, G. (1956). "The Magical Number Seven." Psychological Review.

[25] Wei, J., et al. (2022). "Chain-of-Thought Prompting Elicits Reasoning." NeurIPS 2022.

[26] Schaeffer, R., et al. (2023). "Are Emergent Abilities of LLMs a Mirage?" arXiv:2304.15004.

[27] Brown, T., et al. (2020). "Language Models are Few-Shot Learners." NeurIPS 2020.

[28] Villalobos, P., et al. (2022). "Will We Run Out of Data?" arXiv:2211.04325.

[29] Shumailov, I., et al. (2023). "The Curse of Recursion." arXiv:2305.17493.

[30] Soares, N. (2015). "The Value Learning Problem."

[31] Hadfield-Menell, D., et al. (2016). "Cooperative IRL." NeurIPS 2016.

[32] Hubinger, E., et al. (2019). "Risks from Learned Optimization." arXiv:1902.01856.

[33] Armstrong, S., Bostrom, N. & Shulman, C. (2016). "Racing to the Precipice." AI & Society.

[34] Jimenez, C., et al. (2024). "SWE-bench." ICLR 2024.

[35] Hendrycks, D., et al. (2021). "MMLU." ICLR 2021.

[36] Rein, D., et al. (2023). "GPQA." arXiv:2311.12022.

[37] Silver, D., et al. (2016). "Mastering the Game of Go with Deep Neural Networks." Nature.

[38] Silver, D., et al. (2017). "Mastering Chess and Shogi." arXiv:1712.01815.

[39] Jumper, J., et al. (2021). "AlphaFold." Nature.

[40] Vaswani, A., et al. (2017). "Attention Is All You Need." NeurIPS 2017.

---

## 附录：全文结构索引

为了便于导航，以下是本完整综述文章（Parts 1-5）的树状索引：

```
第一部分：引言与基础概念
├── 1.1 AGI 的定义
├── 1.2 AGI 理论的比较：Morris > Chollet > Legg-Hutter
├── 1.3 主要论文与方法概述
├── 1.4 Morris 的六原则框架
├── 1.5 Chollet 的技能获取效率与 ARC
├── 1.6 Legg-Hutter 的通用智能
├── 1.7 三种定义的比较与综合
└── 1.8 历史背景

第二部分：理论基础
├── 2.1 信息论、算法信息论与 Kolmogorov 复杂度
├── 2.2 Kolmogorov 复杂度的计算
├── 2.3 组合泛化与泛化层次
├── 2.4 Legg-Hutter 的通用智能定义
├── 2.5 Solomonoff 归纳与通用预测
├── 2.6 AIXI 的深度剖析
├── 2.7 形式化理论遗漏的核心问题
├── 2.8 计算主义与智能的关系
└── 2.9 贝叶斯方法与智能推理

第三部分：技术路径
├── 3.1 符号主义 AI
├── 3.2 连接主义 (深度学习)
├── 3.3 混合与神经符号整合
├── 3.4 统一认知架构：OpenCog Hyperon
├── 3.5 统一认知架构：NARS
├── 3.6 符号主义 AI 的复兴
├── 3.7 统一认知架构：SOAR, Sigma, GWT
├── 3.8 LLM 和 LRM 的崛起
├── 3.9 扩展主义路径的根本困境
└── 3.10 符号主义与连接主义的深层整合路径

第四部分：基准与当前状态
├── 4.1 ARC-AGI 与抽象推理
├── 4.2 ARC-AGI-1, -2, -3 的演变
├── 4.3 GPT-5.6 Sol 的技术揭秘
├── 4.4 认知架构实践：短板分析
├── 4.5 AGI 时间线预测
├── 4.6 ARC-AGI 与 AIXI 的潜在连接
├── 4.7 超越 ARC：基准宇宙
├── 4.8 能力分布的不均衡性
├── 4.9 模型的潜在能力 vs 实际部署能力
└── 4.10 从 SWE-bench 看能力评估的陷阱

第五部分：未解决问题、哲学辩论与展望
├── 5.1 理解的本质问题
├── 5.2 组合泛化
├── 5.3 开放结局学习
├── 5.4 计算可行性差距
├── 5.5 中文屋论证与 AGI
├── 5.6 符号接地问题
├── 5.7 意识是 AGI 的必要条件吗？
├── 5.8 价值加载问题
├── 5.9 预测 AGI 时间线的陷阱
└── 5.10 缩放范式的尽头

第六至十章：深度扩展（Part 2-4）
├── 第六章：认知架构的全面比较
├── 第七章：超越基准测量的方法论批评
├── 第八章：扩展的治理与政策分析
├── 第九章：数学基础的深化
├── 第十章：最终的开放式问题
├── 第十一章：基准失败案例深度剖析
├── 第十二章：扩展历史视角
├── 第十三章：认知科学与 AGI 的深层联系
├── 第十四章：具体案例研究
└── 第十五章：AGI 评估的方法论总结

最终增补（Part 5）
├── AGI 研究的关键未解决问题清单
├── AGI 时间线预测元分析
├── AGI 情景分析
├── 最终反思
├── 综合参考文献索引
└── 全文结构索引
```
