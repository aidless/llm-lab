# AGI 深挖——Agent 系统架构

> 从Tool Use到Self-Improving Loop——2026年Agent工程全景
> 涵盖：Claude Code / GPT-5.6 Agent / Gemini Agent / DeepSeek Coder Agent / Stanford AgentFlow

---

## 引子：2026年是"Agent之年"

2026年，Agent从"有趣的概念验证"变成了"AI产业的主要使用形态"。

关键标志：
- OpenAI报告：ChatGPT 65%的tokens现在来源于Agent模式（非直接对话）
- Anthropic：Claude Code在企业开发团队中的采用率超过GitHub Copilot（2026年Q2）
- METR评估显示：自主AI的"任务完成时域"（autonomous task horizon）每4-5个月翻倍
- 多家Agent框架标准化：来自斯坦福的AgentFlow、Anthropic的MCP协议、OpenAI的GPT Actions统一协议

---

## 一、Agent系统的核心架构组件

无论具体实现如何，2026年的Agent系统都由以下五个核心组件构成：

```
┌─────────────────────────────────────────────────────┐
│                   Coordinator                        │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌────────┐ │
│  │  Planner  │ │  Executor │ │  Memory  │ │ Monitor│ │
│  │ (分解任务) │ │ (调用工具) │ │ (上下文)  │ │ (质量)  │ │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘ └───┬────┘ │
│       │             │            │            │       │
│  ┌────▼─────────────▼────────────▼────────────▼─────┐ │
│  │                LLM Core (模型)                     │ │
│  └──────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────┘
```

### 1.1 Planner（任务分解器）

Planner负责将用户的高层指令（"写一个电商网站"）分解为可执行的任务序列。

**2026年SOTA设计**：

**基于ReAct的Tree-of-Thought Planner**
- 不是简单地生成固定步骤
- 在每一步生成3-5个可能的action，评估每个action的预期收益
- 使用BFS/DFS搜索action空间
- 缺点：计算开销大（每次决策需要多次LLM调用）

**逐层分解策略（Hierarchical Task Decomposition）**
- 第一层：将任务分解为阶段（planning/research/coding/testing）
- 第二层：每个阶段分解为子任务
- 第三层：每个子任务分解为具体操作
- 这种分层大幅提升了计划的精确度

**回退与重新规划**
- 这是2026年最大的改进点
- 当subtask失败时，Planner不仅要报告失败，还要分析失败原因并重新规划
- 关键技术：**Reflection Mechanism**（自我反思）——模型在执行后分析自己的输出质量

### 1.2 Executor（工具调用器）

执行器负责调用各种工具/API完成具体任务。

**工具发现的标准化——MCP协议**
- Anthropic推出的**Model Context Protocol (MCP)**在2026年成为行业标准
- 被OpenAI、Google、Microsoft、GitHub全部采纳
- 工具被注册为标准化的MCP Server，模型通过统一接口调用
- 发现（discovery）：模型启动时自动拉取可用工具列表
- 调用（invocation）：JSON-RPC 2.0格式
- 认证（auth）：OAuth 2.0 Device Flow

**2026年工具调用的关键改进**：
1. **Parallel Tool Calling**：模型可以同时发起多个独立工具调用（大幅减少延迟）
2. **Dependency-aware Scheduling**：工具调用之间有依赖（如先搜索再存结果），系统自动排序
3. **Dynamic Tool Composition**：模型可以组合多个工具完成复杂任务（类似Unix管道）
4. **Tool Output Streaming**：大结果可以流式返回，不用等全部完成

**安全边界**：
- **Tool permission sandboxing**：每个工具有独立的权限控制
- **Confirmation gates**：高危操作（写文件、发邮件、删除资源）需要用户确认
- **Tool output validation**：检查工具返回数据是否合理的范围

### 1.3 Memory（记忆系统）

Agent记忆不再只是一个聊天历史。2026年的Agent有**三层记忆架构**：

```
短期记忆 (Working Memory)
  ├── 当前对话上下文（所有消息）
  ├── 当前task栈（正在进行的任务及其状态）
  └── 当前文件的缓存（代码、文档片段）
  
中期记忆 (Episodic Memory)
  ├── 过去的任务记录（完成了什么、怎么做的）
  ├── 用户偏好（用户总是要求什么）
  └── 已解决的错误方案（死路记录）
  
长期记忆 (Semantic Memory)
  ├── 知识图谱（实体、关系、概念）
  ├── 学习到的模式（"这类问题的最佳实践"）
  └── 技能库（已经学会的工具使用模式）
```

**2026年的关键创新**：
- **Memory Synthesis**：不是存储原始对话，而是Agent定期"合成"记忆为更紧凑的表示
- **Hierarchical Memory Retrieval**：先搜索长期记忆摘要，再精确定位到具体内容
- **Forgetting Mechanism**：Agent主动遗忘不再相关的内容（类似睡眠中的记忆整理）

### 1.4 Monitor（质量监控器）

Monitor对Agent的输出进行持续质量评估：

**检查维度**：
1. **Task completion**：任务是否真的完成了？
2. **Correctness**：输出是否正确（语法、逻辑、事实性）？
3. **Safety**：是否包含不安全内容？
4. **Efficiency**：是否做了不必要的工作？

**实现方式**：
- 基于critic model的自动评分
- 与用户期望的对齐检查（"用户真的想要这个吗？"）
- Token budget控制（限制无意义的来回操作）

---

## 二、Claude Code：2026年最成熟的代码Agent

作为2026年最广泛使用的代码Agent，Claude Code的设计代表了Agent工程的最高水平。

### 2.1 核心架构

```
┌────────────────────────────────────────────────┐
│                  Claude Code                    │
│  ┌──────────────────────────────────────────┐   │
│  │          Planning Layer                   │   │
│  │  ┌──────┐ ┌──────────┐ ┌────────────┐    │   │
│  │  │Slash │ │ Subagent │ │  Tool      │    │   │
│  │  │Cmd   │ │ Dispatch │ │  Selection │    │   │
│  │  └──────┘ └──────────┘ └────────────┘    │   │
│  └──────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────┐   │
│  │          Execution Layer                  │   │
│  │  ┌─────────┐ ┌──────┐ ┌───────┐         │   │
│  │  │ Terminal│ │Editor│ │File  │         │   │
│  │  │  (安全) │ │(diff)│ │System│         │   │
│  │  └─────────┘ └──────┘ └───────┘         │   │
│  └──────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────┐   │
│  │          Memory Layer                     │   │
│  │  ┌──────────┐ ┌──────────┐               │   │
│  │  │ Context  │ │Project   │               │   │
│  │  │ Window   │ │Memory    │               │   │
│  │  └──────────┘ └──────────┘               │   │
│  └──────────────────────────────────────────┘   │
└────────────────────────────────────────────────┘
```

### 2.2 关键工程决策

**1. 原子编辑（Atomic Edits）**
- Claude Code的任何代码修改都是原子性的
- 编辑前自动保存原始内容，失败时回滚
- 每次编辑生成diff，用户可以逐块确认

**2. 安全执行（Safe Execution）**
- 所有terminal命令在沙箱中执行
- 高危操作（install、rm、deploy）需要用户确认
- 自动检测命令是否在阻塞等待

**3. Subagent模式**
- 复杂任务自动分解为子任务分配给subagent
- 每个subagent有独立的上下文窗口
- subagent完成后将结果汇总到主agent
- 类似"经理-员工"模式：经理不写代码，只协调

### 2.3 Claude Code对Agent工程的贡献

Claude Code的理念深刻影响了整个Agent领域：
- **MCP协议**：支持动态工具发现和注册
- **Task作为第一公民**：不是没有状态的对话，是有状态的任务管理系统
- **Diff-review loop**：Agent生成→用户审查→修改→确认→提交

---

## 三、AgentFlow（斯坦福）：异步Agent DAG

2026年来自斯坦福的AgentFlow代表了Agent系统的另一种范式——**DAG驱动的异步Agent编排**。

### 3.1 核心思想

不是"一个agent一步步做事"，而是"多个agent并行和串行组合成一个工作流"。

```
AgentFlow DAG 示例（代码审查）:
  代码变更 → [静态检查Agent] → [安全检查Agent] → [合并] → [部署Agent]
                    ↓                              ↓
              [风格审查Agent] ← → [测试Agent] → [报告Agent]
```

### 3.2 关键技术点

**确定性DAG vs LLM驱动的DAG**：
- AgentFlow不鼓励LLM动态决定下一步——这会导致不可预测的行为
- 相反，整个工作流由**确定性DAG定义**，LLM只在节点内部使用
- 节点之间的连接由代码（而非模型）控制

**Agent组合子**：
- `Sequential(A → B)`: A执行完毕后B执行
- `Parallel(A, B, C)`: A/B/C并行执行，全部完成才继续
- `Race(A, B)`: A和B同时执行，取最快返回的结果
- `Retry(A, N)`: A执行N次直到成功
- `MapReduce(A, [items ...])`: 为每个item启动一个A实例，然后汇总

### 3.3 AgentFlow vs Claude Code模式

| 维度 | AgentFlow (DAG) | Claude Code (计划+执行) |
|------|----------------|----------------------|
| 控制流 | 确定性DAG | LLM自主决策 |
| 并行 | 原生支持 | 通过subagent实现 |
| 可预测性 | ★★★★★ | ★★★☆☆ |
| 灵活性 | ★★★☆☆ | ★★★★★ |
| 适合场景 | 生产流水线 | 探索性开发 |

---

## 四、OpenAI Codex Agent / GPT Actions

### 4.1 Codex Agent（2026年重构版）

OpenAI在2026年对Codex Agent做了重大重构：

**架构变更**：
- 从单一模型调用 → 多阶段pipeline
- 从文本终端 → 浏览器化的Agent界面
- 从单一会话 → 持久化项目上下文

**核心创新**：
- **Codex Sandbox**：每个项目有独立的容器化运行环境
- **Git-aware Agent**：自动commit、branch、diff
- **Knowledge Base**：agent在项目中积累的知识持久化

### 4.2 GPT Actions与MCP的融合

2026年，OpenAI的GPT Actions和Anthropic的MCP协议统一了。关键变化：

**统一协议**：
- 任何MCP Server都可以作为GPT Action注册
- 任何GPT Action也可以暴露为MCP Server
- 模型<->工具的通信格式统一为JSON-RPC 2.0

**认证标准**：
- OAuth 2.0 Device Flow成为行业推荐
- 支持API Key、JWT、OAuth三种模式

**工具市场**：
- OpenAI、Anthropic、Google共享工具注册表
- 任何开发者可以发布MCP Server
- 工具可以跨平台使用（一个工具在Claude Code和GPT中都能用）

---

## 五、Multi-Agent编排：2026年的系统工程前沿

### 5.1 为什么Multi-Agent？

单个Agent的瓶颈：
- 上下文窗口有限（即使在10M token下）
- 单模型推理的token预算限制
- 无法同时执行独立任务

Multi-Agent的优势：
- 并行执行独立任务（倍速提升）
- 专业分工（一个agent负责搜索，一个负责分析）
- 独立上下文（每个agent有自己的上下文窗口）
- 审校机制（一个agent写、另一个检查）

### 5.2 主流Multi-Agent模式

**模式1：经理-工人（Manager-Worker）**
- 一个经理agent负责任务分解和进度跟踪
- 多个工人agent并行执行子任务
- 经理agent汇总结果

**模式2：辩论（Debate）**
- 多个agent独立分析同一问题
- 对比分析结果
- 选择最优方案（或综合多个方案）

**模式3：流水线（Pipeline）**
- 每个agent负责流水线中的一个环节
- 前一个agent的输出是下一个agent的输入
- 适合结构化流程（如数据分析报告）

### 5.3 2026年的多Agent治理问题

随着agent数量增多，新的治理问题浮现：

**协调开销**：
- 每增加一个agent，通信开销增加约30%
- 超过5个agent后，协调成本开始超过并行收益
- 经验法则：**2-4个agent是最优的**

**上下文污染**：
- 一个agent的错误可能被其他agent采纳
- 需要独立的验证步骤
- Agent之间的"集体幻觉"是2026年新发现的问题

**收敛问题**：
- 辩论模式可能永远达不成共识
- 需要确定性的终止条件
- "最大迭代次数"和"共识阈值"是关键超参

---

## 六、2026年Agent架构的核心趋势

### 6.1 MCP成为行业标准

MCP协议在2026年的普及率极高：
- 所有主流LLM provider支持（OpenAI、Anthropic、Google、DeepSeek）
- 所有主流IDE集成（VS Code、JetBrains、Cursor）
- GitHub Actions、Vercel、Cloudflare等平台原生支持

### 6.2 从"Agent"到"Agent System"

2026年的认知提升：
- 一个单独的模型调用不构成Agent
- Agent = Model + Tools + Memory + Planner + Monitor
- 整个系统需要作为一个整体来设计和测试

### 6.3 安全与信任

Agent的安全成为2026年的焦点：
- **Tool permission escalation**：攻击者可能通过Agent间接调用高危工具
- **Prompt injection via tool output**：工具返回的数据可能包含恶意指令
- **Model hijacking**：长期运行的Agent可能被劫持
- **Data exfiltration**：Agent可能被诱导读取敏感数据并外传

应对措施：
- 分层权限模型（工具级别、数据级别、操作级别）
- 输出验证（每个工具调用的结果都经过validator检查）
- 用户确认网关（高危操作必须用户手动确认）

### 6.4 METR评估的趋势

METR（Model Evaluation and Threat Research）的持续评估显示：
- 2024年Q1：Agent能完成5分钟的人类任务
- 2025年Q4：Agent能完成45分钟的人类任务
- 2026年Q2：Agent能完成2-3小时的人类任务
- 每4-5个月自主任务时域翻倍

如果这个趋势持续：
- 2027年：8-24小时的人类任务
- 2028年：1-2周的人类任务
- 2029年：月级别的长期任务

---

## 七、Agent工程的核心挑战（未解决）

2026年，以下问题仍在研究中：

**1. 错误恢复**
- 当Agent执行了一个错误操作后，如何自我修复？
- 目前方案：回滚到上一个checkpoint
- 理想方案：部分回滚 + 修正操作

**2. 长期任务**
- 超过8小时的Agent任务如何保持一致性？
- 模型在长时运行中的"注意力漂移"（attention drift）不可忽视
- 需要定期重新评估目标的机制

**3. 跨Agent信任**
- 如何确保两个Agent之间的通信不被篡改？
- 目前方案：所有agent间通信记录到不可变log
- 不够：需要加密验证

**4. Agent与人类的协作模式**
- Agent何时应该问人类？何时应该自主决定？
- "问太多"vs"不问"的平衡
- 个性化策略：根据用户偏好调整自主程度

---

## 结语：Agent架构不会收敛到单一答案

2026年的Agent工程与2023年的模型架构一样——处于"百花齐放"的阶段：
- Claude Code代表**灵活**路线（LLM主导决策）
- AgentFlow代表**确定**路线（代码主导流程）
- GPT Actions代表**平台**路线（生态主导标准）
- 三种路线各有适用场景

Agent的终极形态可能需要：
1. 确定性的骨架（DAG定义整体流程）
2. 灵活性的节点（LLM在节点内自主决策）
3. 标准的生态（MCP作为工具通信协议）

这三者的结合——**不是互相替代，而是互补**。
