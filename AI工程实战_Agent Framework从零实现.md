# AI 工程实战——Agent Framework 从零实现

> 从 LLM API 调用到完整的 Agent 框架：Tool System、Memory、Planning、Multi-Agent
> 2026年——Agent 是 AI 应用的主流范式

---

## 一、Agent 的核心架构

### 1.1 最小 Agent 实现

```
一个 Agent 的最小单元 = 感知 + 思考 + 行动

最简单的实现——单步工具调用:

class SimpleAgent:
    def __init__(self, llm, tools):
        self.llm = llm          # 语言模型
        self.tools = tools      # 工具列表
        self.messages = []      # 对话历史

    def step(self, user_input):
        # 1. 感知: 将用户输入添加到对话
        self.messages.append({"role": "user", "content": user_input})
        
        # 2. 思考: LLM 决定调用哪个工具或直接回复
        response = self.llm.chat(
            messages=self.messages,
            tools=[t.schema for t in self.tools]  # 工具描述
        )
        
        # 3. 行动: 如果 LLM 选择调用工具
        if response.tool_calls:
            for call in response.tool_calls:
                tool = self.get_tool(call.name)
                result = tool.execute(**call.args)
                self.messages.append({
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": str(result)
                })
            # 再次调用 LLM 生成最终回复
            response = self.llm.chat(messages=self.messages)
        
        self.messages.append({
            "role": "assistant",
            "content": response.content
        })
        
        return response.content
```

### 1.2 2026年 Agent 的完整架构

```
┌──────────────────────────────────────────┐
│              Agent 实例                    │
├──────────────────────────────────────────┤
│  状态层:                                  │
│  ┌─────────┐ ┌──────────┐ ┌───────────┐  │
│  │ Short-  │ │ Long-term │ │  Working  │  │
│  │ term    │ │ Memory    │ │  Memory   │  │
│  │ (对话)  │ │ (向量库)  │ │ (上下文)  │  │
│  └─────────┘ └──────────┘ └───────────┘  │
├──────────────────────────────────────────┤
│  推理层:                                  │
│  ┌──────────┐ ┌─────────┐ ┌──────────┐  │
│  │ Planner  │ │  Reason │ │ Decision │  │
│  │ (分解)   │ │ (CoT)   │ │ (选择)   │  │
│  └──────────┘ └─────────┘ └──────────┘  │
├──────────────────────────────────────────┤
│  工具层:                                  │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐  │
│  │ Tool     │ │ Tool     │ │ Tool     │  │
│  │ Registry │ │ Executor │ │ Sandbox  │  │
│  └──────────┘ └──────────┘ └──────────┘  │
├──────────────────────────────────────────┤
│  安全层:                                  │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐  │
│  │ Permission│ │ Input    │ │ Output   │  │
│  │ Check    │ │ Guard    │ │ Filter   │  │
│  └──────────┘ └──────────┘ └──────────┘  │
└──────────────────────────────────────────┘
```

---

## 二、核心组件的工程实现

### 2.1 Tool System

```
Tool Schema 定义 (使用 OpenAI/Anthropic 兼容格式):

{
    "type": "function",
    "function": {
        "name": "search_web",
        "description": "Search the web for information",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query"
                },
                "max_results": {
                    "type": "integer",
                    "description": "Max results to return"
                }
            },
            "required": ["query"]
        }
    }
}

Tool 注册系统:

class ToolRegistry:
    def __init__(self):
        self._tools = {}
    
    def register(self, name=None):
        """装饰器: 将函数注册为工具"""
        def decorator(func):
            tool_name = name or func.__name__
            schema = self._func_to_schema(func)
            self._tools[tool_name] = {
                "func": func,
                "schema": schema,
                "name": tool_name
            }
            return func
        return decorator
    
    def _func_to_schema(self, func):
        """将 Python 函数转换为 tool schema"""
        import inspect
        sig = inspect.signature(func)
        properties = {}
        required = []
        
        for name, param in sig.parameters.items():
            param_schema = {"type": self._type_to_json(param.annotation)}
            if param.default is inspect.Parameter.empty:
                required.append(name)
            else:
                param_schema["default"] = param.default
            properties[name] = param_schema
        
        return {
            "name": func.__name__,
            "description": func.__doc__ or "",
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required
            }
        }

# 使用示例
registry = ToolRegistry()

@registry.register()
def search_web(query: str, max_results: int = 5):
    """Search the web for information"""
    # 实际搜索实现
    return search_results

@registry.register()
def calculate(expression: str):
    """Evaluate a mathematical expression"""
    return eval(expression)  # 注意安全风险

Tool Executor (安全沙箱):

class ToolExecutor:
    def __init__(self, tools):
        self.tools = tools
    
    def execute(self, tool_name, **args):
        if tool_name not in self.tools:
            raise ValueError(f"Unknown tool: {tool_name}")
        
        tool = self.tools[tool_name]
        
        # 安全检查 (2026 标准)
        self._validate_args(tool, args)
        self._check_permission(tool_name)
        
        try:
            # 执行超时 (防止工具 hang)
            with timeout(seconds=30):
                result = tool["func"](**args)
            return {"status": "success", "result": result}
        except Exception as e:
            return {"status": "error", "error": str(e)}
```

### 2.2 Memory System

```
Agent 的 Memory 分为三层:

class AgentMemory:
    def __init__(self, vector_db=None):
        # 第1层: 短期记忆 (当前对话)
        self.short_term = []  # List[Message]
        
        # 第2层: 长期记忆 (向量数据库)
        self.vector_db = vector_db  # Chroma/Pinecone/Milvus
        
        # 第3层: 工作记忆 (当前任务的中间状态)
        self.working = {}  # Dict[str, Any]

Short-Term Memory (对话历史管理):

class ShortTermMemory:
    def __init__(self, max_tokens=32768):
        self.messages = []
        self.max_tokens = max_tokens
        self.token_count = 0
    
    def add(self, message):
        self.messages.append(message)
        self.token_count += self._count_tokens(message)
        
        # 如果超过限制，裁剪最早的对话
        while self.token_count > self.max_tokens:
            removed = self.messages.pop(0)
            self.token_count -= self._count_tokens(removed)
            
            # 保留 system prompt
            if self.messages and self.messages[0]["role"] == "system":
                pass  # system prompt 总是在开头
    
    def summarize(self):
        """压缩对话历史为摘要"""
        if not self._needs_summary():
            return self.messages
        
        summary_prompt = f"""Summarize the following conversation,
        keeping important information like user preferences, 
        decisions made, and key facts discussed:"""
        
        # 用 LLM 压缩对话
        summary = self._call_llm(summary_prompt + str(self.messages))
        
        # 用系统消息替换原始对话
        return [summary]

Long-Term Memory (记忆检索与存储):

class LongTermMemory:
    def __init__(self, vector_db, embedding_fn):
        self.db = vector_db
        self.embed = embedding_fn
    
    def store(self, content, metadata=None):
        """存储重要信息到长期记忆"""
        # 1. 生成 embedding
        embedding = self.embed(content)
        
        # 2. 存储到向量库
        self.db.add(
            embeddings=[embedding],
            documents=[content],
            metadatas=[metadata or {}]
        )
    
    def retrieve(self, query, top_k=5):
        """检索相关记忆"""
        query_embedding = self.embed(query)
        results = self.db.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        return results
    
    def auto_extract(self, messages):
        """自动从对话中提取值得记忆的信息"""
        # 使用 LLM 判断哪些信息值得记忆
        extraction_prompt = """Extract important facts from this conversation 
        that should be remembered for future interactions. 
        Only extract facts that are likely to be useful again."""
        
        # 分析最近的对话
        # ... (调用 LLM 提取)
        # ... (判断是否为新信息)
        # ... (存储到向量库)
```

### 2.3 Planning System

```
Planning 是 Agent 中最关键的组件之一：

class Planner:
    def __init__(self, llm):
        self.llm = llm
    
    def plan(self, task, context=None):
        """将任务分解为可执行的子任务"""
        
        # 生成计划
        plan_prompt = f"""
        Task: {task}
        Context: {context or 'None'}
        
        Break this task into steps. Each step should:
        - Be specific and actionable
        - List required tools
        - Specify expected output
        
        Format:
        Step 1: [action description]
        - Tool: [tool name]
        - Expected output: [description]
        
        Step 2: ...
        """
        
        plan_text = self.llm.generate(plan_prompt)
        plan = self._parse_plan(plan_text)
        
        return plan
    
    def replan(self, original_plan, completed_steps, current_state):
        """根据当前执行情况重新规划"""
        
        replan_prompt = f"""
        Original plan: {original_plan}
        Completed: {completed_steps}
        Current state: {current_state}
        
        Should we continue the plan, modify it, or create a new one?
        """
        
        decision = self.llm.generate(replan_prompt)
        return self._parse_decision(decision)

Hierarchical Planning (分层规划):

class HierarchicalPlanner:
    """
    高层规划: 将任务分解为几个大阶段
    中层规划: 每个大阶段分解为多个子任务
    低层规划: 每个子任务分解为具体工具调用
    """
    
    def plan_high_level(self, task):
        """高层: 2-5 个阶段"""
        return [
            "Research phase",
            "Implementation phase", 
            "Verification phase"
        ]
    
    def plan_mid_level(self, high_level_step):
        """中层: 每个阶段 3-8 个子任务"""
        pass
    
    def plan_low_level(self, mid_level_step):
        """低层: 每个子任务具体到 LLM call"""
        pass
```

---

## 三、Multi-Agent 编排

### 3.1 Agent 间的通信

```
Agent 间消息协议 (参考 A2A):

class AgentMessage:
    def __init__(self, sender, receiver, msg_type, content):
        self.sender = sender    # 发送 Agent ID
        self.receiver = receiver  # 接收 Agent ID
        self.type = msg_type    # request/response/error/broadcast
        self.content = content
        self.timestamp = time.time()
        self.message_id = uuid.uuid4()

class AgentBus:
    """Agent 间的消息总线"""
    def __init__(self):
        self.agents = {}       # {agent_id: Agent}
        self.message_queue = []  # 消息队列
    
    def register(self, agent):
        self.agents[agent.id] = agent
        agent.bus = self
    
    def send(self, message):
        self.message_queue.append(message)
        if message.receiver in self.agents:
            self.agents[message.receiver].receive(message)
    
    def broadcast(self, message):
        """广播到所有 Agent"""
        for agent_id, agent in self.agents.items():
            if agent_id != message.sender:
                agent.receive(message)
```

### 3.2 Supervisor/Worker 模式实现

```
class SupervisorAgent(Agent):
    def __init__(self, llm, workers):
        super().__init__(llm)
        self.workers = workers
    
    def run(self, task):
        # 1. 任务分解
        subtasks = self._decompose(task)
        
        # 2. 分配任务
        results = []
        for subtask in subtasks:
            worker = self._select_worker(subtask)
            result = worker.run(subtask)
            results.append(result)
        
        # 3. 汇总结果
        final = self._synthesize(results)
        return final
    
    def _select_worker(self, subtask):
        """根据子任务选择最合适的 worker"""
        capabilities = [
            (worker, self._match_score(subtask, worker))
            for worker in self.workers
        ]
        return max(capabilities, key=lambda x: x[1])[0]

class WorkerAgent(Agent):
    def __init__(self, llm, name, specialty):
        super().__init__(llm)
        self.name = name
        self.specialty = specialty  # "code", "research", "writing", etc.
    
    def run(self, task):
        return self.execute(task)
```

### 3.3 关键工程陷阱

```
多Agent 系统的常见问题:

1. 协商风暴 (Negotiation Storm):
   两个 Agent 反复讨论无法达成一致
   解决: 设置最大协商轮次，超时自动仲裁

2. 任务漂移:
   子 Agent 在执行过程中偏离了原始目标
   解决: Supervisor 定期检查进度，纠正方向

3. 幻觉放大:
   一个 Agent 的小错误被另一个 Agent 放大
   解决: 每条信息携带置信度分数，低置信度信息标记

4. 资源竞争:
   多个 Agent 同时请求同一个工具（如搜索）
   解决: 工具请求排队，设置超时

5. Agent 死锁:
   Agent A 等 Agent B，Agent B 等 Agent A
   解决: 超时检测，设置依赖图避免循环依赖

生产级 Multi-Agent 的关键指标:

  - Task completion rate
  - Average negotiation rounds
  - Agent utilization (忙碌 vs 等待)
  - Error propagation distance
  - Communication overhead (token cost)
```

---

## 四、实现并部署一个生产级 Agent

### 4.1 从零搭建的推荐路线

```
Phase 1: 单 Agent + 工具集
  - 实现简单的 ReAct 循环
  - 集成 3-5 个工具
  - 验证基本的工具调用能力

Phase 2: 记忆系统
  - 加入向量数据库
  - 实现记忆提取与检索
  - 验证长期记忆效果

Phase 3: Planning
  - 加入任务分解
  - 动态重规划
  - 处理复杂任务

Phase 4: Multi-Agent
  - 消息总线
  - Supervisor/Worker
  - 错误处理与恢复

Phase 5: 生产部署
  - 高可用
  - 状态持久化
  - 监控与日志
  - 安全防护
```

### 4.2 可使用框架对比

```
框架         | 复杂度 | 灵活性 | 稳定性 | 适用场景
-------------|--------|--------|--------|----------
LangGraph    | 中     | 高     | 中     | 自定义工作流
CrewAI       | 低     | 中     | 中     | 快速原型
AutoGen      | 中     | 中     | 高     | Multi-Agent
Semantic Kernel | 高 | 高     | 高     | 企业集成
Dify         | 低     | 低     | 高     | 可视化编排
自研          | 高     | 最高   | -     | 完全定制
```

### 4.3 2026推荐实践

```
构建 Agent 框架的核心原则:

1. 工具优先:
   先定义工具，再构建 Agent
   工具的能力决定了 Agent 的能力上限

2. 记忆是关键:
   Agent 之间最大的差距不在 LLM
   而在记忆系统的设计

3. 安全不是附加:
   权限检查、输入过滤、输出审核应是内置组件
   不是训练后加上的

4. 可观察性:
   每一步都需要有清晰的日志
   Agent 的黑箱是最大的维护障碍
```
