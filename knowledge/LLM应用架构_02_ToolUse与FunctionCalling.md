# LLM应用架构：Tool-Use与Function Calling系统设计

## 1. Function Calling的底层机制

### 1.1 模型端的工具调用协议：OpenAI工具调用格式与规范

Function Calling的底层本质是模型在自回归生成过程中对输出空间的结构化约束。OpenAI在ChatCompletion API中定义了完整的工具调用协议，其核心是将工具函数以JSON Schema的形式注入到对话上下文中，使模型能够在推理时自主决定调用外部工具。

**工具函数定义层**：工具在API中以`tools`参数传递给模型，每个工具条目包含`type`（当前仅支持`"function"`）、`function.name`（函数名称）、`function.description`（函数语义描述）和`function.parameters`（符合JSON Schema规范的参数定义）。模型的system prompt和user message中并不直接包含工具逻辑，而是通过API底层的logit bias和注意力掩码机制，在token生成空间中预先注册了工具调用的特殊token路径。

**模型的选择触发机制**：当模型判断需要调用工具时，它的生成路径会切换到工具调用格式——返回的响应中包含`tool_calls`字段，而非常规的`content`字段。每个`tool_call`对象包含：

- `id`：格式为`call_xxxx`的唯一标识符，用于后续关联工具结果
- `type`：固定为`"function"`
- `function.name`：模型选择的具体函数名称
- `function.arguments`：模型生成的JSON字符串，包含符合参数schema的具体参数值

模型选择工具的决策过程不是显式的"if-else"逻辑，而是通过以下机制的隐式推理：

- **语义对齐**：模型将用户查询与工具`description`字段的语义进行匹配
- **参数填充**：模型从对话上下文和用户输入中提取填充参数所需的实体值
- **置信度阈值**：只有当模型对工具匹配的置信度超过一定阈值时，才会生成`tool_calls`而非直接回复文本

关键观察：模型可能在单次响应中返回零个、一个或多个`tool_calls`。零个`tool_calls`意味着模型认为无需调用工具即可直接回答；多个`tool_calls`意味着模型判断需要并行调用多个独立工具来满足请求。

```json
// OpenAI工具调用的响应结构示例
{
  "choices": [{
    "index": 0,
    "message": {
      "role": "assistant",
      "content": null,
      "tool_calls": [{
        "id": "call_abc123",
        "type": "function",
        "function": {
          "name": "get_weather",
          "arguments": "{\"location\": \"北京\", \"unit\": \"celsius\"}"
        }
      }]
    },
    "finish_reason": "tool_calls"
  }]
}
```

`finish_reason`字段为`"tool_calls"`是客户端必须检测的信号——它指示对话尚未结束，需要执行工具并将结果返回给模型。

**工具结果回传协议**：当客户端执行完工具后，需要将结果以`tool`角色的消息格式追加到对话历史：

```json
{
  "role": "tool",
  "tool_call_id": "call_abc123",
  "content": "{\"temperature\": 22, \"condition\": \"晴\", \"humidity\": 45}"
}
```

回传时`tool_call_id`必须精确匹配原始`tool_calls`中的`id`，这是模型将结果关联到具体调用的唯一纽带。完成回传后，客户端再次调用API，模型会将工具结果作为上下文继续推理——可能再次调用工具，也可能生成最终文本回复。

### 1.2 底层格式分析：Anthropic/Mistral/Google的工具调用差异

不同模型提供商在工具调用的底层协议上存在显著差异，主要体现在消息格式、标识符结构和调用块的组织方式上。

**Anthropic Claude的Tool Use**：

Claude的tool_use采用`content_block`机制，将工具调用作为消息内容块（content block）嵌入到响应结构中。每个响应消息可包含多个内容块，其中`type`为`"tool_use"`的块代表一次工具调用：

```json
{
  "content": [
    {"type": "text", "text": "我来查询天气信息"},
    {
      "type": "tool_use",
      "id": "toolu_12345",
      "name": "get_weather",
      "input": {"location": "北京", "unit": "celsius"}
    }
  ],
  "stop_reason": "tool_use"
}
```

关键差异点：

- **ID前缀不同**：Anthropic使用`toolu_`前缀，OpenAI使用`call_`前缀
- **命名差异**：field名为`name`和`input`，而非OpenAI的`function.name`和`function.arguments`
- **content_block架构**：Claude将文本思考过程与工具调用放在同一层级的content数组中，而非OpenAI的互斥结构（content=null + tool_calls）。这使得Claude可以在调用工具前输出推理过程
- **stop_reason**：`stop_reason: "tool_use"`等同于OpenAI的`finish_reason: "tool_calls"`
- **结果回传**：使用`"type": "tool_result"`的content block，关联`tool_use_id`

**Mistral的工具调用**：

Mistral的API设计最接近OpenAI，但存在细微差异：

```json
{
  "choices": [{
    "message": {
      "role": "assistant",
      "content": "",
      "tool_calls": [{
        "id": "m_call_12345",
        "type": "function",
        "function": {
          "name": "get_weather",
          "arguments": "{\"location\": \"北京\"}"
        }
      }]
    },
    "stop_reason": "tool_calls"
  }]
}
```

Mistral的区别在于：

- 使用`stop_reason`而非`finish_reason`
- ID前缀为`m_call_`（实际格式取决于部署版本）
- 参数传递方式与OpenAI的JSON字符串格式一致
- 当模型生成文本时，content字段包含实际文本；当生成工具调用时，content为空字符串而非null

**Google Gemini的Tool Calling**：

Gemini的工具调用协议差异最大，采用更结构化的FunctionCall对象：

```json
{
  "candidates": [{
    "content": {
      "parts": [
        {"text": "我来查天气"},
        {
          "functionCall": {
            "name": "get_weather",
            "args": {"location": "北京", "unit": "celsius"}
          }
        }
      ],
      "role": "model"
    },
    "finishReason": "STOP"
  }]
}
```

Gemini的独特之处：

- **无finish_reason区分**：Gemini不区分`tool_calls`和`STOP`的finish_reason，工具调用在`STOP`状态下仍可出现
- **Part架构**：文本和工具调用以parts数组共存，类似Claude的content_block
- **args为对象而非字符串**：Gemini直接返回JSON对象而非字符串化的JSON，减少了客户端解析步骤
- **results回传**：使用`functionResponse`类型的part进行回传

```json
{
  "parts": [{
    "functionResponse": {
      "name": "get_weather",
      "response": {
        "name": "get_weather",
        "content": {"temperature": 22, "condition": "晴"}
      }
    }
  }]
}
```

**格式差异的工程影响**：在构建多模型适配层时，必须抽象出以下几层差异：

1. 工具调用块的检测信号（`finish_reason` vs `stop_reason` vs `finishReason`）
2. 工具ID的命名空间隔离（多模型轮询时ID前缀冲突的处理）
3. 参数格式的统一化（JSON字符串 vs JSON对象）
4. 文本与工具调用共存的处理策略

实践中推荐构建统一的`ToolCall`中间表示层，将各模型的原始响应归一化为标准结构，然后再分发到执行引擎。

### 1.3 工具的流式输出

当启用流式输出（`stream: true`）时，工具调用不再是单次返回的完整JSON块，而是被拆分为多个delta片段逐步传输。这带来了显著的工程挑战。

**OpenAI流式工具调用的delta传递机制**：

在stream模式下，模型通过多个chunk逐步构建`tool_calls`。每个chunk的`delta`字段可能包含部分工具调用信息：

```
Chunk 1: delta.tool_calls[0].id = "call_abc123"
Chunk 2: delta.tool_calls[0].function.name = "get_weather"
Chunk 3: delta.tool_calls[0].function.arguments = "{\"loc"
Chunk 4: delta.tool_calls[0].function.arguments = "ation\": \"北京\""
Chunk N: delta.tool_calls[0].function.arguments = "}"
```

核心机制解析：

- **id只在第一个chunk中出现**，后续chunk不再重复ID
- **index字段**是聚合的关键：`tool_calls`数组中的每个`index`值标识一个独立的工具调用流。当存在并行调用时，不同`index`的片段会交错到达
- **arguments是增量式concat**：客户端需要将同一`index`的所有arguments片段按顺序拼接，最后一次性解析JSON
- **function.name也是增量传递**：name字段通常在单个chunk中完整到达，但技术上也是分段传递的

```typescript
// 流式tool_calls的聚合算法
const toolCallAccumulators: Map<number, {
  id: string;
  name: string;
  arguments: string;
}> = new Map();

for await (const chunk of stream) {
  const delta = chunk.choices[0]?.delta;
  if (delta?.tool_calls) {
    for (const tc of delta.tool_calls) {
      const acc = toolCallAccumulators.get(tc.index) || 
                  { id: '', name: '', arguments: '' };
      if (tc.id) acc.id += tc.id;
      if (tc.function?.name) acc.name += tc.function.name;
      if (tc.function?.arguments) acc.arguments += tc.function.arguments;
      toolCallAccumulators.set(tc.index, acc);
    }
  }
}
```

**Anthropic流式工具调用的content_block_start机制**：

Claude的流式协议采用`content_block_start`、`content_block_delta`、`content_block_stop`三阶段事件：

```
Event: content_block_start → index:0, content_block: {type: "tool_use", id: "toolu_12345", name: "get_weather", input: {}}
Event: content_block_delta → index:0, delta: {type: "input_json_delta", partial_json: "{\"loc"}
Event: content_block_delta → index:0, delta: {type: "input_json_delta", partial_json: "ation\": \"北京\", \"unit\": \"celsius\"}"}
Event: content_block_delta → index:0, delta: {type: "input_json_delta", partial_json: "}"}
Event: content_block_stop → index:0
```

Anthropic的特点是：

- `content_block_start`一次性提供name和id，后续只需处理`partial_json`
- `partial_json`的累计方式与OpenAI相同，但事件结构更清晰
- 每个content block有独立的index，支持交错流

**流式工具调用的工程挑战**：

1. **部分到达的处理决策**：是否在arguments片段尚未完整时就开始执行？激进策略可降低延迟但增加错误风险——实践中采用"完整JSON到达再执行"的保守策略
2. **流终止检测**：工具调用的流式响应可能在`finish_reason: "tool_calls"`到来前就已开始传输。客户端需准备好在无finish_reason信号时根据content_block_stop事件判断完整性
3. **并行工具调用的流式交织**：多个工具调用的片段交错到达，需通过index/sequence_number正确分拣
4. **流式中断恢复**：长arguments可能跨越数十个chunk，连接中断时需通过重试机制恢复完整的tool_calls状态

## 2. 工具定义的最佳实践

### 2.1 Function Schema描述的精细控制

工具定义的质量直接影响模型工具调用的准确率。Function Schema不仅是参数约束，更是模型理解工具语义的核心通道。

**命名规范：target_verb_noun格式**

工具名称应采用`{target}_{verb}_{noun}`的三段式结构：

```
create_ticket                 → 目标对象(create) + 操作(ticket)
search_user_by_email          → 搜索(by email) + 目标(user)
calculate_order_shipping      → 计算(calculate) + 业务对象(order) + 范围(shipping)
```

命名原则：

- **自解释性**：名称本身应包含足够信息使模型能判断工具用途，不需额外说明
- **操作前置**：动词前置便于模型快速分类工具类型（查询类 vs 变更类）
- **命名空间隔离**：不同域的工具使用一致前缀（如`user_*`、`order_*`、`payment_*`）
- **避免缩写歧义**：`get_usr_inf`不如`get_user_profile`，缩写可能被模型误解

**参数的description精细控制**：

参数的description是模型理解参数边界的关键，应包含以下信息：

```json
{
  "name": "search_hotels",
  "description": "根据城市、日期和入住人数搜索可用酒店",
  "parameters": {
    "type": "object",
    "properties": {
      "city": {
        "type": "string",
        "description": "城市名称，使用中文标准名称。例如：\"北京\"、\"上海\"、\"广州\"。不接受英文名称或拼音。"
      },
      "check_in": {
        "type": "string",
        "description": "入住日期，格式为YYYY-MM-DD。必须为今天或未来日期。例如：2025-07-26"
      },
      "check_out": {
        "type": "string",
        "description": "退房日期，格式为YYYY-MM-DD。必须在check_in之后。"
      },
      "guests": {
        "type": "integer",
        "description": "入住人数。最小值为1，最大值为10。包含成人和儿童的总人数。"
      },
      "price_range": {
        "type": "object",
        "properties": {
          "min": {"type": "number", "description": "最低价格（元/晚）"},
          "max": {"type": "number", "description": "最高价格（元/晚）"}
        },
        "description": "价格范围过滤，可选参数。不提供则返回所有价格区间。"
      }
    },
    "required": ["city", "check_in", "check_out", "guests"]
  }
}
```

description中应明确标注：

- **格式约束**：日期、时间、编码格式等具体规范
- **允许值示例**：给出2-3个示例帮助模型理解边界
- **依赖关系**：参数之间的逻辑依赖（如check_out > check_in）
- **默认行为**：参数可选时应说明默认值或省略时的行为
- **条件约束**：特定场景下参数值的限制条件

**避免参数冗余——最少参数原则**：

最少参数原则（Minimal Parameter Principle）是指工具定义的参数集合应该是完成工具功能所需的最小完备集。每增加一个非必需参数，模型的选择空间就多一个维度，出错概率也随之增加。

```json
// 反例：冗余参数导致混淆
{
  "name": "get_weather",
  "parameters": {
    "type": "object",
    "properties": {
      "location": {"type": "string"},
      "city": {"type": "string"},           // 与location语义重叠
      "latitude": {"type": "number"},        // 多数场景不需要
      "longitude": {"type": "number"},       // 多数场景不需要
      "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
      "temp_unit": {"type": "string"},       // 与unit重复
      "date": {"type": "string"},            // 扩展了工具边界
      "time": {"type": "string"}
    },
    "required": ["location"]                 // 弱约束导致参数空间过大
  }
}
```

优化后的定义：

```json
{
  "name": "get_weather_forecast",
  "parameters": {
    "type": "object",
    "properties": {
      "location": {
        "type": "string",
        "description": "地点，支持城市名（如\"北京\"）、行政区名或景点名。如果用户提供了经纬度，也接受\"39.9,116.4\"格式。"
      },
      "unit": {
        "type": "string",
        "enum": ["celsius", "fahrenheit"],
        "description": "温度单位，默认为celsius"
      }
    },
    "required": ["location"]
  }
}
```

最少参数原则的实践要点：

- **合并语义重叠参数**：location+city+address应合并为单个location
- **移除低频参数**：在工具定义中仅保留高频使用参数，低频需求通过fallback机制处理
- **利用默认值**：合理设置默认值减少required参数数量
- **拆分复杂工具**：如果工具参数过多（超过5-7个），考虑拆分为多个细粒度工具

### 2.2 描述中的指令注入风险

工具描述是模型理解工具行为的重要上下文，但描述中包含的类指令文本可能带来一系列风险。

**指令注入的机制**：

模型对工具描述的处理方式与用户指令不同——描述嵌入在API协议结构中，模型倾向于将description视为工具的"行为规范"而非"指令"。然而，当description中包含特定模式时，模型可能将其视为隐式规则：

```json
// 高风险模式：描述中包含条件指令
{
  "name": "transfer_money",
  "description": "执行资金转账。重要：只有当用户明确要求转账并且身份验证通过后才能调用此函数。绝对不要在用户未要求时自动转账。"
}
```

这种描述的风险在于：

- **边界溢出**：模型可能将"只有用户明确要求"理解为可自行判断用户意图，在某些模糊场景下错误触发
- **过度遵循**：对用户无害的请求（如"帮我检查账户余额能否转账"）可能被模型理解为潜在转账意图而触发转账工具
- **提示竞争**：description中的指令可能与实际用户意图冲突，模型需在两者间做不确定性权衡

**更安全的描述策略**：

```json
// 低风险模式：专注于功能描述而非条件约束
{
  "name": "transfer_money",
  "description": "执行账户间资金转账。需要转账金额、源账户和目标账户。调用前需完成用户身份的双因素认证。"
}
```

安全原则：

- **描述功能而非条件**：description应描述工具做什么（功能边界），而非在什么条件下做（条件规则）
- **条件约束放在客户端**：模型不应承担条件判断——执行条件应在工具执行的业务逻辑层实现
- **避免"重要""注意""必须"等指令性措辞**：这些词汇暗示模型需要特殊处理，反而增加行为的不可预测性
- **参数约束优先于描述约束**：能用parameter enum约束的就不用description文字约束

**指令注入的防御架构**：

在系统层面，指令注入的防御应遵循以下分层原则：

1. **API层**：description仅作为工具语义标识，不包含执行逻辑
2. **执行层**：所有安全校验在工具执行端实现，不依赖模型判断
3. **策略层**：工具的调用权限、触发条件通过独立的Policy Engine管理

## 3. Tool-Use的执行架构

### 3.1 工具调用的编排模式

工具调用的编排模式决定了Agent系统如何组织多次工具调用的执行流程。三种基本模式——顺序调用、并行调用和递归调用——构成了所有复杂编排的基础。

**顺序调用（Sequential Calling）**：

顺序调用是最直接的执行模式：每个工具的输出作为下一个工具的输入，形成线性链条。

```
User Request → Tool A → Tool B → Tool C → Final Response
```

典型的顺序调用场景包括：

- **多步信息聚合**：先`search_user_by_email`获取用户ID，再用ID调用`get_user_orders`获取订单列表
- **数据管道**：先`fetch_raw_data`，再`transform_data`，最后`save_to_database`
- **依赖链**：先`create_invoice`创建发票，再`send_invoice_email`发送邮件

顺序调用的执行流程：

```
1. 模型返回 Tool A 调用
2. 执行 Tool A，获取结果
3. 将 Tool A 结果（tool role消息）回传给模型
4. 模型基于结果决定下一步——可能返回 Tool B 调用或直接回答
5. 重复直到模型返回content而非tool_calls
```

工程实现上，顺序调用需要一个简单的循环引擎：

```python
def sequential_execution_loop(messages, tools, max_iterations=10):
    for iteration in range(max_iterations):
        response = client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            tools=tools
        )
        
        if not response.choices[0].message.tool_calls:
            return response.choices[0].message.content
        
        messages.append(response.choices[0].message)
        
        for tool_call in response.choices[0].message.tool_calls:
            result = execute_tool(tool_call.function.name, 
                                  json.loads(tool_call.function.arguments))
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(result)
            })
    
    raise MaxIterationsError(f"Reached max iterations: {max_iterations}")
```

**并行调用（Parallel Function Calling）**：

OpenAI在2023年底引入的并行函数调用能力，允许模型在单次响应中同时触发多个独立的工具调用。这些调用互不依赖，可以并发执行。

```
User Request → Tool A ──┐
               Tool B ──┤→ 并发执行 → 聚合结果 → 模型推理 → Final Response
               Tool C ──┘
```

并行调用的触发条件是模型判断用户请求涉及的多个信息维度彼此独立：

```json
// 用户请求："查找北京和上海的天气，并搜索附近的咖啡馆"
// 模型可能返回三个并行工具调用：
"tool_calls": [
  {"id": "call_1", "function": {"name": "get_weather", "arguments": "{\"location\": \"北京\"}"}},
  {"id": "call_2", "function": {"name": "get_weather", "arguments": "{\"location\": \"上海\"}"}},
  {"id": "call_3", "function": {"name": "search_pois", "arguments": "{\"location\": \"北京\", \"type\": \"cafe\"}"}}
]
```

并行调用的工程实现要点：

1. **并发执行**：使用线程池或异步IO并发执行所有工具
2. **无依赖保证**：必须确保工具之间确实无数据依赖。系统应提供依赖声明机制，有依赖的工具不应放在同一批次
3. **结果聚合**：所有并行调用完成后，将所有tool result一次性回传给模型
4. **错误隔离**：单个工具失败不应影响其他工具的执行，但失败信息需要传递给模型

```python
async def parallel_execution(tool_calls):
    async def execute_single(tc):
        result = await execute_tool_async(tc.function.name, 
                                          json.loads(tc.function.arguments))
        return {"tool_call_id": tc.id, "result": result}
    
    tasks = [execute_single(tc) for tc in tool_calls]
    return await asyncio.gather(*tasks, return_exceptions=True)
```

并行调用的优化策略：

- **工具分组**：根据依赖关系将工具分为并行组和串行阶段
- **动态批处理**：相同的工具调用（如批量查询）可合并为单次批处理
- **延迟分配**：并非所有工具调用都需要立即执行——可根据资源优先级进行调度

**递归调用（Agent Loop）**：

递归调用是Agent系统的核心循环模式——模型调用工具→工具返回结果→模型接收结果并决策→循环继续。这个循环构成了Agent的"思考-行动-观察"闭环。

```
while (not终止条件):
  模型推理 → 生成工具调用或文本
  if (工具调用):
    执行工具 → 结果回传 → 继续循环
  else:
    返回文本 → 循环结束
```

递归调用的核心是循环的终止条件设计，不当的终止条件会导致无限循环或过早退出。

### 3.2 循环终止条件

**max_iterations（最大迭代次数）**：

最直接的终止条件——设置Agent循环的最大轮次上限：

```python
MAX_ITERATIONS = 25  # 常见设置范围：10-50

if current_iteration >= MAX_ITERATIONS:
    return {"status": "max_iterations_reached", 
            "partial_result": accumulated_context}
```

max_iterations的选择需要权衡：
- 过小（<10）：复杂任务可能无法完成
- 过大（>50）：成本失控风险增加
- 动态调整：根据token消耗率和任务进展动态调整限额

**Token Limit（Token限额）**：

基于Token消耗的终止条件更加精细：

- **总Token预算**：限制整个Agent会话的token消耗总量（如100K tokens）
- **每轮Token上限**：限制单次模型调用的输出token数
- **无进展检测**：如果连续N轮调用后解决问题的关键指标没有改善，强制终止

```python
if total_tokens_consumed >= TOKEN_BUDGET:
    return {"status": "token_budget_exceeded", "usage": token_usage}
```

**no_more_tool_call（模型自主终止）**：

当模型在响应中返回`content`而非`tool_calls`，且`finish_reason`为`"stop"`时，循环自然终止。这是最优雅的终止方式——模型自身判断已获取足够信息。

然而，模型可能因以下原因过早终止：
- **幻觉满足**：模型认为已有足够信息回答，但实际上关键数据缺失
- **上下文限制**：过长对话导致模型"遗忘"初始目标
- **懒惰模式**：模型倾向于减少工具调用以降低回复复杂度

应对策略是在system prompt中嵌入"持续验证"的指令，并在循环中增加"无工具调用的确认轮次"——模型第一次返回文本时，要求其确认是否需要再次调用工具。

### 3.3 工具注册与发现

**Tool Registry（工具注册中心）**：

在Agent系统中，工具不应是硬编码的if-else分支，而应通过注册中心（Registry）进行集中管理。Tool Registry的核心职责是：

1. **存储**：维护所有可用工具的元数据和可执行引用
2. **检索**：根据查询条件动态筛选和返回匹配的工具
3. **验证**：工具定义变更时的格式校验和冲突检测

```python
class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, ToolDefinition] = {}
    
    def register(self, tool_def: ToolDefinition, executor: Callable):
        self._validate_schema(tool_def)
        self._check_conflicts(tool_def)
        self._tools[tool_def.name] = tool_def
        self._executors[tool_def.name] = executor
    
    def unregister(self, tool_name: str):
        return self._tools.pop(tool_name, None)
    
    def list_tools(self, category: str = None, tags: list[str] = None) -> list[ToolDefinition]:
        tools = self._tools.values()
        if category:
            tools = [t for t in tools if t.category == category]
        if tags:
            tools = [t for t in tools if any(tag in t.tags for tag in tags)]
        return list(tools)
    
    def get_openai_tools_schema(self) -> list[dict]:
        return [t.to_openai_schema() for t in self._tools.values()]
```

**Category与Tag的动态路由发现**：

随着工具数量增长（从十几个到数百个），将所有工具schema一次性注入模型上下文是不可行的。动态路由发现的策略是：

- **Category分层**：将工具按领域划分为粗粒度分类（如`user_management`、`data_query`、`system_admin`）
- **Tag交叉标记**：使用细粒度标签描述工具的能力维度（如`read-only`、`high-cost`、`needs-confirmation`）
- **动态注入**：在每次Agent循环前，根据当前对话上下文和已执行工具的历史，动态选择将哪些工具的schema注入模型

```python
def select_tools_for_context(registry: ToolRegistry, 
                              conversation_context: Context) -> list[ToolDefinition]:
    # 第一阶段：根据对话意图筛选候选工具
    candidates = registry.list_tools(category=context.primary_category)
    
    # 第二阶段：根据已执行工具的依赖追加相关工具
    for tool in context.executed_tools:
        related = registry.list_tools(tags=tool.related_tags)
        candidates.extend(related)
    
    # 第三阶段：限制总注入量
    deduplicated = deduplicate(candidates)
    return deduplicated[:MAX_TOOLS_PER_REQUEST]  # 通常为20-30
```

## 4. 工具执行的可靠性与容错

### 4.1 工具调用中的参数验证

模型生成的`arguments`是JSON字符串，但模型可能生成不符合schema定义的参数值。服务器端验证是防止执行时崩溃的第一道防线。

**验证的层次结构**：

```
layer 1: JSON解析验证          → 参数是否为合法JSON
layer 2: Schema合规性验证       → 参数类型、格式是否符合定义
layer 3: 业务规则验证           → 参数值的业务合理性
layer 4: 权限与安全验证          → 调用者是否有权使用该参数组合
```

**常见参数错误及处理策略**：

| 错误类型 | 示例 | 处理策略 |
|---------|------|---------|
| 类型不匹配 | 传入字符串"123"而非integer | 尝试类型强制转换（"123"→123），失败则返回错误 |
| 必需参数缺失 | 调用时不提供required字段 | 返回明确错误信息标识缺失参数 |
| 枚举值越界 | 传入enum外的值 | 检查是否可选近义词，否则返回支持的值列表 |
| 格式错误 | 日期格式不是YYYY-MM-DD | 尝试解析常见日期格式，失败则返回错误 |
| 参数值越界 | 人数为-1 | 返回参数值范围约束 |

**修复尝试策略**：

在获取验证结果后，系统可选择以下修复路径：

1. **自动修复**：对类型不匹配等容错度高的错误自动修正并执行
2. **要求重新生成**：返回结构化错误信息让模型重新生成参数
3. **降级执行**：使用安全默认值替代非法参数
4. **放弃执行**：对于无法修复的错误，返回明确失败信息给Agent循环

```python
def validated_tool_call(tool_name, arguments_str, tool_schema):
    try:
        args = json.loads(arguments_str)
    except json.JSONDecodeError as e:
        return ToolResult.error(f"JSON解析失败: {e}", 
                                 retryable=True, 
                                 fix_suggestion="请确保arguments是合法的JSON格式")
    
    validator = jsonschema.Draft202012Validator(tool_schema)
    errors = list(validator.iter_errors(args))
    
    if not errors:
        return execute_tool(tool_name, args)
    
    # 尝试修复
    fixed_args = attempt_autofix(args, tool_schema, errors)
    if fixed_args:
        return execute_tool(tool_name, fixed_args)
    
    # 返回结构化错误给模型
    error_detail = {
        "errors": [{"path": list(e.path), "message": e.message} for e in errors],
        "supported_format": tool_schema
    }
    return ToolResult.error(json.dumps(error_detail), retryable=True)
```

### 4.2 工具调用的错误处理框架

**Transient Error与指数退避**：

瞬时错误（网络超时、服务暂时不可用、限流）具有自恢复特征，采用重试策略是最佳实践。

```python
import random
import time

def execute_with_retry(tool_func, args, max_retries=3, base_delay=1.0):
    last_error = None
    
    for attempt in range(max_retries):
        try:
            return tool_func(**args)
        except TransientError as e:
            delay = base_delay * (2 ** attempt) + random.uniform(0, 0.1)
            time.sleep(delay)
            last_error = e
    
    return ToolResult.error(f"经过{max_retries}次重试后仍失败: {last_error}",
                             retryable=False,
                             error_type="transient_exhausted")
```

退避策略的工程要点：

- **基础延迟**：通常设为500ms-2s
- **指数因子**：2x指数增长
- **随机抖动（Jitter）**：增加±10%的随机偏移避免惊群效应
- **最大退避上限**：通常为30-60s，防止无限制等待

**Permanent Error的处理**：

永久性错误（无效参数、资源不存在、权限不足）不因重试而解决，应直接返回Agent循环处理：

```python
class ToolExecutionResult:
    def __init__(self, success: bool, data=None, error: str = None, 
                 error_type: str = None, retryable: bool = False):
        self.success = success
        self.data = data
        self.error = error
        self.error_type = error_type
        self.retryable = retryable
    
    def to_model_message(self):
        """转为模型可理解的错误信息"""
        if self.success:
            return json.dumps(self.data)
        
        if self.error_type == "invalid_arguments":
            return f"参数错误: {self.error}。请根据错误信息调整参数后重试。"
        elif self.error_type == "resource_not_found":
            return f"未找到资源: {self.error}。请检查查询条件是否正确。"
        elif self.error_type == "permission_denied":
            return f"权限不足: {self.error}。请确认您的账户有相应权限。"
        else:
            return f"执行失败: {self.error}"
```

**人机协作分工**：

对于系统无法自动处理的错误，设计人机协作的降级路径：

- **Agent层错误处理**：对于参数错误，模型通常可以自行修正并重新调用
- **用户确认**：对于数据一致性风险（如部分执行成功），返回给用户决策
- **运维告警**：对于系统异常（工具崩溃、第三方API不可用），触发监控告警

### 4.3 工具执行超时

工具执行的时间控制是Agent系统健壮性的重要维度。超时控制包括三个层面：

**单次工具调用的超时**：

```python
async def execute_tool_with_timeout(tool_func, args, timeout=10.0):
    try:
        result = await asyncio.wait_for(tool_func(**args), timeout=timeout)
        return ToolResult.success(result)
    except asyncio.TimeoutError:
        return ToolResult.error(
            f"工具执行超时（{timeout}s）",
            error_type="timeout",
            retryable=True
        )
```

**Agent循环的级联超时**：

Agent循环作为一个整体也需要时间约束：

```python
class AgentLoop:
    def __init__(self, total_timeout=120.0, per_tool_timeout=15.0):
        self.total_timeout = total_timeout
        self.per_tool_timeout = per_tool_timeout
        self.start_time = time.monotonic()
    
    def check_total_timeout(self):
        elapsed = time.monotonic() - self.start_time
        if elapsed >= self.total_timeout:
            raise TotalTimeoutError(f"Agent循环整体超时（{elapsed:.1f}s > {self.total_timeout}s）")
```

**超时后的补偿策略**：

工具执行超时不等于整个任务失败，系统应具备补偿能力：

1. **部分结果返回**：将超时前已获取的部分数据返回给模型
2. **降级服务**：使用备用工具或备用API执行相同功能
3. **缓存查询**：从缓存中获取过时但可接受的近似结果
4. **用户通知**：告知用户当前进度和预计延迟

## 5. 安全与访问控制

### 5.1 工具的权限模型

工具调用的权限控制不应依赖模型的行为一致性，而应在工具执行层实现强制访问控制。

**User vs System Scoped权限分离**：

```python
class ToolPermissionScope(Enum):
    USER = "user"        # 受用户身份和数据域限制
    SYSTEM = "system"    # 系统级操作，需特殊授权
    PUBLIC = "public"    # 公开可访问，无需鉴权

class ToolPermissionGuard:
    def __init__(self):
        self.scope_map = {
            "get_user_profile": ToolPermissionScope.USER,
            "update_user_email": ToolPermissionScope.USER,
            "delete_user_account": ToolPermissionScope.SYSTEM,
            "system_shutdown": ToolPermissionScope.SYSTEM,
            "search_public_info": ToolPermissionScope.PUBLIC,
        }
    
    def check_permission(self, tool_name: str, user_context: UserContext) -> bool:
        scope = self.scope_map.get(tool_name, ToolPermissionScope.SYSTEM)
        
        if scope == ToolPermissionScope.PUBLIC:
            return True
        
        if scope == ToolPermissionScope.USER:
            return user_context.is_authenticated
        
        if scope == ToolPermissionScope.SYSTEM:
            return user_context.has_role("admin") or user_context.has_role("system")
        
        return False
```

**用户身份验证的传入机制**：

在工具执行链中，用户身份应通过请求上下文逐层传递，而非依赖模型生成：

```python
def execute_tool_safe(tool_name: str, arguments: dict, request_context: RequestContext):
    # 从请求上下文而非arguments中获取用户身份
    user_id = request_context.user_id
    
    # 权限校验
    if not permission_guard.check_permission(tool_name, request_context.user):
        return ToolResult.error("权限不足", error_type="permission_denied")
    
    # 注入用户身份到工具执行
    enriched_args = {**arguments, "_user_id": user_id, "_request_id": request_context.request_id}
    return actual_tool_executor.execute(tool_name, enriched_args)
```

关键原则：**模型不应决定"谁"在执行**，模型只决定"执行什么"，身份验证由基础设施层提供。

### 5.2 高风险工具的防护机制

**Confirmation Guard（人机确认守卫）**：

对写操作、资金操作、数据删除等高危工具，引入Human-in-the-Loop的确认机制：

```python
class ConfirmationGuard:
    HIGH_RISK_TOOLS = {
        "transfer_money": {"requires_confirmation": True, "timeout_seconds": 300},
        "delete_user": {"requires_confirmation": True, "timeout_seconds": 60},
        "send_bulk_email": {"requires_confirmation": True, "timeout_seconds": 120},
    }
    
    def requires_confirmation(self, tool_name: str) -> bool:
        return tool_name in self.HIGH_RISK_TOOLS
    
    async def request_confirmation(self, tool_call_info: dict, 
                                    user_id: str) -> ConfirmationResult:
        # 向用户发送确认请求（通过推送、邮件、UI弹窗等）
        confirmation = await self.confirmation_service.send_request(
            user_id=user_id,
            tool_name=tool_call_info["name"],
            parameters=tool_call_info["arguments"],
            expiry=300  # 5分钟过期
        )
        
        # 等待用户响应
        result = await confirmation.wait_for_response()
        return result
```

Guard的执行流程：

1. 模型生成高风险工具的调用请求
2. ConfirmationGuard拦截执行，向用户发送确认请求
3. 用户确认或拒绝（通常有超时限制）
4. 确认通过→执行工具；拒绝→返回拒绝信息给模型
5. 模型基于确认结果继续推理

**预算限制（Token Cost & Rate Quota）**：

Agent循环的无限递归可能导致高额API费用，预算控制是生产系统的关键组件：

```python
class BudgetController:
    def __init__(self):
        self.session_budget = 0.5   # 美元，单次会话预算上限
        self.tool_cost_map = {
            "search_database": {"token_cost": 0.001, "rate_limit": 100},
            "generate_image": {"token_cost": 0.04, "rate_limit": 5},
            "send_email": {"token_cost": 0.002, "rate_limit": 20},
            "call_external_api": {"token_cost": 0.01, "rate_limit": 30},
        }
    
    def check_budget(self, tool_name: str, session_spend: float) -> bool:
        return session_spend < self.session_budget
    
    def check_rate_limit(self, tool_name: str, window_usage: dict) -> bool:
        limit = self.tool_cost_map[tool_name]["rate_limit"]
        return window_usage.get(tool_name, 0) < limit
    
    def before_execution(self, tool_name: str, session: Session) -> bool:
        if not self.check_budget(tool_name, session.total_spend):
            raise BudgetExceededError(f"会话预算已达上限（${session.total_spend:.3f}）")
        
        if not self.check_rate_limit(tool_name, session.window_usage):
            raise RateLimitError(f"工具 {tool_name} 已达到速率限制")
        
        return True
```

### 5.3 工具调用日志

完整的审计日志是安全合规的基础。每条工具调用记录应包含：

```json
{
  "log_id": "log_20250725_001",
  "timestamp": "2025-07-25T10:30:00.123Z",
  "session_id": "sess_abc123",
  "user_id": "user_456",
  "request_id": "req_789",
  
  "tool_call": {
    "id": "call_xyz",
    "name": "transfer_money",
    "arguments": {"amount": 1000, "to_account": "acc_999"},
    "model": "gpt-4o",
    "iteration": 3
  },
  
  "execution": {
    "start_time": "2025-07-25T10:30:00.124Z",
    "end_time": "2025-07-25T10:30:02.456Z",
    "latency_ms": 2332,
    "status": "success",
    "error": null
  },
  
  "security": {
    "permission_check": "passed",
    "confirmation_required": true,
    "confirmation_status": "approved",
    "budget_check": "passed",
    "rate_limit_check": "passed"
  },
  
  "audit": {
    "ip_address": "203.0.113.1",
    "user_agent": "Mozilla/5.0...",
    "compliance_tags": ["pci", "sox"]
  }
}
```

日志的工程要求：

- **异步写入**：日志采集不应阻塞工具执行路径
- **结构化输出**：采用JSON格式以便于查询和分析
- **不可篡改性**：高安全场景下，日志应写入append-only存储或区块链
- **保留策略**：按合规要求设置日志保留期限（通常90天-7年）

## 6. MCP协议与工具标准化

### 6.1 MCP的架构分析

Model Context Protocol（MCP）是由Anthropic提出的开放协议，旨在标准化AI模型与外部工具/数据源的交互方式。MCP本质上是对Function Calling模式的协议化和生态化改造。

**Client-Host-Server三层结构**：

MCP采用三层架构，严格分离关注点：

```
┌──────────────┐     MCP Protocol      ┌──────────────┐
│   MCP Host   │◄─────────────────────►│  MCP Server  │
│  (Claude App │     JSON-RPC 2.0      │  (Tool       │
│   , IDE等)    │                       │   Provider)  │
└──────┬───────┘                       └──────────────┘
       │
       │  SDK
       ▼
┌──────────────┐
│  MCP Client  │
│  (SDK layer) │
└──────────────┘
```

- **MCP Host**：用户交互界面（如Claude Desktop、IDE插件），负责启动和管理与Server的连接
- **MCP Client**：Host内部与Server通信的SDK组件，维护1:1的连接通道
- **MCP Server**：工具提供方，暴露标准化的工具列表和执行接口

**JSON-RPC 2.0协议规范**：

MCP的通信基于JSON-RPC 2.0规范，使用标准化的Request-Response-Notify模式：

```json
// 请求格式
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "tools/list",
  "params": {}
}

// 成功响应
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "tools": [
      {
        "name": "get_weather",
        "description": "查询天气",
        "inputSchema": {
          "type": "object",
          "properties": {
            "location": {"type": "string"}
          }
        }
      }
    ]
  }
}

// 通知（无需响应）
{
  "jsonrpc": "2.0",
  "method": "notifications/tools/list_changed",
  "params": {}
}
```

**传输层：stdio与SSE**：

MCP支持两种传输模式：

1. **stdio传输**：MCP Server作为子进程运行，通过stdin/stdout管道与Client通信
   - 适用于本地工具、无需网络通信的场景
   - 低延迟、进程隔离、生命周期与Host绑定
   - 典型用例：本地文件系统访问、代码分析工具

2. **SSE（Server-Sent Events）传输**：MCP Server通过HTTP端点暴露服务
   - 适用于远程服务、多云部署
   - 支持服务发现、负载均衡
   - 典型用例：第三方API网关、数据库查询服务

```python
# stdio传输模式示例
# Server启动为子进程
server_process = subprocess.Popen(
    ["python", "mcp_server.py"],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    text=True
)

# 通过stdin/stdout交换JSON-RPC消息
request = json.dumps({"jsonrpc": "2.0", "method": "tools/list", "id": 1})
server_process.stdin.write(request + "\n")
response = server_process.stdout.readline()
```

### 6.2 MCP工具的Lifecycle管理

**工具发现的标准化流程**：

MCP中工具的使用遵循标准化的两阶段流程：

```
第一阶段：工具发现（tools/list）
  Client → Server: {"method": "tools/list"}
  Server → Client: {"result": {"tools": [...]}}

第二阶段：工具执行（tools/call）
  Client → Server: {"method": "tools/call", "params": {"name": "...", "arguments": {...}}}
  Server → Client: {"result": {"content": [...]}}
```

**首阶段：Capability Negotiation**：

在`tools/list`之前，MCP还包含一个协商阶段——在Initialize握手过程中交换协议版本和Capabilities：

```json
// Client → Server: Initialize请求
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "initialize",
  "params": {
    "protocolVersion": "2024-11-05",
    "capabilities": {
      "roots": {"listChanged": true},
      "sampling": {}
    },
    "clientInfo": {
      "name": "claude-desktop",
      "version": "1.0.0"
    }
  }
}

// Server → Client: Initialize响应（声明自身能力）
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "protocolVersion": "2024-11-05",
    "capabilities": {
      "tools": {
        "listChanged": true
      }
    },
    "serverInfo": {
      "name": "weather-server",
      "version": "1.0.0"
    }
  }
}
```

Capability Detection是工具系统的关键设计——Client和Server通过这个协商阶段确定双方支持的协议特性，避免调用不兼容的接口。

**工具变更通知**：

当Server的工具列表发生动态变化（如新插件加载、权限变更），Server可以发送通知：

```json
// Server → Client: 通知工具列表已变更
{
  "jsonrpc": "2.0",
  "method": "notifications/tools/list_changed",
  "params": {}
}
```

Client收到通知后，应重新调用`tools/list`获取最新工具列表。

**MCP工具的Lifecycle状态机**：

```
                ┌──────────────┐
                │   注册/启动   │
                └──────┬───────┘
                       ▼
              ┌─────────────────┐
              │    Initialize    │──→ 协商失败→断开连接
              └────────┬────────┘
                       ▼
              ┌─────────────────┐
     ┌────────│   Initialized   │
     │        └────────┬────────┘
     │                 ▼
     │        ┌─────────────────┐
     │◄───────│  Tool Discovery  │◄─── 收到 list_changed 通知
     │        │  (tools/list)   │
     │        └────────┬────────┘
     │                 ▼
     │        ┌─────────────────┐
     │◄───────│  Tool Execution │
     │        │  (tools/call)   │
     │        └────────┬────────┘
     │                 │
     └─────────────────┘
                       ▼
              ┌─────────────────┐
              │     Shutdown    │
              └─────────────────┘
```

### 6.3 MCP与OpenAI Function Calling的对比

**根本定位差异**：

| 维度 | MCP | OpenAI Function Calling |
|------|-----|------------------------|
| 协议层级 | 工具发现协议（应用层） | 模型集成接口（API层） |
| 设计目标 | 工具生态标准化 | 模型推理与工具执行的接口 |
| 传输层 | stdio/SSE | HTTP REST |
| 状态管理 | 有状态（connect/disconnect） | 无状态（request/response） |
| 发现机制 | tools/list自动发现 | 应用代码手动注入 |
| 变更通知 | list_changed推送机制 | 无（需轮询或重启） |

**工具定义的可移植性**：

MCP的`inputSchema`使用JSON Schema标准，与OpenAI的`parameters`规范本质上兼容但存在差异：

```json
// MCP工具定义
{
  "name": "get_weather",
  "description": "查询天气",
  "inputSchema": {
    "type": "object",
    "properties": {
      "location": {"type": "string"}
    },
    "required": ["location"]
  }
}

// 对应的OpenAI Function Calling定义
{
  "type": "function",
  "function": {
    "name": "get_weather",
    "description": "查询天气",
    "parameters": {
      "type": "object",
      "properties": {
        "location": {"type": "string"}
      },
      "required": ["location"]
    }
  }
}
```

核心差异在于：

- **包装层不同**：MCP直接定义工具，OpenAI多一层`type: "function"`和`function`包装
- **元数据扩展**：MCP支持在`result`中的工具条目之外扩展额外元数据
- **多工具批量**：MCP的`tools/list`天然设计为返回完整工具列表，OpenAI需要应用层管理工具注入

**模型特定优化vs通用协议**：

OpenAI Function Calling针对其模型进行了大量优化：
- 模型内部对function token的logit分布做了精细调优
- 支持并行调用（Parallel Function Calling）
- 流式输出的delta格式针对模型生成特征设计

MCP不针对特定模型优化，而是提供通用工具发现协议，其工具定义需要应用层做模型适配转换。MCP的value proposition在于生态和可发现性——工具开发者编写一次MCP Server即可被所有支持MCP的Host使用。

## 7. 大规模Tool-Use系统的架构设计

### 7.1 工具调用代理的微服务化

当Agent系统需要管理数十到数百个工具时，单体架构的瓶颈变得明显——工具注册中心的单一进程负载过高、工具执行路径的耦合导致部署困难、不同工具的扩缩容需求不同。

**每个工具的独立微服务部署**：

```
                          ┌──────────────────────┐
                          │   Tool Router / API   │
                          │       Gateway         │
                          └──────────┬───────────┘
                   ┌──────────────────┼──────────────────┐
                   ▼                  ▼                  ▼
          ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
          │  Search Service │ │  Weather Query  │ │  Order Service  │
          │  (高流量, 3副本) │ │  (低流量, 1副本) │ │  (高流量, 5副本) │
          └─────────────────┘ └─────────────────┘ └─────────────────┘
                   │                  │                   │
                   ▼                  ▼                   ▼
          ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
          │ Elasticsearch   │ │ 第三方天气API    │ │  PostgreSQL    │
          └─────────────────┘ └─────────────────┘ └─────────────────┘
```

**Tool Router的设计**：

Tool Router是微服务化架构的核心组件，负责：
1. **请求路由**：根据`tool_name`将调用请求分发到对应服务的端点
2. **负载均衡**：对多副本服务实现健康检查与流量分发
3. **协议转换**：统一内部调用协议（gRPC/HTTP），对外暴露一致接口
4. **熔断与降级**：对故障服务实现熔断，避免级联故障

```python
class ToolRouter:
    def __init__(self):
        self.route_table = {
            "search_documents": {"service": "search-service", "endpoint": "/v1/search", "timeout": 5.0},
            "query_weather": {"service": "weather-service", "endpoint": "/v1/weather", "timeout": 3.0},
            "create_order": {"service": "order-service", "endpoint": "/v1/orders", "timeout": 10.0},
        }
        self.circuit_breakers = {name: CircuitBreaker() for name in self.route_table}
    
    async def route(self, tool_name: str, arguments: dict, context: RequestContext):
        route_entry = self.route_table.get(tool_name)
        if not route_entry:
            raise ToolNotFoundError(f"未找到工具: {tool_name}")
        
        if self.circuit_breakers[tool_name].is_open():
            raise CircuitBreakerOpenError(f"工具 {tool_name} 的熔断器已打开")
        
        try:
            result = await self._call_service(route_entry, arguments, context)
            self.circuit_breakers[tool_name].record_success()
            return result
        except Exception as e:
            self.circuit_breakers[tool_name].record_failure()
            raise
```

**水平扩缩策略**：

```
高流量工具（如搜索、推荐）→ 预置多副本 + HPA
低流量工具（如管理操作）→ 按需部署 + 冷启动优化
峰值敏感工具（如支付）  → 峰值预留 + 弹性扩缩
```

### 7.2 工具调用的流控与管道设计

**请求级别的多工具并行调用**：

当模型在同一次响应中返回多个`tool_calls`时，系统需要并发执行。在大规模架构中，这需要细致的并发控制：

```python
class ToolExecutionPipeline:
    def __init__(self):
        self.semaphore = asyncio.Semaphore(10)  # 最大并发工具调用数
        self.executor = ThreadPoolExecutor(max_workers=20)
    
    async def execute_parallel(self, tool_calls: list[ToolCall]):
        async with self.semaphore:
            tasks = []
            for tc in tool_calls:
                task = asyncio.create_task(self._execute_with_limits(tc))
                tasks.append(task)
            return await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _execute_with_limits(self, tool_call: ToolCall):
        # 单工具的速率限制
        await self.rate_limiter.acquire(tool_call.function.name)
        return await self.tool_router.route(tool_call.function.name, tool_call.arguments)
```

**批量优化（Batching）**：

对同一类型的高频工具调用，可采用批量聚合策略：

```python
class BatchAggregator:
    def __init__(self, batch_window_ms=100, max_batch_size=50):
        self.batch_window = batch_window_ms / 1000
        self.max_batch_size = max_batch_size
        self.pending = defaultdict(list)
    
    async def submit(self, tool_name: str, arguments: dict) -> asyncio.Future:
        future = asyncio.get_event_loop().create_future()
        self.pending[tool_name].append((arguments, future))
        
        if len(self.pending[tool_name]) >= self.max_batch_size:
            asyncio.create_task(self.flush(tool_name))
        elif len(self.pending[tool_name]) == 1:
            asyncio.create_task(self.delayed_flush(tool_name))
        
        return await future
    
    async def delayed_flush(self, tool_name: str):
        await asyncio.sleep(self.batch_window)
        await self.flush(tool_name)
    
    async def flush(self, tool_name: str):
        batch = self.pending.pop(tool_name, [])
        if not batch:
            return
        args_list = [args for args, _ in batch]
        futures = [future for _, future in batch]
        
        try:
            results = await self.batch_executor.execute_batch(tool_name, args_list)
            for future, result in zip(futures, results):
                future.set_result(result)
        except Exception as e:
            for future in futures:
                future.set_exception(e)
```

### 7.3 工具的Caching策略

缓存是降低Tool-Use系统成本和延迟的关键手段。但工具调用缓存的特殊性在于输入参数的多样性和输出结果的时效性。

**精确匹配缓存**：最简单的缓存策略——同参数同结果。适用于确定性工具如文档查询、代码分析等。

```python
class ExactMatchCache:
    def __init__(self, ttl_seconds=300):
        self.cache = {}
        self.ttl = ttl_seconds
    
    def get_cache_key(self, tool_name: str, arguments: dict) -> str:
        return f"{tool_name}:{hash(frozenset(sorted(arguments.items())))}"
    
    async def get_or_execute(self, tool_name: str, arguments: dict, executor):
        key = self.get_cache_key(tool_name, arguments)
        
        if key in self.cache:
            entry = self.cache[key]
            if time.monotonic() - entry["timestamp"] < self.ttl:
                return entry["result"]
        
        result = await executor(tool_name, arguments)
        self.cache[key] = {"result": result, "timestamp": time.monotonic()}
        return result
```

**Semantic Cache（语义缓存）**：

对于自然语言输入的工具（如搜索、问答），精确匹配的命中率极低。Semantic Cache通过embedding相似度匹配近似查询：

```python
class SemanticCache:
    def __init__(self, threshold=0.92, embedder=None):
        self.entries: list[CacheEntry] = []
        self.threshold = threshold
        self.embedder = embedder or EmbeddingService()
    
    async def lookup(self, tool_name: str, arguments: dict) -> CacheResult:
        # 对需要语义匹配的字段计算embedding
        query_text = self._build_query_text(tool_name, arguments)
        query_embedding = await self.embedder.embed(query_text)
        
        for entry in self.entries:
            similarity = cosine_similarity(query_embedding, entry.embedding)
            if similarity >= self.threshold and entry.tool_name == tool_name:
                return CacheResult(hit=True, result=entry.result, similarity=similarity)
        
        return CacheResult(hit=False)
    
    async def store(self, tool_name: str, arguments: dict, result: any):
        query_text = self._build_query_text(tool_name, arguments)
        embedding = await self.embedder.embed(query_text)
        self.entries.append(CacheEntry(
            tool_name=tool_name,
            embedding=embedding,
            result=result,
            timestamp=time.monotonic()
        ))
```

Semantic Cache特别适用于低变化频率的工具输出：
- 天气查询（同一地点的天气短时间内不剧变）
- 公共知识检索（知识库问答）
- 文档分析（内容不变时结果可复用）
- 汇率查询（短时间内的汇率变化可接受近似值）

**缓存失效策略**：

| 工具类型 | 缓存TTL | 策略 |
|---------|---------|------|
| 天气查询 | 30min | 按时间段更新 |
| 知识检索 | 1h-24h | 按内容版本更新 |
| 用户信息 | 5min | 按事件触发生效 |
| 支付接口 | 0（不缓存） | 无条件直连 |

对于大规模部署，缓存层应作为独立服务部署（如Redis Cluster），与Tool Router分离：

```
Tool Router → Cache Layer (Redis) → Tool Service
                   │
                   ▼
            (未命中时继续执行)
            (命中时直接返回)
```

完整的Tool-Use系统架构应该是多层缓存的组合：L1内存缓存（微秒级）、L2分布式缓存（毫秒级）、L3语义缓存（考虑embedding计算成本）。通过合理的缓存策略，生产系统中工具调用的缓存命中率可达到30%-60%，显著降低延迟和API成本。

---

*本文档从底层协议分析到大规模架构设计，系统性地覆盖了LLM Tool-Use与Function Calling的完整技术栈。每个模块之间既相互独立又有机衔接——从单次工具调用的协议细节，到数千工具并发调用的分布式架构，构成了Agent工具系统设计的完整知识体系。*
