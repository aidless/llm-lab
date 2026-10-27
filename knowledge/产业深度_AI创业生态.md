# 产业深度——AI 创业生态

> 从基础设施到应用层的创业机会，融资节奏，以及大模型的降维打击
> 2026年——AI 创业者面对的是大模型平台化、巨头碾压和资本理性化

---

## 一、2026 年 AI 创业金字塔

### 1.1 创业生态结构

```
AI 创业公司分层 (2026):

  ┌──────────────────────────────────┐
  │    层 1: 基础模型 (Foundation)   │
  │  OpenAI, Anthropic, Google DeepMind│
  │  xAI, Meta, Mistral              │
  │  ~10 家, 融资 10B+               │
  ├──────────────────────────────────┤
  │    层 2: 中间层 (Infra/Tools)    │
  │  vLLM, LangChain, Weights & Biases│
  │  Replicate, Together AI          │
  │  ~500 家, 融资 10M-100M          │
  ├──────────────────────────────────┤
  │    层 3: 应用层 (Applications)   │
  │  Jasper, Perplexity, Notion AI   │
  │  Devin, Factory, Cursor          │
  │  ~10,000 家, 种子轮到 A 轮       │
  ├──────────────────────────────────┤
  │    层 4: AI-Native 垂直领域      │
  │  Harvey (法律), Abridge (医疗)   │
  │  Runway (视频), Descript (音频)  │
  │  ~5,000 家, ARR 百万到千万级     │
  └──────────────────────────────────┘

残酷的现实:

  Layer 3 的公司死亡率: >70% (2 年内)
  Layer 2: ~50% 死亡率 (3 年内)
  Layer 1: 绝大部分只有巨头和极少数人能玩

  AI 创业融资 (全球, 2026):
    总数: ~$120B (2023: ~$50B)
    Layer 1: ~$60B (50%)
    Layer 2: ~$30B (25%)
    Layer 3+4: ~$30B (25%)
```

### 2.2 基础模型创业的生死线

```
做基础模型公司需要的最低启动资金:

  训练 1 次大模型 (2026):
    70B 模型, 10T tokens
    15,000 GPU-hours → ~$150万 (按 $100/GPU-hour)
    
  考虑到实验次数:
    10-20 次训练实验 → $1500-3000万
    团队: 50-100 人 × $20万/年 = $1000-2000万/年
    
  年度烧钱率: ~$1-2亿 (保守估计)
  需要融资: ~$5亿 才能活到有收入

基础模型公司的收入现实:

  排名前 3 的公司: API 收入快速增长
  排名 4-10 的公司: 收入覆盖不了训练成本
  
  Mistral 的策略: 开源 + 托管 API
  xAI: 绑定 X/Twitter 生态
  Meta: 不卖模型, 开源做生态

2026 年生存策略:

  1. 做窄: 垂直领域模型 (医疗, 法律, 代码)
  2. 不做基础模型: 基于 API 做应用
  3. 被收购: 技术能力卖给大厂
  4. 去中国: 国内基础模型公司还有融资窗口 (国产替代)
```

---

## 二、AI 创业的风口与坑

### 2.1 2026 年热点赛道

```
🔥 最热门赛道:

1. AI Agent 基础设施:
   Agent 框架: AutoGen, CrewAI, LangGraph
   Agent 监控: LangSmith, LangFuse
   Agent 安全: Guardrails AI, Lakera
   估值逻辑: 对标 MLOps 市场

2. AI Coding:
   Cursor: ~$2B 估值, 年化 ARR >$100M
   Factory: AI 软件工厂
   Devin: 争议大但带动品类
   Bolt/V0/Lovable: AI 生成应用

3. 垂直 SaaS + AI:
   Harvey (法律): $7B+ 估值
   Abridge (医疗): 年化 ARR $100M+
   Glean (企业搜索): $4B+ 估值
   Notion AI (协作): $10B+

4. AI 安全:
   红队自动化的需求暴增
   模型安全评估 (气闸实验室类)
   Prompt 注入防护

5. 端侧 AI:
   手机 AI 助手: Rabbit AI, Humane 的教训
   Apple Intelligence: 让所有人看清楚了端侧 AI 的可行性

❌ 已经饱和或跑不通的赛道:

1. AI Writing 通用工具:
   Jasper 从 $1.5B 估值跌到几乎归零
   Copy.ai 艰难转型

2. AI 生成图像 (泛化):
   Midjourney, DALL-E 垄断
   Stable Diffusion 开源但商业化难

3. AI Chatbot 白标:
   太多竞品, 无差异化
   大模型的 API 直接做更好

4. AI 客服 (通用):
   Zendesk, Intercom 直接集成了
   独立 AI 客服公司压力大
```

### 2.2 大模型的"降维打击"

```
为什么做 AI 应用越来越难?

2023: GPT-4 发布 → 所有对话 AI 创业公司死了
2024: GPT-4o 多模态 → 多模态创业公司凉了
2025: Claude 4 + GPT-5 → Agent 能力的升级
2026: 基础模型持续的进展 → 应用层的空间越来越窄

具体案例:

  AI 写作助手:
    2023: Jasper 做了 1 亿 ARR
    2024: GPT 直接写得好, 不需要 Jasper
    2025: Claude/ChatGPT 自带写作功能
    结果: Jasper 裁员, 估值暴跌

  AI 客服:
    2023: 很多 startups 做 AI 客服
    2024: Zendesk 集成 AI, Intercom 集成 AI
    2025: OpenAI 直接提供客服 API?
    结论: 底层平台正在吃掉垂直应用

如何对抗降维打击?

  1. 工作流壁垒:
    不是帮你写一个东西
    是帮你跑完整的工作流

  2. 数据网络效应:
    用户越多 → 数据越多 → 模型越好
    代表: Glean, Harvey

  3. 行业纵深:
    不是通用 AI, 是法律/医疗/金融的 AI
    需要行业知识、认证、合规

  4. 集成锁定:
    跟客户的 IT 系统深度绑定
    替换成本很高

  5. 速度:
    大公司决策慢
    startup 可以快速迭代
```

---

## 三、AI 公司的商业模式

### 3.1 AI 公司的收入模型

```
主流 AI 产品定价 (2026):

  API (按 tokens):
    GPT-6: 输入 $15/M tokens, 输出 $60/M tokens
    Claude 4: $15/$60 (接近对等定价)
    Llama 4 托管: $5/$15 (开源, 竞争定价)
    DeepSeek V4: $0.5/$2 (极大价格颠覆)

  SaaS (按月订阅):
    代码助手: $15-30/人/月
    通用 AI 助手: $20-40/人/月
    垂直 AI (法律/医疗): $100-500/人/月
    AI Agent: $200-2000/月 + 按用量

  Consumption (按完成的任务):
    Cursor: $20/月 + Commands 额外
    Factory: 按任务数量
    Devin: $500/月

AI 公司毛利率:

  纯 API 转售: 10-20% (薄利, 规模效应)
  模型上封装: 30-50% (prompt 优化, RAG)
  SaaS + AI: 60-80% (垂直集成, 高附加值)
  自有模型训练: 不确定 (基础设施投资巨大)

关键见解:
  最赚钱的 AI 公司不是卖模型的, 是卖应用的
  OpenAI 2026 年 ARR ~$100 亿, 但利润率低
  Cursor ARR ~$2 亿, 但利润率高 (封装 API)
```

### 3.2 开源模型的商业冲击

```
开源模型的商业模式演变:

  2023: LLaMA 开源 → 小公司可以自己做基础模型
  2024: Mistral 开源 → 开源模型性能追上 GPT-3.5
  2025: Llama 3 开源 → 开源接近 GPT-4 水平
  2026: Llama 4 + DeepSeek V4 → 开源追平商业化 (某些方面)
  
  开源对创业的影响:
    好: 不需要烧几亿美元训练模型
    坏: 开源模型降低了进入门槛, 竞争更激烈

  分化:
    做应用: 开源模型提供了很好的基础
    做基础模型: 只有少数几家公司能获得差异化
    做中间层: 开源生态需要工具和服务

开源公司怎么赚钱?

  Mistral: 开源模型 + 付费 API
  Hugging Face: 开源平台 + 企业服务
  Together AI: 开源模型托管
  Fireworks AI: 推理优化 + 开源模型
  CoreWeave: GPU 算力 + 开源模型优化

  开源模型正在蚕食 API 市场:
    2024: 开源模型占 API 调用的 ~20%
    2026: ~40%+, 且还在增长
    推理成本低的优势 (DeepSeek 等)
```

---

## 四、AI 创业的融资策略

### 4.1 2026 年的融资环境

```
2026 年 AI 融资的特点:

  1. 大资本集中在少数公司:
     OpenAI, Anthropic, xAI 吸走大部分钱
     《The Bitter Lesson》在 VC 市场的展现

  2. 资本要求"看得见的收入":
    2022-2023: 讲故事的融资
    2024-2026: 需要 ARR, 需要收入增长
    
  3. 估值泡沫的破裂:
    很多 AI 公司的估值在 2024-2025 年挤泡沫
    幸存者: 真实收入支撑 + 技术壁垒

不同轮次的策略:

  种子轮:
    - 需要的金额: $2-5M
    - 关键: 做 Demo, 要有用户 (不一定是客户)
    - 估值: $10-30M pre-money
    
  A 轮:
    - 需要的金额: $10-30M
    - 关键: 有可重复的营收, ARR > $1M
    - 估值: $50-150M pre-money
    
  B 轮+:
    - 需要的金额: $30-100M
    - 关键: 快速增长, ARR > $5M + 高毛利
    - 估值: $200M-1B pre-money

AI 公司的特殊估值逻辑:

  AI 公司的倍数: 
    传统 SaaS: 10-15× ARR
    AI SaaS (2026): 5-10× ARR (因为不确定性高)
    AI Infra: 15-25× ARR (增长快, 行业升级)
    Foundation Model: 没法用 ARR 衡量 (看技术壁垒 + 人才)
```

### 4.2 最可能的终局

```
AI 公司的退出路径 (基于 2020-2026 年数据):

  上市 (IPO):
    Open AI: 可能性很大 (~2027-2028)
    Anthropic: 可能 (但可能保持 long-term)
    Perplexity, Scale AI: 已经在预备
    Cursor, Glean: 可能上市
    
  被收购 (M&A):
    大厂需要 AI 团队:
      Microsoft: 收购 Inflection ($650M)
      Amazon: 收购 Adept (~$300M)
      Google: 收购 Character.AI (~$200M)
      NVIDIA: 大量小收购
    
    2026 年: M&A 活跃, 大厂买团队 > 买产品
    
  关门:
    AI 公司的死亡率: ~70%
    原因: 钱烧完了 / 产品跟不上 / 大厂入局
    
  活下去 (但长不大):
    小而美的 AI 公司 (10-30 人)
    专注某一垂直领域
    年收入 $5-50M, 活得舒服
```
