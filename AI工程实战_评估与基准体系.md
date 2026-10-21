# AI 工程实战——评估与基准体系

> 从 Perplexity 到全面能力评估，从静态基准到对抗性测试
> 2026年——没有好的评估，就没有好的模型

---

## 一、为什么评估比训练更难

### 1.1 评估的四大挑战

```
1. 基准饱和 (Benchmark Saturation):
   MMLU: GPT-4 在 2023 年达到 ~86%, Claude 4 在 2026 年达到 ~97%
   天花板效应: 大多数模型在顶级基准上接近满分
   解决方案: 更难的基准, 自适应测试, 过程评估

2. 数据污染 (Data Contamination):
   训练数据中可能包含基准测试的题目
   Llama 3 的教训: 部分 GSM8K 题在训练数据中被发现
   2026 年: 用 ChatGPT 检测污染 + 定期发布新题
   
3. 评估维度不全:
   传统基准只测"做题正确率"
   忽略: 安全性、事实性、创造力、长期规划能力
   2026 年: 100+ 维度的评估体系

4. 评分者偏差:
   人工评估: 成本高, 不一致
   LLM-as-Judge: 自身也有偏差
   2026 年: 多 LLM 投票 + 奖励模型校准
```

### 1.2 评估金字塔

```
┌───────────────────────────────────────────────┐
│          能力层 (What can it do?)              │
│  推理 | 知识 | 代码 | 数学 | 多语言 | 多模态    │
│  工具使用 | 规划 | 创造力 | 安全性              │
├───────────────────────────────────────────────┤
│          行为层 (How does it behave?)          │
│  事实性 | 忠实度 | 一致性 | 长度偏差            │
│  对抗稳健性 | 校准 | 幻觉率                     │
├───────────────────────────────────────────────┤
│          系统层 (How does it perform?)         │
│  延迟 | 吞吐量 | 成本 | 上下文窗口利用           │
│  并发处理 | 扩展性                              │
├───────────────────────────────────────────────┤
│          用户层 (How do users experience?)     │
│  满意度 | 任务完成率 | 偏好胜率 | 保留率         │
│  A/B 测试 | 在线指标                            │
└───────────────────────────────────────────────┘
```

---

## 二、2026 年核心基准全景

### 2.1 基础能力基准

```
推理 (Reasoning):
  基准                | 测量            | 2026年天花板
  ───────────────────|─────────────────|──────────────
  GPQA Diamond       | 博士级科学问题    | ~85%
  MMLU-Pro           | 扩展版 MMLU      | ~90%
  ARC-AGI-2          | 高级推理          | ~60%
  BIG-Bench Hard     | 困难多样化推理     | ~85%
  MATH-500           | 竞赛级数学        | ~95%
  HumanEval          | 代码生成          | ~95% (已饱和)
  SWE-Bench Verified | 真实软件工程       | ~70%

知识 (Knowledge):
  MMLU: ~97% (饱和)
  SimpleQA: 事实性问答
  FreshQA: 需要最新知识的问答
  Wikipedia-based: 知识边界测量

英语评估基准的实际区分度（2026年）:

  基准分类:
    饱和级 (90%+): MMLU, HellaSwag, WinoGrande, PIQA
    有区分度 (70-90%): GPQA, MMLU-Pro, ARC-AGI
    增长级 (<70%): SWE-Bench, GAIA, AgentBench
    
  结论:
    对小模型 (<10B) 仍有参考价值
    对大模型 (>100B) 需要更难的基准
    2026 年评估焦点转向 Agent 能力和长上下文
```

### 2.2 安全评估

```
红队测试分类 (Red Teaming Taxonomy):

  越狱 (Jailbreak):
    直接攻击: "Ignore previous instructions..."
    间接攻击: 角色扮演、虚构场景
    编码攻击: Base64、Leetspeak
    多模态攻击: 图像隐藏文本
    
  有害内容 (Harmful Content):
    毒性: 仇恨言论, 歧视
    违法: 犯罪指导, 武器制造
    敏感: 政治敏感, 医疗建议
    
  偏差 (Bias):
    性别: 职业性别刻板印象
    种族: 肤色/民族偏见
    文化: Western 中心主义

安全评估框架:

  Llama Guard 3 - 分类级别安全
  BeaverTails - 细粒度安全评分
  Anthropic Harmlessness - 无害化基准
  
  2026 年标准:
    需要多个安全分类器交叉验证
    不能依赖单一安全模型
    每次部署前必须跑完整红队套件
```

### 2.3 Agent 能力评估

```
AgentBench (2025+): 
  操作系统: 终端任务完成
  Web 浏览: 信息检索
  SQL 操作: 数据库查询
  购物: 电商任务

GAIA (General AI Assistants):
  真实世界问题
  需要: 多步推理 + 工具使用 + 信息整合
  Level 1-3 难度分级

SWE-Bench (软件工程):
  真实 GitHub Issue → 修复
  验证: 通过对应的单元测试
  2026 年: SWE-Bench Multilingual (Python + JS + Rust + Java)

Tool Bench:
  工具调用: 正确调用工具
  参数提取: 从用户自然语言提取参数
  多工具组合: 串联调用

Agent 评估的挑战:
  环境交互: 每次评估都是不同的执行环境
  随机性: Agent 行动序列不唯一
  成本: 一次完整评估可能消耗 $100+ 的 API 费用
  可复现性: 记录 seed, API calls, 中间状态
```

---

## 三、评估工程实践

### 3.1 自动化评估管道

```
生产级评估流水线:

  def eval_pipeline(model, benchmarks, run_id):
      results = {}
      
      # Phase 1: 基础能力 (快速, 成本低)
      for bench in ["MMLU-Pro", "GPQA"]:
          results[bench] = run_benchmark(model, bench)
          log_progress(run_id, bench, results[bench])
      
      # Phase 2: 安全评估
      safety_results = red_team(model)
      results["safety"] = safety_results
      
      # Phase 3: Agent 评估 (慢, 成本高)
      if budget_remaining:
          results["agent"] = run_agent_eval(model)
      
      # Phase 4: 人工评估 (少量, 随机抽样)
      human_scores = human_eval(model, sample_size=500)
      results["human_feedback"] = human_scores
      
      # 生成报告
      generate_report(run_id, results)
      return results

并行化:

  每个基准可以在独立的 GPU 上运行
  一个 80B 模型的完整评估:
    20 个基准 × 每个 GPU 运行一个
    800 GPU-hours → 40 GPU-hours (20× 并行)
    成本: ~$2,000 一次
    
  建议: 只在 checkpoint 选择阶段跑完整评估
  快速验证阶段跑 3-5 个核心基准就够了
```

### 3.2 数据污染检测

```
污染检测方法:

  1. n-gram 重叠:
    计算基准题与训练数据的 n-gram 重叠率
    如果 MMLU 的数据在训练数据中找到 → 污染
    
  2. Perplexity 异常:
    污染样本的 PPL 通常异常低 (模型见过)
    对比: 新样本 vs 训练中出现过的样本的 PPL
    
  3. 金标准: 发布后评估:
    在模型训练完成后才发布的新基准
    这是唯一没有争议的评估方式
    
  2026 年实践:
    训练数据中保存 1% 作为 held-out 验证集
    提交模型前先检查污染
    发布后评估作为最终标准

Contamination 严重性分级:

  Level 1 (无害):
    基准数据在训练数据中出现 < 0.2%
    对总分影响 < 0.5%
    
  Level 2 (轻微):
    0.2-1% 出现率
    对总分影响 0.5-2%
    需要在报告中披露
    
  Level 3 (严重):
    >1% 出现率
    对总分影响 > 2%
    评估结果不可信
```

### 3.3 LLM-as-Judge 的工程实现

```
挑战: 用 LLM 评估 LLM 本身有偏差

已知偏差:
  位置偏差: 偏好先出现或后出现的回答
  长度偏差: 偏好更长的回答
  权威偏差: 偏好自己的风格
  分数偏移: 某些 LLM 给分普遍高或低

缓解方案:

  class LLMJudge:
      def __init__(self, judge_model="claude-4"):
          self.model = judge_model
      
      def evaluate(self, question, answer_a, answer_b):
          # 1. 消除位置偏差
          score1 = self._judge(question, answer_a, answer_b)
          score2 = self._judge(question, answer_b, answer_a)
          
          # 2. 综合判断
          if score1 == "A wins" and score2 == "B loses":
              return "A wins"    # 一致
          elif score1 == "Tie" or score2 == "Tie":
              return "Tie"       # 一方觉得平
          else:
              return "Tie"       # 不一致，标记为平
      
      def score(self, question, answer, rubric):
          # 给单个回答打分
          prompts = [
              self._build_prompt(question, answer, rubric, "strict"),
              self._build_prompt(question, answer, rubric, "lenient"),
              self._build_prompt(question, answer, rubric, "balanced"),
          ]
          scores = [self._call(p) for p in prompts]
          return median(scores)  # 取中位数

2026 年黄金标准:
  每个评估样本: 3 个不同 LLM Judge + 随机抽样 10% 人工复核
  Judge 模型与评估的模型不能是同一家
  常用: Claude-4 + Gemini 3 + GPT-6
```

---

## 四、训练中的持续评估

### 4.1 评估驱动的训练

```
Training-time Evaluation Pipeline:

  for step in training_loop:
      train_step()
      
      if step % eval_interval == 0:
          # 快速评估 (lightweight)
          current_ppl = eval_ppl(held_out_set)
          log("step", step, "ppl", current_ppl)
          
          if step % full_eval_interval == 0:
              # 全面评估 (heavyweight)
              scores = full_eval(model)
              
              # 检测退步
              for metric, score in scores:
                  if score < best_scores[metric] - threshold:
                      alert(f"Regression in {metric}: {score}")
                      trigger_rollback_check()
              
              # 更新最佳模型
              if scores['composite'] > best_composite:
                  save_checkpoint("best_model")
                  best_composite = scores['composite']

评估频率的建议:

  PPL 评估: 每 500-1000 步 (~10 秒)
  快速基准: 每 5000-10000 步 (~1 分钟)
  完整评估: 每 50K-100K 步 (~2 小时)
  安全评估: 每 100K 步 (持续增多)
```

### 4.2 评估中的致命陷阱

```
1. 过拟合评估基准:
   现象: 在 MMLU 反复调参 → MMLU 分高但实际能力没涨
   解决方案: 
     - 维护私有不公开的验证集
     - 使用 held-out 子集
     - 多样化评估

2. 评估批次不一致:
   现象: batch_size 影响评估结果
   原因: 不同 batch_size 下 attention mask 不同
   解决方案: 固定评估的 batch_size

3. Prompt 敏感性:
   现象: 同一条题目的 prompt 格式不同导致不同结果
   解决方案: 
     - 标准化 prompt (few-shot 格式一致)
     - 运行 5 次取平均值
     - 报告标准差

4. Logprobs 偏见:
   现象: 用 logprobs 评估 vs generate 评估结果不同
   原因: logprobs 是全局最优，generate 是贪心或采样
   解决方案: 始终使用 generate 评估

5. 采样参数不一致:
   现象: temperature=0 vs temperature=0.1
   影响: 在部分基准上差 5%
   解决方案: 固定 temperature=0 (greedy) 作为标准评估
```

---

## 五、前沿评估方法

### 5.1 自适应测试

```
传统基准的缺陷:
  对高端模型区分度不足 (天花板效应)
  对低端模型太困难 (地板效应)

自适应测试 (Computerized Adaptive Testing, CAT):
  根据模型表现动态调整难度
  
  流程:
    1. 从中等难度开始
    2. 正确 → 下一题更难
    3. 错误 → 下一题更简单
    4. 经过 20-30 题后收敛到能力估计值

  优势:
    更少的题目获得更精确的能力估计
    没有天花板/地板效应
    对抗污染 (题库可以很大，每个模型只做子集)

  实现:
    需要有校准过的题库 (IRT 模型)
    2026 年: ETS 正在开发 AI 能力的 CAT 评估
```

### 5.2 过程评估

```
不再只看最终答案，评估推理过程:

  ProcessBench:
    每个推理步骤标注正确/错误
    评估模型是否能自我纠错
    检测"错误推理得出正确答案"的侥幸情况

  步骤审计:
    Reason: 82 = 64 (正确)
    Reason: 剩下的 38 个苹果被分给 5 个人 (错误，应从总数减)
    Detected: 此步有误 → 纠正: 从 150 个苹果开始

  过程评估指标:
    步骤正确率: 每一步的准确性
    错误检测率: 模型发现自身错误的能力
    纠正率: 从错误恢复的概率
    "假对率": 推理错但答案对的比率
```

### 5.3 生态位评估

```
2026 年趋势: 不再追求单一总分
而是评估模型的"生态位适应度":

  一个模型在特定场景下的表现:
    
    代码助手: Code+Debug+API
    写作助手: 创意+事实+风格
    客服: 准确+安全+同理心
    研究助手: 深+准+快

  生态位评估工具:
    SEAL (System Evaluation and Alignment):
      按使用场景定义评估维度
      加权综合评分
      定制化评估基准

  对于你的项目:
    评估不只看论文分数
    重点关注:
      校准能力: 模型知道自身知识边界吗?
      长上下文: 50K+ tokens 的检索准确率
      工具使用: 多工具组合的完成率
```
