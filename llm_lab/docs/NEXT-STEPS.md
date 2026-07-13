# 下一步计划（M5+：把 v0.9.0 推到世界）

> **写这份计划的时刻：** M1–M4 完成。代码 381 测试全绿，治理 / 审计链 / 可观测性 / 内容框架都搭好。但 **v0.9.0 还没 tag**、**一篇博文都没发**、**0 个外部用户**、**0 条 outreach 消息**。
>
> **读这份计划前先决定一件事**：你是要 *把 `llm-lab` 真正推出去*，还是 *把它当作品集 / 学习项目*？两种路径不一样，预算完全不同。下面的"计划 A"是第一种；如果你选第二种，跳到末尾看"计划 B"。

---

## 0. 第一步不是执行，是承诺（1 天）

**这是一个 hard gate**。如果你不打算 30 天内把 v0.9.0 推到 GitHub 上公开，下面的计划都不用读。

- [ ] 决定：**v0.9.0 在 14 天内公开 push 到 GitHub**（不一定要全功能，但 governance + audit chain + observability 必须全）
- [ ] 决定：每周花在 `llm-lab` 上的时间预算（现实：5h / 10h / 20h？）

预算决定后续节奏。我假设 **5–10h/周**（有正职 / 在校的标准情况），下面的计划按这个预算排。20h/周可以加快一倍。

如果你**不打算公开 push**，跳到末尾的"计划 B：归档 / 学习化"。

---

## 1. v0.9.0 发版（5 天工作量，分散在 1 周内）

> **为什么先发版再 outreach**：v0.9.0 之前给 Alice 发消息，她点进 GitHub 看到的是 "draft governance / no release"。**没 release tag 的项目 ≈ 不存在**。先把 release 摆出来，再去拉人。

### Day 1（周一）：commit + tag

```bash
cd F:\TMLR
git add -A
git commit -m "v0.9.0: governance push + observability + tamper-evident audit chain"
git tag -a v0.9.0 -m "First formal release. See CHANGELOG.md and llm_lab/docs/RELEASE-v0.9.0.md."
git push origin main --tags
```

如果 `origin` 没配（私有仓库还没 remote），先决定 push 到哪里：
- **GitHub 公开仓库**（推荐）：最大的"被看到"机会
- **GitHub 私有仓库**（保守）：先 30 天自己 / 朋友 / 同事用，再决定公开
- **GitLab / Codeberg / Gitea**：都行，但要确认 doc 里所有 GitHub 链接的语义

**我的建议**：公开。即使没准备好被世界看到，"有 release tag 的公开仓库"在求职 / 信用 / future 你自己的回看上，都比"私有的实验"价值高 10×。

### Day 1-2：repo 元数据（30 分钟）

- [ ] GitHub repo description: "Local-first LLM eval and orchestration with tamper-evident audit log and security-first design."
- [ ] 主题标签（GitHub topics）: `python` `llm` `evaluation` `security` `audit-log` `openai` `anthropic`
- [ ] 网站 URL（可选）：指向 README
- [ ] 关闭 GitHub Discussions（暂时），打开 Issues
- [ ] 确认 `SECURITY.md` 的 `security@llm-lab.local` 占位邮箱替换成真实地址（用 GitHub 的 `security@` 邮箱转发也行）

### Day 3-5：发布公告（1 小时）

发 1 条 **Show HN** 帖子（Hacker News）。**只发 1 条**，发完就停。

标题候选（**短、具体、禁营销词**）：
- ✅ "Show HN: LLM eval framework with tamper-evident audit log (Python, MIT)"
- ✅ "Show HN: llm-lab – auditable LLM evaluation for security-conscious teams"
- ❌ "I built a tool that..."（HN 死）
- ❌ "The future of..."（HN 死）

帖子正文模板：

```
Hi HN,

`llm-lab` is a Python framework for running and comparing LLM
evaluations, with a focus on the audit trail being defensible.
Three things it does that promptfoo / ragas / deepeval don't:

1. A tamper-evident event log: every row carries a SHA-256 hash
   chained to the previous. `llm-lab verify` walks the chain
   and reports the first break.
2. A published threat model (`THREAT_MODEL.md`) with explicit
   "we defend against X, we don't defend against Y" sections.
3. A CycloneDX SBOM per release, attached as a CI artifact.

Use case: teams that need to use an LLM eval tool but have been
blocked by their security team because the existing tools
don't produce an audit trail.

381 tests pass. Zero new runtime deps for the observability
layer (uses stdlib). MIT license. PRs welcome.

Repo: https://github.com/aidless/llm-lab
Quickstart: [README "5-minute quickstart" section link]
```

**预期结果**：HN 上 0–3 票 / 0–10 评论。**0 票不是失败**——这本来就是低概率事件。**预期 2k–20k GitHub repo views 在 7 天内**（HN 流量和搜索引擎爬虫爬 README 都算）。

---

## 2. 第一轮 outreach（2 周，10 条消息）

按 `docs/OUTREACH-TEMPLATES.md` 的周计划：

### Week 1：列名单 + Tier-1 接触

- [ ] Day 1–2：私下 spreadsheet 里列 5 个 Tier-1（你直接认识的人）+ 5 个 Tier-2（一度分隔）。按 `OUTREACH-TEMPLATES.md` 的 4 类 persona 分类。
- [ ] Day 3：发 **5 条 Alice 模板**消息（LinkedIn DM / Twitter DM / 邮件）。
- [ ] Day 4：发 **5 条 Bob 模板**消息（LinkedIn DM，只发给 security 圈的人）。
- [ ] Day 5–6：发 **2 条 Carol 模板**消息（更难找，但更可能答应）。
- [ ] Day 7：发 5 条 Tier-2 消息（基于 Tier-1 的推荐链）。

### Week 2：跟进 + 第一篇博文

- [ ] Day 8：跟进所有没回复的 Tier-1（"got it, no worries, here's the link if you change your mind"）。
- [ ] Day 9-10：写**第一篇实际博文**（不是骨架）。**M3 那个 hash chain 故事**，钩子最强。按骨架填 1000-1500 字。
- [ ] Day 11：发布到 dev.to + r/Python + LinkedIn（如果有的话）。
- [ ] Day 12-14：等回复。**不要发更多消息**。看数据。

### 决策点：Day 14 晚上

| 信号 | 含义 | 下一步 |
|---|---|---|
| **0/12 答复** | 钉子没对，或者消息没送到对的人 | **停下来重审钉子**。看 PERSONAS.md，重写那 3 个 persona。**不要发第 13 条**。 |
| **1-2/12 答复，但都是"以后再说"** | 钉子 OK，但时机不对 | 等 2 周再追一次。**不要**写新消息轰炸。 |
| **1-2/12 答复，且至少 1 个愿意 POC** | ✅ 钉子对，消息对。去做 POC。 | 跳到 Step 3。 |
| **3+/12 答复** | 🚀 你撞大运了。 | 同上，POC + 立刻写 M4 wedge 博文。 |

**如果 Day 14 是 0/12**，**这是最重要的时刻**。不是继续 outreach 的时刻。**停下来重审**：
- PERSONAS.md 写得对吗？Alice 的"pain point"是你自己脑补的还是从她嘴里听出来的？
- OUTREACH-TEMPLATES.md 的消息读起来像 "我想帮你" 还是 "我在推销"？
- 钉子选对了吗？也许你应该做"快速 eval"而不是"可审计 eval"。

**重审 1 周，再 outreach 一轮**。第二轮如果还 0/10，**这个项目可能不该存在**——不是你的代码不好，是市场不需要。

---

## 3. 第一位外部用户 + 第一次真实 benchmark（Month 2）

### 第一位外部用户的 POC 协议

如果有人愿意试 30 分钟，按 `docs/FINDING-FIRST-USER.md` 的 Week 3 协议：

- 你不 push，不要 PR，不要 Discord。
- 30 分钟 screen-share，**你打字**，他们说 "no, that doesn't work for us"。
- 1 周内写 1 段笔记：什么 broke、什么 surprised them、他们会不会推荐。
- 决定：(a) 加进 ADOPTERS（**先问**） (b) 再做一次 POC (c) 终止。

**不要**因为"总算有人用了"而**降低标准**。第一个用户应该是 real 的人、real 的 use case、real 的反馈。

### 真实 benchmark（与 POC 并行）

现在 `benchmarks/v1-results.json` 是用 **stub LLM** 跑的。**这不真实**。v0.10.0 应该跑真实 provider：

```bash
export OPENAI_API_KEY=sk-...
python benchmarks/self_bench.py --mode all --steps 50 --real \
    --output benchmarks/v1-results.json
```

会得到真实的 p50 / p95 / 成本数字。**这才是 M6 博文 "我跑了 50 个真实 prompt 的 benchmark 报告" 的素材**。

**警告**：openai API 调用需要钱。50 steps × 4 fault × 2 retries ≈ $0.10–0.50。**别用 Claude Opus 做这个 benchmark**，用 `gpt-4o-mini` 或 `claude-haiku-3.5`。

---

## 4. 第一场 meetup talk（Month 3）

按 `docs/talks/2026-local-ml-meetup.md` 的 slide 提纲 + `docs/talks/cfp-abstract.md` 的 CFP 模板。

**先在本地 meetup 讲一次**（不是 CFP 会议）。原因：
- 本地 talk 失败 = 5 个朋友笑你。CFP 会议失败 = 一年都忘不掉。
- 本地 talk 录下来 1 个月后回看，会发现 "slide 7 的图根本看不清"、"demo 慢得要死"、"我在 18 分钟处跑题了 8 分钟"。

**目标**：Month 3 在本地 meetup 讲 1 场，Month 6 之前投稿 1 个 regional conference（PyCon APAC / EuroPython / PyCon US 都行）。

---

## 5. Content cadence（Month 1-12）

按 `docs/CONTENT-CALENDAR.md` 的 12 篇计划，**每月 1 篇**：

| Month | Post | Audience | 钩子 |
|---|---|---|---|
| 1 | 07-治理先于特性 | solo OSS 维护者 | "我删了一个 feature 加了文档" |
| 2 | 08-zero-dep 可观测性 | Python 工具作者 | "三个 one-job 库我没装" |
| 3 | 09-多进程 race 的故事 | security 工程师 | "我故意 disable 修，测试抓住了" |
| 4 | 10-定位 / 钉子 | ML 工程师 | "我没和 promptfoo 比，我有不同的格子" |
| 5 | 11-self-correction 的信誉 | 所有人 | "承认错误比吹嘘更能赢得信任" |
| 6 | 12-真实 benchmark | OSS 维护者 | "如果 5 个数字每 release 出现，没人能黑你" |
| 7 | 13-三个失败模式 | 部署 LLM 的人 | "断网时大多数 eval 工具 crash，我没 crash" |
| 8 | 14-SBOM 8 分钟入门 | Python 维护者 | "我第一次做 SBOM 用了 7 分钟看文档 + 1 分钟命令" |
| 9 | 15-重新设计定位 | OSS 创始人 | "我应该做 best eval tool，但我做了 the one 安全团队会批的" |
| 10 | 16-0 feature 的月份 | OSS 维护者 | "349 → 381 测试，但 0 行新 model 代码" |
| 11 | 17-review 怎么抓到 bug | reviewer | "第二次 review 抓到了，第一次漏了" |
| 12 | 18-年度回顾 | 所有人 | "365 天，0 个付费客户，最有用的教训是 X" |

**纪律**：
- 每月 1 篇，**不能**miss。如果 miss 了一个月，写一段 1 段落的"这个月学到的"补上。**比 1 年里 4 篇完美文章好**。
- 发布前 1 周写完。不要 11 月 30 日赶稿。

---

## 6. 决策点：Month 6 复盘

Month 6 周末坐下来，回答这 4 个问题：

1. **外部用户 ≥ 1？** （F→continue；T→go to step 7）
2. **总 stars ≥ 100？** （指标太硬的话改成 "周下载 ≥ 10" 或 "周 issue ≥ 0.5"）
3. **有外部贡献者 PR 合并？** （这是 governance "core team" 触发的硬指标）
4. **你自己还愿意继续？** （最容易骗自己的问题）

| 结果 | 下一步 |
|---|---|
| **0/4 yes** | 归档项目。写一篇"为什么 `llm-lab` 没成"的复盘博文，把代码归档到 maintenance-only 模式。 |
| **1-2/4 yes** | 继续，但缩小范围。放弃 content cadence，放弃 talks，只做 1 用户支持的 minimum viable。 |
| **3-4/4 yes** | 申请 LF AI & Data Sandbox（满足 star / contrib / docs 要求）。M7+ 进入 growth scale。 |
| **4/4 yes + 用户愿意付费** | 考虑咨询业务（基于 `docs/OUTREACH-TEMPLATES.md` 的"现实可持续性矩阵"，$200-500/h）。 |

---

## 7. 后续 6 个月（如果 6/6 走通了）

| Month | 工作 | 风险 |
|---|---|---|
| 7 | LF AI & Data Sandbox 申请（如果不进 CNCF，走 LF） | 申请失败是 6 个月机会成本 |
| 8 | 第三方安全审计（预算 $5-15k，联系 Trail of Bits / Cure53 / NCC） | 暴露漏洞，**好事** |
| 9 | 第一次 sponsorable 基础设施（GitHub Sponsors / Open Collective） | $50/月 sponsor 不值得花时间；跳过如果 < 5 sponsors |
| 10 | 第一个付费咨询客户（如果有的话），$5-15k 4 周 POC | 决策复杂：是否商业化 |
| 11 | 商业化决定：open core 收费、赞助 only、还是维持纯 OSS | 错一个 = 1 年的方向调整 |
| 12 | 年度博文 + roadmap 重写（v1.0.0 scope） |  |

---

## 计划 B：归档 / 学习化

如果你**决定不公开 push**（正职太忙 / 优先级变了 / 只是学习项目），下面的精简版更现实：

### 2 周内
- [ ] 把 v0.9.0 commit 起来（不 push）
- [ ] 写 **1 篇 "build journal"** 博文：M1-M4 的过程、决策、遇到的问题。dev.to 发布，**算 1 个 portfolio piece**。
- [ ] 把它列入 resume / 个人网站。
- [ ] **停止所有 "growth" 动作**。不 outreach，不写内容，不申请会议。

### 长期
- [ ] 维护模式：bug 修复、依赖更新、security 扫描。**0 新功能**。
- [ ] 如果有 inbound 询问（issue / 邮件），回应。但不主动拉人。
- [ ] 6 个月后重审：要不要转成计划 A？

**这个路径不丢人**。`llm-lab` 已经是个值得写的项目（架构选型、ADR 文化、self-review 流程），作为个人作品集 / 学习记录完全够格。**但**你得**主动选择**这条路，而不是"等有空再 outreach"——后者是慢性失败。

---

## 时间预算（计划 A 假设 5-10h/周）

| 阶段 | 工作量 | 累计 |
|---|---|---|
| Step 1（v0.9.0 发版） | 5h | 5h |
| Step 2（第一轮 outreach + 1 篇博文） | 20h | 25h |
| Step 3（POC + 真实 benchmark） | 15h | 40h |
| Step 4（meetup talk） | 20h | 60h |
| Step 5（content 月度） | 4h/月 × 6 = 24h | 84h |
| Step 6（6 月复盘） | 4h | 88h |

**6 个月总投入：~90h** = 3.5h/周 平均。**可持续。**

如果你只能投入 2h/周：
- 跳过 talks（Step 4）
- Content 改成 2 月 1 篇
- 第一轮 outreach 只发 5 条（不是 12 条）
- 6 月复盘改成 9 月

**任何 1h/周 都比 0h/周 好。** 但 1h/周 意味着接受"小、不快、可持续"。

---

## 你今天可以做的 3 件事

如果今天就决定 plan A：

1. **决定 repo 公开 vs 私有**（30 分钟想清楚）
2. **`git add -A && git commit -m "v0.9.0: governance + observability + audit chain"`** （1 分钟）
3. **写 `security@` 邮箱**（10 分钟，GitHub 提供 `security@<org>.github.io` 转发）

如果决定 plan B（归档）：

1. **commit 起来**（同上）
2. **写 1 篇 build journal 博文**（2-3 小时）
3. **加进 resume / 个人网站**（1 小时）

---

## 不做

以下事我**不**建议：

- ❌ **等 v0.9.0 "完美" 才 push**。它不会完美。先发，发后改。
- ❌ **第一个月发 5 篇博文**。你做不到，第二个月你就放弃。
- ❌ **追求 1000 stars**。500 stars / 5 个外部用户 / 1 个付费客户，**比 5000 stars / 0 用户**有价值 100×。
- ❌ **用 `gpt-4o` 或 `claude-opus` 跑 benchmark**。贵、慢、没必要。
- ❌ **批量 outreach**。一次发 30 条 = 0 答复。一次发 5 条 = 3 答复。
- ❌ **说"我们比 promptfoo 好"**。我们不。我们占不同的格子。说谎损害信任。

---

## 决策点

读完这份计划，你现在的判断是：

- **A**：commit + push + outreach + content，月 5-10 小时
- **B**：commit + build journal + 归档，零后续
- **C**：先 commit + push 公开，但**不**做任何 growth（"let it sit" 模式）
- **D**：先 commit 私有，让 1 个朋友 / 同事试 30 分钟，**再**决定

按你选的路线走。每个都合理。