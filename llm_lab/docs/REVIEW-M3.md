# M3 自评报告 (Review of the audit-chain + SBOM push)

> 自我批评,不是庆功。目的是在下一次合并前找到我能找到的 bug / 漏洞 / 决策错误。
>
> 评估日期: 2026-07-13

## 总体评级: **B+** (在 M2 B+ 基础上有实质进展,但仍有 2 个严重漏洞 + 3 个改进点)

具体完成情况:ruff 干净 / mypy 干净(37 文件)/ pytest 380 通过(+11)/ benchmark 6/6 故障场景通过。但有 **1 个会让 CI 失败的命令错误**、**1 个未文档化的并发竞争 bug**、**2 个我已经发现但没有处理的设计折衷**。

---

## 严重问题(必须在下个 sprint 修)

### S1. CI 的 `cyclonedx-py` 命令错了 — CI 会失败

**症状**: 我在 `.github/workflows/test.yml` 写了:

```yaml
cyclonedx-py environment \
    --output sbom.cdx.json \
    --spec-format 1.5 \
    --output-format JSON \
    --pyproject pyproject.toml
```

`--output` 应该是 `--output-file`,`--spec-format` 应该是 `--spec-version`,而且 `cyclonedx-py` 没有可执行脚本,正确调用是 `python -m cyclonedx_py environment ...`。**版本也错了** —— `cyclonedx-bom 7.3.0` 支持的最高 spec 是 **1.6**,不是 1.5。

**证据**:
```
$ python -m cyclonedx_py environment --output sbom.cdx.json ...
python -m cyclonedx_py: error: unrecognized arguments: --output --spec-format 1.5
```

**已修**: 改成了 `--output-file` / `--spec-version 1.6` / `python -m cyclonedx_py`,并本地测试通过(SBOM 文件成功生成,JSON 有效)。

**根因**: 我从记忆中写命令,没查 cyclonedx-python 的实际 CLI 表面。这是 **M1/M2/M3 期间第一次写 CI 时未验证的命令**。**应该在每次写 shell 命令前先本地跑一遍**——这是流程漏洞。

---

### S2. 哈希链在多进程并发下有 race condition(未文档化也未测试)

**症状**: `append_event()` 的实现是:

```python
async with _connect() as db:
    prev_cursor = await db.execute("SELECT row_hash FROM event_log ORDER BY id DESC LIMIT 1")
    prev_row = await prev_cursor.fetchone()
    prev_hash = prev_row["row_hash"] if prev_row and prev_row["row_hash"] else ""
    
    # INSERT with prev_hash
    
    # compute row_hash
    
    await db.execute("UPDATE event_log SET row_hash = ? WHERE id = ?", (row_hash, new_id))
    await db.commit()
```

**同一个连接里,这个实现是正确的**。但在 **多进程部署**(`uvicorn --workers 4`)下:

1. 进程 A 连接 SQLite,读到 `prev_hash = "abc"`
2. 进程 B 也连上,读到 `prev_hash = "abc"` (因为 A 还没 commit)
3. 进程 A INSERT 行 N+1,prev_hash="abc",row_hash="X"
4. 进程 B INSERT 行 N+2,prev_hash="abc",row_hash="Y"(**prev_hash 应该是 X,不是 abc**)
5. 验证时:行 N+2 的 stored prev_hash="abc",但实际上一行(N+1)的 row_hash="X"。**链断裂,验证报 failure at N+2**。

**为什么 aiosqlite 单进程安全**: `asyncio.to_thread` 把 sync 代码扔到默认 ThreadPoolExecutor,GIL 释放,但每个线程各自有 connection,共享同一个 SQLite 文件。SQLite WAL 的并发模型是 "1 writer + N readers"。两个 writer 同时写,**SQLite 会串行化它们的 COMMIT,但 prev_hash 读取不会被串行化**。

**测试覆盖**: 完全没测这个。`test_chain_*` 是顺序写。

**实际风险**:
- v0.x 文档说 "single-host deployment only",单进程跑没事。
- 但 uvicorn workers / Gunicorn / 多副本 `llm-lab serve` 会触发。运营者想 scale out 第一秒就踩雷。

**该做的(未做)**:
1. 在 `append_event` 外面包 `BEGIN IMMEDIATE` transaction —— SQLite 串行化写。
2. 或者用 advisory lock(单 SQLite 文件 `flock`)防止并发 writer。
3. 或者 ADR-0006 加一句明说 "single-process only; multi-process → use Postgres"。
4. 加一个并发写入的测试(`concurrent.futures.ThreadPoolExecutor` 启 4 个线程各写 100 行,验证完应该全部 OK)。

**为什么我没做**: M3 计划里 ADR-0006 没列这条。我的疏漏。计划审查时也漏了 —— 那时如果我跑过一次 `uvicorn --workers 4` + 跑 `llm-lab verify` 就会立刻发现。

---

## 设计问题(影响可用性,但不是 critical bug)

### D1. 验证是 O(N) 全表扫描

**症状**: `verify_log()` 单次 `SELECT * FROM event_log ORDER BY id`。10 万行还行,100 万行就开始慢(几秒),1000 万行就是分钟级。

**没做的优化**:
- checkpointing: 周期性 (例如每 1000 行) 存一个 `checkpoint_hash` 行,验证从最近的 checkpoint 开始。
- 或者: 按 `intent_id` 局部成链,而不是全局一条链。`verify_log --intent-id X` 只验证一个 run 的链。
- 或者: 把 hash chain 索引化(`prev_hash → row_hash` 的查找索引)以便增量验证。

**该做的**: M+ 时按需。当前单进程 + 几千行级别没问题。**在 ADR-0006 加 "scalability" 章节**明说。

### D2. `compute_row_hash` 依赖 `json.dumps` 的精确行为

**症状**: canonical JSON 依赖 CPython `json.dumps` 的输出格式。特别是 `ensure_ascii=False`(我们用了)对非 ASCII 字符串的 UTF-8 字节序。CPython 一致,但:
- PyPy 可能行为略不同
- 不同 Python 小版本的 float repr 偶尔变(虽然 3.7+ 稳定)

**风险**: 在一台机器上算的 hash,在另一台机器上算不出来。`verify_log` 在跨平台时可能误报 tamper。

**该做的**: ADR-0006 加一句 "CPython 3.10+ only; PyPy not tested"。

### D3. 验证不防 "重算整条链" 的攻击者

THREAT_MODEL.md §P3 我写了:

> Not defended against: an attacker who modifies a row *and* recomputes the chain for all subsequent rows (requires reading every row in order to compute the new chain — high cost but not impossible).

这确实是关键限制 — 攻击者拿到文件系统写权限后,可以重算链让 verify 不报。但我**没写明推荐的 mitigation**:周期性把 SQLite 文件 / 行的 hash 快照 ship 到 S3 object lock 或 immutable log。操作员读这个文档,如果以为 verify = 完整防篡改保护,就会误判风险。

**该做的**: THREAT_MODEL.md §P3 加一行明确: "For tamper-evident verification to be trustworthy, ship the SQLite file (or its last-known hash) to an immutable store (S3 Object Lock, append-only syslog, etc.) on a schedule. verify_log detects tampering after the fact; an immutable external store gives you 'tamper was attempted' attribution."

---

## 次要问题(可以延后,但应该记账)

### m1. 我之前写 `audit_chain_clean` 时硬编码了 observation 字符串

```python
# 之前的错误代码
return {
    "pass": ok,
    "observation": "ok: 3-row chain verifies intact",  # ← 无论 ok 是 True 还是 False 都打印这个
}
```

只在跑 polluted DB 时才暴露:pass=False 但 observation 说 "ok:..."。**测试在干净 DB 下不会发现这个问题**(因为 ok=True 时一切自洽)。

**根因**: 报告字符串应该从 `ok` 计算,而不是固定文本。我已经修了,但**这暴露了一个 review 漏洞** —— 我没在 ship 前用 polluted state 跑 benchmark。

**流程改进**: 跑 bench 前**显式放一些垃圾数据进 llm_lab.db**,确认 benchmark 在污染状态下也输出正确。

### m2. SBOM 的 `cyclonedx-bom` 没进 `pyproject.toml`

`cyclonedx-bom` 是 build-time-only 工具,应该放 `[project.optional-dependencies]` 里(如 `dev` 或单独的 `sbom`)。当前 CI 单独 pip install,本地开发者要跑 SBOM 时也得手动装。

**该做的**:`pyproject.toml` 加 `[project.optional-dependencies.sbom]`,CONTRIBUTING.md 加一行 `pip install -e ".[dev,sbom]"`。

### m3. SBOM 输出位置 `llm_lab/sbom.cdx.json` 进了仓库吗?

当前 CI 配置:`path: llm_lab/sbom.cdx.json` 上传为 artifact。**不会**被 commit 到仓库(因为是 build artifact)。**但 gitignore 没排除它** —— 如果有人在本地跑 SBOM,可能误 commit。

**该做的**: `.gitignore` 加 `sbom.cdx.json`。

### m4. 故障场景 `audit_chain_tamper` 用 `# noqa: SLF001`

`db._connect()` 是私有 API。Benchmark 直接用没问题(它是开发工具),但**这给 "公共 API" 设了一个不好的先例**。如果有人想给 benchmark 加更多 "私有 API 调用",就会形成一种 silent coupling。

**该做的**: 要么给 `db._connect` 加个公共别名,要么在 benchmark 里用一个明确的 "developer API" import path。

### m5. `_fault_audit_chain_clean` 用 `tempfile.TemporaryDirectory()` 但旧场景(`provider_timeout` 等)污染全局 `llm_lab.db`

Bench 中 6 个故障场景里,4 个用默认 `llm_lab.db`,2 个用 tempdir。**不一致**。审计链场景用 tempdir 是因为它们需要洁净状态。但 `sqlite_lock` 场景也用了默认 DB —— 它对状态不敏感(只测并发 append),所以 OK。

**该做的**: 在 `_run_fault()` docstring 里说明哪些场景污染 DB、哪些不污染。**让操作员知道跑完 bench 后他们的 `llm_lab.db` 里多了什么**。

---

## 流程问题(下次要改进的做事方法)

### P1. CI shell 命令 ship 前必须本地跑过

M3 的 SBOM 命令我没本地验证。这是基本失误。**下次每个新 shell 命令(包括 cyclonedx-py、pip-audit、bandit 第一次用时)都先在本地 isolate venv 跑一遍确认 CLI 表面**。

### P2. 自评应该用 polluted state 跑

`audit_chain_clean` 的 bug 我在审查 bench 时才发现,而且只是因为我注意到 observation 字符串奇怪。**应该在审查阶段主动制造 pollution**:
- 跑前先在 `llm_lab.db` 里塞 1000 行垃圾
- 跑 bench,确认 OK / FAIL 的报告仍然自洽
- 跑后才把垃圾删了

### P3. ADR 应该列 known limitation

ADR-0006 没写:
- 多进程并发竞争(S2)
- 跨平台兼容性(D2)
- O(N) 验证开销(D1)
- 防不了"重算整链"攻击者(D3)

我以为这些"以后再说"。**实际上,这些是读者决定"用不用这个工具"的关键信息**。应该至少在 ADR 里标 "Known limitations",让读者知道当前的真实承诺边界。

---

## 整体判断

**v0.9.0 能不能切?**

技术上能。三个 PRD 都已经 deliver。但建议:

1. **今天就修 S1**(已经修了,确认 CI 会通过)
2. **v0.9.0 标签之前再 review 一遍 S2 和 D3** —— 这是 "audit log tamper-evident" 这个钉子能不能兑现的承诺边界
3. 如果不修 S2,**至少在 THREAT_MODEL.md 和 README.md 加明确警告**: "verified on single-process deployments only; multi-process not yet supported"

**我不建议跳过 S2 直接发版**。M3 的整个卖点是 audit chain。如果运营者 scale out 第一天 verify 就 false-alarm,这个钉子就砸了。

如果你的优先级是 "先把 v0.9.0 tag 发出去再迭代",我可以在 v0.10.0 加 M3.5 来修 S2(SQLite BEGIN IMMEDIATE + 并发测试)。

---

## 评分明细

| 维度 | M1 | M2 | M3 |
|---|---|---|---|
| 计划符合度 | A | A | A |
| 代码质量 | A- | A | **B+** (S1 ship 时未验证 + S2 漏并发) |
| 测试覆盖 | A | A | A |
| 文档完整 | A | A | **B+** (ADR-0006 漏 limitations) |
| 安全承诺兑现 | A | A | **B** (D3: 防不了重算整链未明说) |
| 流程纪律 | A | A | **B** (P1: 没本地验 SBOM 命令) |