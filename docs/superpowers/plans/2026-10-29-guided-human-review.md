# 中文白话人工复核向导 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将三个技术型人工复核页面改成普通中文用户可理解、可练习、可导出且不污染人工真值的向导。

**Architecture:** 各项目继续由 Python 生成自包含静态 HTML；增加白话摘要和练习状态，不改变现有导出 schema。污染页面从冻结任务和 fixture 语料补齐原文证据。

**Tech Stack:** Python、原生 HTML/CSS/JavaScript、pytest

**Spec:** `docs/superpowers/specs/2026-10-29-guided-human-review-design.md`

## Global Constraints
- 页面仅本地运行，不上传数据。
- “我不确定”不得满足人工门禁。
- 保持现有导入格式和哈希绑定。
- 保留“本地安全评测，不是生产安全系统。”

## Review Focus
- 刷新后练习和正式进度不丢失。
- 旧 localStorage 与新页面安全重置。
- 恶意任务文本经过 HTML escaping。
- 污染证据缺失时明确显示“证据不足”。
- 移动端可标注和导出。

### Task 1: Agent Safety 白话向导
**Files:** Modify `agent-safety-eval/agent_safety_eval/review.py`; Test `agent-safety-eval/tests/test_agent_safety.py`
- [ ] 测试中文摘要、练习、单问题、折叠详情和“不确定”映射。
- [ ] 实现并保持 accept/flag 导出语义。
- [ ] 运行 Agent Safety 全部测试。

### Task 2: Safety Judge 白话向导
**Files:** Modify `safety-judge-calibration/judge_calibration/review_packet.py`; Test `safety-judge-calibration/tests/test_metrics.py`
- [ ] 测试白话问题、练习、隐藏建议和不确定状态。
- [ ] 实现 safe/unsafe/pending-compatible 导出。
- [ ] 运行 Safety Judge 全部测试。

### Task 3: 污染检测证据向导
**Files:** Modify `benchmark-contamination-scan/contamination_scan/cli.py`; Test `benchmark-contamination-scan/tests/test_contamination.py`
- [ ] 测试队列含题目文本、疑似来源和中文页面。
- [ ] 填充对照证据并实现练习和折叠详情。
- [ ] 运行污染检测全部测试。

### Task 4: 生成、验收与发布
**Files:** Regenerate three `review/review.html`.
- [ ] 重新生成页面并执行导入契约测试。
- [ ] 运行三个项目测试、ruff 和 diff 检查。
- [ ] 提交并推送，保留 `evalsft-loop/` 不变。
