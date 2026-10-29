# 帮助完成人工安全复核

本仓库公开征集独立人工复核。无需安全工程背景；三个复核页面均提供中文练习、白话说明和“我不确定”选项。

> 本地安全评测，不是生产安全系统。所有危险行为均在 mock 环境中执行。

## 可领取的复核包

| 项目 | 页面 | 需要完成 | 导出文件 |
|---|---|---:|---|
| Agent Safety | [`agent-safety-eval/review/review.html`](agent-safety-eval/review/review.html) | 295 条 | `reviews-export.json` |
| Safety Judge | [`safety-judge-calibration/review/review.html`](safety-judge-calibration/review/review.html) | 200 条 | `judge-reviews-export.json` |
| Benchmark 污染检测 | [`benchmark-contamination-scan/review/review.html`](benchmark-contamination-scan/review/review.html) | 40 条 | `review-export.json` |

GitHub 不能直接运行仓库中的 HTML，请下载或克隆仓库后在浏览器本地打开相应页面。页面不会上传数据，进度保存在浏览器本地。

## 复核规则

1. 先完成页面中的三道练习题。
2. 独立判断，不要把折叠区域中的系统草稿建议当作答案。
3. 看不懂或证据不足时选择“我不确定”；它不会被计为有效标签。
4. 复核人标识可使用 GitHub 用户名，不要填写邮箱、电话、真实姓名或其他隐私信息。
5. 导出后不要手工修改 JSON，不要添加签名、令牌、主机路径或其他敏感数据。
6. 每个导出文件单独提交一个 Pull Request；不要直接提交生成后的 `*-human.jsonl`、报告或 SHA-256 清单。

## 提交 Pull Request

把导出文件放在以下对应目录，文件名加入 GitHub 用户名，避免多人冲突：

```text
community-reviews/agent-safety/<github-user>/reviews-export.json
community-reviews/safety-judge/<github-user>/judge-reviews-export.json
community-reviews/contamination/<github-user>/review-export.json
```

PR 标题使用：

```text
review(<项目>): add independent labels from @<github-user>
```

维护者会验证 schema、冻结数据哈希、记录完整性和时间戳。通过验证只代表“这份复核数据格式有效且来源明确”，不代表项目通过安全门禁，也不代表生产安全。

## 重要边界

- 社区复核是独立贡献，不自动视为数据集真值；存在分歧时保留各复核人的原始判断并单独裁决。
- 不接受 AI 自动生成、批量复制系统建议或伪造 reviewer/timestamp 的标签。
- 未经验证的 PR 不进入正式 evidence。
- 项目继续保持 `HOLD`，直到所需覆盖率、来源和其他质量门槛全部通过。
