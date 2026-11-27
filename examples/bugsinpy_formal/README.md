# BugsInPy 正式评估任务 v1

本轮采用 [裁剪题包 v3](#裁剪题包-v3) 的 12 条任务。以下 v1 和 v2 说明保留供复现早期版本。

[任务清单](manifest.json)固定了八条来自 BugsInPy 的真实缺陷。PySnooper/1 用于开发联调。正式任务统一在本地验收过的 Python 3.8.1 容器里运行，公开与保留测试使用同一镜像。

## C 的本地交付

- `../evaluation-data/bugsinpy-formal-v1` 是交给 A 的题包。每题包含出错版本源码和公开测试，根目录的 `manifest.json` 记录缺陷编号、版本、镜像和路径。每轮从对应源码目录的独立副本开始。Agent 的执行环境只挂载公开题包。
- `../evaluation-private/bugsinpy-formal-v1` 由 C 持有。`held-out` 保存保留测试，`preflight.json` 保存版本对照的验收结果和定位文件标签。先用公开及保留测试验证出错版本失败、参考修复通过，再导出公开题包。

评估端已有指定 BugsInPy 检出和各项目的 Git 对象时，可在仓库根目录运行：

```powershell
$env:PYTHONPATH = Join-Path (Get-Location).Path 'src'
python -m active_debug.evaluation.formal_tasks
```

准备程序读取 `../evaluation-data/BugsInPy` 以及同目录下的 `tqdm`、`youtube-dl` 和 `tornado` Git 仓库。数据集需使用清单里的版本，各项目需含其记录中的出错和修复提交。验证镜像复用 [PySnooper 的 Dockerfile](../pysnooper_1/Dockerfile)。现有题包可直接使用，重导出时可用 `--output` 指定新目录。

## 接收 A 的运行结果

开跑前在评估端记录实验计划。必填字段为 `experiment_id`、`model_snapshot`、`max_actions` 和整数数组 `seeds`。把采样参数、工具、token 和时间预算一并记入同一计划，并让四种策略共用。每个种子对应 8 道题 × 4 种策略，共 32 次计划运行。

A 每次运行写一条 JSONL。必填身份字段为 `experiment_id`、`case_id`、`revision`、`seed`、`strategy`、`status`。策略名沿用 `direct_repair`、`free_debug`、`hypothesis_llm`、`hypothesis_eig`。已有最终候选时，填 `status: "completed"` 和 `candidate_dir`，路径相对于 JSONL 所在目录或用绝对路径。明确提交初始源码时可填 `candidate_dir: null`。同时记录 `actions`、`seconds`、`stop_reason`，有实测 token 时记录 `input_tokens` 和 `output_tokens`。如收集定位指标，记录修复前预测的 `predicted_file`。

超时、异常或最终确认缺失的运行分别用 `timeout`、`error`、`missing` 标记。预算耗尽时，如有最终候选，仍填写 `completed`，并把 `stop_reason` 写为 `budget_exhausted`，由 C 照常判定。尚未收到的计划运行也单列为缺失。

```powershell
python -m active_debug.evaluation.formal_results --plan C:/path/to/plan.json --runs C:/path/to/runs.jsonl
```

报告按「缺陷、源码版本、种子」配对。修复率以计划次数为分母，缺失、超时和异常均无修复成功记录。成本只在有对应数值的运行中统计并显示样本数，耗时包含独立验证。报告也会列出 EIG 和其余三种策略的配对差，以及配对记录齐全时按缺陷重采样的 95% 区间。

## 裁剪题包 v2

与 A 对齐单题规模后，同一批 8 条缺陷另有小规格题包 `../evaluation-data/bugsinpy-focused-v2`。每道题只向 A 提供一个 `source/buggy.py` 和公开测试，分为 6 道基础题、2 道进阶题。它们是按 BugsInPy 原缺陷裁剪的练习，源码及测试范围以这一版的清单为准。完整项目题包 v1 仍可单独复用。

裁剪题的 `revision` 是实际交给 A 的源码哈希，`upstream_revision` 单独记录来源。A 写运行记录时填写裁剪题包清单里的 `revision`。如收集定位结果，填修复前预测的 `predicted_function`，顶层函数用函数名，方法用 `Class.method`。C 用保留的函数标签评分。

裁剪题的保留测试和验收记录位于 `../evaluation-private/bugsinpy-focused-v2`。现有评估入口可以直接读取这两套目录：

```powershell
python -m active_debug.evaluation.formal_results `
  --bundle ../evaluation-data/bugsinpy-focused-v2 `
  --private-root ../evaluation-private/bugsinpy-focused-v2 `
  --plan C:/path/to/plan.json --runs C:/path/to/runs.jsonl
```

## 裁剪题包 v3

本轮正式题包是 `../evaluation-data/bugsinpy-focused-v3`，包含 10 道基础题和 2 道进阶题。原 8 道题的源码、测试和哈希保持一致，新增 `tqdm/2`（ANSI 颜色重置）、`black/17`（空源码解码）、`cookiecutter/1`（UTF-8 模板配置）和 `youtube-dl/33`（带小数秒的时间戳）。新增四题都已用裁剪后的出错版和参考修复版验收。

A 使用 v3 清单中的 `revision`。每个种子对应 12 道题 × 4 种策略，共 48 次运行。C 使用 v3 的私有目录评分，函数定位仍用 `predicted_function`。

```powershell
$env:PYTHONPATH = Join-Path (Get-Location).Path 'src'
python -m active_debug.evaluation.formal_results `
  --bundle ../evaluation-data/bugsinpy-focused-v3 `
  --private-root ../evaluation-private/bugsinpy-focused-v3 `
  --plan C:/path/to/plan.json --runs C:/path/to/runs.jsonl
```
