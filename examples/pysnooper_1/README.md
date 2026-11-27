# PySnooper 的第 1 条缺陷

[BugsInPy 记录](https://github.com/soarsmu/BugsInPy/tree/11c5f1eea954a42132cfd06bf257766a7963e0fd/projects/PySnooper/bugs/1)提供了出错版本和测试位置。候选目录从 [出错版本 e21a311](https://github.com/cool-RR/PySnooper/tree/e21a31162f4c54be693d8ca8260e42393b39abd3)开始，应用候选补丁后交给评估端。公开测试 `public/test_chinese.py` 单独复现中文日志写入失败。默认文本编码为 ASCII，日志应仍能用 UTF-8 写入和读回。

保留测试由评估端提供，放在 Git 仓库和候选目录之外。候选目录、公开测试目录和保留测试目录互相独立。官方修复版本仅用于评估端校验任务。

在仓库根目录用 PowerShell 构建验证环境：

```powershell
docker build -t active-debug-pysnooper-1 -f examples/pysnooper_1/Dockerfile .
```

首次使用时，在仓库旁边准备数据集记录和上游源码：

```powershell
New-Item -ItemType Directory -Force ../evaluation-data
git clone https://github.com/soarsmu/BugsInPy.git ../evaluation-data/BugsInPy
git -C ../evaluation-data/BugsInPy switch --detach 11c5f1eea954a42132cfd06bf257766a7963e0fd
git clone --no-checkout https://github.com/cool-RR/PySnooper.git ../evaluation-data/PySnooper
```

按编号加载任务并复现公开失败：

```powershell
$env:PYTHONPATH = Join-Path (Get-Location).Path 'src'
python -m active_debug.evaluation.bugsinpy PySnooper/1
```

输出 `reproduced: true` 表示公开测试在出错版本上按预期失败。可加 `--source-out ../evaluation-data/pysnooper-1-buggy` 把该版本的源码导出到一个新目录。导出的源码没有 Git 历史；公开测试在本页旁边的 `public` 目录。当前复现入口只配置了 `PySnooper/1`。

拿到候选补丁后，把它应用到导出源码的独立副本。用候选目录和保留测试目录运行独立验证：

```powershell
$env:PYTHONPATH = Join-Path (Get-Location).Path 'src'
python -m active_debug.evaluation.project_validator `
  --candidate 'C:/path/to/candidate' `
  --public-suite 'examples/pysnooper_1/public' `
  --held-out-suite 'C:/path/to/private-suite' `
  --image active-debug-pysnooper-1
```

输出 JSON 包含 `passed`、`public` 和 `held_out`。两套测试都通过且没有跳过的测试时，退出码为 0。测试失败或超时返回 1；运行环境、收集或命令错误返回 2。默认每套测试限时 30 秒，可用 `--timeout` 调整。判定结果可交给汇总流程，测试详情留在评估环境。
