# 实验协议

## 任务、切分与可见性

本地 task.json 提供 name、public 整数对及 held_out 整数对，源代码为相邻 buggy.py，入口 solve(x)。test_buggy.py 展示公开失败，默认 pytest 仅收集 tests/。清单由评估代码读取，Agent 只得到源码和公开对，探针临时目录只复制当前源码。Mock 含针对示例的修复规则，该任务只用于工程验收。真实实验将保留测试、真因标签和参考补丁置于 Agent 无权访问的评估环境；按项目或缺陷族划分开发集和测试集，记录排重与筛除原因。

第一个项目级开发样例是 [BugsInPy 的 PySnooper 缺陷](../examples/pysnooper_1/README.md)。评估端接收冻结的项目目录，分别运行公开和保留的 pytest 测试，用同一判定标准验证候选。

当前正式样本使用 [BugsInPy 裁剪题包 v3 的十二条任务](../examples/bugsinpy_formal/README.md#裁剪题包-v3)，覆盖 tqdm、youtube-dl、Tornado、Black 和 Cookiecutter。每题在裁剪后的出错版本上复现公开和保留失败，并在相同范围内的参考修复上通过两套测试。PySnooper 属于独立的开发集。样本按可离线复现和缺陷类型筛选，统计结论限于所选任务和裁剪范围。

## 配对与成本

每个任务、版本、随机种子应在四种策略下分别从相同初态运行。固定模型快照、采样参数、工具集合、运行环境与提示上下文上限。以同一最终验证套件评定结果。当前演示中，复现、Provider 选择、修复、探针、应用和公开验证每次各记 1 次；确定性本地计算不单独计数。验证阶段用相同的每进程超时与用例数量，不扣 Agent 行动。总耗时含最终验证。行动数不能代表等价 token、CPU 或货币成本；正式比较前增加实际输入输出 token、工具时间、总墙钟上限与重试费用记录。

## 结果与分析

每轮运行写独立 UUID 文件。start 保存策略、任务、配置与 Provider，action/response 通过 index 对齐，prior/update 保存结构化诊断证据，finish 保存最终 verdict 对应的 validation、actions、seconds、stop_reason。error 表示异常中断。用 summarize 汇总同策略的完整 RunResult；正式实验须按运行清单补记缺失或异常，禁止仅以正常退出子集作为成功率分母。

修复成功以所有验证用例通过为准。定位准确率需先定义文件或函数粒度、预测提交时刻和允许的真因集合。拟报告每题配对的成功差、成本分布与区间；对超时和预算耗尽预注册计分规则。消融时固定候选、先验及预算，只改变选择器或成本权重，记录选择轨迹。真实模型输出、数据集版本、预算和随机种子冻结后再开始正式实验。

评估实现按 README 的成员 C 分工维护。`dataset_loader.py` 加载和导出任务，`task_filter.py` 检查用例隔离及任务能否入选，`patch_validator.py` 独立验证候选，`benchmark_runner.py` 读取 A 的运行记录并调用验证器，`metrics.py` 汇总单策略指标，`statistical_analysis.py` 按任务、版本和种子配对。正式题包的命令入口见上面的运行说明，验收测试集中在 `tests/test_evaluation.py`。
