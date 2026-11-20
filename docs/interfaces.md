# 跨模块契约

| 生产者 → 消费者 | 输入与返回 | 不变量 |
|---|---|---|
| C → A | DebugTask(name, source, public) | 只含源码和公开复现用例 |
| A → B | FailureTrace(case, actual) → list[Hypothesis] | 名称唯一，非负信念之和为 1 |
| B → A | Experiment(argument, predictions, cost) | 探针输入为整数；确定性选择器要求每个候选假设都有输出预测 |
| A → B | run_experiment(experiment) → Observation | stdout、stderr 为文本；超时置 timed_out，返回码为 -1 |
| B → A | update(state, probe, obs) → DiagnosisState | 追加历史，返回新状态；未解释或执行异常时保留先验 |
| Provider → A | repair(task, failure, state, feedback) → Patch | 单文件完整替换文本；state 在两个 baseline 中为 None |
| A → C | apply_patch(patch) → PatchResult；verify_patch() → TestResult | 前者编译成功才替换当前候选，后者仅检验公开输入 |
| C → 汇总 | validate(source, cases, timeout) → TestResult；benchmark → RunResult | checked 是实际执行数，passed 要求所有用例通过 |

下例可在安装项目后从根目录执行。算法测试可以直接构造 Observation，从而在没有子进程时独立测试 B。

```python
from pathlib import Path
from active_debug.agent.tool_executor import ToolExecutor
from active_debug.diagnosis import hypothesis_generator, hypothesis_manager
from active_debug.evaluation.dataset_loader import load
from active_debug.schemas import Experiment, FailureTrace

task, held_out = load(Path("examples/simple_bug"))
executor = ToolExecutor(task.source, task.public, timeout=2)
actual = executor.run_experiment(Experiment(task.public.argument))
state = hypothesis_manager.initialize(
    hypothesis_generator.generate(FailureTrace(task.public, actual))
)
assert len(state.hypotheses) == 2
```

Provider 还提供 choose(state, options) 和 free_probe(task, failure, feedback)；返回候选实验或在自由模式返回 None 转入修复。预算不足时 Actions.call 在执行前抛 BudgetExhausted，由策略循环转换为停止原因。任务清单错误、未知候选与未实现的适配器显式报错，应由批处理层记录为运行失败。共享字段变更须同时更新此表和调用方的集成测试。
