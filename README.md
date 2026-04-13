# ShiftGuard-MoE Minimal Experiment

这是一个最小可运行的 MoE workload shift 实验脚手架，目标是：

- 使用 `deepseek-ai/DeepSeek-V2-Lite-Chat` 做真实推理前向。
- 抓取每个 `DeepSeekV2Moe` 层的 `top-k expert` 路由序列。
- 用这些真实路由离线重放 `LRU` 和 `LFU` expert cache。
- 在 `stable / block_shift / medium_shift / severe_shift` 四种 workload 下比较命中率变化，验证 workload shift 时传统策略会退化。

## 目录

- `run_shift_experiment.py`: 一条命令启动实验。
- `shiftguard_moe/runner.py`: DeepSeek 路由采样器，支持真实模型和 `tiny-random` 冒烟模式。
- `shiftguard_moe/policies.py`: LRU / LFU 与分层 cache simulator。
- `shiftguard_moe/workload.py`: 内置多领域 prompt bank 和 workload shift 生成器。
- `shiftguard_moe/experiment.py`: 实验编排、结果汇总和图表导出。

## 当前设计

1. 先对唯一 prompt 集合执行一次模型前向，拿到每个 prompt 的 expert 访问序列。
2. 再把这些 trace 按不同 workload 顺序重放，避免重复跑同一条 prompt。
3. cache 是按层独立模拟的，因为不同层的 expert 权重不是同一份参数。
4. `stable` 是无漂移基线；`block_shift` 表示纯分块领域切换；`medium_shift` 和 `severe_shift` 逐步增加窗口内混乱程度。

## 运行方式

### 1. 本地快速冒烟

这不会下载真实模型，会构造一个很小的随机 DeepSeekV2 模型来验证整条 pipeline。

```bash
python run_shift_experiment.py ^
  --mode tiny-random ^
  --output-dir outputs/tiny_smoke ^
  --cache-capacity 4 ^
  --prompts-per-domain 4 ^
  --num-windows 4 ^
  --window-size 4
```

### 2. 真实 DeepSeek-V2-Lite-Chat

建议在有 GPU 的环境运行。

```bash
python run_shift_experiment.py ^
  --mode real ^
  --model-id deepseek-ai/DeepSeek-V2-Lite-Chat ^
  --device-map auto ^
  --dtype bfloat16 ^
  --output-dir outputs/deepseek_real ^
  --cache-capacity 8 ^
  --prompts-per-domain 8 ^
  --num-windows 6 ^
  --window-size 6
```

默认 `--max-new-tokens 0`，表示只做 prompt prefill 前向，这已经足够用于采样路由并分析 expert cache。若你想顺带看看生成阶段，也可以设置一个小值，例如 `--max-new-tokens 8`。

## 自定义数据集

你可以传入自己的 JSONL，每行格式如下：

```json
{"domain": "code", "text": "请分析一个 Python 推理服务的队列抖动问题。"}
{"domain": "math", "text": "请证明一个递推序列单调收敛。"}
```

运行时加上：

```bash
python run_shift_experiment.py ^
  --mode real ^
  --dataset-path path/to/custom_prompts.jsonl ^
  --output-dir outputs/custom_run
```

如果你想完全固定 workload 的时序，而不是运行时随机采样，再加上：

```bash
python run_shift_experiment.py ^
  --mode real ^
  --dataset-path data/custom_prompts.jsonl ^
  --workload-plan-path data/custom_workload_plan.json ^
  --output-dir outputs/custom_plan_run
```

要求：

- 至少包含两个 `domain`。
- 同一 `domain` 下最好有多条 prompt，这样窗口切换更稳定。
- 如果你想复现实验，建议保留固定 `--seed`。

## 输出文件

运行完成后，`output-dir` 下会生成：

- `prompt_bank.jsonl`: 唯一 prompt 集合。
- `prompt_traces.json`: 每条 prompt 的 expert 路由访问序列。
- `workloads.json`: 不同场景的 prompt 顺序。
- `prompt_metrics.csv`: 每个 prompt 位置的命中率。
- `prompt_hit_rates.png`: 按 prompt 位置绘制的滚动命中率曲线。
- `window_hit_rates.png`: shift 后平均命中率对比图。
- `summary.json`: 聚合统计。

## 分析脚本

实验输出目录生成后，可以再运行分析脚本，基于 `summary.json` 和 `prompt_metrics.csv` 重新绘图并生成一份 Markdown 摘要：

```bash
python analyze_experiment_outputs.py ^
  --input-dir outputs/custom_plan_run
```

默认会把分析结果写到 `outputs/custom_plan_run/analysis/`，包括：

- `scenario_summary.csv`: 便于后续表格分析的聚合结果。
- `scenario_total_hit_rate.png`: 不同场景下 LRU/LFU 的总命中率对比。
- `scenario_drop_vs_stable.png`: 每个场景相对 stable 的退化幅度。
- `window_trends_by_policy.png`: 按 window 观察不同场景的命中率演化。
- `prompt_rolling_hit_rate.png`: 按 prompt 位置绘制滚动命中率。
- `analysis_report.md`: 自动生成的实验结论摘要。

`summary.json` 里最值得看的是：

- `total_hit_rate`
- `first_window_hit_rate`
- `post_shift_hit_rate`
- `drop_vs_stable`

如果 `block_shift / medium_shift / severe_shift` 下 `drop_vs_stable` 明显为正，同时 `post_shift_hit_rate` 持续下降，就说明 workload shift 确实让传统 cache 策略退化了。

## 说明

- 当前环境只有 CPU，所以这里默认把 CLI 的默认模式设成了 `tiny-random`，方便直接冒烟。
- 真实 `DeepSeek-V2-Lite-Chat` 权重较大，建议在有显存的机器上运行。
- 这个实现是“最小系统”，重点在可验证 workload shift 对 expert cache 的影响，而不是做完整 serving 系统。
- `data/custom_workload_plan.json` 里提供了一套更贴近“验证传统策略失效”的固定场景：
  - `stable`: 长时间单领域，作为基线。
  - `lru_break`: `code -> math -> medicine -> creative -> code` 的阶段切换，用来冲掉近期性。
  - `lfu_break`: 先长时间 `code` 预热，再切到其他领域，用来放大旧热点计数对 `LFU` 的拖累。
  - `mixed_shift`: 窗口内显式混合多领域，逐步提高混乱度。
# ShiftGuard-MoE
