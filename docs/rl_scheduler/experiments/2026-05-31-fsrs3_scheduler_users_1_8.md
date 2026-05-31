# FSRS3 Scheduler vs FSRS6 Baseline 实验报告

Date: 2026-05-31

## 问题

从间隔重复算法的角度看，本项目研究的是：在给定用户记忆模型、每日新卡/复习上限和时间成本约束下，调度器如何在“长期记住更多卡片”和“花更少复习时间”之间取得 Pareto 最优折中。

RL Scheduler 实验把这个问题形式化为调度策略搜索和外部评估：训练或选择一组候选策略，然后在统一仿真环境中与 FSRS 基线的 desired-retention frontier 对比，主要看 scheduler-only hypervolume、same-budget memory lift AUC 和 same-target time saved AUC。

本次实验回答一个更窄的问题：如果 FSRSv3 也使用与 FSRS6 baseline 相同的 CMA-ES 预算，为每个用户独立选择最大化 2D hypervolume 的 16 个 desired-retention 点，它能否在 FSRS6 和 LSTM 两个评估环境中超过 FSRS6 baseline frontier？

## Runs

| item | value |
| --- | --- |
| config | `experiments/rl_scheduler/configs/fsrs3_scheduler_users_1_8.toml` |
| run root | `artifacts/rl_scheduler/fsrs3_scheduler_users_1_8/fsrs3_scheduler_users_1_8_v1` |
| scheduler under test | `fsrs3` |
| baseline scheduler | `fsrs6` |
| users | 1-8 |
| evaluation envs | `fsrs6`, `lstm` |
| simulation horizon | 1825 days |
| engine/device | batched CUDA |
| deck / learn limit / review limit | 10000 / 10 / 9999 |
| daily cost limit | 720 minutes |
| behavior priority | `new-first` |
| scheduler priority | `low_retrievability` |
| seed | 42 |

Analysis summary:

- `artifacts/rl_scheduler/fsrs3_scheduler_users_1_8/fsrs3_scheduler_users_1_8_v1/analyze-pareto/analyze_pareto_outputs/analysis_summary.json`
- `artifacts/rl_scheduler/fsrs3_scheduler_users_1_8/fsrs3_scheduler_users_1_8_v1/analyze-pareto/analyze_pareto_outputs/analysis.md`

## Baselines And DR Selection

FSRS6 baseline 当前不是在这个实验里重新训练记忆模型。FSRS6 的模型参数来自 `../srs-benchmark/result/FSRS-6-recency.jsonl`；本实验使用已存在的 FSRS6 16-DR baseline manifest：

`artifacts/rl_scheduler/baseline_dr_selection/fsrs6_users_1_8_16dr_pop16_gen5.json`

该 manifest 的 16 个 per-user desired-retention 点由 CMA-ES 在 `fsrs6` selection environment 中选择，目标是 2D hypervolume，维度为 `memorized_average` 和 `negative_time_average`。预算为 population 16、generations 5、`max_lanes_per_batch = 8192`，起点为 `uniform_anchor`。FSRS6 的 anchor HV sum 为 `3,655,082.46`，selected HV sum 为 `3,687,030.14`，增益为 `31,947.67`，即 `+0.874%`。

为了公平比较，本次为 FSRSv3 使用相同的选择流程和预算生成候选 DR manifest：

`artifacts/rl_scheduler/baseline_dr_selection/fsrs3_users_1_8_16dr_pop16_gen5.json`

FSRS3 的记忆模型参数来自 `../srs-benchmark/result/FSRSv3.jsonl`。FSRS3 DR selection 同样在 `fsrs6` selection environment 中优化 2D hypervolume，target count 为 16，retention range 为 `0.50..0.98`，CMA-ES population 16、generations 5、`max_lanes_per_batch = 8192`。FSRS3 的 anchor HV sum 为 `1,398,088.09`，selected HV sum 为 `1,432,591.71`，增益为 `34,503.62`，即 `+2.468%`。

这意味着本实验里的“训练”主要是 desired-retention portfolio selection，而不是重新拟合 FSRS3/FSRS6 的记忆模型参数。

## Stage Status

| stage | passed | notes |
| --- | --- | --- |
| preflight | yes | CUDA available |
| stage-baseline | yes | copied/staged FSRS6 baseline records |
| sweep | yes | 256 lanes, 256 logs, 40.2 s, 363.0 user-days/s |
| build-pareto | yes | 8 result files, 8 plots |
| analyze-pareto | yes | comparison `fsrs3:fsrs6` |

## Results

Primary metric is scheduler-only hypervolume delta against the staged FSRS6 baseline frontier. Positive HV delta is better. Same-budget memory lift AUC integrates memorized-card lift over the common covered time-budget interval. Same-target time saved AUC integrates saved time over the common covered memory-target interval; positive relative time-save AUC means the scheduler reaches the same memory targets faster than FSRS6 on the common covered span.

| env | HV delta sum | HV / baseline | frontier points | memory lift AUC | relative memory lift AUC | memory coverage | time saved AUC | relative time saved AUC | target coverage |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- |
| fsrs6 | -77,933.61 | -2.521% | 116 | 21.01 | +0.337% | 77/112, 42.079% span | 1.33 | +3.059% | 77/112, 72.280% span |
| lstm | -196,832.07 | -6.113% | 117 | -22.04 | -0.344% | 81/114, 36.408% span | -4.30 | -31.340% | 80/114, 72.634% span |

Per-user hypervolume deltas:

| user | fsrs6 env HV delta | lstm env HV delta |
| ---: | ---: | ---: |
| 1 | -2,325.62 | 866.86 |
| 2 | -79,719.07 | -176,897.07 |
| 3 | -1,160.44 | -2,788.82 |
| 4 | 2,465.71 | -5,483.09 |
| 5 | 4,538.92 | -165.93 |
| 6 | 556.08 | -11,475.92 |
| 7 | -682.86 | -882.37 |
| 8 | -1,606.34 | -5.73 |

## Interpretation

在相同 DR-selection budget 下，FSRS3 没有超过 FSRS6 baseline 的 primary Pareto hypervolume。FSRS6 environment 中 FSRS3 的 HV delta 为 `-2.521%`；LSTM environment 中为 `-6.113%`，外部泛化更弱。

FSRS6 environment 的 common-span AUC 诊断显示一个局部优势：FSRS3 的 same-budget memory lift AUC 为 `+21.01`，relative memory lift AUC 为 `+0.337%`；same-target time saved AUC 为 `+1.33`，relative time saved AUC 为 `+3.059%`。但该优势只覆盖 77/112 个 common targets，且不足以抵消整体 frontier HV 的损失。

LSTM environment 中三个主要比较都不支持 FSRS3：HV delta 为负，relative memory lift AUC 为 `-0.344%`，relative time saved AUC 为 `-31.340%`。这说明 FSRS3 在 FSRS6 selection environment 里优化出的 16 个 DR 点没有在 LSTM 外部记忆环境中保持优势。

同用户同 DR 的 pairwise delta 为 0 对，因为 FSRS3 和 FSRS6 的 16 个 DR 是各自独立优化得到的 per-user sets，不共享完全相同的 desired-retention 网格。因此本实验应以 frontier-level 指标为准，而不是同 DR 点比较。

## GPU Monitor

Formal sweep wrote GPU monitor artifacts:

`artifacts/rl_scheduler/fsrs3_scheduler_users_1_8/fsrs3_scheduler_users_1_8_v1/sweep/gpu_monitor/summary.json`

| metric | value |
| --- | ---: |
| shared-memory spill detected | false |
| shared-memory peak summed bytes | 167,112,704 |
| shared-memory peak single adapter bytes | 148,008,960 |
| `nvidia-smi` peak memory used | 3,035 MiB |
| samples | 11 |

The monitor does not indicate VRAM spill.

## Checks

These checks passed after the implementation and experiment updates:

```bash
uv run ruff format
uv run python -m unittest tests.test_select_fsrs6_baseline_drs tests.test_batched_sweep_config tests.test_rl_scheduler_unified_workflow_config
uv run pyright
```

The run inspection also passed:

```bash
uv run python experiments/rl_scheduler/inspect_run.py \
  --run-root artifacts/rl_scheduler/fsrs3_scheduler_users_1_8/fsrs3_scheduler_users_1_8_v1
```

## Limitations

This report covers users 1-8 only. The FSRS3 and FSRS6 memory-model weights are imported from `srs-benchmark`; this experiment does not retrain those weights. FSRS3 DR selection is optimized in the FSRS6 environment, while LSTM is an external-generalization evaluation. The AUC diagnostics are computed on common covered frontier spans, so their sign and magnitude should be interpreted together with coverage and primary HV.
