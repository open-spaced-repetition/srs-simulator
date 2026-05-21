  # Single-Card Tradeoff 架构重构计划

  ## 摘要

  目标是对 experiments/single_card_tradeoff 做保守分层重构：保持现有 CLI 命令、CSV 字段、checkpoint/artifact 兼容，优先打断循环依赖、抽出库层能
  力、降低 tradeoff.py / oracle_frontier.py / uvfa_ppo.py 的职责密度。

  建议将本文保存为 docs/single_card_tradeoff/plans/single_card_tradeoff_arch_refactor_plan.md。重构按小步提交推进，每步都应能独立通过 focused
  tests，避免一次性大迁移。

  ## 重构原则

  - 不改变用户可见命令：现有 uv run experiments/single_card_tradeoff/*.py ... 继续可用。
  - 不改变默认实验语义：默认 retentions、cost weights、GPU monitor、DP cache、multi-user batching 行为保持一致。
  - 不改变现有 checkpoint 和 CSV schema；新增 typed API 只作为内部更稳的表达。
  - 对外常用 import 保持兼容：例如 uvfa_ppo.PolicyValueNet、oracle_frontier.FSRS6GridOracle 先作为 re-export 保留。
  - 先抽“无行为变化”的纯搬迁模块，再改调度结构；每阶段都做回归测试。

  ## 分阶段实施

  ### 阶段 0：建立基线和防护

  - 记录当前内部依赖图，确认已知循环依赖：tradeoff、uvfa_ppo、oracle_frontier、oracle_interval_distill、oracle_retention_distill、
    uvfa_ppo_rnn_interval。
  - 增加一个轻量 import 回归测试，覆盖核心模块可以被单独 import，且不会依赖 CLI side effects。
  - 用小规模 smoke 命令保存当前 tradeoff.py 的 CSV 行为样本，作为后续重构对比基线。

  ### 阶段 1：抽出稳定常量、类型和结果工具

  - 新增共享模块：
      - defaults.py：集中 DEFAULT_TARGET_RETENTIONS、MIN_TARGET_RETENTION、固定间隔、scalarization weights、single-card scheduler 名称、默认
        artifact path。
      - types.py：集中 SimMetrics、SchedulerPoint、BatchRow、MemoryTargetRegretAucSummary 这类跨模块数据结构。
      - results.py：集中 CSV row 构造、Pareto frontier、same-target time-saved AUC、CSV 写入相关逻辑。
  - retention_space.py 只保留 retention validation，默认 retention 值从 defaults.py 导出或 re-export。
  - __init__.py 改为从 defaults.py lazy/export，避免 import 包时加载 tradeoff.py。
  - 更新 uvfa_ppo.py、oracle_frontier.py 等模块，不再从 tradeoff.py 读取常量。

  ### 阶段 2：把训练/模型库层从 CLI 中抽出

  - 新增模型库模块：
      - single_card_env.py：移动 FSRS6SingleCardBatch 和环境观测/rollout 相关纯逻辑。
      - policy_net.py：移动 PolicyValueNet、ResidualBlock、QuadraticFeatureMap、ZeroValueHead。
      - metrics.py：如未放入 types.py，集中 SimMetrics 和 scalar objective。
  - uvfa_ppo.py 保留为训练入口：参数解析、调用训练函数、写 CSV/plot/checkpoint。
  - 在 uvfa_ppo.py 中 re-export 老符号，保证现有测试和其他脚本暂时无需同步大改。
  - checkpoint 写入字段保持不变，只把 payload 构造拆成可测试函数。

  ### 阶段 3：拆分 oracle 算法和 oracle CLI

  - 新增 oracles.py 或 oracles/ 包，移动：
      - FSRS6GridOracle
      - FSRS6StationaryFiniteOracle
      - FSRS6BatchedStationaryFiniteOracle
      - FSRS6AverageRewardOracle
      - FSRS6IntervalOracle
      - oracle solution / transition cache dataclass
  - oracle_frontier.py 变成薄 CLI：parse args、构造 config、调用 oracle、写结果。
  - oracle_frontier.py 保留 re-export，兼容现有 from ...oracle_frontier import FSRS6GridOracle。
  - oracle DP cache 继续作为独立模块，但调用方逐步显式传入 cache config。

  ### 阶段 4：重构 tradeoff.py 的 scheduler 执行分发

  - 新增 evaluator registry，替代 tradeoff.py 末尾长串 if scheduler_name == ...。
  - 定义内部接口：
      - 输入：environment、scheduler spec、retention grid、user contexts、seed、args/context。
      - 输出：统一的 TradeoffRow 或可写 CSV row。
  - 分组实现 evaluator：
      - baseline/vectorized scheduler evaluator：fsrs6、fsrs3、hlr、lstm、fixed、anki_sm2、memrise。
      - oracle evaluator：finite、infinite、stationary finite、interval。
      - distilled policy evaluator：UVFA、RNN interval、oracle distill variants。
      - ADR evaluator：FSRS6 ADR policy expansion 和 batching。
  - tradeoff.py 最终只保留：CLI、参数校验、环境/user context 构造、调用 registry、写 outputs。

  ### 阶段 5：消除隐藏全局状态

  - 新增 SingleCardRuntimeContext，显式携带 torch_device、dp_cache_config、monitor 输出目录、repo root。
  - load_single_card_fsrs6_config() 改为只加载 FSRS/button usage 配置，不再设置 DP cache 全局默认。
  - CLI 入口负责从 args 构造 runtime context；需要兼容旧调用时，短期保留 wrapper，但新代码不再依赖全局默认。
  - oracle_dp_cache stats 可以继续全局计数，但 cache config 必须显式传递到 oracle 构造。

  ### 阶段 6：整理配置运行器和报告生成

  - run_tradeoff_config.py 抽出 command builder，消除 single-user 与 multi-user 命令拼装重复。
  - generate_experiment_report.py 暂不大改算法逻辑，只把读取 artifact、聚合 summary、render Markdown 分成小函数/模块。
  - 所有报告脚本继续读取现有 artifact 路径，不迁移历史产物。

  ## 公共接口和兼容性

  - CLI：所有现有命令、参数名、默认值保持不变。
  - CSV：results.csv、regret_auc.csv、summary CSV 字段保持不变；新增 typed row 只在内部使用。
  - Checkpoint：policy_type、model_state_dict、cost_weights、action_retentions、FSRS config payload 字段保持不变。
  - Imports：旧路径至少保留一个重构周期，并在模块顶部通过 re-export 兼容。
  - README：只有当命令、默认路径或示例行为变化时才更新；本计划默认不触发 README 示例变更。

  ## 测试计划

  - 每阶段运行 focused tests：
      - uv run python -m unittest tests.test_single_card_tradeoff_multiuser
      - uv run python -m unittest tests.test_oracle_dp_cache
      - uv run python -m unittest tests.test_oracle_interval_oracle
      - uv run python -m unittest tests.test_single_card_policy_value_net
  - 增加测试：
      - shared defaults 不 import tradeoff.py。
      - 老 import 路径 re-export 与新模块对象一致。
      - evaluator registry 对每类 scheduler 选择正确 evaluator。
      - AUC/CSV 工具搬迁前后输出一致。
  - Smoke 回归：
      - 小规模 tradeoff.py --env fsrs6_default --sched fsrs6_default,fixed --days 30 --particles 16 --target-retentions 0.5 --fixed-intervals 8
        --no-plot --no-regret-auc。

  - 内部 import 图不再存在核心循环依赖。
  - tradeoff.py 明显瘦身，主要职责变成 CLI 编排，而不是实现所有 scheduler 分支。
  - oracle_frontier.py 和 uvfa_ppo.py 可作为薄入口使用，核心类从库模块导入。