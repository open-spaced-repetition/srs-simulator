from __future__ import annotations

from pathlib import Path

from simulator.scheduler_catalog import PolicySource, schedulers_for_policy_source

MIN_TARGET_RETENTION = 0.50
DEFAULT_TARGET_RETENTIONS = [
    0.50,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
    0.85,
    0.90,
    0.93,
    0.96,
    0.98,
]

DEFAULT_FIXED_INTERVALS = [8, 16, 32, 64, 128, 256, 512]
DEFAULT_SCALARIZATION_TRAIN_COST_WEIGHTS = [
    0,
    1,
    2,
    4,
    8,
    16,
    32,
    64,
    128,
    256,
    512,
    1024,
]
DEFAULT_SCALARIZATION_EVAL_COST_WEIGHTS = [
    0,
    1,
    2,
    4,
    8,
    16,
    32,
    48,
    64,
    96,
    128,
    192,
    256,
    320,
    384,
    512,
    1024,
]

DEFAULT_UVFA_PPO_POLICY = Path("artifacts/single_card_tradeoff/uvfa_ppo_policy.pt")
DEFAULT_FSRS6_ORACLE_DISTILL_POLICY = Path(
    "artifacts/single_card_tradeoff/fsrs6_oracle_distill_policy.pt"
)
DEFAULT_UVFA_PPO_RNN_INTERVAL_POLICY = Path(
    "artifacts/single_card_tradeoff/uvfa_ppo_rnn_interval_policy.pt"
)
DEFAULT_FSRS6_ORACLE_INTERVAL_DISTILL_POLICY = Path(
    "artifacts/single_card_tradeoff/fsrs6_oracle_interval_distill_policy.pt"
)
DEFAULT_FSRS6_ORACLE_RETENTION_DISTILL_POLICY = Path(
    "artifacts/single_card_tradeoff/fsrs6_oracle_retention_distill_policy.pt"
)
DEFAULT_FSRS6_ORACLE_CONTINUOUS_STATIONARY_FINITE_DISTILL_POLICY = Path(
    "artifacts/single_card_tradeoff/"
    "fsrs6_oracle_continuous_stationary_finite_distill_policy.pt"
)
DEFAULT_FSRS6_ORACLE_INFINITE_DISTILL_POLICY = Path(
    "artifacts/single_card_tradeoff/fsrs6_oracle_infinite_distill_policy.pt"
)
DEFAULT_FSRS6_ORACLE_STATIONARY_FINITE_DISTILL_POLICY = Path(
    "artifacts/single_card_tradeoff/fsrs6_oracle_stationary_finite_distill_policy.pt"
)
DEFAULT_FSRS6_ADR_TRAIN_RUN_ROOT = Path(
    "artifacts/rl_scheduler/fsrs6_adr_portfolio_users_1_8/"
    "fsrs6_adr_portfolio_users_1_8_pop16_v1"
)

FSRS6_ORACLE_SCHEDULER = "fsrs6_oracle"
FSRS6_ORACLE_INFINITE_SCHEDULER = "fsrs6_oracle_infinite"
FSRS6_ORACLE_STATIONARY_FINITE_SCHEDULER = "fsrs6_oracle_stationary_finite"
FSRS6_ORACLE_DISTILL_SCHEDULER = "fsrs6_oracle_distill"
FSRS6_ORACLE_INFINITE_DISTILL_SCHEDULER = "fsrs6_oracle_infinite_distill"
FSRS6_ORACLE_STATIONARY_FINITE_DISTILL_SCHEDULER = (
    "fsrs6_oracle_stationary_finite_distill"
)
FSRS6_ORACLE_INTERVAL_SCHEDULER = "fsrs6_oracle_interval"
FSRS6_ORACLE_INTERVAL_BILINEAR_ACTION_SCHEDULER = (
    "fsrs6_oracle_interval_bilinear_action"
)
FSRS6_ORACLE_INTERVAL_DISTILL_SCHEDULER = "fsrs6_oracle_interval_distill"
FSRS6_ORACLE_RETENTION_DISTILL_SCHEDULER = "fsrs6_oracle_retention_distill"
FSRS6_ORACLE_CONTINUOUS_RETENTION_SCHEDULER = "fsrs6_oracle_continuous_retention"
FSRS6_ORACLE_CONTINUOUS_STATIONARY_FINITE_SCHEDULER = (
    "fsrs6_oracle_continuous_stationary_finite"
)
FSRS6_ORACLE_CONTINUOUS_STATIONARY_FINITE_DISTILL_SCHEDULER = (
    "fsrs6_oracle_continuous_stationary_finite_distill"
)
UVFA_PPO_SCHEDULER = "uvfa_ppo"
UVFA_PPO_RNN_INTERVAL_SCHEDULER = "uvfa_ppo_rnn_interval"
FSRS6_ADR_SCHEDULERS = frozenset(schedulers_for_policy_source(PolicySource.FSRS6_ADR))
