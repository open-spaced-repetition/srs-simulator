from __future__ import annotations

# ruff: noqa: E402
# pyright: reportPrivateImportUsage=false

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
import os
from pathlib import Path
import sys
import time

import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("MPLBACKEND", "Agg")

from experiments.single_card_tradeoff.core.config import (  # noqa: E402
    SingleCardFSRS6Config,
    add_single_card_fsrs6_config_args,
    configure_oracle_dp_cache_from_args,
    load_single_card_fsrs6_config,
)
from experiments.single_card_tradeoff.oracles import (  # noqa: E402
    FSRS6AverageRewardOracle,
)
from experiments.single_card_tradeoff.oracles.dp_cache import (  # noqa: E402
    OracleDPCacheConfig,
)
from experiments.single_card_tradeoff.core.defaults import (  # noqa: E402
    DEFAULT_SCALARIZATION_TRAIN_COST_WEIGHTS,
    DEFAULT_TARGET_RETENTIONS,
)
from experiments.single_card_tradeoff.core.retention_space import (  # noqa: E402
    validate_retention_values,
)
from experiments.single_card_tradeoff.cli.uvfa_ppo import (  # noqa: E402
    DEFAULT_LEARNING_RATE,
    DEFAULT_MAX_GRAD_NORM,
    DEFAULT_NETWORK,
    DEFAULT_ORACLE_D_GRID_SIZE,
    DEFAULT_ORACLE_S_GRID_SIZE,
    DEFAULT_TRAIN_ENVS,
    FSRS6SingleCardBatch,
    PolicyValueNet,
    evaluate_policy,
    fsrs_config_kwargs,
    parse_csv_floats,
    row_from_metrics,
    scalar_objective,
    write_csv,
)
from simulator.defaults import DEFAULT_DAYS, DEFAULT_DECK_SIZE, DEFAULT_SEED  # noqa: E402
from simulator.scheduler_spec import format_float  # noqa: E402

DEFAULT_DISTILL_EPOCHS = 64
DEFAULT_STEPS_PER_EPOCH = 64
DEFAULT_EVAL_PARTICLES = 10_000
DEFAULT_AGREEMENT_ENVS = 4096
DEFAULT_AGREEMENT_STEPS = 256
DEFAULT_DISTILL_OBS_MODE = "oracle_stationary"
DEFAULT_DISTILL_HIDDEN_SIZE = 16
DEFAULT_DISTILL_NETWORK_DEPTH = 2
DEFAULT_INFINITE_MAX_ITERATIONS = 128
DEFAULT_INFINITE_TOLERANCE = 1e-10


@dataclass(frozen=True)
class DistillStats:
    epochs: int
    steps_per_epoch: int
    transitions: int
    runtime_s: float
    final_loss: float
    final_action_agreement: float


class AverageRewardOracleGuide:
    def __init__(
        self,
        *,
        cost_weights: Sequence[float],
        action_retentions: Sequence[float],
        s_grid_size: int,
        d_grid_size: int,
        device: torch.device,
        max_iterations: int,
        tolerance: float,
        progress: bool,
        fsrs_config: SingleCardFSRS6Config | None = None,
        cache_config: OracleDPCacheConfig | None = None,
    ) -> None:
        self.device = device
        self.cost_weights = torch.tensor(
            list(cost_weights), device=device, dtype=torch.float32
        )
        self.oracle = FSRS6AverageRewardOracle(
            action_retentions=action_retentions,
            s_grid_size=s_grid_size,
            d_grid_size=d_grid_size,
            device=device,
            cache_config=cache_config,
            **fsrs_config_kwargs(fsrs_config),
        )
        solution = self.oracle.solve_average_reward_policies(
            cost_weights,
            max_iterations=max_iterations,
            tolerance=tolerance,
            progress=progress,
        )
        if not all(solution.converged):
            failed = [
                format_float(weight)
                for weight, converged in zip(
                    cost_weights,
                    solution.converged,
                    strict=True,
                )
                if not converged
            ]
            raise RuntimeError(
                "Average-reward oracle did not converge for cost weights: "
                + ",".join(failed)
            )
        self.policy = solution.policy.to(device=device)
        self.gains = solution.gains.to(device=device)
        self.iterations = solution.iterations
        self.residuals = solution.residuals

    def labels(self, env: FSRS6SingleCardBatch) -> torch.Tensor:
        return self.oracle.labels(
            policies=self.policy,
            cost_weights=self.cost_weights,
            s=env.s,
            d=env.d,
            goal_weight=env.goal_weight,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Distill the infinite-horizon average-reward FSRS-6 oracle into a "
            "UVFA policy."
        ),
        allow_abbrev=False,
    )
    add_single_card_fsrs6_config_args(parser)
    parser.add_argument("--days", type=int, default=DEFAULT_DAYS)
    parser.add_argument("--deck-scale", type=int, default=DEFAULT_DECK_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--torch-device", default=None)
    parser.add_argument(
        "--cost-weights",
        default=",".join(
            format_float(value) for value in DEFAULT_SCALARIZATION_TRAIN_COST_WEIGHTS
        ),
        help="Comma-separated scalarization weights for the oracle teacher.",
    )
    parser.add_argument(
        "--action-retentions",
        default=",".join(format_float(value) for value in DEFAULT_TARGET_RETENTIONS),
        help="Discrete desired-retention actions available to the oracle teacher.",
    )
    parser.add_argument("--train-envs", type=int, default=DEFAULT_TRAIN_ENVS)
    parser.add_argument("--epochs", type=int, default=DEFAULT_DISTILL_EPOCHS)
    parser.add_argument("--steps-per-epoch", type=int, default=DEFAULT_STEPS_PER_EPOCH)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument(
        "--obs-mode",
        choices=[
            "oracle_stationary",
            "oracle",
            "oracle_rho",
            "oracle_rho4",
            "oracle_rho3",
            "rich",
            "basic",
        ],
        default=DEFAULT_DISTILL_OBS_MODE,
    )
    parser.add_argument(
        "--network",
        choices=["mlp", "residual"],
        default=DEFAULT_NETWORK,
    )
    parser.add_argument(
        "--network-depth",
        type=int,
        default=DEFAULT_DISTILL_NETWORK_DEPTH,
    )
    parser.add_argument("--hidden-size", type=int, default=DEFAULT_DISTILL_HIDDEN_SIZE)
    parser.add_argument(
        "--oracle-s-grid-size",
        type=int,
        default=DEFAULT_ORACLE_S_GRID_SIZE,
    )
    parser.add_argument(
        "--oracle-d-grid-size",
        type=int,
        default=DEFAULT_ORACLE_D_GRID_SIZE,
    )
    parser.add_argument(
        "--oracle-infinite-max-iterations",
        type=int,
        default=DEFAULT_INFINITE_MAX_ITERATIONS,
    )
    parser.add_argument(
        "--oracle-infinite-tolerance",
        type=float,
        default=DEFAULT_INFINITE_TOLERANCE,
    )
    parser.add_argument("--max-grad-norm", type=float, default=DEFAULT_MAX_GRAD_NORM)
    parser.add_argument("--eval-particles", type=int, default=DEFAULT_EVAL_PARTICLES)
    parser.add_argument("--agreement-envs", type=int, default=DEFAULT_AGREEMENT_ENVS)
    parser.add_argument("--agreement-steps", type=int, default=DEFAULT_AGREEMENT_STEPS)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(
            "artifacts/single_card_tradeoff/fsrs6_oracle_infinite_distill_results.csv"
        ),
    )
    parser.add_argument(
        "--model-out",
        type=Path,
        default=Path(
            "artifacts/single_card_tradeoff/fsrs6_oracle_infinite_distill_policy.pt"
        ),
    )
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def resolve_torch_device(raw: str | None) -> torch.device:
    if raw:
        return torch.device(raw)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _build_env(
    args: argparse.Namespace,
    *,
    device: torch.device,
    cost_weights: Sequence[float],
    action_retentions: Sequence[float],
    env_count: int,
    seed: int,
    fsrs_config: SingleCardFSRS6Config | None,
) -> FSRS6SingleCardBatch:
    return FSRS6SingleCardBatch(
        days=args.days,
        env_count=env_count,
        cost_weights=cost_weights,
        action_retentions=action_retentions,
        device=device,
        dtype=torch.float32,
        seed=seed,
        exact_memory=False,
        goal_norm_max=max(cost_weights),
        obs_mode=args.obs_mode,
        **fsrs_config_kwargs(fsrs_config),
    )


def train_distilled_policy(
    args: argparse.Namespace,
    *,
    device: torch.device,
    cost_weights: Sequence[float],
    action_retentions: Sequence[float],
    fsrs_config: SingleCardFSRS6Config | None,
    cache_config: OracleDPCacheConfig | None = None,
) -> tuple[PolicyValueNet, AverageRewardOracleGuide, DistillStats]:
    torch.manual_seed(args.seed)
    start = time.perf_counter()
    env = _build_env(
        args,
        device=device,
        cost_weights=cost_weights,
        action_retentions=action_retentions,
        env_count=args.train_envs,
        seed=args.seed,
        fsrs_config=fsrs_config,
    )
    model = PolicyValueNet(
        env.obs_dim,
        env.action_count,
        args.hidden_size,
        architecture=args.network,
        depth=args.network_depth,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, eps=1e-5)
    guide = AverageRewardOracleGuide(
        cost_weights=cost_weights,
        action_retentions=action_retentions,
        s_grid_size=args.oracle_s_grid_size,
        d_grid_size=args.oracle_d_grid_size,
        device=device,
        max_iterations=args.oracle_infinite_max_iterations,
        tolerance=args.oracle_infinite_tolerance,
        progress=not args.no_progress,
        fsrs_config=fsrs_config,
        cache_config=cache_config,
    )

    obs = env.obs()
    final_loss = 0.0
    final_action_agreement = 0.0
    for epoch in range(args.epochs):
        loss_sum = 0.0
        correct = 0
        total = 0
        for _ in range(args.steps_per_epoch):
            label = guide.labels(env)
            logits, _ = model(obs)
            loss = nn.functional.cross_entropy(logits, label)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()

            with torch.no_grad():
                pred = torch.argmax(logits, dim=1)
                batch_total = int(label.numel())
                correct += int((pred == label).sum().item())
                total += batch_total
                loss_sum += float(loss.item()) * batch_total
                next_obs, _, done = env.step(label)
                if done.any():
                    env.reset_indices(done.nonzero(as_tuple=False).squeeze(1))
                    next_obs = env.obs()
                obs = next_obs

        final_loss = loss_sum / float(max(1, total))
        final_action_agreement = correct / float(max(1, total))
        if not args.no_progress:
            print(
                f"epoch={epoch + 1}/{args.epochs} "
                f"ce={final_loss:.5f} "
                f"teacher_action_agreement={final_action_agreement:.4f}",
                flush=True,
            )

    runtime_s = time.perf_counter() - start
    return (
        model,
        guide,
        DistillStats(
            epochs=args.epochs,
            steps_per_epoch=args.steps_per_epoch,
            transitions=args.epochs * args.steps_per_epoch * args.train_envs,
            runtime_s=runtime_s,
            final_loss=final_loss,
            final_action_agreement=final_action_agreement,
        ),
    )


@torch.inference_mode()
def estimate_teacher_action_agreement(
    model: PolicyValueNet,
    guide: AverageRewardOracleGuide,
    *,
    args: argparse.Namespace,
    device: torch.device,
    cost_weights: Sequence[float],
    action_retentions: Sequence[float],
    fsrs_config: SingleCardFSRS6Config | None,
) -> float:
    if args.agreement_steps <= 0:
        return 0.0
    env = _build_env(
        args,
        device=device,
        cost_weights=cost_weights,
        action_retentions=action_retentions,
        env_count=args.agreement_envs,
        seed=args.seed + 60_000,
        fsrs_config=fsrs_config,
    )
    correct = 0
    total = 0
    model.eval()
    obs = env.obs()
    for _ in range(args.agreement_steps):
        label = guide.labels(env)
        logits, _ = model(obs)
        pred = torch.argmax(logits, dim=1)
        correct += int((pred == label).sum().item())
        total += int(label.numel())
        next_obs, _, done = env.step(label)
        if done.any():
            env.reset_indices(done.nonzero(as_tuple=False).squeeze(1))
            next_obs = env.obs()
        obs = next_obs
    return correct / float(max(1, total))


def save_model(
    path: Path,
    *,
    model: PolicyValueNet,
    guide: AverageRewardOracleGuide,
    args: argparse.Namespace,
    cost_weights: Sequence[float],
    action_retentions: Sequence[float],
    stats: DistillStats,
    eval_teacher_action_agreement: float,
    fsrs_config: SingleCardFSRS6Config,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "policy_type": "fsrs6_oracle_infinite_distill",
            "model_state_dict": model.state_dict(),
            "cost_weights": list(cost_weights),
            "action_retentions": list(action_retentions),
            "days": args.days,
            "obs_dim": model.obs_dim,
            "obs_mode": args.obs_mode,
            "hidden_size": args.hidden_size,
            "network": args.network,
            "network_depth": args.network_depth,
            "guide_policy": "average_reward_oracle",
            "oracle_s_grid_size": args.oracle_s_grid_size,
            "oracle_d_grid_size": args.oracle_d_grid_size,
            "oracle_infinite_max_iterations": args.oracle_infinite_max_iterations,
            "oracle_infinite_tolerance": args.oracle_infinite_tolerance,
            "oracle_infinite_gains": [float(value) for value in guide.gains.tolist()],
            "oracle_infinite_iterations": list(guide.iterations),
            "oracle_infinite_residuals": list(guide.residuals),
            **fsrs_config.checkpoint_payload(),
            "distill_epochs": stats.epochs,
            "distill_steps_per_epoch": stats.steps_per_epoch,
            "train_updates": 0,
            "train_transitions": stats.transitions,
            "train_runtime_s": stats.runtime_s,
            "final_ce_loss": stats.final_loss,
            "final_teacher_action_agreement": stats.final_action_agreement,
            "eval_teacher_action_agreement": eval_teacher_action_agreement,
        },
        path,
    )


def main() -> None:
    args = parse_args()
    cache_config = configure_oracle_dp_cache_from_args(args)
    if args.days <= 1:
        raise SystemExit("--days must be > 1.")
    if args.deck_scale <= 0:
        raise SystemExit("--deck-scale must be > 0.")
    if args.train_envs <= 0 or args.eval_particles <= 0:
        raise SystemExit("--train-envs and --eval-particles must be > 0.")
    if args.epochs < 0:
        raise SystemExit("--epochs must be >= 0.")
    if args.steps_per_epoch <= 0:
        raise SystemExit("--steps-per-epoch must be > 0.")
    if args.network_depth <= 0:
        raise SystemExit("--network-depth must be > 0.")
    if args.oracle_s_grid_size < 8 or args.oracle_d_grid_size < 8:
        raise SystemExit("--oracle grid sizes must be >= 8.")
    if args.oracle_infinite_max_iterations <= 0:
        raise SystemExit("--oracle-infinite-max-iterations must be > 0.")
    if args.oracle_infinite_tolerance <= 0.0:
        raise SystemExit("--oracle-infinite-tolerance must be > 0.")
    if args.agreement_envs <= 0:
        raise SystemExit("--agreement-envs must be > 0.")
    if args.agreement_steps < 0:
        raise SystemExit("--agreement-steps must be >= 0.")

    device = resolve_torch_device(args.torch_device)
    cost_weights = parse_csv_floats(args.cost_weights, name="--cost-weights")
    if any(value < 0.0 for value in cost_weights):
        raise SystemExit("--cost-weights must be >= 0.")
    action_retentions = parse_csv_floats(
        args.action_retentions,
        name="--action-retentions",
    )
    validate_retention_values(action_retentions, name="--action-retentions")
    fsrs_config = load_single_card_fsrs6_config(args)

    model, guide, stats = train_distilled_policy(
        args,
        device=device,
        cost_weights=cost_weights,
        action_retentions=action_retentions,
        fsrs_config=fsrs_config,
        cache_config=cache_config,
    )
    agreement = estimate_teacher_action_agreement(
        model,
        guide,
        args=args,
        device=device,
        cost_weights=cost_weights,
        action_retentions=action_retentions,
        fsrs_config=fsrs_config,
    )
    save_model(
        args.model_out,
        model=model,
        guide=guide,
        args=args,
        cost_weights=cost_weights,
        action_retentions=action_retentions,
        stats=stats,
        eval_teacher_action_agreement=agreement,
        fsrs_config=fsrs_config,
    )

    rows = []
    for cost_weight in cost_weights:
        start = time.perf_counter()
        metrics = evaluate_policy(
            model,
            args=args,
            device=device,
            cost_weight=cost_weight,
            action_retentions=action_retentions,
            particles=args.eval_particles,
            seed=args.seed + 70_000 + int(round(cost_weight * 10.0)),
            goal_norm_max=max(cost_weights),
            obs_mode=args.obs_mode,
            fsrs_config=fsrs_config,
        )
        runtime_s = time.perf_counter() - start
        rows.append(
            row_from_metrics(
                args=args,
                scheduler="fsrs6_oracle_infinite_distill",
                scheduler_spec="fsrs6_oracle_infinite_distill",
                goal_cost_weight=cost_weight,
                metrics=metrics,
                particles=args.eval_particles,
                seed=args.seed,
                runtime_s=runtime_s,
                scalar=scalar_objective(metrics, cost_weight),
            )
        )

    write_csv(args.out, rows)
    print(f"Wrote CSV: {args.out}")
    print(f"Wrote model: {args.model_out}")
    print(
        f"Distill: epochs={stats.epochs} steps_per_epoch={stats.steps_per_epoch} "
        f"transitions={stats.transitions} runtime_s={stats.runtime_s:.2f} "
        f"device={device} "
        f"final_ce={stats.final_loss:.5f} train_agreement="
        f"{stats.final_action_agreement:.4f} eval_agreement={agreement:.4f}"
    )


if __name__ == "__main__":
    main()
