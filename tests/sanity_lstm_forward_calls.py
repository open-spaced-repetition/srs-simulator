from pathlib import Path

from simulator.behavior import StochasticBehavior
from simulator.benchmark_loader import load_benchmark_weights
from simulator.cost import StatefulCostModel
from simulator.models.lstm import LSTMModel
from simulator.schedulers import FSRS6Scheduler
from simulator.vectorized import simulate_lstm_vectorized


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    weights = load_benchmark_weights(
        repo_root=repo_root,
        benchmark_root=None,
        environment="fsrs6",
        user_id=1,
        partition_key="0",
        overrides=None,
    )

    scheduler = FSRS6Scheduler(
        weights=weights, desired_retention=0.9, priority_mode="low_retrievability"
    )
    env = LSTMModel(user_id=1, benchmark_root=None, device="cpu")
    behavior = StochasticBehavior()
    cost_model = StatefulCostModel()

    env.reset_forward_calls()
    stats = simulate_lstm_vectorized(
        days=60,
        deck_size=500,
        environment=env,
        scheduler=scheduler,
        behavior=behavior,
        cost_model=cost_model,
        seed=0,
        device="cpu",
        progress=False,
    )

    print("total_reviews", stats.total_reviews)
    print("forward_calls", env.forward_calls)
    print("forward_calls_per_day", env.forward_calls / 60)


if __name__ == "__main__":
    main()
